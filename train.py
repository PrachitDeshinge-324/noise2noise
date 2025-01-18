# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import tensorflow as tf
import numpy as np
import os
import dnnlib
import dnnlib.tflib as tflib
from dnnlib.tflib.autosummary import autosummary
import dnnlib.tflib.tfutil as tfutil
import dnnlib.util as util

import config

from util import save_image, save_snapshot
from validation import ValidationSet
from dataset import create_dataset

class AugmentGaussian:
    def __init__(self, validation_stddev, train_stddev_rng_range):
        self.validation_stddev = validation_stddev
        self.train_stddev_range = train_stddev_rng_range

    def add_train_noise_tf(self, x):
        (minval, maxval) = self.train_stddev_range
        shape = tf.shape(x)
        rng_stddev = tf.random.uniform(
            shape=[1, 1, 1], 
            minval=minval/255.0, 
            maxval=maxval/255.0
        )
        return x + tf.random.normal(shape) * rng_stddev

    def add_validation_noise_np(self, x):
        return x + np.random.normal(size=x.shape)*(self.validation_stddev/255.0)

class AugmentPoisson:
    def __init__(self, lam_max):
        self.lam_max = lam_max

    def add_train_noise_tf(self, x):
        chi_rng = tf.random.uniform(
            shape=[1, 1, 1], 
            minval=0.001, 
            maxval=self.lam_max
        )
        return tf.random.poisson(
            lam=chi_rng*(x+0.5), 
            shape=tf.shape(x),
            dtype=x.dtype
        )/chi_rng - 0.5

    def add_validation_noise_np(self, x):
        chi = 30.0
        return np.random.poisson(chi*(x+0.5))/chi - 0.5

def compute_ramped_down_lrate(i, iteration_count, ramp_down_perc, learning_rate):
    ramp_down_start_iter = iteration_count * (1 - ramp_down_perc)
    if i >= ramp_down_start_iter:
        t = ((i - ramp_down_start_iter) / ramp_down_perc) / iteration_count
        smooth = (0.5+np.cos(t * np.pi)/2)**2
        return learning_rate * smooth
    return learning_rate

class Trainer(tf.keras.Model):
    def __init__(self, network, noise2noise=True):
        super().__init__()
        self.network = network
        self.noise2noise = noise2noise
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    @tf.function
    def train_step(self, noisy_input, noisy_target, clean_target):
        with tf.GradientTape() as tape:
            denoised = self.network(noisy_input, training=True)
            target = noisy_target if self.noise2noise else clean_target
            loss = tf.reduce_mean(tf.square(target - denoised))
        
        gradients = tape.gradient(loss, self.network.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_variables))
        return loss, denoised

def create_network():
    """Create autoencoder network using Keras layers."""
    model = tf.keras.Sequential([
        # Convert from NCHW to NHWC
        tf.keras.layers.Permute((2, 3, 1)),
        
        # Encoder
        tf.keras.layers.Conv2D(48, 3, padding='same', data_format='channels_last'),
        tf.keras.layers.LeakyReLU(0.1),
        tf.keras.layers.Conv2D(48, 3, padding='same', data_format='channels_last'),
        tf.keras.layers.LeakyReLU(0.1),
        tf.keras.layers.MaxPooling2D(2, data_format='channels_last'),
        
        tf.keras.layers.Conv2D(48, 3, padding='same', data_format='channels_last'),
        tf.keras.layers.LeakyReLU(0.1),
        tf.keras.layers.MaxPooling2D(2, data_format='channels_last'),
        
        tf.keras.layers.Conv2D(48, 3, padding='same', data_format='channels_last'),
        tf.keras.layers.LeakyReLU(0.1),
        tf.keras.layers.MaxPooling2D(2, data_format='channels_last'),
        
        tf.keras.layers.Conv2D(48, 3, padding='same', data_format='channels_last'),
        tf.keras.layers.LeakyReLU(0.1),
        tf.keras.layers.MaxPooling2D(2, data_format='channels_last'),
        
        # Bottleneck
        tf.keras.layers.Conv2D(48, 3, padding='same', data_format='channels_last'),
        tf.keras.layers.LeakyReLU(0.1),
        tf.keras.layers.Conv2D(48, 3, padding='same', data_format='channels_last'),
        tf.keras.layers.LeakyReLU(0.1),
        
        # Decoder
        tf.keras.layers.UpSampling2D(2, data_format='channels_last'),
        tf.keras.layers.Conv2D(96, 3, padding='same', data_format='channels_last'),
        tf.keras.layers.LeakyReLU(0.1),
        tf.keras.layers.Conv2D(96, 3, padding='same', data_format='channels_last'),
        tf.keras.layers.LeakyReLU(0.1),
        
        tf.keras.layers.UpSampling2D(2, data_format='channels_last'),
        tf.keras.layers.Conv2D(96, 3, padding='same', data_format='channels_last'),
        tf.keras.layers.LeakyReLU(0.1),
        tf.keras.layers.Conv2D(96, 3, padding='same', data_format='channels_last'),
        tf.keras.layers.LeakyReLU(0.1),
        
        tf.keras.layers.UpSampling2D(2, data_format='channels_last'),
        tf.keras.layers.Conv2D(96, 3, padding='same', data_format='channels_last'),
        tf.keras.layers.LeakyReLU(0.1),
        tf.keras.layers.Conv2D(96, 3, padding='same', data_format='channels_last'),
        tf.keras.layers.LeakyReLU(0.1),
        
        tf.keras.layers.UpSampling2D(2, data_format='channels_last'),
        tf.keras.layers.Conv2D(64, 3, padding='same', data_format='channels_last'),
        tf.keras.layers.LeakyReLU(0.1),
        tf.keras.layers.Conv2D(32, 3, padding='same', data_format='channels_last'),
        tf.keras.layers.LeakyReLU(0.1),
        
        # Output
        tf.keras.layers.Conv2D(3, 3, padding='same', data_format='channels_last'),
        
        # Convert back from NHWC to NCHW
        tf.keras.layers.Permute((3, 1, 2))
    ])
    return model

def train(
    submit_config: dnnlib.SubmitConfig,
    iteration_count: int,
    eval_interval: int,
    minibatch_size: int,
    learning_rate: float,
    ramp_down_perc: float,
    noise: dict,
    validation_config: dict,
    train_tfrecords: str,
    noise2noise: bool):
    noise_augmenter = dnnlib.util.call_func_by_name(**noise)
    validation_set = ValidationSet(submit_config)
    validation_set.load(**validation_config)

    # Initialize network with proper architecture
    network = create_network()
    # Build model with sample input shape
    network.build(input_shape=(None, 3, 256, 256))  # [batch, channels, height, width]
    trainer = Trainer(network, noise2noise)
    
    # Create dataset
    dataset = create_dataset(train_tfrecords, minibatch_size, 
                           noise_augmenter.add_train_noise_tf)

    # Training loop
    for i in range(iteration_count):
        noisy_input, noisy_target, clean_target = next(dataset)
        loss, denoised = trainer.train_step(noisy_input, noisy_target, clean_target)

        if i % eval_interval == 0:
            # Validation code
            validation_set.evaluate(network, i, noise_augmenter.add_validation_noise_np)
            
        # Update learning rate
        lrate = compute_ramped_down_lrate(i, iteration_count, ramp_down_perc, learning_rate)
        trainer.optimizer.learning_rate = lrate

    # Save final model
    network.save(os.path.join(submit_config.run_dir, 'final_model'))
