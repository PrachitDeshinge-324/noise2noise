# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import tensorflow as tf

def parse_tfrecord_tf(record):
    features = {
        'shape': tf.io.FixedLenFeature([3], tf.int64),
        'data': tf.io.FixedLenFeature([], tf.string)
    }
    example = tf.io.parse_single_example(record, features)
    data = tf.io.decode_raw(example['data'], tf.uint8)
    return tf.reshape(data, example['shape'])

# [c,h,w] -> [h,w,c]
def chw_to_hwc(x):
    return tf.transpose(x, perm=[1, 2, 0])

# [h,w,c] -> [c,h,w]
def hwc_to_chw(x):
    return tf.transpose(x, perm=[2, 0, 1])

def resize_small_image(x):
    shape = tf.shape(x)
    return tf.cond(
        tf.logical_or(
            tf.less(shape[2], 256),
            tf.less(shape[1], 256)
        ),
        true_fn=lambda: hwc_to_chw(tf.image.resize(chw_to_hwc(x), size=[256,256], method='bicubic')),
        false_fn=lambda: tf.cast(x, tf.float32)
     )

def random_crop_noised_clean(x, add_noise):
    """Apply random crop with TF2 compatible operations."""
    # First resize if needed
    x = resize_small_image(x)
    
    # Create random crop using tf.image operations
    x_hwc = chw_to_hwc(x)  # Convert to HWC for tf.image operations
    cropped_hwc = tf.image.random_crop(x_hwc, size=[256, 256, 3])
    cropped = hwc_to_chw(cropped_hwc)  # Back to CHW
    
    # Normalize to [-0.5, 0.5] range
    cropped = cropped / 255.0 - 0.5
    
    # Apply noise
    return (add_noise(cropped), add_noise(cropped), cropped)

def create_dataset(train_tfrecords, minibatch_size, add_noise):
    """Create input pipeline using tf.data with TF2 best practices."""
    print('Setting up dataset source from', train_tfrecords)
    
    dset = tf.data.TFRecordDataset(train_tfrecords)
    dset = dset.repeat()
    dset = dset.prefetch(tf.data.AUTOTUNE)
    dset = dset.map(parse_tfrecord_tf, num_parallel_calls=tf.data.AUTOTUNE)
    dset = dset.shuffle(buffer_size=1000)
    
    # Use tf.data.Dataset.map with the updated random_crop function
    dset = dset.map(
        lambda x: random_crop_noised_clean(x, add_noise),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    dset = dset.batch(minibatch_size)
    return iter(dset)  # Return iterator directly instead of initializable iterator

