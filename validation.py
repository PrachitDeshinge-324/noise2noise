# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import os
import sys
import numpy as np
import PIL.Image
import tensorflow as tf

import dnnlib
import dnnlib.submission.submit as submit
import dnnlib.tflib.tfutil as tfutil
from dnnlib.tflib.autosummary import autosummary

import util
import config

def list_image_files(directory):
    """List all image files in directory recursively."""
    image_files = []
    for root, _, files in os.walk(directory):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                full_path = os.path.join(root, f)
                print(f"Found image: {full_path}")
                image_files.append(full_path)
    return sorted(image_files)

class ValidationSet:
    def __init__(self, submit_config):
        self.images = None
        self.submit_config = submit_config
        return

    def load(self, dataset_dir):
        """Load validation images."""
        if isinstance(dataset_dir, dict):
            dataset_dir = dataset_dir.get('dataset_dir')
            
        dataset_dir = os.path.abspath(dataset_dir)
        print(f"Loading validation images from: {dataset_dir}")
        
        if not os.path.exists(dataset_dir):
            raise RuntimeError(f"Validation directory does not exist: {dataset_dir}")
            
        # List directory contents for debugging
        print("\nDirectory contents:")
        for item in os.listdir(dataset_dir):
            print(f"  {item}")
            
        # Use recursive file finding
        fnames = list_image_files(dataset_dir)
        
        if not fnames:
            raise RuntimeError(f'No validation images found in {dataset_dir}')
            
        print(f"\nLoading {len(fnames)} validation images...")
        
        images = []
        for fname in fnames:
            try:
                im = PIL.Image.open(fname).convert('RGB')
                arr = np.array(im, dtype=np.float32)
                reshaped = arr.transpose([2, 0, 1]) / 255.0 - 0.5
                images.append(reshaped)
                print(f"Successfully loaded: {os.path.basename(fname)}")
            except Exception as e:
                print(f'Error loading {fname}: {str(e)}')
                
        if not images:
            raise RuntimeError('No valid images could be loaded!')
            
        self.images = images
        print(f"\nSuccessfully loaded {len(self.images)} validation images")

    def evaluate(self, net, iteration, noise_func):
        avg_psnr = 0.0
        for idx in range(len(self.images)):
            orig_img = self.images[idx]
            w = orig_img.shape[2]
            h = orig_img.shape[1]

            noisy_img = noise_func(orig_img)
            pred255 = util.infer_image(net, noisy_img)
            orig255 = util.clip_to_uint8(orig_img)
            assert (pred255.shape[2] == w and pred255.shape[1] == h)

            sqerr = np.square(orig255.astype(np.float32) - pred255.astype(np.float32))
            s = np.sum(sqerr)
            cur_psnr = 10.0 * np.log10((255*255)/(s / (w*h*3)))
            avg_psnr += cur_psnr

            util.save_image(self.submit_config, pred255, "img_{0}_val_{1}_pred.png".format(iteration, idx))

            if iteration == 0:
                util.save_image(self.submit_config, orig_img, "img_{0}_val_{1}_orig.png".format(iteration, idx))
                util.save_image(self.submit_config, noisy_img, "img_{0}_val_{1}_noisy.png".format(iteration, idx))
        avg_psnr /= len(self.images)
        print ('Average PSNR: %.2f' % autosummary('PSNR_avg_psnr', avg_psnr))


def validate(submit_config: dnnlib.SubmitConfig, noise: dict, dataset: dict, network_snapshot: str):
    noise_augmenter = dnnlib.util.call_func_by_name(**noise)
    validation_set = ValidationSet(submit_config)
    validation_set.load(**dataset)

    ctx = dnnlib.RunContext(submit_config, config)

    tfutil.init_tf(config.tf_config)

    with tf.device("/gpu:0"):
        net = util.load_snapshot(network_snapshot)
        validation_set.evaluate(net, 0, noise_augmenter.add_validation_noise_np)
    ctx.close()

def infer_image(network_snapshot: str, image: str, out_image: str):
    tfutil.init_tf(config.tf_config)
    net = util.load_snapshot(network_snapshot)
    im = PIL.Image.open(image).convert('RGB')
    arr = np.array(im, dtype=np.float32)
    reshaped = arr.transpose([2, 0, 1]) / 255.0 - 0.5
    pred255 = util.infer_image(net, reshaped)
    t = pred255.transpose([1, 2, 0])  # [RGB, H, W] -> [H, W, RGB]
    PIL.Image.fromarray(t, 'RGB').save(os.path.join(out_image))
    print ('Inferred image saved in', out_image)
