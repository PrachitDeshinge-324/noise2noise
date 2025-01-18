# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import glob
import os
import sys
import argparse
import struct
import PIL.Image
import numpy as np
from collections import defaultdict
from pathlib import Path

size_stats = defaultdict(int)
format_stats = defaultdict(int)

# Try importing TensorFlow, fallback to pure Python implementation if it fails
try:
    import tensorflow as tf
    USE_TF = True
except:
    print("Warning: TensorFlow import failed, using pure Python implementation")
    USE_TF = False

def load_image(fname):
    global format_stats, size_stats
    im = PIL.Image.open(fname)
    format_stats[im.mode] += 1
    if (im.width < 256 or im.height < 256):
        size_stats['< 256x256'] += 1
    else:
        size_stats['>= 256x256'] += 1
    arr = np.array(im.convert('RGB'), dtype=np.uint8)
    assert len(arr.shape) == 3
    return arr.transpose([2, 0, 1])

def shape_feature(v):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=v))

def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

examples='''examples:

  python %(prog)s --input-dir=./kodak --out=imagenet_val_raw.tfrecords
'''

def write_record_python(writer, image):
    """Write a record using pure Python"""
    # Write shape
    shape = image.shape
    writer.write(struct.pack('III', *shape))
    # Write data
    writer.write(image.tobytes())

def find_images(directory):
    """Recursively find all images in a directory"""
    extensions = {'.jpeg', '.jpg', '.png', '.tif', '.tiff'}
    images = []
    
    # Print directory contents for debugging
    print(f"\nDebug: Scanning directory structure of {directory}")
    print("Found files and directories:")
    for root, dirs, files in os.walk(directory):
        indent = '  ' * (root[len(directory):].count(os.sep))
        print(f"{indent}{os.path.basename(root)}/")
        for f in files:
            print(f"{indent}  {f}")
            if Path(f).suffix.lower() in extensions:
                images.append(os.path.join(root, f))
    
    return sorted(images)

def main():
    parser = argparse.ArgumentParser(
        description='Convert a set of image files into a TensorFlow tfrecords training set.',
        epilog=examples,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input-dir", help="Directory containing ImageNet images")
    parser.add_argument("--out", help="Filename of the output tfrecords file")
    args = parser.parse_args()

    if args.input_dir is None:
        print ('Must specify input file directory with --input-dir')
        sys.exit(1)
    if args.out is None:
        print ('Must specify output filename with --out')
        sys.exit(1)

    input_dir = os.path.abspath(args.input_dir)
    if not os.path.exists(input_dir):
        print(f'ERROR: Input directory does not exist: {input_dir}')
        sys.exit(1)
    
    print(f'Loading image list from {input_dir}')
    images = find_images(input_dir)
    
    if not images:
        print(f'ERROR: No images found in {input_dir}')
        sys.exit(1)
        
    print(f'\nFound {len(images)} total images')
    print('\nFirst 5 images found:')
    for img in images[:5]:
        print(f'  {img}')

    outdir = os.path.dirname(args.out)
    os.makedirs(outdir, exist_ok=True)

    if USE_TF:
        writer = tf.io.TFRecordWriter(args.out)
        for (idx, imgname) in enumerate(images):
            print (idx, imgname)
            image = load_image(imgname)
            feature = {
                'shape': shape_feature(image.shape),
                'data': bytes_feature(tf.compat.as_bytes(image.tostring()))
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())
    else:
        with open(args.out, 'wb') as writer:
            # Write number of images
            writer.write(struct.pack('I', len(images)))
            for (idx, imgname) in enumerate(images):
                print (idx, imgname)
                image = load_image(imgname)
                write_record_python(writer, image)

    print ('Dataset statistics:')
    print ('  Formats:')
    for key in format_stats:
        print ('    %s: %d images' % (key, format_stats[key]))
    print ('  width,height buckets:')
    for key in size_stats:
        print ('    %s: %d images' % (key, size_stats[key]))
    writer.close()

if __name__ == "__main__":
    main()