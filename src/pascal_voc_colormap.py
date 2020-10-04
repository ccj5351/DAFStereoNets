# !/usr/bin/env python3
# -*-coding:utf-8-*-
# @file: colorize.py
# @brief:
# @author: Changjiang Cai, ccai1@stevens.edu, caicj5351@gmail.com
# @version: 0.0.1
# @creation date: 20-03-2019
# @last modified: Thu 21 Mar 2019 11:51:33 AM EDT

""" see source code func colorize() at https://gist.github.com/jimfleming/c1adfdb0f526465c99409cc143dea97b"""

#def colorize(value, vmin=None, vmax=None, cmap=None):

import tensorflow as tf
import numpy as np

def pascali_voc_color_map(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap


def pascal_voc_label2RGB(label):
    """
    A utility function for TensorFlow that maps a grayscale image to a matplotlib
    colormap for use with TensorBoard image summaries.
    Arguments:
      - label: 2D Tensor of shape [height, width] or 3D Tensor of shape
        [height, width, 1], with uint8 type.
    
    Returns a 3D tensor of shape [height, width, 3].
    """
    # quantize
    indices = tf.to_int32(tf.squeeze(label))

    # get pascal voc colormap
    cmap = pascali_voc_color_map(N=256, normalized=False)
    colors = tf.constant(cmap, dtype=tf.uint8)
    # gather
    value = tf.gather(colors, indices)
    return value
