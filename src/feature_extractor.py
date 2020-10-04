# !/usr/bin/env python3
# -*-coding:utf-8-*-
# @file: feature_extractor.py
# @brief:
# @version: 0.0.1
# @creation date: 06-03-2019
# @last modified:

# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Extracts features for different models."""
import functools
import tensorflow as tf

#from deeplab.core import nas_network
#from deeplab.core import resnet_v1_beta
#from deeplab.core import xception
from tensorflow.contrib.slim.nets import resnet_utils
#from nets.mobilenet import mobilenet_v2


slim = tf.contrib.slim

# Mean pixel value.
_MEAN_RGB = [123.15, 115.90, 103.06]


def _preprocess_subtract_imagenet_mean(inputs, dtype=tf.float32):
  """Subtract Imagenet mean RGB value."""
  mean_rgb = tf.reshape(_MEAN_RGB, [1, 1, 1, 3])
  num_channels = tf.shape(inputs)[-1]
  # We set mean pixel as 0 for the non-RGB channels.
  mean_rgb_extended = tf.concat(
      [mean_rgb, tf.zeros([1, 1, 1, num_channels - 3])], axis=3)
  return tf.cast(inputs - mean_rgb_extended, dtype=dtype)


def _preprocess_zero_mean_unit_range(inputs, dtype=tf.float32):
  """Map image values from [0, 255] to [-1, 1]."""
  preprocessed_inputs = (2.0 / 255.0) * tf.to_float(inputs) - 1.0
  return tf.cast(preprocessed_inputs, dtype=dtype)

"""
_PREPROCESS_FN = {
    'mobilenet_v2': _preprocess_zero_mean_unit_range,
    'resnet_v1_50': _preprocess_subtract_imagenet_mean,
    'resnet_v1_50_beta': _preprocess_zero_mean_unit_range,
    'resnet_v1_101': _preprocess_subtract_imagenet_mean,
    'resnet_v1_101_beta': _preprocess_zero_mean_unit_range,
    'xception_41': _preprocess_zero_mean_unit_range,
    'xception_65': _preprocess_zero_mean_unit_range,
    'xception_71': _preprocess_zero_mean_unit_range,
    'nas_pnasnet': _preprocess_zero_mean_unit_range,
    'nas_hnasnet': _preprocess_zero_mean_unit_range,
}
"""


def mean_pixel(model_variant=None):
  """Gets mean pixel value.

  This function returns different mean pixel value, depending on the input
  model_variant which adopts different preprocessing functions. We currently
  handle the following preprocessing functions:
  (1) _preprocess_subtract_imagenet_mean. We simply return mean pixel value.
  (2) _preprocess_zero_mean_unit_range. We return [127.5, 127.5, 127.5].
  The return values are used in a way that the padded regions after
  pre-processing will contain value 0.

  Args:
    model_variant: Model variant (string) for feature extraction. For
      backwards compatibility, model_variant=None returns _MEAN_RGB.

  Returns:
    Mean pixel value.
  """
  if model_variant in ['resnet_v1_50',
                       'resnet_v1_101'] or model_variant is None:
    return _MEAN_RGB
  else:
    return [127.5, 127.5, 127.5]
