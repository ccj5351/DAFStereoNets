# !/usr/bin/env python3
# -*-coding:utf-8-*-
# @file: common_flags.py
# @brief:
# @version: 0.0.1
# @creation date: 05-03-2019
# @last modified: Tue 12 Mar 2019 03:28:29 PM EDT

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
"""Provides flags that are common to scripts.

Common flags from train/eval/vis/export_model.py are collected in this script.
"""
import collections
import copy
import json
import tensorflow as tf

flags = tf.app.flags

# Flags for input preprocessing.

flags.DEFINE_integer('min_resize_value', None,
                     'Desired size of the smaller image side.')

flags.DEFINE_integer('max_resize_value', None,
                     'Maximum allowed size of the larger image side.')

flags.DEFINE_integer('resize_factor', None,
                     'Resized dimensions are multiple of factor plus one.')

FLAGS = flags.FLAGS

# Constants

# Perform semantic segmentation predictions.
OUTPUT_TYPE = 'semantic'

# Semantic segmentation item names.
LABELS_CLASS = 'labels_class'
IMAGE = 'image'
HEIGHT = 'height'
WIDTH = 'width'
IMAGE_NAME = 'image_name'
LABEL = 'label'
ORIGINAL_IMAGE = 'original_image'

# Test set name.
TEST_SET = 'test'
