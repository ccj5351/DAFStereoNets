# !/usr/bin/env python3
# -*-coding:utf-8-*-
# @file: setup.py
# @brief:
# @author: Changjiang Cai, ccai1@stevens.edu, caicj5351@gmail.com
# @version: 0.0.1
# @creation date: 28-10-2019
# @last modified: Tue 29 Oct 2019 02:00:37 PM EDT

from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension

ext_modules = [Extension('writeKT15ErrorLogColor', ['writeKT15ErrorLogColor.pyx'])]
setup(
        ext_modules = cythonize(ext_modules)
    )
