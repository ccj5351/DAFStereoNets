# !/usr/bin/env python3
# -*-coding:utf-8-*-
# @file: im2col.py
# @brief:
# @author: Changjiang Cai, ccai1@stevens.edu, caicj5351@gmail.com
# @version: 0.0.1
# @creation date: 07-11-2019
# @last modified: Thu 07 Nov 2019 02:46:31 PM EST

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

"""
image to column network
"""

class im2col_layer(nn.Module):
     def __init__(self, k, d = 2, is5D = True):
         super(im2col_layer, self).__init__()  
         self.k = k # kernel size
         self.d = d # dilation
         self.padding = self.d * (self.k - 1) // 2
         self.im2col = nn.Unfold(self.k, self.d, self.padding, stride = 1)
         self.is5D = is5D
         #NOTE:
         """
         PyTorch im2col (i.e., nn.Unfold) flattens each k by k
         block into a column which conains C*(k*k) values, where k*k is a
         continugous chunk, with C be the Channel dimension.
         """
     
     def forward(self, x):
         N, C, H, W = x.size()[:]
         if self.is5D:
             return self.im2col(x).view(N, C, self.k*self.k, H, W)
         else:
             return self.im2col(x).view(N, C*self.k*self.k, H, W)
         
