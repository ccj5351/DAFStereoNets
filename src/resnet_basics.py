# !/usr/bin/env python3
# -*-coding:utf-8-*-
# @file: resnet_basics.py
# @brief:
# @author: Changjiang Cai, ccai1@stevens.edu, caicj5351@gmail.com
# @version: 0.0.1
# @creation date: 24-03-2020
# @last modified: Tue 24 Mar 2020 02:26:31 AM EDT


# > see this tutorial: An Overview of ResNet and its Variants,
# > at https://towardsdatascience.com/an-overview-of-resnet-and-its-variants-5281e2f56035;

import torch
import torch.nn as nn
#import torch.nn.functional as F

""" Original Residual Block;
    code adapted from 
    https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/deep_residual_network/main.py
"""
class PreActiResidualBlock(nn.Module):
    #here we use original residual block:
    # x --> Weight --> BN --> ReLU --> Weight --> BN --> ReLU
    # |                                                   |
    # |                                                   |
    # |                                                   |
    # |-----------------------------------------------> Addition --> x_new
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(PreActiResidualBlock, self).__init__()
        self.my_func = nn.Sequential(
                nn.BatchNorm2d(in_channels), #bn1
                nn.ReLU(inplace=True), #relu1
                self.conv3x3(in_channels, out_channels, stride), #conv1
                nn.BatchNorm2d(out_channels), #bn2
                nn.ReLU(inplace=True), #relu2
                self.conv3x3(out_channels, out_channels, stride), #conv2
                )
        
    # 3x3 convolution
    def conv3x3(self, in_channels, out_channels, stride=1):
        return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                stride=stride, padding=1, bias=False)
    
    def forward(self, x):
        rem = x
        x = self.my_func(x)
        return x + rem

# Residual block
class OriginalResidualBlock(nn.Module):
    """ here we use original residual block:
         x --> Weight --> BN --> ReLU --> Weight --> BN --> ReLU
         |                                           |
         |                                           |
         |                                           |
         |-----------------------------------------------> Addition --> x_new
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(OriginalResidualBlock, self).__init__()
        self.my_func = nn.Sequential(
                self.conv3x3(in_channels, out_channels, stride), #conv1
                nn.BatchNorm2d(out_channels), #bn1
                nn.ReLU(inplace=True), #relu1
                self.conv3x3(out_channels, out_channels, stride), #conv2
                nn.BatchNorm2d(out_channels), #bn2
                )
        self.relu = nn.ReLU(inplace=True) #relu2
        
    # 3x3 convolution
    def conv3x3(self, in_channels, out_channels, stride=1):
        return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                stride=stride, padding=1, bias=False)
    
    def forward(self, x):
        rem = x
        print ("??? rem shape = ", rem.shape)
        x = self.my_func(x)
        print ("??? x shape = ", x.shape)
        x += rem
        x = self.relu(x)
        return x