# !/usr/bin/env python3
# -*-coding:utf-8-*-
# @file: net_init.py
# @brief:
# @author: Changjiang Cai, ccai1@stevens.edu, caicj5351@gmail.com
# @version: 0.0.1
# @creation date: 13-01-2020
# @last modified: Mon 13 Jan 2020 04:34:59 PM EST

import torch
import torch.nn as nn
import numpy as np

""" the function is copied from 
    https://github.com/wyf2017/DSMnet/blob/master/models/util_conv.py
"""
# weight init
def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)

""" the function is adapted from 
    https://github.com/wyf2017/DSMnet/blob/master/models/util_conv.py
"""

#NOTE: this function works well for all most the cases, except GANet, due to sync BN;
def net_init_v0(net):
    for m in net.modules():
    #for idx, (n, m) in enumerate(net.named_modules()):
        if isinstance(m, nn.Linear):
            m.weight.data = fanin_init(m.weight.data.size())
        elif isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
            m.weight.data.normal_(0, np.sqrt(2.0 / n))
            if m.bias is not None:
                m.bias.data.zero_()

        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, np.sqrt(2.0 / n))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
            n = m.kernel_size[0] * m.out_channels
            m.weight.data.normal_(0, np.sqrt(2.0 / n))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            #print ("[????] net_init_bn, idx ", idx, " - ", n)
            m.weight.data.fill_(1)
            m.bias.data.zero_()


#from src.baselines.GANet.libs.sync_bn.modules.sync_bn import BatchNorm2d, BatchNorm3d, BatchNorm1d
from src.modules.sync_batchnorm import SynchronizedBatchNorm3d as BatchNorm3d 
from src.modules.sync_batchnorm import SynchronizedBatchNorm2d as BatchNorm2d 
from src.modules.sync_batchnorm import SynchronizedBatchNorm1d as BatchNorm1d 
def net_init_SyncBN(net):
    for m in net.modules():
    #for idx, (n, m) in enumerate(net.named_modules()):
        #print ("[????] net_init_sync, idx ", idx, " - ", n)
        if isinstance(m, nn.Linear):
            m.weight.data = fanin_init(m.weight.data.size())
        elif isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
            m.weight.data.normal_(0, np.sqrt(2.0 / n))
            if m.bias is not None:
                m.bias.data.zero_()

        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, np.sqrt(2.0 / n))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
            n = m.kernel_size[0] * m.out_channels
            m.weight.data.normal_(0, np.sqrt(2.0 / n))
            if m.bias is not None:
                m.bias.data.zero_()
        
        elif isinstance(m, BatchNorm3d) or isinstance(m, BatchNorm2d) or isinstance(m, BatchNorm1d):
            #print ("[????] net_init_sync_bn, idx ", idx, " - ", n)
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)