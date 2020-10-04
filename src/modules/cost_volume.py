# !/usr/bin/env python3
# -*-coding:utf-8-*-
# @file: cost_volume.py
# @brief:
# @author: Changjiang Cai, ccai1@stevens.edu, caicj5351@gmail.com
# @version: 0.0.1
# @creation date: 30-01-2020
# @last modified: Thu 30 Jan 2020 05:20:05 PM EST

import torch
from torch.autograd import Variable
import torch.nn as nn

#-------------------
#NOTE: this function is adapted from PSMNet code ??? TOO SLOW???;
#-------------------
def get_costVolume(x, y, d):
    """
    args: 
        x : left feature,  in size [N,C,H,W]
        y : right feature, in size [N,C,H,W]
        d : disparity range
    return:
        cost: cost volume in size [N,2C,D,H,W]
    """
    # matching
    assert(x.is_contiguous() == True)
    N0, C0, H0, W0 = x.size()[:]
    cost = Variable(torch.FloatTensor(N0, C0*2, d, H0, W0).zero_()).cuda()
    #cost = torch.tensor((), dtype=torch.float32).new_zeros((N0,2*C0,d,H0,W0),requires_grad=True).cuda()
    
    for i in range(0, d):
        if i > 0:
            cost[:, :C0, i, :, i:] = x[:, :, :, i:]
            cost[:, C0:, i, :, i:] = y[:, :, :, :-i]
        else:
            cost[:, :C0, i, :, :] = x
            cost[:, C0:, i, :, :] = y
    
    return cost.contiguous()


#NOTE: faster!! But consume a little more memory than the above one???
def cost_volume_faster(x, y, d):
    """
    args:
        x : left feature,  in size [N,C,H,W]
        y : right feature, in size [N,C,H,W]
        d : disparity range
    return:
        cost: cost volume in size [N,2C,D,H,W]
    """
    N0, C0, H0, W0 = x.size()[:]
    cv_list = []
    # Pads the input tensor boundaries with zero.
    # padding = (padding_left, padding_right, padding_top, padding_bottom) 
    # along the [H, W] dim; 
    #y_pad = nn.ZeroPad2d((d, 0, 0, 0))(y)
    #NOTE: updated d to d-1 for left padding!!!
    y_pad = nn.ZeroPad2d((d-1, 0, 0, 0))(y)
    #print ('[???] y_pad shape: ', y_pad.shape)
    
    #NOTE: reversed() is necessary!!!
    for i in reversed(range(0, d)):
        x_slice = x
        #Note added by CCJ:
        #Note that you don’t need to use torch.narrow or select, 
        #but instead basic indexing will do it for you.
        y_slice = y_pad[:,:,:, i:i+W0]
        xy_temp = torch.cat((x_slice, y_slice), 1)
        cv_list.append(xy_temp)
    
    #Stacks a list of rank-R tensors into one rank-(R+1) tensor.
    cv = torch.stack(cv_list, 2)
    #assert(cv.is_contiguous() == True)
    #print ("[???] cv shape = ", cv.shape)
    return cv
