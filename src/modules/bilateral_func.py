# !/usr/bin/env python3
# -*-coding:utf-8-*-
# @file: module_bilateral.py
# @brief:
# @author: Changjiang Cai, ccai1@stevens.edu, caicj5351@gmail.com
# @version: 0.0.1
# @creation date: 15-10-2019
# @last modified: Tue 12 Nov 2019 04:07:36 PM EST

import numpy as np
import src.pfmutil as pfm
import sys
import math
import torch
import torch.nn as nn

__all__ = ['bilateralFilter', 
        'get_space_gaussian_filter', 
        'get_gaussian_filter_width_from_sigma',
        'im2col_func',
        ]


#---------------------------------
""" some utility functions """
#---------------------------------

def im2col_func(x, k, d = 2, is5D = True):
    """
    args:
       x : input, [N,C,H,W]
       k:  window size
    return:
       y : in shape [N, C, (k*k), H, W] if is5D == True, 
           otherwise in shape [N, C*(k*k), H, W]
    """
    N, C, H, W = x.size()[:]
    p = d * (k-1) // 2
    unfold = nn.Unfold(kernel_size= k, dilation = d, padding = p, stride = 1)
    #NOTE:
    """
    PyTorch im2col (i.e., nn.Unfold) flattens each k by k
    block into a column which conains C*(k*k) values, where k*k is a
    continugous chunk, with C be the Channel dimension
    """
    #return unfold(x).view(N, C, k*k, H, W)
    if is5D:
        return unfold(x).view(N, C, k*k, H, W)
    else:
        return unfold(x).view(N, C*k*k, H, W)

# define the window width to be the 3 time the spatial std. dev. to 
# be sure that most of the spatial kernel is actually captured
def get_gaussian_filter_width_from_sigma(sigma_s):
    return int(3*sigma_s  + 1)

def get_space_gaussian_filter(sigma_s = 0.1):
    """
    make a simple Gaussian function taking the squared radius
    """
    gaussian = lambda r2, sigma : np.exp( -0.5*r2/sigma**2)
                        
    # define the window width to be the 3 time the spatial std. dev. to 
    # be sure that most of the spatial kernel is actually captured
    win_width = get_gaussian_filter_width_from_sigma(sigma_s)
    #print ('win_size = {} x {}'.format(2*win_width+1, 2*win_width + 1))
    
    """ generate gaussian filter for space weight """
    sw_filter = np.zeros(( 2*win_width + 1, 2*win_width + 1))
    for shft_y in range(-win_width,win_width+1):
        for shft_x in range(-win_width,win_width+1):
            # compute the spatial weight
            y_idx = shft_y + win_width
            x_idx = shft_x + win_width
            #print (type(x_idx), (type(y_idx)))
            sw = gaussian(shft_x**2+shft_y**2, sigma_s) # scalar sw
            #print ("sw_filter[%d, %d] = %f" %(y_idx, x_idx, sw))
            sw_filter[y_idx, x_idx] = sw
            #print ("sw_filter = ", sw_filter)
    return sw_filter.astype(np.float32)


""" module : bilateral filter leveraging embedding feature """
def bilateralFilter(embed_in, x_in, sigma_s = 0.7, sigma_v = 0.1, isCUDA = True, dilation = 1):
    """
    using im2col to change the filtering to element-wise matrix multiplication
    Performs standard bilateral filtering of an input image.
    Padding is taken care by the inside function im2col_pytorch()!!!
    
    Args:
         embed_in (Tensor), embeding for generatin mask, in size [N,F,H,W]
         x_in     (Tensor), input array, in size [N,C,H,W], 
                            apply the bilateral filter to it;
         sg_filter  (Tensor)   spatial gaussia filter, given sigma_s as spatial gaussian std. dev.
         win_width    (float)  spatical gaussin filter half window size, determined by space filter std dev sigma_s;
         sigma_s      (float)  space gaussian std. dev.
         sigma_v      (float)  value gaussian std. dev.
         reg_constant (float)  optional regularization constant for pathalogical cases

    Returns:
         result (Tensor) bilateral-filtered x_in, in size [N,C,H,W]
    """    
    win_width = get_gaussian_filter_width_from_sigma(sigma_s)
    k = 2 * win_width + 1
    sg_filter = torch.from_numpy(get_space_gaussian_filter(sigma_s)).view(1,1,k*k,1,1)
    if isCUDA:
        sg_filter = sg_filter.cuda()
    #print ('[???]sg_filter :', sg_filter.requires_grad)

    reg_constant=1e-20 # regularization constant

    N,F,H,W = embed_in.shape[:]
    #print ('embed_in shape = ', embed_in.shape)
    
    """ apply im2col for embedding masking """
    embed_im2col = im2col_func(embed_in, k, dilation, is5D = True) # in shape [N, F, k*k, H, W]
    # broadcasting
    embed_im2col -= embed_in.view(N,F,1,H,W)
    rg_filter = torch.exp_(-0.5*torch.sum(embed_im2col.pow_(2), dim = 1, keepdim= True) / (sigma_v**2)) # [N,1,k*k,H,W]
    
    
    #NOTE:updated due to the gradient error!!!
    # change the inplace *= to out-of-place one, due to the error: gradient computation has been modified by an inplace operation
    #rg_filter *= sg_filter # broadcasting
    rg_filter = rg_filter*sg_filter # broadcasting
    

    """
    implement eq (4) in W. Harley' paper 
    "Segmentation-Aware Convolutional Networks Using Local Attention Masks (ICCV'17)"
    """
    wgt_sum = torch.ones([N, 1, H, W])*reg_constant
    if isCUDA:
        wgt_sum = wgt_sum.cuda()
    """ broadcasting """
    #unrolled for element-wise multiplication:
    N, C, H, W = x_in.size()[:]
    #print ('[???] x_in shape = ', x_in.shape)
    x_in_im2col = im2col_func(x_in, k, dilation, is5D = True) # in shape [N, C, k*k, H, W]
    
    #(N, C, k*k, H, W ) * (N, 1, k*k, H, W) => (N, C, k*k, H, W)
    #print ("[**] x shape, rg_filter shape :", x_in_im2col.shape, rg_filter.shape)
    wgt_sum += torch.sum(rg_filter, axis=2) # N x 1 x H x W 
    x_in_im2col *= rg_filter # N x C x k*k x H x W
    result = torch.sum(x_in_im2col, axis = 2) # N x C x H x W
    # normalize the result and return
    # N x C x H x W 
    result /= wgt_sum
    return result
