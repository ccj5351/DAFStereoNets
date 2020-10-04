# !/usr/bin/env python3
# -*-coding:utf-8-*-
# @file: filter_corr1D.py
# @brief:
# @author: Changjiang Cai, ccai1@stevens.edu, caicj5351@gmail.com
# @version: 0.0.1
# @creation date: 02-02-2020
# @last modified: Mon 03 Feb 2020 02:08:50 AM EST

import torch
import torch.nn as nn
from .im2col import im2col_layer

class filtered_correlation1D(nn.Module):
    def __init__(self, maxdisp, kernel_size = 5, dilation = 2):
        """
        args:
            max_disp: disparity range
            kernel_size: e.g., 5 x 5
        """
        super(filtered_correlation1D, self).__init__()
        self.maxdisp = maxdisp
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.im2col = im2col_layer(k = self.kernel_size, d = dilation, is5D = False)

    def forward(self, x, y, filters, filters_biases):
        """
        args:
            x: left feature, in [N,C,H,W]
            y: right feature, in [N,C,H,W]
            filters: dynamic filters, in shape [N, k*k, H, W]
            filters_biases: used as additive biases after the filtering, in shape [N, 1, H, W];
        return:
            corr: correlation map in size [N,D,H,W]
        """
    
        """
        #NOTE: make sure x means left image, y means right image,
        # so that we have the pixel x in left image, 
        # and the corresponding match pixel in right image has x-d 
        # (that means disparity d = shifting to left by d pixels). 
        # Otherwise, we have to chagne the shift direction!!!
        """
        # Pads the input tensor boundaries with zero.
        # padding = (padding_left, padding_right, padding_top, padding_bottom) along the [H, W] dim; 
        #y_pad = nn.ZeroPad2d((self.maxdisp-1, 0, 0, 0))(y)
        #NOTE: updated maxdisp to maxdisp-1 for left padding!!!
        y_pad = nn.ZeroPad2d((self.maxdisp-1, 0, 0, 0))(y)
        # input width
        W0 = x.size()[3]
        corr_tensor_list = []
        # filters: [N, k*k, H, W]
        #N, F, H, W = filters.size()[:] # here F = k*k
        #filters = filters.view(N,1,F,H,W)
        #NOTE: reversed() is necessary!!!
        for d in reversed(range(self.maxdisp)):
            x_slice = x
            #added by CCJ:
            #Note that you don’t need to use torch.narrow or select, 
            #but instead basic indexing will do it for you.
            y_slice = y_pad[:,:,:,d:d+W0]
            
            """ first k*k sum, then channel sum, Memory Consuming!!!"""
            # Newly added by CCJ on Feb 3rd, 2020:
            #x_slice_im2col = self.im2col(x_slice) # in shape [N, C, k*k, H, W]
            #x_slice_im2col *= self.im2col(y_slice)
            #NOTE:broadcasting: [N, C, k*k, H, W] * [N,1,k*k, H, W] ==> [N, C, k*k, H, W]
            #xy_cor_sum_kk = torch.sum(x_slice_im2col*filters, axis = 2, keepdim=False) # [N, C, H, W]
            #xy_cor = torch.sum(xy_cor_sum_kk, axis = 1, keepdim = True) + filters_biases # [N, 1, H, W]
            
            """ first channel sum, then k*k sum, could save Memory Consuming!!!"""
            # Newly added by CCJ on Feb 3rd, 2020:
            xy_slice = torch.sum(x_slice*y_slice, axis = 1, keepdim=True) # [N, 1, H, W]
            xy_slice_im2col = self.im2col(xy_slice) # in shape [N, 1*k*k, H, W]
            #NOTE: [N, 1*k*k, H, W] * [N, k*k, H, W] ==> [N, k*k, H, W]
            xy_cor_sum_kk = torch.sum(xy_slice_im2col*filters, axis = 1, keepdim=True) # [N, 1, H, W]
            xy_cor = xy_cor_sum_kk + filters_biases # [N, 1, H, W]

            #CosineSimilarity
            #cos = nn.CosineSimilarity(dim=1, eps=1e-08)
            #xy_cor = torch.unsqueeze(cos(x_slice,y_slice),1)
            corr_tensor_list.append(xy_cor)
        corr = torch.cat(corr_tensor_list, dim = 1)
        #print ("[???] corr shape: ", corr.shape)
        return corr
