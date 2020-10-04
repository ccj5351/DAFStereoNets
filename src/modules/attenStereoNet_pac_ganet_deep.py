# !/usr/bin/env python3
# -*-coding:utf-8-*-
# @file: attenStereoNet_embed_sga.py
# @brief:
# @author: Changjiang Cai, ccai1@stevens.edu, caicj5351@gmail.com
# @version: 0.0.1
# @creation date: 16-10-2019
# @last modified: Mon 30 Mar 2020 07:11:20 PM EDT

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .embednetwork import embed_net
from .pac import  PacConv2d

#NOTE: Updated by CCJ on 2020/07/17, 17:35;
#PAC works well for all most the cases, except GANet, 
# due to the sync_bn BatchNorm 2d or 3d used by GANet;
from ..baselines.GANet.models.GANet_deep_syncbn_v2 import (
    GANet, BasicConv
    )

from src.net_init import net_init_v0

############################################
""" adapted from GANet paper code """
############################################

"""
our network
"""
# updated: using Python Inheritance:
#class AttenStereoNet(nn.Module):
class AttenStereoNet(GANet):
    def __init__(self, maxdisp=192, 
            kernel_size = 5,
            isPAC = True, 
            isEmbed = False,
            pac_in_channels = 64, # e.g., == cost_volume_channel
            pac_out_channels = 64,
            dilation = 2,
            cost_filter_grad = True,
            native_impl = True
            ):
        
        super(AttenStereoNet, self).__init__(maxdisp = maxdisp)
        
        self.isPAC = isPAC # True of False
        self.isEmbed = isEmbed # True of False
        self.k = kernel_size
        self.pac_in_ch = pac_in_channels
        self.pac_out_ch = pac_out_channels
        self.d = dilation
        self.pad = dilation * (kernel_size - 1) // 2
        self.cost_filter_grad = cost_filter_grad
        """ pixel-adaptive convolution (PAC) network """
        if self.isPAC:
            self.pacconv = PacConv2d(self.pac_in_ch, self.pac_out_ch, kernel_size = self.k, 
                                stride = 1, padding = self.pad, dilation = self.d,
                                native_impl = native_impl
                                )
            if native_impl:
                print(' Enable Native_implement Pixel-Adaptive Convolution (NPAC) Network!!!')
            else:
                print(' Enable (Filter_implemetn) Pixel-Adaptive Convolution (PAC) Network!!!')

        else:
            print ('[!!!] No PAC Network!!')
            self.pacconv = None

        """ embedding network """
        if self.isEmbed:
            print(' PAC adapting feture f comes from Another Embedding Network!!!')
            self.embednet = embed_net()
            
        else:
            self.embednet = None
        
        net_init_v0(self)
        print ("[***] AttenStereoNet weights inilization done!")
        """ the GANet initilization is omitted due to inheritance from GANet """

    def forward(self, x, y):
        """
           args:
              x: left images, in size [N,C,H,W]
              y: right images, in size [N,C,H,W]
        """
        #----------------------
        # extract feature
        #----------------------
        #print ("[???] x.size = ", x.size())
        f_x = self.feature(x)
        rem = f_x
        f_x = self.conv_x(f_x)
        f_y = self.feature(y)
        f_y = self.conv_y(f_y)


        #--------------
        # guidance
        #--------------
        g = self.conv_start(x) # C=32
        x1 = self.conv_refine(rem) # C=32
        x1 = F.interpolate(x1, [x1.size()[2]*3,x1.size()[3]*3], 
                mode='bilinear', align_corners=False)
        x1 = self.bn_relu(x1)
        g = torch.cat((g, x1), 1) # C=64
        g = self.guidance(g)
        

        # feture concatenation to generate cost volume
        # cost volume, in size [N, C, D/3, H/3, W/3]
        #NOTE: +1 is following the GANet paper's func GetCostVolume(), TODO: double check it;
        cv = self.cv(f_x, f_y, self.maxdisp//3 +1) 

        
        pac_guide_fea = None 
        if self.isPAC:
            if not self.isEmbed:
                pac_guide_fea = f_x
            else:
                # downscale x to [N,C,H/3, W/3] then fed into embeddingnet,
                # because the cost volume generated below is in shape [N,C,D/3, H/3, W/3]
                x_scale = F.interpolate(x, [x.size()[2]//3, x.size()[3]//3], 
                        mode='bilinear', align_corners=False)
                pac_guide_fea = self.embednet(x_scale)
                #print ('[???] embed pac_guide_fea shape', pac_guide_fea.shape)
            
            #print ('[???] cv shape', cv.shape)
            #N, C, D, H, W = cv.size()[:]
            D = cv.size()[2]
            
            # NOTE: this might be the memory consuming!!!
            with torch.set_grad_enabled(self.cost_filter_grad):
                for d in range(0,D):
                    cv_d_slice = cv[:,:,d,:,:].contiguous()
                    #print ('[???] cv_d_slice shape', cv_d_slice.shape)
                    cv[:,:,d,:,:] = self.pacconv(
                            input_2d = cv_d_slice,
                            input_for_kernel = pac_guide_fea)

        # make sure the contiguous memeory
        cv = cv.contiguous()

        if self.training:
            disp0, disp1, disp2 = self.cost_agg(cv, g)
            return disp0, disp1, disp2, pac_guide_fea
        else:
            disp2 = self.cost_agg(cv, g)
            return disp2, pac_guide_fea
