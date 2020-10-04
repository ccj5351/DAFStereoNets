# !/usr/bin/env python3
# -*-coding:utf-8-*-
# @file: attenStereoNet_embed_sga.py
# @brief:
# @author: Changjiang Cai, ccai1@stevens.edu, caicj5351@gmail.com
# @version: 0.0.1
# @creation date: 16-10-2019
# @last modified: Sat 23 Nov 2019 11:58:12 PM EST

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .dfn import filterGenerator, DynamicFilterLayerOneChannel, DynamicFilterLayer

#NOTE: Updated by CCJ on 2020/07/17, 17:35;
# DFN works well for all most the cases, except GANet, due to the sync_bn BatchNorm 2d or 3d used by GANet;
from ..baselines.GANet.models.GANet_deep_syncbn_v2 import (
    GANet, BasicConv
    )

from src.net_init import net_init_v0, net_init_SyncBN
# see this problem: 
# 1) Question about SyncBN open-mmlab/mmdetection#933, at https://github.com/open-mmlab/mmdetection/issues/933;
# 2) Multi-GPU training process stuck randomly #219, at https://github.com/facebookresearch/Detectron/issues/219;

############################################
""" adapted from GANet paper code """
############################################

"""
our network
"""
# updated: using Python Inheritance:
#class AttenStereoNet(nn.Module):
class AttenStereoNet(GANet):
    def __init__(self, 
            maxdisp = 192,
            kernel_size = 5,
            crop_img_h = 256,
            crop_img_w = 512,
            isDFN = True, 
            dilation = 2,
            cost_filter_grad = False
            ):
        super(AttenStereoNet, self).__init__(maxdisp = maxdisp)
        self.isDFN = isDFN # True of False
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.cost_filter_grad = cost_filter_grad
        


        """ dynamic filter network """
        if self.isDFN:
            print(' Enable Dynamic Filter Network!!!')
            self.dfn_generator = filterGenerator(F = 32, 
                    dynamic_filter_size=(kernel_size, kernel_size), 
                    #img_size = (crop_img_h//4, crop_img_w//4), # due to 1/4 downsampling in PSMNet;
                    in_channels = 3,
                    #NOTE: Updated by CCJ on 2020/07/17, 3:47AM;
                    # net_init() function works well for all most the cases, except GANet, 
                    # due to sync BN used by GANet;
                    # So here instead, we will use net_init_SyncBN();
                    is_sync_bn=True
                    )
            #the module layer:
            self.dfn_layer = DynamicFilterLayer(kernel_size, dilation)
        else:
            print ('[!!!] No dfn_generator and dfn_layer!!')
            self.dfn_generator = None
            self.dfn_layer = None
        
        """ the followind initilization is omitted due to inheritance from GANet """
        #net_init_SyncBN(self)
        print ("[***] AttenStereoNet_dfn_ganet_deep weights inilization done!")


    def forward(self, x, y):
        """
           args:
              x: left images, in size [N,C,H,W]
              y: right images, in size [N,C,H,W]
        """
        #----------------------
        # extract feature
        #----------------------
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

        if not self.isDFN:
            dfn_filter = None
            dfn_bias = None

        else: # using DFN
            # downscale x to [N,C,H/3, W/3] then fed into embeddingnet,
            # because the cost volume generated below is in shape [N,C,D/3, H/3, W/3]
            x_scale = F.interpolate(x, [x.size()[2]//3, x.size()[3]//3], 
                    mode='bilinear', align_corners=False)
            dfn_filter, dfn_bias = self.dfn_generator(x_scale) 
            #print ('[???] cv shape', cv.shape, "device = ", dfn_filter.get_device())
            #N, C, D, H, W = cv.size()[:]
            D = cv.size()[2]
            
            # NOTE: this is the memory problem ???
            with torch.set_grad_enabled(self.cost_filter_grad):
                for d in range(0,D):
                    #print ('bilateral filtering cost volume slice %d/%d' %(d+1, D))
                    cv_d_slice = cv[:,:,d,:,:].contiguous()
                    cv[:,:,d,:,:] = self.dfn_layer(cv_d_slice, dfn_filter, dfn_bias)
            
            # make sure the contiguous memeory
            cv = cv.contiguous()
            #print ('[???] done dfn(cv), device = ', cv.get_device())

        if self.training:
            #print ('[???] to do cost_agg, device = ', cv.get_device())
            disp0, disp1, disp2 = self.cost_agg(cv, g)
            #print ('[???] finished cost_agg, device = ', disp0.get_device())
            return disp0, disp1, disp2, dfn_filter, dfn_bias
        else:
            disp2 = self.cost_agg(cv, g)
            return disp2, [dfn_filter, dfn_bias]