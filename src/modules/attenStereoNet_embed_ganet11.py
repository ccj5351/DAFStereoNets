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

#from ..baselines.GANet.libs.GANet.modules.GANet import SGA
from .embednetwork import embed_net
from .bilateral import bilateralFilter
#from .bilateral_func import bilateralFilter

from ..baselines.GANet.models.GANet11 import (
    GANet, BasicConv
    )

from src.net_init import net_init

############################################
""" adapted from GANet paper code """
############################################


"""
our network
"""
# updated: using Python Inheritance:
#class AttenStereoNet(nn.Module):
class AttenStereoNet(GANet):
    def __init__(self, maxdisp=192, sigma_s = 0.7, # 1.7: 13 x 13; 0.3 : 3 x 3;
                 sigma_v = 0.1, isEmbed = True, 
                 dilation = 1,
                 cost_filter_grad = False
                 ):
        super(AttenStereoNet, self).__init__(maxdisp = maxdisp)
        #self.maxdisp = maxdisp

        self.isEmbed = isEmbed # True of False
        self.sigma_s = sigma_s
        self.sigma_v = sigma_v
        self.dilation = dilation
        self.cost_filter_grad = cost_filter_grad

        """ embedding network """
        if self.isEmbed:
            print(' Enable Embedding Network!!!')
            self.embednet = embed_net()
            #the module layer:
            self.bifilter = bilateralFilter(sigma_s, sigma_v, isCUDA = True, dilation = self.dilation)
        else:
            self.embednet = None
            self.bifilter = None
        
        """ the followind initilization is omitted due to inheritance from GANet """
        net_init(self)
        print ("[***] AttenStereoNet weights inilization done!")


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

        if not self.isEmbed:
            embed = None

        else: # using embedding
            # downscale x to [N,C,H/3, W/3] then fed into embeddingnet,
            # because the cost volume generated below is in shape [N,C,D/3, H/3, W/3]
            x_scale = F.interpolate(x, [x.size()[2]//3, x.size()[3]//3], 
                    mode='bilinear', align_corners=False)
            embed = self.embednet(x_scale)
            
            #print ('[???] cv shape', cv.shape)
            #N, C, D, H, W = cv.size()[:]
            D = cv.size()[2]
            
            # NOTE: this is the memory problem ???
            with torch.set_grad_enabled(self.cost_filter_grad):
                for d in range(0,D):
                    #print ('bilateral filtering cost volume slice %d/%d' %(d+1, D))
                    cv_d_slice = cv[:,:,d,:,:]
                    cv[:,:,d,:,:] = self.bifilter(embed, cv_d_slice)
            
            # make sure the contiguous memeory
            cv = cv.contiguous()

        if self.training:
            disp0, disp1 = self.cost_agg(cv, g)
            return disp0, disp1, embed
        else:
            disp1 = self.cost_agg(cv, g)
            return disp1, embed