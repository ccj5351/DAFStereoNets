# !/usr/bin/env python3
# -*-coding:utf-8-*-
# @file: attenStereoNet_pac_psm.py
# @brief:
# @author: Changjiang Cai, ccai1@stevens.edu, caicj5351@gmail.com
# @version: 0.0.1
# @creation date: 17-12-2019
# @last modified: Sat 01 Feb 2020 07:28:24 PM EST
from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math
#from .psmnet_submodule import *
from .psmnet_submodule import PSMNet, disparityregression
from .embednetwork import embed_net
from .pac import  PacConv2d
#from src.net_init import net_init_v0
from src.modules.cost_volume import cost_volume_faster

"""
our network
"""
# adapted from PSMNet:
# updated: using Python Inheritance:
#class AttenStereoNet(nn.Module):
class AttenStereoNet(PSMNet):
    def __init__(self, maxdisp=192,
            kernel_size = 9,
            isPAC = True, 
            isEmbed = False,
            pac_in_channels = 64, # e.g., == cost_volume_channel
            pac_out_channels = 64,
            dilation = 1,
            cost_filter_grad = True,
            native_impl = True
            ):
        super(AttenStereoNet, self).__init__(maxdisp=maxdisp)
        #self.maxdisp = maxdisp
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

        """ the PSMNet initilization is omitted due to inheritance from PSMNet """

    def forward(self, left, right):

        x = self.feature_extraction(left) # left feature, in size [N,C,H/4,W/4]
        y = self.feature_extraction(right)# right feature, in size [N,C,H/4,W/4]
        # matching volume, in size [N,2C,D/4, H/4, W/4];
        cost = cost_volume_faster(x, y, self.maxdisp//4)
        
        pac_guide_fea = None 
        if self.isPAC:
            if not self.isEmbed:
                pac_guide_fea = x
            else:
                # downscale x to [N,C,H/4, W/4] then fed into embeddingnet,
                # because the cost volume generated below is in shape [N,C,D/4, H/4, W/4]
                left_scale = F.interpolate(left, [left.size()[2]//4, left.size()[3]//4], 
                        mode='bilinear', align_corners=True)
                #print ('[???] left shape', left.shape)
                #print ('[???] left_scale shape', left_scale.shape)
                """ embed shape [2, 64, 64, 128]"""
                pac_guide_fea = self.embednet(left_scale)
                #print ('[???] embed pac_guide_fea shape', pac_guide_fea.shape)
            
            N, C, D, H, W = cost.size()[:]
            #print ('[???] cost size = ', cost.size())
            
            # NOTE: this might be the memory consuming!!!
            # NO sure this torch.no_grad() will distory the training or not !!!!
            #with torch.set_grad_enabled(False):
            #with torch.set_grad_enabled(True):
            #print ("[***???] self.cost_filter_grad = ", self.cost_filter_grad)
            with torch.set_grad_enabled(self.cost_filter_grad):
                for d in range(0,D):
                #for d in range(0,1):
                    cv_d_slice = cost[:,:,d,:,:].contiguous()
                    #print ('[???] cv_d_slice shape', cv_d_slice.shape)
                    cost[:,:,d,:,:] = self.pacconv(
                            input_2d = cv_d_slice,
                            input_for_kernel = pac_guide_fea)

        cost = cost.contiguous()

        cost0 = self.dres0(cost)
        cost0 = self.dres1(cost0) + cost0

        out1, pre1, post1 = self.dres2(cost0, None, None)
        out1 = out1+cost0

        out2, pre2, post2 = self.dres3(out1, pre1, post1)
        out2 = out2+cost0

        out3, pre3, post3 = self.dres4(out2, pre1, post2)
        out3 = out3+cost0

        cost1 = self.classif1(out1)
        cost2 = self.classif2(out2) + cost1
        cost3 = self.classif3(out3) + cost2

        if self.training:
            # updated by CCJ: due to deprecated warning!
            #cost1 = F.upsample(cost1, [self.maxdisp,left.size()[2],left.size()[3]], mode='trilinear')
            #cost2 = F.upsample(cost2, [self.maxdisp,left.size()[2],left.size()[3]], mode='trilinear')
            cost1 = F.interpolate(cost1, [self.maxdisp, left.size()[2], left.size()[3]],
                                  mode='trilinear', align_corners=True)
            cost2 = F.interpolate(cost2, [self.maxdisp, left.size()[2], left.size()[3]],
                                  mode='trilinear', align_corners=True)

            cost1 = torch.squeeze(cost1, 1)
            pred1 = F.softmax(cost1, dim=1)
            pred1 = disparityregression(self.maxdisp)(pred1)

            cost2 = torch.squeeze(cost2, 1)
            pred2 = F.softmax(cost2, dim=1)
            pred2 = disparityregression(self.maxdisp)(pred2)

        cost3 = F.interpolate(cost3, [self.maxdisp, left.size()[2], left.size()[3]],
                              mode='trilinear', align_corners=True)
        cost3 = torch.squeeze(cost3, 1)
        pred3 = F.softmax(cost3, dim=1)
        # For your information: This formulation 'softmax(c)' learned "similarity"
        # while 'softmax(-c)' learned 'matching cost' as mentioned in the paper.
        # However, 'c' or '-c' do not affect the performance because feature-based 
        # cost volume provided flexibility.
        pred3 = disparityregression(self.maxdisp)(pred3)

        if self.training:
            return pred1, pred2, pred3, pac_guide_fea
        else:
            return pred3, pac_guide_fea
