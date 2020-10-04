# !/usr/bin/env python3
# -*-coding:utf-8-*-
# @file: attenStereoNet_dyn_psm.py
# @brief:
# @author: Changjiang Cai, ccai1@stevens.edu, caicj5351@gmail.com
# @version: 0.0.1
# @creation date: 08-12-2019
# @last modified: Thu 30 Jan 2020 04:58:35 PM EST
from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math

#from .psmnet_submodule import *
from .psmnet_submodule import PSMNet, disparityregression

#from .embednetwork import embed_net
#from .bilateral import bilateralFilter
from .dfn import filterGenerator, DynamicFilterLayerOneChannel, DynamicFilterLayer
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
            kernel_size = 5,
            crop_img_h = 256,
            crop_img_w = 512,
            isDFN = True, 
            dilation = 2,
            cost_filter_grad = True
            ):
        super(AttenStereoNet, self).__init__(maxdisp=maxdisp)
        #self.maxdisp = maxdisp
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
                    in_channels = 3)
            #the module layer:
            self.dfn_layer = DynamicFilterLayer(kernel_size, dilation)
        else:
            print ('[!!!] No dfn_generator and dfn_layer!!')
            self.dfn_generator = None
            self.dfn_layer = None

        """ the followind initilization is omitted due to inheritance from PSMNet """


    def forward(self, left, right):

        x = self.feature_extraction(left) # left feature, in size [N,C,H/4,W/4]
        y = self.feature_extraction(right) # right feature, in size [N,C,H/4,W/4]

        # matching volume, in size [N,2C,D/4, H/4, W/4]; 
        cost = cost_volume_faster(x, y, self.maxdisp // 4)
        
        
        if not self.isDFN:
            dfn_filter = None
            dfn_bias = None
        else:
            # downscale x to [N,C,H/4, W/4] then fed into dfn_generator,
            # because the cost volume generated below is in shape [N,C,D/4, H/4, W/4]
            left_scale = F.interpolate(left, [left.size()[2]//4, left.size()[3]//4], 
                    mode='bilinear', align_corners=True)
            #print ('[???] left shape', left.shape)
            #print ('[???] left_scale shape', left_scale.shape)
            dfn_filter, dfn_bias = self.dfn_generator(left_scale)
            #N, C, D, H, W = cost.size()[:]
            D = cost.size()[2]
            #print ('[???] cost size = ', cost.size())
            
            # NOTE: this might be the memory consuming!!!
            # NO sure this torch.no_grad() will distory the training or not !!!!
            #with torch.set_grad_enabled(False):
            #with torch.set_grad_enabled(True):
            with torch.set_grad_enabled(self.cost_filter_grad):
                for d in range(0, D):
                #for d in range(0,1):
                    #print ('DFN filtering cost volume slice %d/%d' %(d+1, D))
                    # apply DFN filter to cost volume [N,C,H,W];
                    cv_d_slice = cost[:,:,d,:,:].contiguous()
                    #print ('[???] cv_d_slice shape', cv_d_slice.shape)
                    cost[:,:,d,:,:] = self.dfn_layer(cv_d_slice, dfn_filter, dfn_bias)
             
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

        #cost3 = F.upsample(cost3, [self.maxdisp,left.size()[2],left.size()[3]], mode='trilinear')
        cost3 = F.interpolate(cost3, [self.maxdisp, left.size()[2], left.size()[3]],
                              mode='trilinear', align_corners=True)
        cost3 = torch.squeeze(cost3, 1)
        pred3 = F.softmax(cost3, dim=1)
        # For your information: This formulation 'softmax(c)' learned "similarity"
        # while 'softmax(-c)' learned 'matching cost' as mentioned in the paper.
        # However, 'c' or '-c' do not affect the performance because feature-based cost volume provided flexibility.
        pred3 = disparityregression(self.maxdisp)(pred3)

        if self.training:
            return pred1, pred2, pred3, dfn_filter, dfn_bias
        else:
            return pred3, [dfn_filter, dfn_bias]
