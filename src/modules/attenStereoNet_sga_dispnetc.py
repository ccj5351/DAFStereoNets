# !/usr/bin/env python3
# -*-coding:utf-8-*-
# @file: attenStereoNet_sga_dispnetc.py
# @brief:
# @author: Changjiang Cai, ccai1@stevens.edu, caicj5351@gmail.com
# @version: 0.0.1
# @creation date: 17-03-2020
# @last modified: Wed 18 Mar 2020 08:35:52 PM EDT
from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math
#from .psmnet_submodule import *
from .psmnet_submodule import PSMNet, disparityregression

#from .bilateral import bilateralFilter
#from src.net_init import net_init_v0
from .embednetwork import embed_net
from .sga_11 import SGA_CostAggregation
from ..baselines.DispNet.models.dispnet import DispNet
"""
our network
"""
# adapted from DispNet:
class AttenStereoNet(DispNet):
    def __init__(self, maxdisp=192, 
            is_sga_guide_from_img = True,
            #is_quarter_size = True, # feature in 1/4 image size (i.e., H/4 x W/4) or 1/3 size (i.e., H/3 x W/3)
            downsample_scale = 4, # dummy one!!!
            is_lga = False, # generate LGA(Local Guided Aggregation) weights or not
            cost_filter_grad = False
            ):
        # due to TWO consecutive downsampling, so here maxdisp=40, 
        # actually means 4*maxdisp=160 in the original input image pair;
        super(AttenStereoNet, self).__init__( 
                is_corr = True, 
                maxdisp_corr = maxdisp //4, # for maxdisp_corr!! 
                corr_func_type = 'correlation1D_map_V1',
                is_bn = True,
                is_relu = True)
        #self.downsample_scale = downsample_scale # dummy one!!!
        self.downsample_scale = 4
        print ("SGA + DispNetC: set downsample_scale = %d" % self.downsample_scale)

        self.is_sga_guide_from_img = is_sga_guide_from_img # True of False
        self.cost_filter_grad = cost_filter_grad
        #self.is_quarter_size = is_quarter_size
        self.is_lga =  is_lga
        
        if self.is_sga_guide_from_img:
            print('is_sga_guide_from_img = True !!!')
            self.embednet = None
        else:
            """ embedding network """
            print('is_sga_guide_from_img = False !!!')
            print('SGA_CostAggregation uses Embedding Network!!!')
            self.embednet = embed_net()
            
        self.sga_costAgg = SGA_CostAggregation(
            self.is_sga_guide_from_img,
            #self.is_quarter_size,
            self.downsample_scale,
            self.is_lga,
            cost_volume_in_channels = 1
            )
        """ the followind initilization is omitted due to inheritance from PSMNet """


    def forward(self, left, right):
        """ DispNetC """
        #left image, in size [N, 3, H, W]
        out_x = self.conv1(left)
        shortcut_c1 = out_x
        out_x = self.conv2(out_x)
        shortcut_c2 = out_x
        
        #right image
        out_y = self.conv1(right)
        out_y = self.conv2(out_y)
        # correlation map, in [N, D, H/4, W/4]
        out = self.corr_func(out_x, out_y)
        out_redir = self.conv_redir(out_x)

        if self.is_sga_guide_from_img:
            g_in = None
        else:
            # downscale x to [N,C,H/4, W/4] then fed into embeddingnet,
            # because the cost volume generated below is in shape [N,C,D/4, H/4, W/4]
            left_scale = F.interpolate(left, [left.size()[2]//4, left.size()[3]//4], 
                    mode='bilinear', align_corners=True)
            #print ('[???] left shape', left.shape)
            #print ('[???] left_scale shape', left_scale.shape)
            """ embed shape [2, 64, 64, 128]"""
            g_in = self.embednet(left_scale)
            #print ('[???] embed shape', embed.shape)
            
        """ apply SGA_CostAggregation() """
        # NOTE: this might be the memory consuming!!!
        with torch.set_grad_enabled(self.cost_filter_grad):
            out = self.sga_costAgg(
                out[:,None,:], # add C=1 dimension;
                g_in, img_for_g = left) 
            #print ('[???] cost shape', out.shape)
            out = torch.squeeze(out, dim=1)
            #print ('[???] cost shape', out.shape)
        


        out = self.conv3a(torch.cat((out,out_redir),dim=1))
        # commen parts
        out = self.conv3b(out)
        shortcut_c3 = out
        out = self.conv4a(out)
        out = self.conv4b(out)
        shortcut_c4 = out
        out = self.conv5a(out)
        out = self.conv5b(out)
        shortcut_c5 = out
        out = self.conv6a(out)
        out = self.conv6b(out)
        disp6 = self.conv_disp6(out) # in size [N, 1, H/64, W/64]
        
        # decoder 5
        out = self.upc5(out)
        out = self.ic5(torch.cat((out, shortcut_c5, self.upconv_disp6(disp6)), dim=1))
        disp5 = self.conv_disp5(out) # in size [N, 1, H/32, W/32]

        # decoder 4
        out = self.upc4(out)
        out = self.ic4(torch.cat((out, shortcut_c4, self.upconv_disp5(disp5)), dim=1))
        disp4 = self.conv_disp4(out) # in size [N, 1, H/16, W/16]
        
        # decoder 3
        out = self.upc3(out)
        out = self.ic3(torch.cat((out, shortcut_c3, self.upconv_disp4(disp4)), dim=1))
        disp3 = self.conv_disp3(out) # in size [N, 1, H/8, W/8]
        #print ("[???] disp3: ", disp3.shape)
        
        # decoder 2
        out = self.upc2(out)
        out = self.ic2(torch.cat((out, shortcut_c2, self.upconv_disp3(disp3)), dim=1))
        disp2 = self.conv_disp2(out) # in size [N, 1, H/4, W/4]
        #print ("[???] disp2: ", disp2.shape)
        
        # decoder 1
        out = self.upc1(out)
        out = self.ic1(torch.cat((out, shortcut_c1, self.upconv_disp2(disp2)), dim=1))
        disp1 = self.conv_disp1(out) # in size [N, 1, H/2, W/2]
        #print ("[???] disp1: ", disp1.shape)
        
        # added by CCJ for original size disp as output
        # NOTE: disp3 is in 1/8 size ==> interpolated to full size;
        H3, W3 = disp3.size()[2:]
        H0, W0 = 8*H3, 8*W3
        disp_out1 = torch.squeeze(
                F.interpolate(disp3, [H0, W0], mode='bilinear', align_corners = True),
                1)# squeeze disp [N, 1, H, W] to [N, H, W]
        
        # NOTE: disp2 is in 1/4 size ==> interpolated to full size;
        disp_out2 = torch.squeeze(
                F.interpolate(disp2, [H0, W0], mode='bilinear', align_corners = True),
                1)# squeeze disp [N, 1, H, W] to [N, H, W]
        
        # NOTE: disp1 is in 1/2 size ==> interpolated to full size;
        disp_out3 = torch.squeeze(
                F.interpolate(disp1, [H0, W0], mode='bilinear', align_corners = True),
                1)# squeeze disp [N, 1, H, W] to [N, H, W]
        
        if self.training:
            return disp_out1, disp_out2, disp_out3, g_in
        else:
            return disp_out3, g_in