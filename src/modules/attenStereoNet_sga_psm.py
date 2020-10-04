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
#from .bilateral import bilateralFilter
from src.net_init import net_init_v0
from src.modules.cost_volume import cost_volume_faster
from .sga_11 import SGA_CostAggregation

"""
our network
"""
# adapted from PSMNet:
# updated: using Python Inheritance:
#class AttenStereoNet(nn.Module):
class AttenStereoNet(PSMNet):
    def __init__(self, maxdisp=192, 
            is_sga_guide_from_img = True,
            #is_quarter_size = True, # feature in 1/4 image size (i.e., H/4 x W/4) or 1/3 size (i.e., H/3 x W/3)
            downsample_scale = 4, # dummy one!!!
            is_lga = False, # generate LGA(Local Guided Aggregation) weights or not
            cost_filter_grad = False
            ):
        super(AttenStereoNet, self).__init__(maxdisp = maxdisp)
        #self.downsample_scale = downsample_scale # dummy one!!!
        self.downsample_scale = 4
        print ("SGA + PSMNet: set downsample_scale = %d" % self.downsample_scale)

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
            cost_volume_in_channels = 64
            )

        """ the followind initilization is omitted due to inheritance from PSMNet """
        net_init_v0(self)
        print ("[***] attenStereoNet_sga_psmnet weights inilization done!")


    """ follow PSM.forward(...) """
    def forward(self, left, right):
        x = self.feature_extraction(left) # left feature
        y = self.feature_extraction(right) # right feature

        # matching volume, in size [N,2C,D/4, H/4, W/4];
        cost = cost_volume_faster(x, y, self.maxdisp//4)
        
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
            cost = self.sga_costAgg(cost, g_in, img_for_g = left) 
            #print ('[???] cost shape', cost.shape)
        
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
            return pred1, pred2, pred3, g_in
        else:
            return pred3, g_in
