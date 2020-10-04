from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math


from ..baselines.GCNet.models.gcnet import GCNet
from .embednetwork import embed_net
#from .bilateral import bilateralFilter
from src.modules.cost_volume import cost_volume_faster
from .sga_11 import SGA_CostAggregation
from src.net_init import net_init_v0

"""
our network
"""
# adapted from GCNet:
# updated: using Python Inheritance:
#class AttenStereoNet(nn.Module):
class AttenStereoNet(GCNet):
    def __init__(self, 
            maxdisp=192,
            is_sga_guide_from_img = True,
            #is_quarter_size = True, # feature in 1/4 image size (i.e., H/4 x W/4) or 1/3 size (i.e., H/3 x W/3)
            downsample_scale = 4,
            is_lga = False, # generate LGA(Local Guided Aggregation) weights or not
            cost_filter_grad = False,
            is_kendall_version = True, # excatly following the structure in Kendall's GCNet paper;
            is_quarter_size_cost_volume_gcnet = False # cost volume in quarter image size, i.e., [D/4, H/4, W/4]
            ):
        
        super(AttenStereoNet, self).__init__(
            maxdisp = maxdisp,
            #newly added arguments:
            is_kendall_version = is_kendall_version,
            is_quarter_size_cost_volume_gcnet = is_quarter_size_cost_volume_gcnet
            )
        
        self.downsample_scale = downsample_scale # dummy one!!!
        self.downsample_scale = 4 if is_quarter_size_cost_volume_gcnet else 2
        print ("SGA + GCNet: set downsample_scale = %d" % self.downsample_scale)
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
        
        """ the followind initilization is omitted due to inheritance from GCNet """
        net_init_v0(self)
        print ("[***] attenStereoNet_sga_gcnet weights inilization done!")

    
    def forward(self, imgLeft, imgRight):
        N, C, H, W = imgLeft.size()[:]
        assert C == 3, 'should be RGB images as input'
        
        #NOTE: newly added for quarter size cost volume;
        # add one downsample operation:
        if self.is_quarter_size_cost_volume_gcnet:
            img_ds_scale = 2
            imgl = F.interpolate(imgLeft,  [H//2, W//2], mode='bilinear', align_corners=True)
            imgr = F.interpolate(imgRight, [H//2, W//2], mode='bilinear', align_corners=True)
        else:
            img_ds_scale = 1
            imgl = imgLeft
            imgr = imgRight
        
        # feature extraction; 
        f_imgl = self.feature_extraction(imgl)
        f_imgr = self.feature_extraction(imgr)

        # cost volume
        cv = cost_volume_faster(f_imgl, f_imgr, d = self.maxdisp//(2*img_ds_scale))
        #print ("[???] cv shape: ", cv.shape)

        if self.is_sga_guide_from_img:
            g_in = None
        else:
            # downscale x to [N,C,H/4, W/4] then fed into embeddingnet,
            # because the cost volume generated below is in shape [N,C,D/4, H/4, W/4]
            left_scale = F.interpolate(imgLeft, [H//4, W//4], mode='bilinear', align_corners=True)
            """ embed shape [2, 64, 64, 128]"""
            g_in = self.embednet(left_scale)
            #print ('[???] embed shape', embed.shape)
            
        """ apply SGA_CostAggregation() """
        # NOTE: this might be the memory consuming!!!
        with torch.set_grad_enabled(self.cost_filter_grad):
            cv = self.sga_costAgg(cv, g_in, img_for_g = imgLeft) 
            #print ('[???] cost shape', cv.shape)
        
        cv = cv.contiguous()
        
        # cost volume aggregation
        if self.is_kendall_version:
            out = self.cost_aggregation_kendall(cv)
        else:
            out = self.cost_aggregation(cv)
        
        out = out.view(N, self.maxdisp//img_ds_scale, H//img_ds_scale, W//img_ds_scale)
        #NOTE: This is right!!! Updated on 04/12/2020;
        # We should upsample the cost volume (now in quarter size) to full size before the soft-argmin operation;
        # which can gaurantee that the regressed disparity range should be in [0, D) (instead of in [0, D/4));
        if self.is_quarter_size_cost_volume_gcnet:
            # corresponding to the first downsampling at the beginning to the input image pair;
            out = out[:,None,...] # add channel C first, i.e., chang [N,D,H,W] to [N,C=1,D,H,W];
            out = F.interpolate(out, [self.maxdisp, H, W], mode='trilinear', align_corners=True) # in size [N,C=1,D,H,W];
            out = torch.squeeze(out, 1) # in size [N,D,H,W]
        prob = F.softmax(out, 1)
        #disp = self.disparityregression(prob, maxdisp=self.maxdisp//img_ds_scale)
        #NOTE: This is right!!! Updated on 04/12/2020;
        disp = self.disparityregression(prob, maxdisp=self.maxdisp)
        return disp, g_in