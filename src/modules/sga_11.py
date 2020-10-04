# !/usr/bin/env python3
# -*-coding:utf-8-*-
# @file: attenStereoNet_embed_sga_11.py
# @brief:
# @author: Changjiang Cai, ccai1@stevens.edu, caicj5351@gmail.com
# @version: 0.0.1
# @creation date: 16-10-2019
# @last modified: Mon 11 May 2020 01:02:03 AM EDT

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ..baselines.GANet.libs.GANet.modules.GANet import SGA
#from .embednetwork import embed_net
#from .bilateral import bilateralFilter
from ..baselines.GANet.libs.sync_bn.modules.sync_bn import BatchNorm2d, BatchNorm3d
from ..baselines.GANet.libs.GANet.modules.GANet import DisparityRegression
#from ..baselines.GANet.libs.GANet.modules.GANet import GetCostVolume
from ..baselines.GANet.libs.GANet.modules.GANet import LGA, LGA2, LGA3

############################################
""" adapted from GANet paper code """
############################################

class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, bn=True, relu=True, **kwargs):
        super(BasicConv, self).__init__()
#        print(in_channels, out_channels, deconv, is_3d, bn, relu, kwargs)
        self.relu = relu
        self.use_bn = bn
        if is_3d:
            if deconv:
                self.conv = nn.ConvTranspose3d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv3d(in_channels, out_channels, bias=False, **kwargs)
            self.bn = BatchNorm3d(out_channels)
        else:
            if deconv:
                self.conv = nn.ConvTranspose2d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
            self.bn = BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x


class Conv2x(nn.Module):

    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, concat=True, bn=True, relu=True):
        super(Conv2x, self).__init__()
        self.concat = concat

        kernel = 3
        if deconv and is_3d: 
            #kernel = (3, 4, 4)
            #updated by CCJ:
            kwargs = {
                'stride': 2, 
                'padding': 1,
                'output_padding':1
            }

        elif deconv:
            #kernel = 4
            kwargs = {
                'stride': 2, 
                'padding': 1,
                'output_padding':1
            }
        else:
            kwargs = {
                'stride': 2, 
                'padding': 1,
            }
        #self.conv1 = BasicConv(in_channels, out_channels, deconv, is_3d, bn=True, relu=True, kernel_size=kernel, stride=2, padding=1)
        self.conv1 = BasicConv(in_channels, out_channels, deconv, is_3d, bn=True, relu=True, kernel_size=kernel, **kwargs)

        if self.concat: 
            self.conv2 = BasicConv(out_channels*2, out_channels, False, is_3d, bn, relu, kernel_size=3, stride=1, padding=1)
        else:
            self.conv2 = BasicConv(out_channels, out_channels, False, is_3d, bn, relu, kernel_size=3, stride=1, padding=1)

    def forward(self, x, rem):
        x = self.conv1(x)
        #print ("[???] x size = ", x.size())
        #print ("[???] rem size = ", rem.size())
        assert(x.size() == rem.size())
        if self.concat:
            x = torch.cat((x, rem), 1)
        else: 
            x = x + rem
        x = self.conv2(x)
        return x


class Feature(nn.Module):
    def __init__(self, 
        #is_quarter_size = True
        downsample_scale = 3
        ):

        super(Feature, self).__init__()
        #if not is_quarter_size:
        assert downsample_scale in [2, 3, 4], "downsample_scale should be 2, 3, or 4!!!"
        if downsample_scale == 3:
            print ("[***] Feature() to 1/3 image size for original GANet !!!")
            self.conv_start = nn.Sequential(
                # Added by CCJ:
                # Convolution In/Out Size: O = floor{(W - F + 2P)/S + 1}
                BasicConv(3, 32, kernel_size=3, padding=1),
                BasicConv(32, 32, kernel_size=5, stride=3, padding=2), #in size [H/3, W/3]
                BasicConv(32, 32, kernel_size=3, padding=1))
        
        elif downsample_scale == 4:
            print ("[***] Feature() to 1/4 image size for PSMNet etc !!!")
            self.conv_start = nn.Sequential(
                # Added by CCJ:
                # Convolution In/Out Size: O = floor{(W - F + 2P)/S + 1}
                BasicConv(3, 32, kernel_size=3, padding=1),
                BasicConv(32, 32, kernel_size=3, stride=2, padding=1), #in size [H/2, W/2]
                BasicConv(32, 32, kernel_size=3, stride=2, padding=1), #in size [H/4, W/4]
                BasicConv(32, 32, kernel_size=3, padding=1))
        
        elif downsample_scale == 2:
            print ("[***] Feature() to 1/2 image size for GCNet etc !!!")
            self.conv_start = nn.Sequential(
                # Added by CCJ:
                # Convolution In/Out Size: O = floor{(W - F + 2P)/S + 1}
                BasicConv(3, 32, kernel_size=3, padding=1),
                BasicConv(32, 32, kernel_size=3, stride=2, padding=1), #in size [H/2, W/2]
                BasicConv(32, 32, kernel_size=3, padding=1))
        
        #else:
        #    raise Exception("No suitable downsample_scale value found ...")
        
        self.conv1a = BasicConv(32, 48, kernel_size=3, stride=2, padding=1)
        self.conv2a = BasicConv(48, 64, kernel_size=3, stride=2, padding=1)
        self.conv3a = BasicConv(64, 96, kernel_size=3, stride=2, padding=1)
        self.conv4a = BasicConv(96, 128, kernel_size=3, stride=2, padding=1)

        self.deconv4a = Conv2x(128, 96, deconv=True)
        self.deconv3a = Conv2x(96, 64, deconv=True)
        self.deconv2a = Conv2x(64, 48, deconv=True)
        self.deconv1a = Conv2x(48, 32, deconv=True)

        self.conv1b = Conv2x(32, 48) # default: k=3,s=2,p=1
        self.conv2b = Conv2x(48, 64)
        self.conv3b = Conv2x(64, 96)
        self.conv4b = Conv2x(96, 128)

        self.deconv4b = Conv2x(128, 96, deconv=True)
        self.deconv3b = Conv2x(96, 64, deconv=True)
        self.deconv2b = Conv2x(64, 48, deconv=True)
        self.deconv1b = Conv2x(48, 32, deconv=True)

    def forward(self, x):
        x = self.conv_start(x)
        rem0 = x
        x = self.conv1a(x)
        rem1 = x
        x = self.conv2a(x)
        rem2 = x
        x = self.conv3a(x)
        rem3 = x
        x = self.conv4a(x)
        rem4 = x
        x = self.deconv4a(x, rem3)
        rem3 = x

        x = self.deconv3a(x, rem2)
        rem2 = x
        x = self.deconv2a(x, rem1)
        rem1 = x
        x = self.deconv1a(x, rem0)
        rem0 = x

        x = self.conv1b(x, rem1)
        rem1 = x
        x = self.conv2b(x, rem2)
        rem2 = x
        x = self.conv3b(x, rem3)
        rem3 = x
        x = self.conv4b(x, rem4)

        x = self.deconv4b(x, rem3)
        x = self.deconv3b(x, rem2)
        x = self.deconv2b(x, rem1)
        x = self.deconv1b(x, rem0)

        return x

class Guidance_11(nn.Module):
    def __init__(
        self, 
        #is_quarter_size = True,
        # could be: 
        # 2: Half size, i.e., [H/2, W/2]
        # 3: 1/3 size, i.e., [H/3, W/3]
        # 4: quarter size, i.e., [H/4, W/4]
        downsample_scale = 3, 
        is_lga = False):

        super(Guidance_11, self).__init__()
        #self.is_quarter_size = is_quarter_size
        
        assert downsample_scale in [2, 3, 4], "downsample_scale should be 2, 3, or 4!!!"
        self.is_lga = is_lga
        self.conv0 = BasicConv(64, 16, kernel_size=3, padding=1)

        #if not is_quarter_size:
        if downsample_scale == 3:
            print ("[***] Guidance_11() module to 1/3 image size for original GANet !!!")
            self.conv1 = nn.Sequential(
                BasicConv(16, 32, kernel_size=5, stride=3, padding=2),#in size [H/3, W/3]
                BasicConv(32, 32, kernel_size=3, padding=1))
        elif downsample_scale == 4:
            print ("[***] Guidance_11() module to 1/4 image size for PSMNet etc !!!")
            self.conv1 = nn.Sequential(
                BasicConv(16, 32, kernel_size=3, stride=2, padding=1),#in size [H/2, W/2]
                BasicConv(32, 32, kernel_size=3, stride=2, padding=1))#in size [H/4, W/4]
        elif downsample_scale == 2:
            print ("[***] Guidance_11() module to 1/2 image size for GCNet etc !!!")
            self.conv1 = nn.Sequential(
                BasicConv(16, 32, kernel_size=3, stride=2, padding=1),#in size [H/2, W/2]
                BasicConv(32, 32, kernel_size=3, stride=1, padding=1))#in size [H/2, W/2]
        #else:
        #    raise Exception("No suitable downsample_scale value found ...")


        self.conv2 = BasicConv(32, 32, kernel_size=3, padding=1)
        self.conv3 = BasicConv(32, 32, kernel_size=3, padding=1)

#        self.conv11 = Conv2x(32, 48)
        self.conv11 = nn.Sequential(BasicConv(32, 48, kernel_size=3, stride=2, padding=1),
                                    BasicConv(48, 48, kernel_size=3, padding=1))
        self.conv12 = BasicConv(48, 48, kernel_size=3, padding=1)
        self.conv13 = BasicConv(48, 48, kernel_size=3, padding=1)
        self.conv14 = BasicConv(48, 48, kernel_size=3, padding=1)

        self.weight_sg1 = nn.Conv2d(32, 640, (3, 3), (1, 1), (1, 1), bias=False)
        self.weight_sg2 = nn.Conv2d(32, 640, (3, 3), (1, 1), (1, 1), bias=False)
        self.weight_sg3 = nn.Conv2d(32, 640, (3, 3), (1, 1), (1, 1), bias=False)

        self.weight_sg11 = nn.Conv2d(48, 960, (3, 3), (1, 1), (1, 1), bias=False)
        self.weight_sg12 = nn.Conv2d(48, 960, (3, 3), (1, 1), (1, 1), bias=False)
        self.weight_sg13 = nn.Conv2d(48, 960, (3, 3), (1, 1), (1, 1), bias=False)
        self.weight_sg14 = nn.Conv2d(48, 960, (3, 3), (1, 1), (1, 1), bias=False)
        
        if self.is_lga:
            self.weight_lg1 = nn.Sequential(BasicConv(16, 16, kernel_size=3, padding=1),
                                            nn.Conv2d(16, 75, (3, 3), (1, 1), (1, 1) ,bias=False))
            self.weight_lg2 = nn.Sequential(BasicConv(16, 16, kernel_size=3, padding=1),
                                            nn.Conv2d(16, 75, (3, 3), (1, 1), (1, 1) ,bias=False))

    def forward(self, x):
        x = self.conv0(x)
        rem = x
        x = self.conv1(x)
        sg1 = self.weight_sg1(x)
        x = self.conv2(x)
        sg2 = self.weight_sg2(x)
        x = self.conv3(x)
        sg3 = self.weight_sg3(x)

        x = self.conv11(x)
        sg11 = self.weight_sg11(x)
        x = self.conv12(x)
        sg12 = self.weight_sg12(x)
        x = self.conv13(x)
        sg13 = self.weight_sg13(x)
        x = self.conv14(x)
        sg14 = self.weight_sg14(x)
        if self.is_lga:
            lg1 = self.weight_lg1(rem)
            lg2 = self.weight_lg2(rem)
        else:
            lg1 = None
            lg2 = None
       
        return dict([
            ('sg1', sg1),
            ('sg2', sg2),
            ('sg3', sg3),
            ('sg11', sg11),
            ('sg12', sg12),
            ('sg13', sg13),
            ('sg14', sg14),
            ('lg1', lg1),
            ('lg2', lg2)])


class Disp(nn.Module):

    def __init__(self, maxdisp=192):
        super(Disp, self).__init__()
        self.maxdisp = maxdisp
        self.softmax = nn.Softmin(dim=1)
        self.disparity = DisparityRegression(maxdisp=self.maxdisp)
#        self.conv32x1 = BasicConv(32, 1, kernel_size=3)
        self.conv32x1 = nn.Conv3d(32, 1, (3, 3, 3), (1, 1, 1), (1, 1, 1), bias=False)

    def forward(self, x):
        x = F.interpolate(self.conv32x1(x), [self.maxdisp+1, x.size()[3]*3, x.size()[4]*3], 
                mode='trilinear', align_corners=False)
        x = torch.squeeze(x, 1)
        x = self.softmax(x)

        return self.disparity(x)

class DispAgg(nn.Module):

    def __init__(self, maxdisp=192):
        super(DispAgg, self).__init__()
        self.maxdisp = maxdisp
        self.LGA3 = LGA3(radius=2)
        self.LGA2 = LGA2(radius=2)
        self.LGA = LGA(radius=2)
        self.softmax = nn.Softmin(dim=1)
        self.disparity = DisparityRegression(maxdisp=self.maxdisp)
#        self.conv32x1 = BasicConv(32, 1, kernel_size=3)
        self.conv32x1=nn.Conv3d(32, 1, (3, 3, 3), (1, 1, 1), (1, 1, 1), bias=False)

    def lga(self, x, g):
        g = F.normalize(g, p=1, dim=1)
        x = self.LGA2(x, g)
        return x

    def forward(self, x, lg1, lg2):
        x = F.interpolate(self.conv32x1(x), [self.maxdisp+1, x.size()[3]*3, x.size()[4]*3], mode='trilinear', 
                align_corners=False)
        x = torch.squeeze(x, 1)
        assert(lg1.size() == lg2.size())
        x = self.lga(x, lg1)
        x = self.softmax(x)
        x = self.lga(x, lg2)
        x = F.normalize(x, p=1, dim=1)
        return self.disparity(x)

class SGABlock(nn.Module):
    def __init__(self, channels=32, refine=False):
        super(SGABlock, self).__init__()
        self.refine = refine
        if self.refine:
            self.bn_relu = nn.Sequential(BatchNorm3d(channels),
                                         nn.ReLU(inplace=True))
            self.conv_refine = BasicConv(channels, channels, is_3d=True, kernel_size=3, padding=1, relu=False)
#            self.conv_refine1 = BasicConv(8, 8, is_3d=True, kernel_size=1, padding=1)
        else:
            self.bn = BatchNorm3d(channels)
        self.SGA=SGA()
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x, g):
        rem = x
        #NOTE:
        #Comments added by CCJ:
        # split g channel C (e.g., C= 640) to 4 parts, corresponding to four directions (left, right, up and down), 
        # each with C/4 ( e.g., = 640/4=160) size along channel dim, i.e., dim=1;
        # each C/4=160-dim vector is further divided into 32 x 5, where 32 is the same as input x channel, 
        # and 5 means w0, w1, ..., w4 in Eq (5) in GANet CVPR paper, s.t. w0 + w1 + ... + w4 = 1.0, 
        # this why F.normalize() is applied along dim=5, that is normalize those five values, s.t. w0 + w1 + ... + w4 = 1.0 !!!
        k1, k2, k3, k4 = torch.split(g, (x.size()[1]*5, x.size()[1]*5, x.size()[1]*5, x.size()[1]*5), 1)
        k1 = F.normalize(k1.view(x.size()[0], x.size()[1], 5, x.size()[3], x.size()[4]), p=1, dim=2)
        k2 = F.normalize(k2.view(x.size()[0], x.size()[1], 5, x.size()[3], x.size()[4]), p=1, dim=2)
        k3 = F.normalize(k3.view(x.size()[0], x.size()[1], 5, x.size()[3], x.size()[4]), p=1, dim=2)
        k4 = F.normalize(k4.view(x.size()[0], x.size()[1], 5, x.size()[3], x.size()[4]), p=1, dim=2)
        x = self.SGA(x, k1, k2, k3, k4)
        if self.refine:
            x = self.bn_relu(x)
            x = self.conv_refine(x)
        else:
            x = self.bn(x)
        assert(x.size() == rem.size())
        x += rem
        return self.relu(x)    
#        return self.bn_relu(x)


class CostAggregation_11(nn.Module):
    def __init__(self, 
        cost_volume_in_channels = 64, # for DispNetC channels = 1, for PSMNet channels = 64;
        ):
        super(CostAggregation_11, self).__init__()
        print ("[***] cost_volume_in_channels = ", cost_volume_in_channels)
        self.conv_start = BasicConv(cost_volume_in_channels, 32, is_3d=True, kernel_size=3, padding=1, relu=False)
        self.conv_end = BasicConv(32, cost_volume_in_channels, is_3d=True, kernel_size=3, padding=1, relu=True, bn=False)

        self.conv1a = BasicConv(32, 48, is_3d=True, kernel_size=3, stride=2, padding=1)
        self.conv2a = BasicConv(48, 64, is_3d=True, kernel_size=3, stride=2, padding=1)

        self.deconv1a = Conv2x(48, 32, deconv=True, is_3d=True, relu=False)
        self.deconv2a = Conv2x(64, 48, deconv=True, is_3d=True)
        
        self.sga1 = SGABlock(refine=True)
        self.sga2 = SGABlock(refine=True)

        self.sga11 = SGABlock(channels=48, refine=True)
        self.sga12 = SGABlock(channels=48, refine=True)

    def forward(self, x, g):
        """
           args:
              x: cost volume, in size [N,C,D,H,W];
              g : guidance, in size [N, C2, H, W], where C2 = 20*C=20*32=640;
           return 
        """
        x = self.conv_start(x) # C=32
        x = self.sga1(x, g['sg1'])
        rem0 = x
        x = self.conv1a(x)
        x = self.sga11(x, g['sg11'])
        rem1 = x
        #print ("[???] rem1 size:", rem1.size())
        x = self.conv2a(x)
        #print ("[???] after conv2a(x) size:", x.size())
        x = self.deconv2a(x, rem1) #???
        x = self.sga12(x, g['sg12'])
        x = self.deconv1a(x, rem0)
        x = self.sga2(x, g['sg2'])
        #added by CCJ:
        x = self.conv_end(x)
        return x

""" generate input signal g, which is fed into the Guidance() block, 
    in order to generate the weights (in 4 directions in total) for SGA Block;
"""
class GetInput4Guidance(nn.Module):
    def __init__(self, 
        #is_quarter_size = True
        downsample_scale = 3
        ):
        super(GetInput4Guidance, self).__init__()
         
        assert downsample_scale in [2, 3, 4], "downsample_scale should be 2, 3, or 4!!!"
        self.conv_start = nn.Sequential(BasicConv(3, 16, kernel_size=3, padding=1),
                                        BasicConv(16, 32, kernel_size=3, padding=1))
        self.conv_refine = nn.Conv2d(32, 32, (3, 3), (1,1), (1,1), bias=False) #just convolution, no bn and relu
        self.feature = Feature(downsample_scale)
        #self.inter_C = 4 if is_quarter_size else 3
        self.inter_C = downsample_scale
        self.bn_relu = nn.Sequential(BatchNorm2d(32),
                                     nn.ReLU(inplace=True))
    
    def forward(self, x):
        """
           args:
              x: input image, in size [N,3,H,W];
           return:
              g : signal for Guidance() module, in size [N, C=64, H, W];
        """
        g = self.conv_start(x)	
        x = self.feature(x) # in size [N, C=32, H/3, W/3]
        x = self.conv_refine(x)
        x = F.interpolate(x, [x.size()[2] * self.inter_C, x.size()[3] * self.inter_C], 
                          mode='bilinear', align_corners=False)
        x = self.bn_relu(x)
        g = torch.cat((g, x), 1)
        return g


"""
SGA module, adapted from GANet_11 code;
"""
class SGA_CostAggregation(nn.Module):
    def __init__(self, 
        is_guide_from_img, 
        #is_quarter_size, # feature in 1/4 image size (i.e., H/4 x W/4) or 1/3 size (i.e., H/3 x W/3)
        downsample_scale, # feature in 1/2, 1/3, or 1/2 image size (i.e., H/4 x W/4, H/3 x W/3, or H/2 x W/2)
        is_lga, # generate LGA(Local Guided Aggregation) weights or not
        cost_volume_in_channels # input cost volume feature channels 
        ):
        super(SGA_CostAggregation, self).__init__()
        self.is_guide_from_img = is_guide_from_img
        if is_guide_from_img:
            #self.get_g_from_img = GetInput4Guidance(is_quarter_size)
            self.get_g_from_img = GetInput4Guidance(downsample_scale)
        else:
            self.get_g_from_img = None
        self.guidance = Guidance_11(downsample_scale, is_lga)
        self.cost_agg = CostAggregation_11(cost_volume_in_channels)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (BatchNorm2d, BatchNorm3d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

     
    
    def forward(self, cv, g_in = None, img_for_g = None):
        """
           args:
              cv: cost volume, in size [N,C1,D,H,W]
              g_in : input for guidance module, in size [N, C2=64, H, W]
              img_for_g: input image for generating guide input g_in; in size [N,3,H,W]
        """
        #--------------
        # guidance
        #--------------
        if g_in is None:
            assert self.is_guide_from_img, 'No g provided!!!'
            g_in = self.get_g_from_img(img_for_g) # in size [N, 64, H, W]
            #print("[???] g_in shape", g_in.size())
        
        g_out = self.guidance(g_in) 
        #for k,v in g_out.items():
        #    if v is not None:
                #print("[???] g_out[%s] has shape" %k, v.size())
        
        # after the guidance(), g_out in size [N, 4*C3=640, H, W], 
        # with C3=640/4=160=5*32;
        # Note: 640/4=160=5*32, and 32 corresponds the convolved 
        # cost volume (for changing its C=64 to C=32);
        assert cv.ndim == 5, "Should be a 5D tenor!!!" 
        return self.cost_agg(cv, g_out).contiguous()
