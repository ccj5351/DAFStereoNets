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

from ..baselines.GANet.libs.GANet.modules.GANet import SGA
from .embednetwork import embed_net
from .bilateral import bilateralFilter
#from .bilateral_func import bilateralFilter
from ..baselines.GANet.libs.sync_bn.modules.sync_bn import BatchNorm2d, BatchNorm3d
from ..baselines.GANet.libs.GANet.modules.GANet import DisparityRegression
from ..baselines.GANet.libs.GANet.modules.GANet import GetCostVolume
from ..baselines.GANet.libs.GANet.modules.GANet import LGA, LGA2, LGA3

from src.modules.cost_volume import cost_volume_faster
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
        
        if deconv and is_3d: 
            kernel = (3, 4, 4)
        elif deconv:
            kernel = 4
        else:
            kernel = 3
        self.conv1 = BasicConv(in_channels, out_channels, deconv, is_3d, bn=True, relu=True, kernel_size=kernel, stride=2, padding=1)

        if self.concat: 
            self.conv2 = BasicConv(out_channels*2, out_channels, False, is_3d, bn, relu, kernel_size=3, stride=1, padding=1)
        else:
            self.conv2 = BasicConv(out_channels, out_channels, False, is_3d, bn, relu, kernel_size=3, stride=1, padding=1)

    def forward(self, x, rem):
        x = self.conv1(x)
        assert(x.size() == rem.size())
        if self.concat:
            x = torch.cat((x, rem), 1)
        else: 
            x = x + rem
        x = self.conv2(x)
        return x


class Feature(nn.Module):
    def __init__(self):
        super(Feature, self).__init__()

        self.conv_start = nn.Sequential(
            BasicConv(3, 32, kernel_size=3, padding=1),
            BasicConv(32, 32, kernel_size=5, stride=3, padding=2),
            BasicConv(32, 32, kernel_size=3, padding=1))
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

class Guidance(nn.Module):
    def __init__(self):
        super(Guidance, self).__init__()

        self.conv0 = BasicConv(64, 16, kernel_size=3, padding=1)
        self.conv1 = nn.Sequential(
            BasicConv(16, 32, kernel_size=5, stride=3, padding=2),
            BasicConv(32, 32, kernel_size=3, padding=1))

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

        lg1 = self.weight_lg1(rem)
        lg2 = self.weight_lg2(rem)
       
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
        x = F.interpolate(self.conv32x1(x), [self.maxdisp+1, x.size()[3]*3, x.size()[4]*3], mode='trilinear', align_corners=False)
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
        x = F.interpolate(self.conv32x1(x), [self.maxdisp+1, x.size()[3]*3, x.size()[4]*3], mode='trilinear', align_corners=False)
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
        # split g channel C (e.g., C= 640) to 4 parts, each with C/4 ( e.g., = 640/4=160) size along channel dim, i.e., dim=1;
        # each C/4=160-dim vector is further divided into 32 x 5, where 32 is the same as input x channel, 
        # and 5 means w0, w1, ..., w4 in Eq (5) in GANet CVPR paper, s.t. w0 + w1 + ... + w4 = 1.0, 
        # this why F.normalize() is applied along dim=5, to normalize those five values, s.t. w0 + w1 + ... + w4 = 1.0 !!!
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

class GetCostVolume4BilaterFilterV0(nn.Module):
    def __init__(self, maxdisp):
        super(GetCostVolume4BilaterFilterV0, self).__init__()
        self.maxdisp = maxdisp + 1

    def forward(self, x, y):
        """
          args: 
              x : left feature,  in size [N,C,H,W]
              y : right feature, in size [N,C,H,W]
          return:
              cost: cost volume in size [N, D, C,H,W], (which can be 
                    further reshape to [N*D,C,H,W] for bilateral fitlering)
        """
        assert(x.is_contiguous() == True)
        with torch.cuda.device_of(x):
            num, channels, height, width = x.size()
            cost = x.new().resize_(self.maxdisp, num, channels * 2, height, width).zero_()
            for i in range(self.maxdisp):
                if i > 0 :
                    cost[i, :, :x.size()[1], :,i:]  = x[:,:,:,i:]
                    cost[i, :, x.size()[1]:, :,i:]  = y[:,:,:,:-i]
                else:
                    cost[i, :, :x.size()[1], :,:]   = x
                    cost[i, :, x.size()[1]:, :,:]   = y

            cost = cost.contiguous()
        return cost

class GetCostVolume4BilaterFilter(nn.Module):
    def __init__(self, maxdisp):
        super(GetCostVolume4BilaterFilter, self).__init__()
        self.maxdisp = maxdisp + 1

    def forward(self, x, y):
        """
          args: 
              x : left feature,  in size [N,C,H,W]
              y : right feature, in size [N,C,H,W]
          return:
              cost: cost volume in size [N, D, C,H,W], (which can be 
                    further reshape to [N*D,C,H,W] for bilateral fitlering)
        """
        assert(x.is_contiguous() == True)
        with torch.cuda.device_of(x):
            num, channels, height, width = x.size()
            #cost = x.new().resize_(self.maxdisp, num, channels * 2, height, width).zero_()
            cost = x.new().resize_(num, channels * 2, self.maxdisp, height, width).zero_()
            for i in range(self.maxdisp):
                if i > 0 :
                    cost[:, :x.size()[1], i, :,i:]   = x[:,:,:,i:]
                    cost[:, x.size()[1]:, i, :,i:]   = y[:,:,:,:-i]
                else:
                    cost[:, :x.size()[1], i, :,:]   = x
                    cost[:, x.size()[1]:, i, :,:]   = y
            
            cost = cost.contiguous()
        return cost

class CostAggregation(nn.Module):
    def __init__(self, maxdisp=192):
        super(CostAggregation, self).__init__()
        self.maxdisp = maxdisp


        self.conv_start = BasicConv(64, 32, is_3d=True, kernel_size=3, padding=1, relu=False)

        self.conv1a = BasicConv(32, 48, is_3d=True, kernel_size=3, stride=2, padding=1)
        self.conv2a = BasicConv(48, 64, is_3d=True, kernel_size=3, stride=2, padding=1)
#        self.conv3a = BasicConv(64, 96, is_3d=True, kernel_size=3, stride=2, padding=1)

        self.deconv1a = Conv2x(48, 32, deconv=True, is_3d=True, relu=False)
        self.deconv2a = Conv2x(64, 48, deconv=True, is_3d=True)
#        self.deconv3a = Conv2x(96, 64, deconv=True, is_3d=True)

        self.conv1b = Conv2x(32, 48, is_3d=True)
        self.conv2b = Conv2x(48, 64, is_3d=True)
#        self.conv3b = Conv2x(64, 96, is_3d=True)

        self.deconv1b = Conv2x(48, 32, deconv=True, is_3d=True, relu=False)
        self.deconv2b = Conv2x(64, 48, deconv=True, is_3d=True)
#        self.deconv3b = Conv2x(96, 64, deconv=True, is_3d=True)
        self.deconv0b = Conv2x(8, 8, deconv=True, is_3d=True)
        
        self.sga1 = SGABlock(refine=True)
        self.sga2 = SGABlock(refine=True)
        self.sga3 = SGABlock(refine=True)

        self.sga11 = SGABlock(channels=48, refine=True)
        self.sga12 = SGABlock(channels=48, refine=True)
        self.sga13 = SGABlock(channels=48, refine=True)
        self.sga14 = SGABlock(channels=48, refine=True)

        self.disp0 = Disp(self.maxdisp)
        self.disp1 = Disp(self.maxdisp)
        self.disp2 = DispAgg(self.maxdisp)


    def forward(self, x, g):
        
        x = self.conv_start(x)
        x = self.sga1(x, g['sg1'])
        rem0 = x
       
        if self.training:
            disp0 = self.disp0(x)

        x = self.conv1a(x)
        x = self.sga11(x, g['sg11'])
        rem1 = x
        x = self.conv2a(x)
        rem2 = x
#        x = self.conv3a(x)
#        rem3 = x

#        x = self.deconv3a(x, rem2)
#        rem2 = x
        x = self.deconv2a(x, rem1)
        x = self.sga12(x, g['sg12'])
        rem1 = x
        x = self.deconv1a(x, rem0)
        x = self.sga2(x, g['sg2'])
        rem0 = x
        if self.training:
            disp1 = self.disp1(x)

        x = self.conv1b(x, rem1)
        x = self.sga13(x, g['sg13'])
        rem1 = x
        x = self.conv2b(x, rem2)
#        rem2 = x
#        x = self.conv3b(x, rem3)

#        x = self.deconv3b(x, rem2)
        x = self.deconv2b(x, rem1)
        x = self.sga14(x, g['sg14'])
        x = self.deconv1b(x, rem0)
        x = self.sga3(x, g['sg3'])

        disp2 = self.disp2(x, g['lg1'], g['lg2'])
        if self.training:
            return disp0, disp1, disp2
        else:
            return disp2

# > see this tutorial: An Overview of ResNet and its Variants,
# > at https://towardsdatascience.com/an-overview-of-resnet-and-its-variants-5281e2f56035;

""" Original Residual Block;
    code adapted from 
    https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/deep_residual_network/main.py
"""
class PreActiResidualBlock(nn.Module):
    #here we use original residual block:
    # x --> Weight --> BN --> ReLU --> Weight --> BN --> ReLU
    # |                                                   |
    # |                                                   |
    # |                                                   |
    # |-----------------------------------------------> Addition --> x_new
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(PreActiResidualBlock, self).__init__()
        self.my_func = nn.Sequential(
                nn.BatchNorm2d(in_channels), #bn1
                nn.ReLU(inplace=True), #relu1
                self.conv3x3(in_channels, out_channels, stride), #conv1
                nn.BatchNorm2d(out_channels), #bn2
                nn.ReLU(inplace=True), #relu2
                self.conv3x3(out_channels, out_channels, stride), #conv2
                )
        
    # 3x3 convolution
    def conv3x3(self, in_channels, out_channels, stride=1):
        return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                stride=stride, padding=1, bias=False)
    
    def forward(self, x):
        rem = x
        x = self.my_func(x)
        return x + rem

# Residual block
class OriginalResidualBlock(nn.Module):
    """ here we use original residual block:
         x --> Weight --> BN --> ReLU --> Weight --> BN --> ReLU
         |                                           |
         |                                           |
         |                                           |
         |-----------------------------------------------> Addition --> x_new
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(OriginalResidualBlock, self).__init__()
        self.my_func = nn.Sequential(
                self.conv3x3(in_channels, out_channels, stride), #conv1
                nn.BatchNorm2d(out_channels), #bn1
                nn.ReLU(inplace=True), #relu1
                self.conv3x3(out_channels, out_channels, stride), #conv2
                nn.BatchNorm2d(out_channels), #bn2
                )
        self.relu = nn.ReLU(inplace=True) #relu2
        
    # 3x3 convolution
    def conv3x3(self, in_channels, out_channels, stride=1):
        return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                stride=stride, padding=1, bias=False)
    
    def forward(self, x):
        rem = x
        print ("??? rem shape = ", rem.shape)
        x = self.my_func(x)
        print ("??? x shape = ", x.shape)
        x += rem
        x = self.relu(x)
        return x

"""
our network
"""
class AttenStereoNet(nn.Module):
    def __init__(self, maxdisp=192, sigma_s = 0.7, # 1.7: 13 x 13; 0.3 : 3 x 3;
                 sigma_v = 0.1, isEmbed = True, 
                 dilation = 1,
                 cost_filter_grad = False
                 ):
        super(AttenStereoNet, self).__init__()
        self.maxdisp = maxdisp
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
            # the function version
            #self.bifilter = bilateralFilter
        else:
            self.embednet = None
            self.bifilter = None
        
        """ SGA layer, semi-global aggregation layer """
        self.conv_start = nn.Sequential(BasicConv(3, 16, kernel_size=3, padding=1),
                                        BasicConv(16, 32, kernel_size=3, padding=1))
        self.conv_refine = nn.Conv2d(32, 32, (3, 3), (1,1), (1,1), bias=False) #just convolution, no bn and relu
        self.feature = Feature()
        #self.residualBlk = nn.Sequential(
        #                    OriginalResidualBlock(in_channels = 3, out_channels = 16, stride=1),
        #                    OriginalResidualBlock(in_channels = 16, out_channels = 32, stride=1)
        #                    )

        #self.conv3d_start = BasicConv(cv_in_channels, 32, is_3d=True, kernel_size=3, padding=1, relu=False)
        self.conv_x = BasicConv(32, 32, kernel_size=3, padding=1) # with default bn=True, relu=True
        self.conv_y = BasicConv(32, 32, kernel_size=3, padding=1) # with default bn=True, relu=True
        if self.isEmbed:
            self.cv = GetCostVolume4BilaterFilter(int(self.maxdisp/3))
            #self.cv = GetCostVolume4BilaterFilterV0(int(self.maxdisp/3))
        else:
            self.cv = GetCostVolume(int(self.maxdisp/3))
        
        self.guidance = Guidance()
        
        self.cost_agg = CostAggregation(self.maxdisp)
        self.bn_relu = nn.Sequential(BatchNorm2d(32), nn.ReLU(inplace=True))

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (BatchNorm2d, BatchNorm3d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    
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
        if not self.isEmbed:
            # [N, C, D/3, H/3, W/3]
            cv = self.cv(f_x, f_y)
            embed = None
            #embed_scale = None

        else: # using embedding
            # [D/3, N, C, H/3, W/3]
            cv = self.cv(f_x, f_y)
            # downscale x to [N,C,H/3, W/3] then fed into embeddingnet,
            # because the cost volume generated below is in shape [N,C,D/3, H/3, W/3]
            x_scale = F.interpolate(x, [x.size()[2]//3, x.size()[3]//3], 
                    mode='bilinear', align_corners=False)
            #print ('[???] x shape', x.shape)
            #print ('[???] x_scale shape', x_scale.shape)
            embed = self.embednet(x_scale)
            
            #print ('[???] cv shape', cv.shape)
            #D, N, C, H, W = cv.size()[:]
            #cv_filtered = cv.new().resize_(N, C, D, H, W).zero_()
            N, C, D, H, W = cv.size()[:]
            
            # NOTE: this is the memory problem ???
            # NO sure this torch.no_grad() will distory the training or not !!!!
            #with torch.no_grad():
            with torch.set_grad_enabled(self.cost_filter_grad):
                for d in range(0,D):
                #for d in range(0,1):
                    #print ('bilateral filtering cost volume slice %d/%d' %(d+1, D))
                    # apply bilateral filter to cost volume [N,C,H,W];
                    cv_d_slice = cv[:,:,d,:,:]
                    #print ('[???] cv_d_slice shape', cv_d_slice.shape)
                    cv[:,:,d,:,:] = self.bifilter(embed, cv_d_slice)
                    #cv[:,:,d,:,:] = self.bifilter(embed, cv_d_slice, self.sigma_s, self.sigma_v, isCUDA = True, dilation = 1)
                    #print ('[???] cv_d_filtered shape', cv_d_filtered.shape)
             
            
            #print ('[???] embed shape', embed.shape)
            #cv = self.bifilter(embed, cv.view(D*N, C,H,W)).view(D,N,C,H,W)
            #print ('[???] cv after filtered shape', cv.shape)
            
            # upsample to original size
            #embed_scale = F.interpolate(embed, [x.size()[2], 
            #                            x.size()[3]], 
            #                            mode='bilinear', 
            #                            align_corners=False)
            
            # make sure the contiguous memeory
            cv = cv.contiguous()

        if self.training:
            disp0, disp1, disp2 = self.cost_agg(cv, g)
            return disp0, disp1, disp2, embed
        else:
            disp2 = self.cost_agg(cv, g)
            return disp2, embed