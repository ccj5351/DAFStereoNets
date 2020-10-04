# !/usr/bin/env python3
# -*-coding:utf-8-*-
# @file: basic.py
# @brief:
# @author: Changjiang Cai, ccai1@stevens.edu, caicj5351@gmail.com
# @version: 0.0.1
# @creation date: 12-12-2019
# @last modified: Thu 02 Apr 2020 04:47:10 PM EDT

# > see: the code is adapoted from https://github.com/zyf12389/GC-Net/blob/master/gc_net.py
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from .net_init import net_init

from src.modules.cost_volume import cost_volume_faster

def convbn(in_planes, out_planes, kernel_size, stride, pad):
    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=pad, bias=False),
                         nn.BatchNorm2d(out_planes))

def convbn_3d(in_planes, out_planes, kernel_size, stride, pad):
    return nn.Sequential(nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, padding=pad, stride=stride, bias=False),
                         nn.BatchNorm3d(out_planes))

def deconvbn_3d(in_planes, out_planes, kernel_size=3, stride=2):
    return nn.Sequential(nn.ConvTranspose3d(in_planes, out_planes, kernel_size=kernel_size, 
                         padding=1, output_padding=1, stride=2, bias=False),
                         nn.BatchNorm3d(out_planes))

class BasicBlock(nn.Module):  #basic block for Conv2d
    def __init__(self, in_planes, planes,stride=1, kernel_size=3):
        super(BasicBlock,self).__init__()
        self.convbn1 = convbn(in_planes, planes,kernel_size,stride=stride,pad =1)
        self.convbn2 = convbn(planes, planes,kernel_size,stride=1,pad =1)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        residual = x
        out = self.relu(self.convbn1(x))
        out = self.convbn2(out)
        out += residual
        out = self.relu(out)
        return out

class Conv3DBlock(nn.Module):
    def __init__(self,in_planes,planes,stride=1,kernel_size =3):
        super(Conv3DBlock, self).__init__()
        self.convbn_3d_1 = convbn_3d(in_planes, planes, kernel_size, stride=stride, pad = 1)
        self.convbn_3d_2 = convbn_3d(planes, planes, kernel_size, stride=1, pad = 1)
        self.convbn_3d_3 = convbn_3d(planes, planes, kernel_size, stride=1, pad = 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.convbn_3d_1(x))
        out = self.relu(self.convbn_3d_2(out))
        out = self.relu(self.convbn_3d_3(out))
        return out

class Conv3DBlock_Kendall(nn.Module):
    def __init__(self,in_planes,planes,stride=1,kernel_size =3):
        super(Conv3DBlock_Kendall, self).__init__()
        self.convbn_3d_1 = convbn_3d(in_planes, planes, kernel_size, stride=stride, pad = 1)
        self.convbn_3d_2 = convbn_3d(planes, planes, kernel_size, stride=1, pad = 1)
        self.convbn_3d_3 = convbn_3d(planes, planes, kernel_size, stride=1, pad = 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.convbn_3d_1(x))
        out2 = out
        out = self.relu(self.convbn_3d_2(out))
        out = self.relu(self.convbn_3d_3(out))
        return out, out2

class GCNet(nn.Module):
    def __init__(self, maxdisp = 192, 
                is_kendall_version = True, # excatly following the structure in Kendall's GCNet paper;
                is_quarter_size_cost_volume_gcnet = False):

        super(GCNet, self).__init__()
        #----------------------
        # hyper-paremeters
        #----------------------
        self.maxdisp = maxdisp
        self.is_kendall_version = is_kendall_version
        self.is_quarter_size_cost_volume_gcnet = is_quarter_size_cost_volume_gcnet
        print ("[***] is_quarter_size_cost_volume_gcnet = ", self.is_quarter_size_cost_volume_gcnet)
        print ("[***] is_kendall_version = ", self.is_kendall_version)
        self.F = 32
        self.first_kernel_size = 5
        self.kernel_size = 3
        self.num_res_block = 8
        self.num_3d_block = 1

        self.relu = nn.ReLU(inplace=True)
        self.block = BasicBlock
        
        # first 2D Conv, with BN and ReLU;
        self.convbn0 = convbn(3, self.F, self.first_kernel_size, stride = 2,pad = 2)
        # Residual Block
        self.res_block = self._make_layer(self.block, self.F, self.F, self.num_res_block, stride=1)
        #last conv2d, No ReLU or BN
        self.conv1 = nn.Conv2d(self.F, self.F, self.kernel_size, 1, 1) # kernel_size = 3

        #conv3d
        self.conv3dbn_1 = convbn_3d(2*self.F, self.F, self.kernel_size, stride= 1, pad = 1)
        self.conv3dbn_2 = convbn_3d(self.F, self.F, self.kernel_size, stride= 1, pad = 1)

        #conv3d sub_sample block
        if self.is_kendall_version:
            self.block_3d_1 = Conv3DBlock_Kendall(2*self.F, 2*self.F, stride=2, kernel_size=self.kernel_size)
            self.block_3d_2 = Conv3DBlock_Kendall(2*self.F, 2*self.F, stride=2, kernel_size=self.kernel_size)
            self.block_3d_3 = Conv3DBlock_Kendall(2*self.F, 2*self.F, stride=2, kernel_size=self.kernel_size)
            self.block_3d_4 = Conv3DBlock_Kendall(2*self.F, 4*self.F, stride=2, kernel_size=self.kernel_size)
        else:
            self.block_3d_1 = Conv3DBlock(self.F, 2*self.F, stride=2, kernel_size=self.kernel_size)
            self.block_3d_2 = Conv3DBlock(2*self.F, 2*self.F, stride=2, kernel_size=self.kernel_size)
            self.block_3d_3 = Conv3DBlock(2*self.F, 2*self.F, stride=2, kernel_size=self.kernel_size)
            self.block_3d_4 = Conv3DBlock(2*self.F, 4*self.F, stride=2, kernel_size=self.kernel_size)
        
        
        #deconv3d, with BN and ReLU
        self.deconvbn1 = deconvbn_3d(4*self.F, 2*self.F, self.kernel_size, stride=2)
        self.deconvbn2 = deconvbn_3d(2*self.F, 2*self.F, self.kernel_size, stride=2)
        self.deconvbn3 = deconvbn_3d(2*self.F, 2*self.F, self.kernel_size, stride=2)
        self.deconvbn4 = deconvbn_3d(2*self.F, self.F, self.kernel_size, stride=2)

        #last deconv3d, no BN or ReLU
        self.deconv5 = nn.ConvTranspose3d(self.F, 1, self.kernel_size, stride=2, padding=1, output_padding=1)
        
        net_init(self)
        print ("[***]GCNet weights inilization done!")
    
    def _make_layer(self, block, in_planes, planes, num_block, stride):
        strides = [stride] + [1] *(num_block-1)
        layers = []
        for s in strides:
            layers.append(block(in_planes,planes, s))
        return nn.Sequential(*layers)
    
    def feature_extraction(self, x):
        """
        args: 
            x : input image,  in size [N,C=3,H,W]
        return:
            y: feature in size [N,F, H/2, W/2]
        """
        y = self.relu(self.convbn0(x))
        y = self.res_block(y)
        y = self.conv1(y)
        return y

    def cost_aggregation(self, cv):
        """
        args: 
            cv : input cost volume,  in size [N,F,D,H,W]
        return:
            out:  regularized cost volume vis 3D CNN layers, in size [N,F,D, H, W]
        """
        out = self.relu(self.conv3dbn_1(cv)) # conv3d_19
        out = self.relu(self.conv3dbn_2(out)) # conv3d_20
        
        #conv3d block
        res_l20 = out # from layer conv3d_20;
        out = self.block_3d_1(out) # conv3d_21,22,23
        res_l23 = out
        out = self.block_3d_2(out) # conv3d_24,25,26
        res_l26 = out
        out = self.block_3d_3(out) # conv3d_27,28,29
        res_l29 = out
        out = self.block_3d_4(out) # conv3d_30,31,32
        #print ("[???] after conv3d_32 out shape = ", out.shape)
        
        #deconv3d
        #print ("[???] res_l29: ", res_l29.shape)
        out = self.relu(self.deconvbn1(out) + res_l29)
        out = self.relu(self.deconvbn2(out) + res_l26)
        out = self.relu(self.deconvbn3(out) + res_l23)
        out = self.relu(self.deconvbn4(out) + res_l20)
        #last deconv3d, no BN or ReLU
        out = self.deconv5(out) # [N, 1, D, H, W]
        #print ("[???] out shape = ", out.shape)
        return out


    def cost_aggregation_kendall(self, cv):
        """
        Note: exactly following the structure in Kendall's paper:
        args: 
            cv : input cost volume,  in size [N,F,D,H,W]
        return:
            out:  regularized cost volume vis 3D CNN layers, in size [N,F,D, H, W]
        """
        out = self.relu(self.conv3dbn_1(cv)) # conv3d_19
        out = self.relu(self.conv3dbn_2(out)) # conv3d_20
        
        #conv3d block
        res_l20 = out # from layer conv3d_20;
        out, out_21 = self.block_3d_1(cv) # conv3d_21,22,23, NOTE: from 18 in Kendall's paper;
        res_l23 = out
        out, out_24 = self.block_3d_2(out_21) # conv3d_24,25,26, NOTE: from 21 in Kendall's paper;
        res_l26 = out
        out, out_27 = self.block_3d_3(out_24) # conv3d_27,28,29, NOTE: from 24 in Kendall's paper;
        res_l29 = out
        out, _ = self.block_3d_4(out_27) # conv3d_30,31,32, NOTE: from 27 in Kendall's paper;
        #print ("[???] after conv3d_32 out shape = ", out.shape)
        
        #deconv3d
        #print ("[???] res_l29: ", res_l29.shape)
        out = self.relu(self.deconvbn1(out) + res_l29)
        out = self.relu(self.deconvbn2(out) + res_l26)
        out = self.relu(self.deconvbn3(out) + res_l23)
        out = self.relu(self.deconvbn4(out) + res_l20)
        #last deconv3d, no BN or ReLU
        out = self.deconv5(out) # [N, 1, D, H, W]
        #print ("[???] out shape = ", out.shape)
        return out

    def disparityregression(self, x, maxdisp = None):
        #with torch.cuda.device_of(x):
        N, D, H, W = x.size()[:]
        
        if maxdisp is None:
            maxdisp = self.maxdisp
        assert (D == maxdisp)
        disp = torch.tensor(np.array(range(maxdisp)), dtype=torch.float32, 
                requires_grad=False).cuda().view(1, maxdisp,1,1)
        disp = disp.repeat(N,1,H,W)
        disp = torch.sum(x*disp, 1)
        #print ("[???] disp : ", disp.size())
        return disp

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
        # cost volume aggregation
        if self.is_kendall_version:
            out = self.cost_aggregation_kendall(cv)
        else:
            out = self.cost_aggregation(cv)
            
        out = out.view(N, self.maxdisp // img_ds_scale, H // img_ds_scale, W//img_ds_scale)
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
        
        #NOTE: This is wrong!!!
        #if self.is_quarter_size_cost_volume_gcnet:
        #    # NOTE: newly added for SGA: upsampling operation, 
        #    # corresponding to the first downsampling at the beginning to the input image pair;
        #    disp = F.interpolate(disp[:,None,...], [H, W], mode='bilinear', align_corners=True)
        #    disp = torch.squeeze(disp,1) # [N,H,W]
        return disp

