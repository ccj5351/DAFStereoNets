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
        #self.conv1=nn.Conv2d(in_planes,planes,kernel_size,stride=stride,padding=1)
        #self.bn1=nn.BatchNorm2d(planes)
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
        #self.conv1 = nn.Conv3d(in_planes, planes, kernel_size, stride=stride, padding=1)
        #self.bn1 = nn.BatchNorm3d(planes)
        self.convbn_3d_1 = convbn_3d(in_planes, planes, kernel_size, stride=stride, pad = 1)
        self.convbn_3d_2 = convbn_3d(planes, planes, kernel_size, stride=1, pad = 1)
        self.convbn_3d_3 = convbn_3d(planes, planes, kernel_size, stride=1, pad = 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.convbn_3d_1(x))
        out = self.relu(self.convbn_3d_2(out))
        out = self.relu(self.convbn_3d_3(out))
        return out

class disparityregression(nn.Module):
    def __init__(self, maxdisp):
        super(disparityregression, self).__init__()
        self.disp = Variable(torch.Tensor(np.reshape(np.array(range(maxdisp)),[1,maxdisp,1,1])).cuda(), requires_grad=False)
    
    def forward(self, x):
        disp = self.disp.repeat(x.size()[0],1,x.size()[2],x.size()[3])
        out = torch.sum(x*disp,1)
        return out


class GCNet(nn.Module):
    def __init__(self, maxdisp = 192, is_quarter_size_cost_volume_gcnet = False ):
        super(GCNet, self).__init__()
        #----------------------
        # hyper-paremeters
        #----------------------
        #self.height=height
        #self.width=width
        self.maxdisp = maxdisp
        self.is_quarter_size_cost_volume_gcnet = is_quarter_size_cost_volume_gcnet
        print ("[***] is_quarter_size_cost_volume_gcnet = ", self.is_quarter_size_cost_volume_gcnet)
        self.F = 32
        self.first_kernel_size = 5
        self.kernel_size = 3
        self.num_res_block = 8
        self.num_3d_block = 1

        self.relu = nn.ReLU(inplace=True)
        self.block = BasicBlock
        
        # first 2D Conv, with BN and ReLU;
        #self.conv0 = nn.Conv2d(3,self.F, self.first_kernel_size, 2, 2) # kernel_size = 5
        #self.bn0=nn.BatchNorm2d(32)
        self.convbn0 = convbn(3, self.F, self.first_kernel_size, stride = 2,pad = 2)
        # Residual Block
        self.res_block = self._make_layer(self.block, self.F, self.F, self.num_res_block, stride=1)
        #last conv2d, No ReLU or BN
        self.conv1 = nn.Conv2d(self.F, self.F, self.kernel_size, 1, 1) # kernel_size = 3

        #conv3d
        #self.conv3d_1=nn.Conv3d(64,32,3,1,1)
        #self.bn3d_1=nn.BatchNorm3d(32)
        self.conv3dbn_1 = convbn_3d(2*self.F, self.F, self.kernel_size, stride= 1, pad = 1)
        self.conv3dbn_2 = convbn_3d(self.F, self.F, self.kernel_size, stride= 1, pad = 1)

        #conv3d sub_sample block
        self.block_3d_1 = Conv3DBlock(self.F, 2*self.F, stride=2, kernel_size=self.kernel_size)
        self.block_3d_2 = Conv3DBlock(2*self.F, 2*self.F, stride=2, kernel_size=self.kernel_size)
        self.block_3d_3 = Conv3DBlock(2*self.F, 2*self.F, stride=2, kernel_size=self.kernel_size)
        self.block_3d_4 = Conv3DBlock(2*self.F, 4*self.F, stride=2, kernel_size=self.kernel_size)
        
        #deconv3d, with BN and ReLU
        #self.deconv1=nn.ConvTranspose3d(128,64,3,2,1,1)
        #self.debn1=nn.BatchNorm3d(64)
        self.deconvbn1 = deconvbn_3d(4*self.F, 2*self.F, self.kernel_size, stride=2)
        self.deconvbn2 = deconvbn_3d(2*self.F, 2*self.F, self.kernel_size, stride=2)
        self.deconvbn3 = deconvbn_3d(2*self.F, 2*self.F, self.kernel_size, stride=2)
        self.deconvbn4 = deconvbn_3d(2*self.F, self.F, self.kernel_size, stride=2)

        #last deconv3d, no BN or ReLU
        self.deconv5 = nn.ConvTranspose3d(self.F, 1, self.kernel_size, stride=2, padding=1, output_padding=1)
        
        """ code from PSMNet """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1]*m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

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

        #print ("[???] imgLeft shape: ", imgLeft.shape)
        imgl0=self.relu(self.convbn0(imgl))
        imgr0=self.relu(self.convbn0(imgr))

        imgl_block=self.res_block(imgl0)
        imgr_block=self.res_block(imgr0)
        #print ("[???] imgl_block shape: ", imgl_block.shape)

        imgl1=self.conv1(imgl_block)
        imgr1=self.conv1(imgr_block)
        #print ("[???] imgl1 shape: ", imgl1.shape)
        # cost volume
        #cv = self.get_costVolume(imgl1,imgr1)
        #cv = self.cost_volume(imgl1,imgr1)
        #cv = self.cost_volume_faster(imgl1,imgr1)
        cv = cost_volume_faster(imgl1, imgr1, d = self.maxdisp//(2*img_ds_scale))
        #print ("[???] cv shape: ", cv.shape)
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

        out = out.view(N, self.maxdisp // img_ds_scale, H // img_ds_scale, W//img_ds_scale)
        prob = F.softmax(out,1)
        #disp = self.disparityregression(prob)
        disp = self.disparityregression(prob, maxdisp=self.maxdisp//img_ds_scale)
        if self.is_quarter_size_cost_volume_gcnet:
            # NOTE: newly added for SGA: upsampling operation, 
            # corresponding to the first downsampling at the beginning to the input image pair;
            disp = F.interpolate(disp[:,None,...], [H, W], mode='bilinear', align_corners=True)
            disp = torch.squeeze(disp,1) # [N,H,W]
        return disp



    def _make_layer(self, block, in_planes, planes, num_block, stride):
        strides = [stride] + [1] *(num_block-1)
        layers = []
        for s in strides:
            layers.append(block(in_planes,planes, s))
        return nn.Sequential(*layers)
    
    #-------------------
    #NOTE: this function is adapted from PSMNet code ??? TOO SLOW???;
    #-------------------
    def get_costVolume(self, x, y):
        """
        args: 
            x : left feature,  in size [N,C,H/2,W/2]
            y : right feature, in size [N,C,H/2,W/2]
        return:
            cost: cost volume in size [N,2C,D/2,H/2,W/2]
        """
        # matching
        assert(x.is_contiguous() == True)
        N0, C0, H0, W0 = x.size()[:]
        cost = torch.tensor((), dtype=torch.float32).new_zeros((N0,2*C0,self.maxdisp//2,H0,W0),requires_grad=True).cuda()
        #cost = x.new().resize_(N0,C0*2, self.maxdisp//2,H0, W0).zero_()
        #cost = Variable(torch.FloatTensor(N0,C0*2,self.maxdisp//2,H0,W0).zero_()).cuda()
        for i in range(self.maxdisp//2):
            if i > 0:
                cost[:, :C0, i, :, i:] = x[:, :, :, i:]
                cost[:, C0:, i, :, i:] = y[:, :, :, :-i]
            else:
                cost[:, :C0, i, :, :] = x
                cost[:, C0:, i, :, :] = y

        return cost.contiguous()

    #NOTE: faster!! But consume memory than the above one???
    def cost_volume_faster_not_used(self, x, y):
        """
        args:
            x : left feature,  in size [N,C,H,W]
            y : right feature, in size [N,C,H,W]
        return:
            cost: cost volume in size [N,2C,D,H,W]
        """
        N0, C0, H0, W0 = x.size()[:]
        cv_list = []
        # Pads the input tensor boundaries with zero.
        # padding = (padding_left, padding_right, padding_top, padding_bottom) 
        # along the [H, W] dim; 
        y_pad = nn.ZeroPad2d((self.maxdisp//2, 0, 0, 0))(y)

        for d in reversed(range(self.maxdisp//2)):
            x_slice = x
            
            #Note added by CCJ:
            #Note that you don’t need to use torch.narrow or select, 
            #but instead basic indexing will do it for you.
            y_slice = y_pad[:,:,:,d:d+W0]
            xy_temp = torch.cat((x_slice, y_slice), 1)
            cv_list.append(xy_temp)
        
        #Stacks a list of rank-R tensors into one rank-(R+1) tensor.
        cv = torch.stack(cv_list, 2)
        #assert(cv.is_contiguous() == True)
        #print ("[???] cv shape = ", cv.shape)
        return cv

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
