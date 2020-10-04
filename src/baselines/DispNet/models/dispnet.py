# !/usr/bin/env python3
# -*-coding:utf-8-*-
# @file: dispnet.py
# @brief:
# @author: Changjiang Cai, ccai1@stevens.edu, caicj5351@gmail.com
# @version: 0.0.1
# @creation date: 10-01-2020
# @last modified: Fri 31 Jan 2020 02:19:37 AM EST

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, zeros_
from .net_init import net_init
from torch.autograd import Variable

#Convolution In/Out Size:
#O = floor{(W - F + 2P)/S + 1}

#def correlation1D_map(x, y, maxdisp=40):
class correlation1D_map_V1(nn.Module):
    def __init__(self, maxdisp):
        super(correlation1D_map_V1, self).__init__()
        self.maxdisp = maxdisp

    def forward(self, x, y):
        """
        args:
            x: left feature, in [N,C,H,W]
            y: right feature, in [N,C,H,W]
            max_disp: disparity range
        return:
            corr: correlation map in size [N,D,H,W]
        """
    
        """
        #NOTE: make sure x means left image, y means right image,
        # so that we have the pixel x in left image, 
        # and the corresponding match pixel in right image has x-d 
        # (that means disparity d = shifting to left by d pixels). 
        # Otherwise, we have to chagne the shift direction!!!
        """
        # Pads the input tensor boundaries with zero.
        # padding = (padding_left, padding_right, padding_top, padding_bottom) along the [H, W] dim; 
        #y_pad = nn.ZeroPad2d((self.maxdisp-1, 0, 0, 0))(y)
        #NOTE: updated maxdisp to maxdisp-1 for left padding!!!
        y_pad = nn.ZeroPad2d((self.maxdisp-1, 0, 0, 0))(y)
        # input width
        W0 = x.size()[3]
        corr_tensor_list = []
        #NOTE: reversed() is necessary!!!
        for d in reversed(range(self.maxdisp)):
            x_slice = x
            #added by CCJ:
            #Note that you don’t need to use torch.narrow or select, 
            #but instead basic indexing will do it for you.
            y_slice = y_pad[:,:,:,d:d+W0]
            #xy_cor = torch.mean(x_slice*y_slice, dim=1, keepdim=True)
            #NOTE: change the mean to sum!!!
            xy_cor = torch.sum(x_slice*y_slice, dim=1, keepdim=True)
            #CosineSimilarity
            #cos = nn.CosineSimilarity(dim=1, eps=1e-08)
            #xy_cor = torch.unsqueeze(cos(x_slice,y_slice),1)
            corr_tensor_list.append(xy_cor)
        corr = torch.cat(corr_tensor_list, dim = 1)
        #print ("[???] corr shape: ", corr.shape)
        return corr

# corr1d
# code is adapted from https://github.com/wyf2017/DSMnet/blob/master/models/util_conv.py
class Corr1d_V2(nn.Module):
    def __init__(self, kernel_size=1, stride=1, D=1, simfun=None):
        super(Corr1d_V2, self).__init__()
        
        self.kernel_size = kernel_size
        self.stride = stride
        self.D = D
        if(simfun is None):
            self.simfun = self.simfun_default
        else: # such as simfun = nn.CosineSimilarity(dim=1)
            self.simfun = simfun
    
    def simfun_default(self, fL, fR):
        return (fL*fR).sum(dim=1)
        
    def forward(self, fL, fR):
        bn, c, h, w = fL.shape
        D = self.D
        stride = self.stride
        kernel_size = self.kernel_size
        corrmap = Variable(torch.zeros(bn, D, h, w).type_as(fL.data))
        corrmap[:, 0] = self.simfun(fL, fR)
        for i in range(1, D):
            if(i >= w): break
            idx = i*stride
            corrmap[:, i, :, idx:] = self.simfun(fL[:, :, :, idx:], fR[:, :, :, :-idx])
        if(kernel_size>1):
            assert kernel_size%2 == 1
            m = nn.AvgPool2d(kernel_size, stride=1, padding=kernel_size//2)
            corrmap = m(corrmap)
        return corrmap



""" NO BatchNorm Version """
def downsample_conv(in_planes, out_planes, kernel_size = 3):
    return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=2, padding = (kernel_size-1)//2),
            nn.ReLU(inplace=True)
           )

def conv3x3(in_planes, out_planes):
    return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=True), #bias=True by default;
            nn.ReLU(inplace=True)
           )

def conv1x1(in_planes, out_planes):
    return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
           )

#O = (W − 1)×S − 2P + K + output_padding
def upconv3x3(in_planes, out_planes):
    return nn.Sequential(
            nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True)
           )

def upconv4x4(in_planes, out_planes):
    return nn.Sequential(
            nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, output_padding=0),
            nn.ReLU(inplace=True)
           )

#2d convolution with padding, bn and activefun
def downsample_conv_bn(in_planes, out_planes, kernel_size = 3, is_relu = True, is_bn = True):
    if is_bn:
        bias = False
    else:
        bias = True
    assert kernel_size % 2 == 1
    myconv2d = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=2, padding = (kernel_size-1)//2, bias = bias)
    if (not is_bn and not is_relu):
        return myconv2d
    layers = []
    layers.append(myconv2d)
    if is_bn:
        layers.append(nn.BatchNorm2d(out_planes))
        #print ("[**] downsample_conv_bn() Enable BN")
    else:
        print ("[**] downsample_conv_bn() DISABLE BN !!!")
    if is_relu:
        layers.append(nn.ReLU(inplace=True))
    else:
        print ("[**] downsample_conv_bn() DISABLE ReLU !!!")
    return nn.Sequential(*layers)


def conv3x3_bn(in_planes, out_planes, is_relu = True, is_bn = True):
    if is_bn:
        bias = False
    else:
        bias = True
    myconv2d = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=bias) 
    if (not is_bn and not is_relu):
        return myconv2d
    layers = []
    layers.append(myconv2d)
    if is_bn:
        layers.append(nn.BatchNorm2d(out_planes))
        #print ("[**] conv3x3_bn() Enable BN")
    else:
        print ("[**] conv3x3_bn() DISABLE BN !!!")
    if is_relu:
        layers.append(nn.ReLU(inplace=True))
        #print ("[**] conv3x3_bn() Enable ReLU")
    else:
        print ("[**] conv3x3_bn() DISABLE ReLU !!!")
    return nn.Sequential(*layers)

def conv1x1_bn(in_planes, out_planes, is_relu = True, is_bn = True):
    if is_bn:
        bias = False
    else:
        bias = True
    myconv2d = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias= bias)
    if (not is_bn and not is_relu):
        return myconv2d
    layers = []
    layers.append(myconv2d)
    if is_bn:
        layers.append(nn.BatchNorm2d(out_planes))
        #print ("[**] conv1x1_bn() Enable BN")
    else:
        print ("[**] conv1x1_bn() DISABLE BN !!!")
    if is_relu:
        layers.append(nn.ReLU(inplace=True))
        #print ("[**] conv1x1_bn() Enable ReLU")
    else:
        print ("[**] conv1x1_bn() DISABLE ReLU !!!")
    return nn.Sequential(*layers)


def upconv3x3_bn(in_planes, out_planes, is_relu = True, is_bn = True):
    if is_bn:
        bias = False
    else:
        bias = True
    myconv2d = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=2, padding=1, output_padding=1, bias= bias)
    if (not is_bn and not is_relu):
        return myconv2d
    layers = []
    layers.append(myconv2d)
    if is_bn:
        layers.append(nn.BatchNorm2d(out_planes))
        #print ("[**] upconv3x3_bn() Enable BN")
    else:
        print ("[**] upconv3x3_bn() DISABLE BN !!!")
    if is_relu:
        layers.append(nn.ReLU(inplace=True))
        #print ("[**] upconv3x3_bn() Enable ReLU")
    else:
        print ("[**] upconv3x3_bn() DISABLE ReLU !!!")
    return nn.Sequential(*layers)


def upconv4x4_bn(in_planes, out_planes, is_relu = True, is_bn = True):
    if is_bn:
        bias = False
    else:
        bias = True
    myconv2d = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, output_padding=0, bias=bias)
    if (not is_bn and not is_relu):
        return myconv2d
    layers = []
    layers.append(myconv2d)
    if is_bn:
        layers.append(nn.BatchNorm2d(out_planes))
        #print ("[**] upconv4x4_bn() Enable BN")
    else:
        print ("[**] upconv4x4_bn() DISABLE BN !!!")
    if is_relu:
        layers.append(nn.ReLU(inplace=True))
        #print ("[**] upconv4x4_bn() Enable ReLU")
    else:
        print ("[**] upconv4x4_bn() DISABLE ReLU !!!")
    return nn.Sequential(*layers)


class DispNet(nn.Module):
    # due to TWO consecutive downsampling, so here maxdisp_corr=40, 
    # actually means 4*maxdisp_corr=160 in the original input image pair;
    def __init__(self, is_corr = True, maxdisp_corr = 40, 
            corr_func_type = 'correlation1D_map_V1',
            is_bn = True,
            is_relu = True):
        super(DispNet, self).__init__()
        """ 
        DispNetS: DispNet Simple, with is_corr = False
        DispNetC: DispNet Correlation, with is_corr = True
        """

        self.is_corr = is_corr
        self.corr_func_type = corr_func_type
        self.is_relu = is_relu
        self.is_bn = is_bn
        print ("[***] DispNet using corr_func_type : ", corr_func_type)
        print ("[***] DispNet using is_relu : ", is_relu)
        print ("[***] DispNet using is_bn : ", is_bn)
        print ("[***] DispNet using maxdisp_corr : ", maxdisp_corr)
        if self.is_corr:
            if str(corr_func_type).lower() == 'correlation1d_map_v1':
                self.corr_func = correlation1D_map_V1(maxdisp_corr)
            elif str(corr_func_type).lower() == 'corr1d_v2':
                self.corr_func = Corr1d_V2(D=maxdisp_corr)
            else:
                raise Exception("No suitable corr1D() found ...")

        else:
            self.corr_func = None
        self.maxdisp_corr = maxdisp_corr
        
        """ encoder """
        if self.is_corr: 
            """ DispNetC """
            self.conv1 = downsample_conv_bn(3,64,kernel_size = 7,is_relu = self.is_relu,is_bn = self.is_bn)
            self.conv_redir = conv1x1_bn(128, 64,is_relu = self.is_relu,is_bn = self.is_bn)
            self.conv3a = downsample_conv_bn(64+self.maxdisp_corr,256, kernel_size = 5,is_relu = self.is_relu,is_bn = self.is_bn)
        else:
            """ DispNetS """
            self.conv1 = downsample_conv_bn(6, 64, kernel_size = 7, is_relu = self.is_relu, is_bn = self.is_bn)
            self.conv3a = downsample_conv_bn(128, 256, kernel_size = 5, is_relu = self.is_relu, is_bn = self.is_bn)
        
        #conv2
        self.conv2 = downsample_conv_bn(64, 128, 5, is_relu = self.is_relu, is_bn = self.is_bn)
        #conv3b
        self.conv3b = conv3x3_bn(256, 256, is_relu = self.is_relu, is_bn = self.is_bn)
        
        #conv4a&b
        self.conv4a = downsample_conv_bn(256, 512, kernel_size = 3, is_relu = self.is_relu, is_bn = self.is_bn)
        self.conv4b = conv3x3_bn(512, 512, is_relu = self.is_relu, is_bn = self.is_bn)
        #conv5a&b
        self.conv5a = downsample_conv_bn(512, 512, kernel_size = 3, is_relu = self.is_relu, is_bn = self.is_bn)
        self.conv5b = conv3x3_bn(512, 512, is_relu = self.is_relu, is_bn = self.is_bn)
        #conv6a&b
        self.conv6a = downsample_conv_bn(512, 1024, kernel_size = 3, is_relu = self.is_relu, is_bn = self.is_bn)
        self.conv6b = conv3x3_bn(1024, 1024, is_relu = self.is_relu, is_bn = self.is_bn)
        self.upconv_disp6 = upconv4x4_bn(1, 1, is_relu = self.is_relu, is_bn = self.is_bn)
        #NOTE: ??? Maybe Batch Normalization will force to rescale the disparity be in distribution N(0, 1);
        #self.conv_disp6 = conv3x3_bn(1024,1, is_relu = self.is_relu, is_bn = self.is_bn)
        # so here, we try to disable the BN, and see what will happen!!!
        self.is_bn_for_disp_conv = False
        self.conv_disp6 = conv3x3_bn(1024,1, is_relu = self.is_relu, is_bn = self.is_bn_for_disp_conv)


        """ decoder """
        #deconv5
        self.upc5 = upconv4x4_bn(1024, 512, is_relu = self.is_relu, is_bn = self.is_bn)
        self.ic5 = conv3x3_bn(512+512+1, 512, is_relu = self.is_relu, is_bn = self.is_bn) # in/out channels = 1025/512
        self.upconv_disp5 = upconv4x4_bn(1, 1, is_relu = self.is_relu, is_bn = self.is_bn)
        #self.conv_disp5 = conv3x3_bn(512,1, is_relu = self.is_relu, is_bn = self.is_bn)
        self.conv_disp5 = conv3x3_bn(512,1, is_relu = self.is_relu, is_bn = self.is_bn_for_disp_conv)
        #deconv4
        self.upc4 = upconv4x4_bn(512, 256, is_relu = self.is_relu, is_bn = self.is_bn)
        self.ic4 = conv3x3_bn(512+256+1, 256, is_relu = self.is_relu, is_bn = self.is_bn) # in/out channels = 769/256
        self.upconv_disp4 = upconv4x4_bn(1, 1, is_relu = self.is_relu, is_bn = self.is_bn)
        #self.conv_disp4 = conv3x3_bn(256, 1, is_relu = self.is_relu, is_bn = self.is_bn)
        self.conv_disp4 = conv3x3_bn(256, 1, is_relu = self.is_relu, is_bn = self.is_bn_for_disp_conv)
        #deconv3
        self.upc3 = upconv4x4_bn(256, 128, is_relu = self.is_relu, is_bn = self.is_bn)
        self.ic3 = conv3x3_bn(256+128+1, 128, is_relu = self.is_relu, is_bn = self.is_bn) # in/out channels = 385/128
        self.upconv_disp3 = upconv4x4_bn(1, 1, is_relu = self.is_relu, is_bn = self.is_bn)
        #self.conv_disp3 = conv3x3_bn(128, 1, is_relu = self.is_relu, is_bn = self.is_bn)
        self.conv_disp3 = conv3x3_bn(128, 1, is_relu = self.is_relu, is_bn = self.is_bn_for_disp_conv)
        #deconv2
        self.upc2 = upconv4x4_bn(128, 64, is_relu = self.is_relu, is_bn = self.is_bn)
        self.ic2 = conv3x3_bn(128+64+1, 64, is_relu = self.is_relu, is_bn = self.is_bn) # in/out channels = 193/64
        self.upconv_disp2 = upconv4x4_bn(1, 1, is_relu = self.is_relu, is_bn = self.is_bn)
        #self.conv_disp2 = conv3x3_bn(64,1, is_relu = self.is_relu, is_bn = self.is_bn)
        self.conv_disp2 = conv3x3_bn(64,1, is_relu = self.is_relu, is_bn = self.is_bn_for_disp_conv)
        #deconv1
        self.upc1 = upconv4x4_bn(64, 32, is_relu = self.is_relu, is_bn = self.is_bn)
        self.ic1 = conv3x3_bn(64+32+1, 32, is_relu = self.is_relu, is_bn = self.is_bn) # in/out channels = 97/32
        self.upconv_disp1 = upconv4x4_bn(1, 1, is_relu = self.is_relu, is_bn = self.is_bn)
        #self.conv_disp1 = conv3x3_bn(32,1, is_relu = self.is_relu, is_bn = self.is_bn)
        self.conv_disp1 = conv3x3_bn(32,1, is_relu = self.is_relu, is_bn = self.is_bn_for_disp_conv)
        
        net_init(self)
        print ("[***]DispNet weights inilization done!")

    def forward(self, x, y):
        if not self.is_corr:
            """ DispNetS """
            x = torch.cat((x, y), dim=1)
            out = self.conv1(x)
            shortcut_c1 = out
            out = self.conv2(out)
            shortcut_c2 = out
            out = self.conv3a(out)
        else:
            """ DispNetC """
            #left image
            out_x = self.conv1(x)
            shortcut_c1 = out_x
            out_x = self.conv2(out_x)
            shortcut_c2 = out_x
            
            #right image
            out_y = self.conv1(y)
            out_y = self.conv2(out_y)
            # correlation map
            out = self.corr_func(out_x, out_y)
            out_redir = self.conv_redir(out_x)
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
        # added by CCJ for original size disp as output
        # NOTE: disp6 is in 1/64 size ==> interpolated to full size;
        H6, W6 = disp6.size()[2:]
        H0, W0 = 64*H6, 64*W6
        
        # decoder 5
        out = self.upc5(out)
        out = self.ic5(torch.cat((out, shortcut_c5, self.upconv_disp6(disp6)), dim=1))
        # disp6 will be used in the above line;
        # only after that, can we do torch.squeeze, and so do the following disp5,disp4,...,disp1
        disp6 = torch.squeeze(
                F.interpolate(disp6, [H0, W0], mode='bilinear', align_corners = True),
                1)# squeeze disp [N, 1, H, W] to [N, H, W]
        disp5 = self.conv_disp5(out) # in size [N, 1, H/32, W/32]

        # decoder 4
        out = self.upc4(out)
        out = self.ic4(torch.cat((out, shortcut_c4, self.upconv_disp5(disp5)), dim=1))
        disp5 = torch.squeeze(
                F.interpolate(disp5, [H0, W0], mode='bilinear', align_corners = True),
                1)# squeeze disp [N, 1, H, W] to [N, H, W]
        disp4 = self.conv_disp4(out) # in size [N, 1, H/16, W/16]
        
        # decoder 3
        out = self.upc3(out)
        out = self.ic3(torch.cat((out, shortcut_c3, self.upconv_disp4(disp4)), dim=1))
        disp4 = torch.squeeze(
                F.interpolate(disp4, [H0, W0], mode='bilinear', align_corners = True),
                1)# squeeze disp [N, 1, H, W] to [N, H, W]
        disp3 = self.conv_disp3(out) # in size [N, 1, H/8, W/8]
        
        # decoder 2
        out = self.upc2(out)
        out = self.ic2(torch.cat((out, shortcut_c2, self.upconv_disp3(disp3)), dim=1))
        disp3 = torch.squeeze(
                F.interpolate(disp3, [H0, W0], mode='bilinear', align_corners = True),
                1)# squeeze disp [N, 1, H, W] to [N, H, W]
        disp2 = self.conv_disp2(out) # in size [N, 1, H/4, W/4]
        
        # decoder 1
        out = self.upc1(out)
        out = self.ic1(torch.cat((out, shortcut_c1, self.upconv_disp2(disp2)), dim=1))
        disp2 = torch.squeeze(
                F.interpolate(disp2, [H0, W0], mode='bilinear', align_corners = True),
                1)# squeeze disp [N, 1, H, W] to [N, H, W]
        disp1 = self.conv_disp1(out) # in size [N, 1, H/2, W/2]
        disp1 = torch.squeeze(
                F.interpolate(disp1, [H0, W0], mode='bilinear', align_corners = True),
                1)# squeeze disp [N, 1, H, W] to [N, H, W]
        
        if self.training:
            return disp1, disp2, disp3, disp4, disp5, disp6
        else:
            return disp1

def scale_pyramid(img, num_scales=6, is_value_scaled = False):
    assert img.dim() == 4, "input tensor should be a 4D tensor"
    scaled_imgs = []
    # img : [N, C, H, W]
    #print ("[???] scale_pyramid: input size: ", img.size())
    h = img.size()[2]
    w = img.size()[3]
    for i in range(1, num_scales+1):
        ratio = 2 ** i
        nh = h // ratio
        nw = w // ratio
        #print ("[???] nh, nw :", nh, nw)
        #print (img.dtype)
        if not is_value_scaled:
            v_scale = 1.0
        else:
            v_scale = 1.0 / ratio
        scaled_imgs.append(
                v_scale*F.interpolate(img, size=[nh, nw], 
                    mode='bilinear', 
                    align_corners = True))
    return scaled_imgs
