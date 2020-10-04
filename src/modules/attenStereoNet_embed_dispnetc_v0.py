from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math


from .embednetwork import embed_net
from .bilateral import bilateralFilter
from ..baselines.DispNet.models.dispnet import (
        correlation1D_map_V1,
        downsample_conv_bn,
        conv3x3_bn,
        upconv3x3_bn,
        upconv4x4_bn,
        conv1x1_bn)
from src.net_init import net_init
#from torch.nn.parameter import Parameter

#NOTE:
"""
our network, haves disparity maps as out at 3 different scales: 
 - 1/4 (i.e., H/4 X W/4), 
 - 1/2 (i.e., H/2 X W/2), and
 - 1/1 (i.e., H X W), 
"""

class AttenStereoNet(nn.Module):
    def __init__(self, maxdisp=192, 
            sigma_s = 0.7, # 1.7: 13 x 13; 0.3 : 3 x 3;
            sigma_v = 0.1, 
            isEmbed = True, 
            dilation = 2,
            cost_filter_grad = True
            ):
        super(AttenStereoNet, self).__init__()
        self.maxdisp = maxdisp
        self.isEmbed = isEmbed # True of False
        self.sigma_s = sigma_s
        self.sigma_v = sigma_v
        #TODO:
        #if not fixed sigma_v:
        #from torch.nn.parameter import Parameter
        #self.sigma_v = Parameter(sigma_v * torch.ones(out_channels))
        
        self.dilation = dilation
        self.cost_filter_grad = cost_filter_grad
        """ embedding network """
        if self.isEmbed:
            print(' Enable Embedding Network!!!')
            self.embednet = embed_net()
            #the module layer:
            self.bifilter = bilateralFilter(sigma_s, sigma_v, isCUDA = True, dilation = self.dilation)
        else:
            self.embednet = None
            self.bifilter = None


        """ the following is from DispNetC """
        # due to TWO consecutive downsampling, so here maxdisp=40, 
        # actually means 4*maxdisp=160 in the original input image pair;
        self.maxdisp_corr = maxdisp // 4
        self.is_bn = True
        self.is_relu = True
        print ("[***] DispNet using is_relu : ", self.is_relu)
        print ("[***] DispNet using is_bn : ", self.is_bn)
        print ("[***] DispNet using maxdisp_corr : ", self.maxdisp_corr)
        self.corr_func = correlation1D_map_V1(self.maxdisp_corr)

        
        """ DispNetC: encoder """
        #conv1
        self.conv1 = downsample_conv_bn(3,64,kernel_size = 7,is_relu = self.is_relu,is_bn = self.is_bn)
        #conv2
        self.conv2 = downsample_conv_bn(64, 128, 5, is_relu = self.is_relu, is_bn = self.is_bn)
        #conv3a&b
        self.conv_redir = conv1x1_bn(128, 64,is_relu = self.is_relu,is_bn = self.is_bn)
        self.conv3a = downsample_conv_bn(64+self.maxdisp_corr,256, kernel_size = 5,is_relu = self.is_relu,is_bn = self.is_bn)
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
        self.conv_disp6 = conv3x3_bn(1024,1, is_relu = self.is_relu, is_bn = self.is_bn)

        """ DispNetC: decoder """
        #deconv5
        self.upc5 = upconv4x4_bn(1024, 512, is_relu = self.is_relu, is_bn = self.is_bn)
        self.ic5 = conv3x3_bn(512+512+1, 512, is_relu = self.is_relu, is_bn = self.is_bn) # in/out channels = 1025/512
        self.upconv_disp5 = upconv4x4_bn(1, 1, is_relu = self.is_relu, is_bn = self.is_bn)
        self.conv_disp5 = conv3x3_bn(512,1, is_relu = self.is_relu, is_bn = self.is_bn)
        #deconv4
        self.upc4 = upconv4x4_bn(512, 256, is_relu = self.is_relu, is_bn = self.is_bn)
        self.ic4 = conv3x3_bn(512+256+1, 256, is_relu = self.is_relu, is_bn = self.is_bn) # in/out channels = 769/256
        self.upconv_disp4 = upconv4x4_bn(1, 1, is_relu = self.is_relu, is_bn = self.is_bn)
        self.conv_disp4 = conv3x3_bn(256, 1, is_relu = self.is_relu, is_bn = self.is_bn)
        #deconv3
        self.upc3 = upconv4x4_bn(256, 128, is_relu = self.is_relu, is_bn = self.is_bn)
        self.ic3 = conv3x3_bn(256+128+1, 128, is_relu = self.is_relu, is_bn = self.is_bn) # in/out channels = 385/128
        self.upconv_disp3 = upconv4x4_bn(1, 1, is_relu = self.is_relu, is_bn = self.is_bn)
        self.conv_disp3 = conv3x3_bn(128, 1, is_relu = self.is_relu, is_bn = self.is_bn)
        #deconv2
        self.upc2 = upconv4x4_bn(128, 64, is_relu = self.is_relu, is_bn = self.is_bn)
        self.ic2 = conv3x3_bn(128+64+1, 64, is_relu = self.is_relu, is_bn = self.is_bn) # in/out channels = 193/64
        self.upconv_disp2 = upconv4x4_bn(1, 1, is_relu = self.is_relu, is_bn = self.is_bn)
        self.conv_disp2 = conv3x3_bn(64,1, is_relu = self.is_relu, is_bn = self.is_bn)
        #deconv1
        self.upc1 = upconv4x4_bn(64, 32, is_relu = self.is_relu, is_bn = self.is_bn)
        self.ic1 = conv3x3_bn(64+32+1, 32, is_relu = self.is_relu, is_bn = self.is_bn) # in/out channels = 97/32
        self.upconv_disp1 = upconv4x4_bn(1, 1, is_relu = self.is_relu, is_bn = self.is_bn)
        self.conv_disp1 = conv3x3_bn(32,1, is_relu = self.is_relu, is_bn = self.is_bn)
       
        net_init(self)
        print ("[***]DispNet weights inilization done!")




    def forward(self, x, y):
        """ DispNetC """
        #left image, in size [N, 3, H, W]
        out_x = self.conv1(x)
        shortcut_c1 = out_x
        out_x = self.conv2(out_x)
        shortcut_c2 = out_x
        
        #right image
        out_y = self.conv1(y)
        out_y = self.conv2(out_y)
        # correlation map, in [N, D, H/4, W/4]
        out = self.corr_func(out_x, out_y)
        out_redir = self.conv_redir(out_x)
        """ Try 1: adding our filtering module here """
        if not self.isEmbed:
            embed = None
        else:
            # downscale x to [N,C,H/4, W/4] then fed into embeddingnet,
            # because the cost volume generated below is in shape [N,C,D/4, H/4, W/4]
            left_scale = F.interpolate(x, [x.size()[2]//4, x.size()[3]//4], 
                    mode='bilinear', align_corners=True)
            #print ('[???] left shape', x.shape)
            #print ('[???] left_scale shape', left_scale.shape)
            """ embed shape [2, 64, 64, 128]"""
            embed = self.embednet(left_scale)
            #print ('[???] embed shape', embed.shape)

            """ correlation map in shape, [N, C=maxdisp/4, H/4, W/4]"""
            #print ('[???] correlation shape', out.shape)
            # NOTE: this might be the memory consuming!!!
            with torch.set_grad_enabled(self.cost_filter_grad):
                out = self.bifilter(embed, out.contiguous())
                out = out.contiguous()
                #print ('[???] filtered correlation shape', out.shape)
                out_redir = self.bifilter(embed, out_redir.contiguous())
                out_redir = out_redir.contiguous()
                #print ('[???] filtered out_redir shape', out_redir.shape)

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
        
        # decoder 2
        out = self.upc2(out)
        out = self.ic2(torch.cat((out, shortcut_c2, self.upconv_disp3(disp3)), dim=1))
        disp2 = self.conv_disp2(out) # in size [N, 1, H/4, W/4]
        
        # decoder 1
        out = self.upc1(out)
        out = self.ic1(torch.cat((out, shortcut_c1, self.upconv_disp2(disp2)), dim=1))
        disp1 = self.conv_disp1(out) # in size [N, 1, H/2, W/2]
        
        # added by CCJ for original size disp as output
        # NOTE: disp1 is in half size
        H1, W1 = disp1.size()[2:]
        # NOTE: disp0 is in original image size
        disp0 = F.interpolate(disp1, [2*H1, 2*W1], mode='bilinear', align_corners = True)
        
        # squeeze disp [N, 1, H, W] to [N, H, W]
        disp0 = torch.squeeze(disp0, 1) #1/1 scale
        #print ("[???] disp1 shape = ", disp1.shape)
        disp1 = torch.squeeze(disp1, 1) # 1/2 scale
        #print ("[???] disp1 shape = ", disp1.shape)
        disp2 = torch.squeeze(disp2, 1) # 1/4 scale
        #disp3 = torch.squeeze(disp3, 1)
        #disp4 = torch.squeeze(disp4, 1)
        #disp5 = torch.squeeze(disp5, 1)
        #disp6 = torch.squeeze(disp6, 1)
        
        if self.training:
            #return disp0, embed, [disp1, disp2, disp3, disp4, disp5, disp6]
            return disp2, disp1, disp0, embed
        else:
            # disp0 is in original image size
            # disp1 is in half size
            #return disp0, embed, disp1
            return disp0, embed
