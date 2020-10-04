# !/usr/bin/env python3
# -*-coding:utf-8-*-
# @file: dfn.py
# @brief:
# @author: Changjiang Cai, ccai1@stevens.edu, caicj5351@gmail.com
# @version: 0.0.1
# @creation date: 04-12-2019
# @last modified: Wed 29 Jan 2020 09:51:43 PM EST

# >see: https://github.com/ccj5351/dfn/blob/master/experiment_steerableFilter_tensorflow.ipynb
# >see: NIPS 2016 - dynamic filter network paper at: https://arxiv.org/pdf/1605.09673v2.pdf
# the code is adopted from the above link

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .im2col import im2col_layer
import math
from src.net_init import net_init_v0
from src.net_init import net_init_SyncBN

####################
### utility ########
####################
# > see https://github.com/pytorch/pytorch/issues/7965
from torch.nn.modules.utils import _pair

class Conv2dUntiedBias(nn.Module):
    def __init__(self, height, width, in_channels, out_channels, 
            stride = 1, kernel_size=3, padding=0, dilation=1, groups=1, 
            is_relu = False, scopename = 'convUntiedBias'):
        super(Conv2dUntiedBias, self).__init__()
        print("in=%3d, out=%3d, strid=%d, pad =%d, kernel=%2d, isRelu=%s, %s"%(
            in_channels, out_channels, stride, padding, kernel_size, is_relu, scopename))
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)

        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups))
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, *kernel_size))
        self.bias = nn.Parameter(torch.Tensor(out_channels, height, width))
        #self.reset_parameters()
        self.he_initialize()
        self.is_relu = is_relu
        self.relu = nn.LeakyReLU(0.01, inplace = True)

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    """
    torch.nn.init.kaiming_normal_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu'): 
      Fills the input Tensor with values according to the method described 
      in Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification - He, K. et al. (2015), 
      using a normal distribution. The resulting tensor will have values sampled from 
      N(0, std^2), where std = sqrt(2/( (1+a^2)* fan_in )),
      Also known as He initialization!
    Argus:
       mode : either 'fan_in' (default) or 'fan_out'. Choosing 'fan_in' preserves 
           the magnitude of the variance of the weights in the forward pass. 
           Choosing 'fan_out' preserves the magnitudes in the backwards pass.
    """
    def he_initialize(self):
        nn.init.kaiming_normal_(
                self.weight,
                mode='fan_out', 
                nonlinearity='relu')
        # copy the bias argus in nn.Conv2d():
        # the learnable bias of the module of shape (out_channels).
        # the values are sampled from U(−sqrt(k), sqrt(k))
        # where k = 1/(C_in * kernel_h * kernel_w)
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        output = F.conv2d(
                x, self.weight, bias = None, stride = self.stride,
                padding = self.padding, dilation = self.dilation, 
                groups = self.groups)
        #print ("[????] output shape = ", output.size())
        
        # add untied bias
        #print ("[????] bias shape = ", self.bias.size())
        output += self.bias.unsqueeze(0).repeat(x.size(0), 1, 1, 1)
        
        if self.is_relu:
            output = self.relu(output)
        return output

####################
### utility ########
####################
class BasicConv2d(nn.Module):
    def __init__(self, 
            in_channels, out_channels, stride,
            padding = 1,
            dilation = 1,
            kernel_size = 3,
            isRelu=True,
            scopename = 'conv'
            ):
        
        super(BasicConv2d, self).__init__()
        print("in=%3d, out=%3d, strid=%d, pad =%d, kernel=%2d, isRelu=%s, %s" %(
            in_channels, out_channels, stride, padding, kernel_size, isRelu, scopename))
        self.is_relu = isRelu
        self.conv = nn.Conv2d( 
                in_channels = in_channels, 
                out_channels = out_channels,
                kernel_size = kernel_size,
                stride = stride,
                padding = padding,
                dilation = dilation,
                bias = True)

        self.relu = nn.LeakyReLU(0.01, inplace = True)

    def forward(self, x):
        x = self.conv(x)
        if self.is_relu:
            x = self.relu(x)
        return x

####################
### utility ########
####################
class BasicDeConv2d(nn.Module):
    def __init__(self, 
            in_channels, out_channels, stride,
            padding = 0,
            # see https://pytorch.org/docs/stable/nn.html#convolution-layers
            output_padding = 0, #Additional size added to one side of each dimension in the output shape;
            dilation = 1,
            kernel_size = 3,
            isRelu=True,
            scopename = 'deconv'
            ):
        
        super(BasicDeConv2d, self).__init__()
        print("in=%3d, out=%3d, strid=%d, pad =%d, kernel=%2d, isRelu=%s, %s" %(
            in_channels, out_channels, stride, padding, kernel_size, isRelu, scopename))
        self.is_relu = isRelu
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels,  
                kernel_size = kernel_size,
                stride = stride,
                padding = padding,
                output_padding = output_padding,
                dilation = dilation,
                bias=True,
                )
        
        self.relu = nn.LeakyReLU(0.01, inplace = True)

    def forward(self, x):
        x = self.deconv(x)
        if self.is_relu:
            x = self.relu(x)
        return x


###############################
#  filter-generating network  #
###############################
""" see "Dynamic Filter Networks" (NIPS 2016)
        by Bert De Brabandere*, Xu Jia*, Tinne Tuytelaars and Luc Van Gool
"""
class filterGenerator(nn.Module):
    """ model initialization """
    def __init__(self, F = 32, 
            dynamic_filter_size=(9,9), 
            #img_size = (256, 512), 
            in_channels = 3,
            is_sync_bn = False
            ):
        super(filterGenerator, self).__init__()
        self.F = F
        self.kernel_h = dynamic_filter_size[0]
        self.kernel_w = dynamic_filter_size[1]
        self.in_channels = in_channels
        """encoder"""
        self.conv1 = BasicConv2d(self.in_channels, self.F, stride = 1, scopename='conv1') # 32
        # > see: https://pytorch.org/docs/stable/nn.html#convolution-layers for input-output size relationship;
        # downsampling here
        # O = (W - F + 2P )/ S + 1
        # O = (W - 3 + 2P)/2 + 1 = (W -3 + 2P + 2)/2 = W/2 + (2P -1)/2
        self.conv2 = BasicConv2d(self.F, self.F, stride = 2, scopename='conv2') # 32
        self.conv3 = BasicConv2d(self.F, 2*self.F, stride = 1, scopename='conv3') # 64
        self.conv4 = BasicConv2d(2*self.F, 2*self.F, stride = 1, scopename='conv4') # 64
       
        """ !!!Problem: using untie_biases will requrie the input image size is 
            the same as the one during training!
        """
        # untie_biases
        #self.conv5 = Conv2dUntiedBias(
        #        height = img_size[0] // 2, # due to the stride = 2 in conv2
        #        width = img_size[1] // 2, # due to the stride = 2 in conv2
        #        in_channels = 2*self.F, out_channels= 4*self.F, stride = 1, 
        #        kernel_size = 3, padding = 1, is_relu = True, scopename = 'conv5_UntiedBias') # 128
        """ just use regular convolution """
        self.conv5 = BasicConv2d(2*self.F, 4*self.F, stride = 1, scopename='conv5') # 64
        """decoder"""
        self.conv6 = BasicConv2d(4*self.F, 2*self.F, stride = 1, scopename='conv6') # 64
        self.conv7 = BasicConv2d(2*self.F, 2*self.F, stride = 1, scopename='conv7') # 64

        # deconv
        self.deconv8 = BasicDeConv2d(2*self.F, 2*self.F, stride = 2, padding = 1, 
                output_padding = 1, 
                scopename='deconv8') # F=64, S=2

        self.conv9 = BasicConv2d(2*self.F, 2*self.F, stride = 1, scopename='conv9') # 64
        self.conv10 = BasicConv2d(2*self.F, 4*self.F, stride = 1, kernel_size = 1, padding = 0, scopename='conv10') # 128, K = 1
        
        """filter-generating layers"""
        # NOTE:
        # 1) For example, filter kernel size = [13, 13],
        # then the out_channels will be 13*13 + 1 (for bias) = 170;
        # 2) for stereo prediction, if just horizontal filter is used, that is, filter size = 1 x 13;
        # then the out_channels will be 1*13 + 1 (for bias) = 14;
        self.conv11 = BasicConv2d(4*self.F, self.kernel_h*self.kernel_w + 1, 
                stride = 1, kernel_size = 1, padding = 0, scopename='conv11') #170, K=1
        self.soft_max = nn.Softmax2d() #you softmax over the 2nd dimension, given input in size [N,C,H,W]
        
        if is_sync_bn:
            net_init_SyncBN(self)
            print ("[***] calling net_init_SyncBN(): filterGenerator weights inilization done!")
        else:
            net_init_v0(self)
            print ("[***] calling net_init_v0(): filterGenerator weights inilization done!")

    
    def forward(self, x):
        assert (x.size()[1] == self.in_channels)
        """encoder"""
        #x_in = x
        x = self.conv1(x)
        #print ('conv1:', x.size())
        x = self.conv2(x)
        #print ('conv2:', x.size())
        x = self.conv3(x)
        #print ('conv3:', x.size())
        x = self.conv4(x)
        #print ('conv4:', x.size())
        ## mid
        x = self.conv5(x)
        #print ('conv5:', x.size())
        """decoder"""
        x = self.conv6(x)
        #print ('conv6:', x.size())
        x = self.conv7(x)
        #print ('conv7:', x.size())
        x = self.deconv8(x)
        #print ('deconv8:', x.size())
        x = self.conv9(x)
        #print ('conv9:', x.size())
        x = self.conv10(x)
        #print ('conv10:', x.size())
        """filter-generating layers"""
        x = self.conv11(x)
        #print ('conv11:', x.size())
        # filter weights
        filters = x[:,0:-1,:,:] # [N, k*k, H, W]
        # filter bias
        #NOTE: we just want to extract the last slice along dim1, but using [:,-1:,:,:], results in 3D tensor, 
        # instead of 4D tensor, which we actually want!!!
        filters_biases = x[:,self.kernel_h*self.kernel_w:self.kernel_h*self.kernel_w+1,:,:] # [N, 1, H, W]
        # or equivalently could be:
        #filters_biases = x[:,-1:,:,:] # [N, 1, H, W]

        ## softmax along feature channel
        filters = self.soft_max(filters) # [N, k*k, H, W]
        #print ("[???] dfn_filters size = ", filters.size())
        #print ("[???] dfn_filters_biases size = ", filters_biases.size())
        return filters, filters_biases

""" NOT USED YET """
class DynamicFilterLayerOneChannel(nn.Module):
    def __init__(self, kernel_size = 9, dilation = 1):
        super(DynamicFilterLayerOneChannel, self).__init__()
        self.kernel_size = kernel_size
        self.im2col = im2col_layer(k = self.kernel_size, d = dilation, is5D = False)

    #####################################
    def forward(self, x_in, filters, filters_biases):
        """ 
         args:
          x_in : input tensor, could be slice or slices of cost volume, in shape [N,C=1,H,W]
          filters: fynimic filters, in shape [N, k*k, H, W]
          filters_biases: used as additive biases after the filtering, in shape [N, 1, H, W];
        """
        assert (x_in.size()[1] == 1)
        #k = self.kernel_size
        x_in_im2col = self.im2col(x_in) # in shape [N, C*k*k, H, W], with C = 1;
        output = torch.sum(x_in_im2col * filters, axis=1, keepdim= True) + filters_biases
        return output


class DynamicFilterLayer(nn.Module):
    def __init__(self, kernel_size = 9, dilation = 1):
        super(DynamicFilterLayer, self).__init__()
        self.kernel_size = kernel_size 
        self.im2col = im2col_layer(k = self.kernel_size, d = dilation, is5D = True)

    #####################################
    def forward(self, x_in, filters, filters_biases):
        """ 
         args:
          x_in : input tensor, could be slice or slices of cost volume, in shape [N, C, H, W]
          filters: dynamic filters, in shape [N, k*k, H, W]
          filters_biases: used as additive biases after the filtering, in shape [N, 1, H, W];
        """
        #k = self.kernel_size
        N, F, H, W = filters.size()[:]
        #print ("[???] filters size = ", filters.size())
        filters = filters.view(N,1,F,H,W)
        x_in_im2col = self.im2col(x_in) # in shape [N, C, k*k, H, W]
        #NOTE:broadcasting: [N, C, k*k, H, W] * [N,1,k*k, H, W] ==> [N, C, k*k, H, W]
        output = torch.sum(x_in_im2col * filters, axis = 2, keepdim=False) + filters_biases
        return output