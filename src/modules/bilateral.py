# !/usr/bin/env python3
# -*-coding:utf-8-*-
# @file: module_bilateral.py
# @brief:
# @author: Changjiang Cai, ccai1@stevens.edu, caicj5351@gmail.com
# @version: 0.0.1
# @creation date: 15-10-2019
# @last modified: Mon 29 Jun 2020 02:06:18 PM PDT

import numpy as np
import src.pfmutil as pfm
import sys
import math
import torch
import torch.nn as nn
from .im2col import im2col_layer

__all__ = ['bilateralFilter', 
        'get_space_gaussian_filter', 
        'get_gaussian_filter_width_from_sigma',
        ]
# try to use pyinn.im2col for speed up 
#try:
#    import pyinn as P
#    has_pyinn = True
#    print ('[!!!] Using pyinn for im2col()')
#except ImportError:
#    P = None
#    has_pyinn = False
#    pass

try:
    import src.tools.pyinn.im2col as P
    has_pyinn = True
    print ('[!!!] Found pyinn im2col()')

except ImportError:
    has_pyinn = False
    print ('[!!!] Cannot import pyinn im2col()')
    pass

#---------------------------------
""" some utility functions """
#---------------------------------

# define the window width to be the 3 time the spatial std. dev. to 
# be sure that most of the spatial kernel is actually captured
def get_gaussian_filter_width_from_sigma(sigma_s):
    return int(3*sigma_s  + 1)

def get_space_gaussian_filter(sigma_s = 0.1):
    """
    make a simple Gaussian function taking the squared radius
    """
    gaussian = lambda r2, sigma : np.exp( -0.5*r2/sigma**2)
                        
    # define the window width to be the 3 time the spatial std. dev. to 
    # be sure that most of the spatial kernel is actually captured
    win_width = get_gaussian_filter_width_from_sigma(sigma_s)
    #print ('win_size = {} x {}'.format(2*win_width+1, 2*win_width + 1))
    
    """ generate gaussian filter for space weight """
    sw_filter = np.zeros(( 2*win_width + 1, 2*win_width + 1))
    for shft_y in range(-win_width,win_width+1):
        for shft_x in range(-win_width,win_width+1):
            # compute the spatial weight
            y_idx = shft_y + win_width
            x_idx = shft_x + win_width
            #print (type(x_idx), (type(y_idx)))
            sw = gaussian(shft_x**2+shft_y**2, sigma_s) # scalar sw
            #print ("sw_filter[%d, %d] = %f" %(y_idx, x_idx, sw))
            sw_filter[y_idx, x_idx] = sw
            #print ("sw_filter = ", sw_filter)
    return sw_filter.astype(np.float32)


""" module : bilateral filter leveraging embedding feature """
class bilateralFilter(nn.Module):
    def __init__(self,  sigma_s = 0.7, sigma_v = 0.1, isCUDA = True, dilation = 1):
        super(bilateralFilter, self).__init__()
        """
        Member Args:
            self.sg_filter  (Tensor)   spatial gaussia filter, given sigma_s as spatial gaussian std. dev.
            self.win_width    (float)  spatical gaussin filter half window size, determined by space filter std dev sigma_s;
            self.sigma_s      (float)  space gaussian std. dev.
            self.sigma_v      (float)  value gaussian std. dev.
            self.reg_constant (float)  optional regularization constant for pathalogical cases
        """
        self.sigma_s = sigma_s
        self.win_width = get_gaussian_filter_width_from_sigma(self.sigma_s)
        self.sigma_v = sigma_v
        #self.inv_sigma_v_squ = - 0.5 / (sigma_v ** 2)
        self.k = 2 * self.win_width + 1
        
        """ added for pyinn.im2col() """
        self.d = dilation
        self.pad = self.d * (self.k - 1) // 2
        
        self.isCUDA = isCUDA
        self.sg_filter = torch.from_numpy(get_space_gaussian_filter(self.sigma_s)).view(1,1,self.k*self.k,1,1)
        #print ('[???]sg_filter :', self.sg_filter.requires_grad)
        
        #if self.isCUDA:
            # cuda() and to('cuda') are going to do the same thing, but the later is more flexible;
            # With the explicit call, you can also use multiple cuda devices – e.g. to('cuda:0') is 
            # different from to('cuda:1'). The simpler cuda() call will just use the DEFAULT cuda device.
            #self.sg_filter = self.sg_filter.cuda()
            #self.sg_filter = self.sg_filter.to("cuda:0")

        self.im2col = im2col_layer(k = self.k, d = dilation, is5D = True)
        self.reg_constant=1e-20 # regularization constant


    def forward(self, embed_in, x_in):
        """
        using im2col to change the filtering to element-wise matrix multiplication
        Performs standard bilateral filtering of an input image.
        Padding is taken care by the inside function im2col_pytorch()!!!

        Args:
            embed_in (Tensor), embeding for generatin mask, in size [N,F,H,W]
            x_in     (Tensor), input array, in size [N,C,H,W], 
                            apply the bilateral filter to it;

        Returns:
            result (Tensor) bilateral-filtered x_in, in size [N,C,H,W]
        """
    
        N,F,H,W = embed_in.shape[:]
        #print ('embed_in shape = ', embed_in.shape)
        
        #NOTE: newly added for multiple GPUs
        if self.isCUDA:
            # cuda() and to('cuda') are going to do the same thing, but the later is more flexible;
            # With the explicit call, you can also use multiple cuda devices – e.g. to('cuda:0');
            # The simpler cuda() call will just use the DEFAULT cuda device.
            #self.sg_filter = self.sg_filter.to("cuda:0")
            self.sg_filter = self.sg_filter.to(x_in.device)
        
        """ apply im2col for embedding masking """
        # The PAC paper said that 'Use PyINN if possible (about 15% faster)',
        # But CCJ tried that and found that is not the case, at least for CCJ's machine!
        if has_pyinn and self.d == 1:
            embed_im2col = P.im2col(embed_in, (self.k, self.k), 
                    stride = 1, padding = self.pad).view(N,F,-1,H,W)
            #print (embed_im2col.shape)
        else:
            embed_im2col = self.im2col(embed_in) # in shape [N, F, k*k, H, W]


        #print ("sigma_v ", sigma_v)
        #embed_tile = embed_in.view(N,F,1,H,W)
        # broadcasting
        embed_im2col -= embed_in.view(N,F,1,H,W)
        #rg_filter = torch.exp(-0.5*torch.sum((embed_im2col - embed_tile)**2, dim = 1, keepdim= True) / (self.sigma_v**2)) # [N,1,k*k,H,W]
        rg_filter = torch.exp_(-0.5*torch.sum(embed_im2col.pow_(2), dim = 1, keepdim= True) / (self.sigma_v**2)) # [N,1,k*k,H,W]
        #rg_filter = torch.exp_(torch.sum(embed_im2col.pow_(2), dim = 1, keepdim= True).mul_(-0.5/(self.sigma_v**2))) # [N,1,k*k,H,W]
        
        if 0: # show
            print ('rg_filter shape = ', rg_filter.shape)
            for i in range(0, self.k*self.k):
                shft_x = i % (2*self.win_width+1) - self.win_width
                shft_y = i / (2*self.win_width+1) - self.win_width
                print("show shift [%d][%d] " %( shft_y, shft_x))
                pfm.show(rg_filter[0,0,i,:,:].cpu().numpy())
        #print ('rg_filter :', rg_filter[0,0,:,20,20])
        # in shape (N, 1, k*k, H, W)
        #print ("[**] ", rg_filter.shape, self.sg_filter.shape)
        #Updated on 2019/11/08: change it to in-place operation
        #wgt_filter = rg_filter * self.sg_filter # broadcasting
        
        #NOTE:updated due to the gradient error!!!
        #change the inplace *= to out-of-place one, due to the error: gradient computation has been modified by an inplace operation
        #rg_filter *= self.sg_filter # broadcasting
        rg_filter = rg_filter*self.sg_filter # broadcasting
        
        

        """
        implement eq (4) in W. Harley' paper 
        "Segmentation-Aware Convolutional Networks Using Local Attention Masks (ICCV'17)"
        """
        wgt_sum = torch.ones([N, 1, H, W])*self.reg_constant
        if self.isCUDA:
            wgt_sum = wgt_sum.cuda()
        """ broadcasting """
        #unrolled for element-wise multiplication:
        N, C, H, W = x_in.size()[:]
        #print ('[???] x_in shape = ', x_in.shape)
        x_in_im2col = self.im2col(x_in) # in shape [N, C, k*k, H, W]
        
        #(N, C, k*k, H, W ) * (N, 1, k*k, H, W) => (N, C, k*k, H, W)
        #print ("[**] x shape, wgt_filter shape :", x_in_im2col.shape, rg_filter.shape)
        #result = x_in_im2col * wgt_filter 
        wgt_sum += torch.sum(rg_filter, axis=2) # N x 1 x H x W 
        x_in_im2col *= rg_filter # N x C x k*k x H x W
        result = torch.sum(x_in_im2col, axis = 2) # N x C x H x W
        # normalize the result and return
        # N x C x H x W 
        result /= wgt_sum
        return result
