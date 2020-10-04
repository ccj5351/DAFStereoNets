# !/usr/bin/env python3
# -*-coding:utf-8-*-
# @file:
# @brief:
# @author: Changjiang Cai, ccai1@stevens.edu, caicj5351@gmail.com
# @version: 0.0.1
# @creation date: 29-09-2019
# @last modified: Sun 26 Jan 2020 03:06:57 AM EST

# > see: http://jamesgregson.ca/bilateral-filtering-in-python.html
import numpy as np
import src.pfmutil as pfm
import sys
import math
import torch
import torch.nn as nn

def filter_bilateral( img_in, sigma_s, sigma_v, reg_constant=1e-8 ):
    """Simple bilateral filtering of an input image

    Performs standard bilateral filtering of an input image. If padding is desired,
    img_in should be padded prior to calling

    Args:
        img_in       (ndarray) monochrome input image
        sigma_s      (float)   spatial gaussian std. dev.
        sigma_v      (float)   value gaussian std. dev.
        reg_constant (float)   optional regularization constant for pathalogical cases

    Returns:
        result       (ndarray) output bilateral-filtered image

    Raises: 
        ValueError whenever img_in is not a 2D float32 valued np.ndarray
    """

    # check the input
    if not isinstance( img_in, np.ndarray ) or img_in.dtype != 'float32' or img_in.ndim != 2:
        raise ValueError('Expected a 2D np.ndarray with float32 elements')


    # make a simple Gaussian function taking the squared radius
    gaussian = lambda r2, sigma: (np.exp( -0.5*r2/sigma**2 )*3).astype(int)*1.0/3.0

    # define the window width to be the 3 time the spatial std. dev. to 
    # be sure that most of the spatial kernel is actually captured
    win_width = int( 3*sigma_s+1 )

    # initialize the results and sum of weights to very small values for
    # numerical stability. not strictly necessary but helpful to avoid
    # wild values with pathological choices of parameters
    wgt_sum = np.ones( img_in.shape )*reg_constant
    result  = img_in*reg_constant

    # accumulate the result by circularly shifting the image across the
    # window in the horizontal and vertical directions. within the inner
    # loop, calculate the two weights and accumulate the weight sum and 
    # the unnormalized result image
    for shft_x in range(-win_width,win_width+1):
        for shft_y in range(-win_width,win_width+1):
            # compute the spatial weight
            w = gaussian( shft_x**2+shft_y**2, sigma_s )

            # shift by the offsets
            off = np.roll(img_in, [shft_y, shft_x], axis=[0,1] )

            # compute the value weight
            tw = w*gaussian( (off-img_in)**2, sigma_v )

            # accumulate the results
            result += off*tw
            wgt_sum += tw

    # normalize the result and return
    return result/wgt_sum 


def gaussian_weights(sigma):
    kr = math.ceil(sigma*3)
    ks = int(kr*2+1)
    k = np.zeros((ks,ks))
    for i in range(0,ks):
        for j in range(0,ks):
            y = i-kr
            x = j-kr
            k[i,j] = math.exp( - (x*x+y*y)/ (2*sigma*sigma) )
    
    return k.astype(np.float32)



def filter_bilateral_with_embeding( x_in, embed_in, sigma_s = 10.0, sigma_v = 0.1, reg_constant=1e-20, 
        hard_lambda = 1.0):
    """Simple bilateral filtering of an input image

    Performs standard bilateral filtering of an input image. If padding is desired,
    embed_in should be padded prior to calling

    Args:
        x_in       (ndarray)   input array, in size [H,W,C], could be any input, like, RGB image, gray image, or cost volume, or cost volume slice
        embed_in   (ndarray)   embeding feature, used for generatin mask, in size [H,W,F] (same to PyTorch)
        sigma_s      (float)   spatial gaussian std. dev.
        sigma_v      (float)   value gaussian std. dev.
        reg_constant (float)   optional regularization constant for pathalogical cases
        hard_lambda (float)    hardness of mask based on embed_in, see parameter lambda in equ (3) in W. Harley' paper "Segmentation-Aware Convolutional Networks Using Local Attention Masks (ICCV'17)"

    Returns:
        result       (ndarray) output bilateral-filtered image

    Raises: 
        ValueError whenever embed_in is not a 3D float32 valued np.ndarray
    """

    # check the input
    if not isinstance( embed_in, np.ndarray ) or embed_in.dtype != 'float32' or embed_in.ndim != 3:
        raise ValueError('Expected embed_in a ３D np.ndarray with float32 elements')
    if not isinstance( x_in, np.ndarray ) or x_in.dtype != 'float32' or x_in.ndim != 3:
        raise ValueError('Expected x_in a ３D np.ndarray with float32 elements')
    H, W, F = embed_in.shape[:]
    _,_, C = x_in.shape[:]
    #print ('embed_in shape = ', embed_in.shape)
    #print ('x_in shape = ', x_in.shape)
    # make a simple Gaussian function taking the squared radius
    #gaussian = lambda r2, sigma, c: (np.exp( c*-0.5*r2/sigma**2 )*3).astype(int)*1.0/3.0
    gaussian = lambda r2, sigma, c: np.exp( c*-0.5*r2/sigma**2)

    # define the window width to be the 3 time the spatial std. dev. to 
    # be sure that most of the spatial kernel is actually captured
    win_width = int( 3*sigma_s+1 )
    #print ('win_size = {} x {}'.format(2*win_width+1, 2*win_width + 1))

    # initialize the results and sum of weights to very small values for
    # numerical stability. not strictly necessary but helpful to avoid
    # wild values with pathological choices of parameters
    wgt_sum = np.ones([H,W])*reg_constant
    #print ("showing initial wgt_sum ")
    #pfm.show(wgt_sum)
    result  = x_in * reg_constant

    L2_dist_suqre = lambda x,y: ( np.sum((x-y)**2, axis = -1))
    # accumulate the result by circularly shifting the image across the
    # window in the horizontal and vertical directions. within the inner
    # loop, calculate the two weights and accumulate the weight sum and 
    # the unnormalized result image

    """ generate gaussian filter for space weight """
    sw_filter = np.zeros(( 2*win_width + 1, 2*win_width + 1))
    for shft_y in range(-win_width,win_width+1):
        for shft_x in range(-win_width,win_width+1):
            # compute the spatial weight
            y_idx = shft_y + win_width
            x_idx = shft_x + win_width
            #print (type(x_idx), (type(y_idx)))
            sw = gaussian(shft_x**2+shft_y**2, sigma_s, hard_lambda) # scalar sw
            #print ("sw_filter[%d, %d] = %f" %(y_idx, x_idx, sw))
            sw_filter[y_idx, x_idx] = sw
    #print ("sw_filter = ", sw_filter)
    
    # verify the sw_filter
    #sw_filter2 = gaussian_weights(sigma_s)
    #print (sw_filter2)
    #sys.exit()



    rw_accum = []
    for shft_y in range(-win_width,win_width+1):
        for shft_x in range(-win_width,win_width+1):
            # compute the spatial weight
            #sw = gaussian( shft_x**2+shft_y**2, sigma_s, hard_lambda) # scalar sw
            sw = sw_filter[shft_y + win_width, shft_x + win_width]
            print ("sw[%d][%d] = %f" %(shft_y + win_width, shft_x + win_width, sw))
            
            # shift by the offsets
            off_embed = np.roll(embed_in, [shft_y, shft_x, 0], axis=[0,1,2])
            off_x = np.roll(x_in, [shft_y, shft_x, 0], axis=[0,1,2])

            # range weight (rw) : [H, W]
            rw = gaussian(L2_dist_suqre(off_embed, embed_in),sigma_v, hard_lambda)
            #print ('rw shape = ', rw.shape)
            rw_accum.append(rw)
            
            #NOTE: debugging
            #sw = 1.0
            tw =sw*rw # in shape [H, W]

            # accumulate the results
            #NOTE:
            # off_x in shape [H,W,C] (note: here C could be ndisp if x_in = cost volume)
            result += off_x * np.expand_dims(tw, axis = -1) # [H,W,C]
            wgt_sum += tw

    if 0:
        rw_all = np.stack(rw_accum, axis=-1)
        print ('rw_all shape = ', rw_all.shape)
        for i in range(0, rw_all.shape[2]):
            shft_x = i % (2*win_width+1) - win_width
            shft_y = i / (2*win_width+1) - win_width
            print("show shift [%d][%d] " %( shft_y, shft_x))
            pfm.show(rw_all[:,:,i])
    
    # normalize the result and return
    return result/ np.expand_dims(wgt_sum, axis = -1), wgt_sum  # in shape [H,W,C] 


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
    return sw_filter.astype(np.float32), win_width


def im2col_pytorch(x, k, d):
    """
    args:
         x : input, [N,C,H,W]
         k : window size
         d : dilation
    return:
         y : in shape [N, C*(k*k), H, W]
    """
    N, C, H, W = x.size()[:]
    # 3D tensor in size [N, (k*k)*C, L], here L = H*W
    #print ('padding = ', k//2)
    #print ('[???]dilation = {}'.format(d))
    p = d*(k-1)//2
    unfold = nn.Unfold(kernel_size= k, dilation = d, padding = p, stride = 1)
    #NOTE:
    """
    PyTorch im2col (i.e., nn.Unfold) flattens each k by k
    block into a column which conains C*(k*k) values, where k*k is a
    continugous chunk, with C be the Channel dimension.
    """
    #return unfold(x).view(N, C, k*k, H, W)
    return unfold(x).view(N, C*k*k, H, W)



def bilateral_embeding_im2col_pytorch( 
        x_in_np, 
        embed_in_np, 
        sg_filter_np,
        win_width,
        dilation,
        sigma_v = 0.1,
        device_str='cuda:0'
        ):

    """ using im2col to change the filtering to matrix multiplication

    Performs standard bilateral filtering of an input image. 
    Padding is taken care by the inside function im2col_pytorch()!!!

    Args:
        x_in_np       (ndarray)   input array, in size [H,W,C], could be any input, like, RGB image, gray image, or cost volume, or cost volume slice
        embed_in_np   (ndarray)   embeding feature, used for generatin mask, in size [H,W,F] (same to PyTorch)
        sg_filter_np  (ndarray)   spatial gaussia filter, given sigma_s as spatial gaussian std. dev.
        win_width    (float)   spatical gaussin filter half window size; 
        sigma_v      (float)   value gaussian std. dev.

    Returns:
        result       (Tensor) output bilateral-filtered image

    Raises: 
        ValueError whenever embed_in is not a 3D float32 valued np.ndarray
    """
    
    # check the input
    if not isinstance( embed_in_np, np.ndarray ) or embed_in_np.dtype != 'float32' or embed_in_np.ndim != 3:
        raise ValueError('Expected embed_in_np a ３D np.ndarray with float32 elements')
    if not isinstance( x_in_np, np.ndarray ) or x_in_np.dtype != 'float32' or x_in_np.ndim != 3:
        raise ValueError('Expected x_in_np a ３D np.ndarray with float32 elements')
    
    N = 1 # batch size
    k = 2*win_width + 1
    H, W, F = embed_in_np.shape[:]
    _,_, C = x_in_np.shape[:]
    
    #print ('[??] embed_in_np shape = ', embed_in_np.shape)
    #print ('[??] x_in_np shape = ', x_in_np.shape)
    assert (embed_in_np.shape[:2] == x_in_np.shape[:2])
    
    """ you can either use cpu or gpu to run this bilateral filtering """
    #device = torch.device("cuda:0")
    #device = torch.device("cpu:0")
    device = torch.device(device_str)
    # H x W x C ==> N x C x H x  W (here N = 1 for batch size)
    x_in = torch.from_numpy(x_in_np.transpose(2,0,1)).to(device).view(N,C,H,W)
    embed_in = torch.from_numpy(embed_in_np.transpose(2,0,1)).to(device).view(N,F,H,W)
    sg_filter = torch.from_numpy(sg_filter_np).to(device)
    reg_constant = 1e-20
    _, result, filter_sum = apply_batch_bilateral_filter( 
            embed_in, sg_filter, 
            win_width, sigma_v, device, 
            dilation,
            x_in, 
            reg_constant)
    
    return result, filter_sum


def apply_batch_bilateral_filter(
        embed_in,
        sg_filter, 
        win_width, 
        sigma_v,
        device, 
        dilation,
        x_in = None, 
        reg_constant=1e-20 # regularization constant
        ):

    """ using im2col to change the filtering to element-wise matrix multiplication

    Performs standard bilateral filtering of an input image. 
    Padding is taken care by the inside function im2col_pytorch()!!!

    Args:
        embed_in   (Tensor)   embeding for generatin mask, in size [N,F,H,W]
        sg_filter  (Tensor)   spatial gaussia filter, given sigma_s as spatial gaussian std. dev.
        win_width    (float)  spatical gaussin filter half window size; 
        sigma_v      (float)  value gaussian std. dev.
        reg_constant (float)  optional regularization constant for pathalogical cases
        x_in       (Tensor)   input array, in size [N,C,H,W],
                              if not None, then apply the bilateral filter to it;

    Returns:
        result (Tensor) output bilateral filter and/or bilateral-filtered x_in
    """
    
    k = 2*win_width + 1
    N,F,H,W = embed_in.shape[:]
    #print ('embed_in shape = ', embed_in.shape)
    
    
    """ apply im2col for embedding masking """
    embed_im2col = im2col_pytorch(embed_in, k, dilation) # in shape [N, (k*k*F), H, W]
    embed_im2col = embed_im2col.view(N, F, k*k, H, W)
    #print ("sigma_v ", sigma_v)
    """ change to in-place operation to save GPU memory """
    embed_tile = embed_in.view(N,F,1,H,W)
    embed_im2col -= embed_tile # broadcasting
    rg_filter = torch.exp_(-0.5*torch.sum(embed_im2col.pow_(2), dim = 1, keepdim= True) / (sigma_v**2)) # [N,1,k*k,H,W]
    #print ('rg_filter :', rg_filter[0,0,:,20,20])
    sg_filter = sg_filter.view(1, 1, k*k, 1, 1)
    # in shape (N, 1, k*k, H, W)
    rg_filter *= sg_filter # broadcasting

    if x_in is None:
        result = None
    else:
        """
        emplement eq (4) in W. Harley' paper 
        "Segmentation-Aware Convolutional Networks Using Local Attention Masks (ICCV'17)"
        """
        filter_sum = torch.ones([N, 1, H, W], device = device)*reg_constant
        """ broadcasting """
        #unrolled for element-wise multiplication:
        N,C, H, W = x_in.shape[:]
        x_in_im2col = im2col_pytorch(x_in, k, dilation) # in shape [N, C*k*k,H,W]
        x_in_im2col = x_in_im2col.view(N,C,k*k, H,W) # in shape [N,C,k*k,H,W]
        
        #(N, C, k*k, H, W ) * (N, 1, k*k, H, W) => (N, C, k*k, H, W)
        result = x_in_im2col * rg_filter
        result = torch.sum(result, axis = 2) # N x C x H x W
        filter_sum += torch.sum(rg_filter, axis=2) # N x 1 x H x W 
        # normalize the result and return
        # N x C x H x W 
        result = result / filter_sum
        
        if 0: # show
            print ('rg_filter shape = ', rg_filter.shape)
            for i in range(0, k*k):
                shft_x = i % (2*win_width+1) - win_width
                shft_y = i / (2*win_width+1) - win_width
                print("show shift [%d][%d] " %( shft_y, shft_x))
                pfm.show(rg_filter[0,0,i,:,:].cpu().numpy())
    
    return rg_filter, result, filter_sum
