# !/usr/bin/env python3
# -*-coding:utf-8-*-
# @file: bilateral_filter_np.py
# @brief:
# @author: Changjiang Cai, ccai1@stevens.edu, caicj5351@gmail.com
# @version: 0.0.1
# @creation date: 26-01-2020
# @last modified: Sun 26 Jan 2020 03:06:15 AM EST

# > see: http://jamesgregson.ca/bilateral-filtering-in-python.html
import numpy as np
import src.pfmutil as pfm
import sys
import math
import torch
import torch.nn as nn


#NOTE: This file is not really used in the project. It is just used to verify the pytorch version!!!
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

""" im2col operation """
#> see https://stackoverflow.com/questions/30109068/implement-matlabs-im2col-sliding-in-python;
from skimage.util import view_as_windows
def im2col(x, k = 3):
    """ args:
            x: input, in shape [H,W,C]
            k: kernel size, k x k
        return:
            y: in shape [H,W,k*k*C]
    """
    H,W,C = x.shape[:]
    pad_h = k/2
    pad_w = k/2
    """ padding """
    x = np.pad(x, ((pad_h, pad_h), (pad_w, pad_w), (0,0)), 
            mode = 'constant',
            #mode= 'edge'
            )

    window_shape = (k, k, C)
    x = np.squeeze(view_as_windows(x, window_shape, step = 1)) #in shape (H, W, k, k, C)
    x = np.reshape(x, [H, W, k*k, C]) #in shape (H, W, k*k, C)
    return x

def get_gaussian_filter_width_from_sigma(sigma_s):
    return int(3*sigma_s  + 1)

def filter_bilateral_with_embeding_im2col(
        x_in, embed_in, sigma_s = 10.0, sigma_v = 0.1, 
        reg_constant=1e-20, 
        hard_lambda = 1.0):
    """ using im2col to change the filtering to matrix multiplication

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
    #win_width = int( 3*sigma_s+1 )
    win_width = get_gaussian_filter_width_from_sigma(sigma_s)

    print ('win_size = {} x {}'.format(2*win_width+1, 2*win_width + 1))

    # initialize the results and sum of weights to very small values for
    # numerical stability. not strictly necessary but helpful to avoid
    # wild values with pathological choices of parameters
    wgt_sum = np.ones([H,W, 1])*reg_constant
    #print ("showing initial wgt_sum ")

    L2_dist_suqre = lambda x,y: ( np.sum((x-y)**2, axis = -1))
    L2_dist_suqre_keepdim = lambda x,y: ( np.sum((x-y)**2, axis = -1, keepdims=True))
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
    
    """ apply im2col for embedding masking """
    # space gaussian filter, 1-D vector
    sg_filter = np.reshape(sw_filter, [1, 1, -1, 1]) # in shape (H, W, k*k,1)
    print ('sg_filter shape = ', sg_filter.shape)
    
    # range gaussian filter, in shape H x W x k x k
    # e.g., F = 64 = embedding_channel_number
    embed_im2col = im2col(embed_in, k = 2*win_width + 1) # in shape [H, W, k*k,F]
    embed_tile = np.reshape(embed_in, [H,W, 1, F]) # in shape [H,W,1, F]
    # broadcasting along dim=2
    # in shape [H,W,k*k, 1]
    rg_filter = gaussian(
            L2_dist_suqre_keepdim(embed_im2col, embed_tile), 
            sigma_v, hard_lambda)
    
    # e.g., C = ndisp = 256; 
    x_in_im2col = im2col(x_in, k = 2*win_width + 1) # in shape [H,W,k*k,C]
    
    """
    emplement eq (4) in W. Harley' paper 
    "Segmentation-Aware Convolutional Networks Using Local Attention Masks (ICCV'17)"
    """
    #unrolled for element-wise multiplication:
    #(H x W x k*k x 1)*(H x W x k*k x 1) =>(H x W x k*k x 1)
    wgt_filter = rg_filter * sg_filter
    #( H x W x k*k x C) * (H x W x k*k x 1) ==> (H x W x k*k x C)
    result = x_in_im2col * wgt_filter 
    result = np.sum(result, axis = 2) # H x W x C
    wgt_sum += np.sum(wgt_filter, axis=2)# H x W x 1
    # normalize the result and return
    result = result / wgt_sum
    
    if 0:
        print ('rg_filter shape = ', rg_filter.shape)
        for i in range(0, rg_filter.shape[2]):
            shft_x = i % (2*win_width+1) - win_width
            shft_y = i / (2*win_width+1) - win_width
            print("show shift [%d][%d] " %( shft_y, shft_x))
            pfm.show(rg_filter[:,:,i,0])
    
    return result, np.squeeze(wgt_sum)





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
