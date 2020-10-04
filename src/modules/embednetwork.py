# !/usr/bin/env python3
# -*-coding:utf-8-*-
# @file: embednetwork.py
# @brief:
# @author: Changjiang Cai, ccai1@stevens.edu, caicj5351@gmail.com
# @version: 0.0.1
# @creation date: 19-02-2019
# @last modified: Sat 23 Nov 2019 10:00:35 PM EST
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

#from src.commons import *
#from src import common_flags as common

""" How to keep the shape of input and output same when dilation conv?
# see https://discuss.pytorch.org/t/how-to-keep-the-shape-of-input-and-output-same-when-dilation-conv/14338
Given:
 o = output
 p = padding
 k = kernel_size
 s = stride
 d = dilation
To get output:
 o = [i + 2*p - k - (k-1)*(d-1)]/s + 1
If we want the "same" size, and with s = 1, k = 3 for our project here,
we can get:
==> o = i = i + 2*p - 3 - 2*(d-1) + 1 
==> p = d 
"""
def im2col(x, k = 3, r = 1, is5D = True):
    """
    args:
       x : input, [N, C, H, W] (Note: in PyTorch is [N C H W], while in TensorFlow is [N H W C];
       k : filter size, corresponding to window in size [k, k]
       r: equivalent to rate in dilated (a.k.a. Atrous) convolutions;
    Note: 
        we let r = p due to the SAME size of output and input;
    """
    #troch.nn.Unfold():Extracts sliding local blocks from a batched input tensor
    N,C,H,W = x.size()[:]
    unfold = nn.Unfold(kernel_size= k, dilation = r, padding = r, stride = 1)
    #NOTE: updated on 2019/10/02: (k*k)*C to C*(k*k), due to (k*k) is continugous chunk; 
    #NOTE: that means, C*(k*k) can be reshape to [C, k*k] for correctly accessing the array elements;
    #output = unfold(x) # 3D tensor in size [N, (k*k)*C, L], here L = H*W
    output = unfold(x) # 3D tensor in size [N, C*(k*k), L], here L = H*W
    # reshape to 4D tensor: [N, C*(k*k), H, W]
    #NOTE: can also be reshape to 5D tensor [N,C,k*k, H,W] if you want!!!
    if is5D:
        return output.view(N, C, k*k, H, W)
    else:
        return output.view(N, C*(k*k), H, W)

def im2dist(x, k = 3, r = 1, isL2Dist = False):
    """
    args:
       x : input, [N, C, H, W] (Note: in PyTorch is [N C H W], while in TensorFlow is [N H W C];
       k : filter size, corresponding to window in size [k, k]
       r: equivalent to rate in dilated (a.k.a. Atrous) convolutions;
       isL2Dist: True or False, if True then return the L2 distance, L1 otherwise;
    return:
       dist: in shape [N, k*k, H, W]
    """

    N, C, H, W = x.size()[:]
    """ C-chunk is continuous in memory """
    # in shape [N, H, W, C*(k*k)], i.e, (x^1_1,...,x^1_c), (...), (x^{k*k}_1, ..., x^{k*k}_c);
    im2col_patch = im2col(x, k, r) # in size (N, C, k*k, H, W)
    #print ('im2col_patch shape = ', im2col_patch.shape)

    """ tile: broadcasting involved """
    #im2col_tile = x.repeat(1,1,k*k,1).view(N,C,k*k, H, W)
    im2col_tile = x.view(N,C,1,H,W)
    #print ('im2col_tile shape = ', im2col_tile.shape)
    
    im2col_patch -= im2col_tile
    if not isL2Dist:
        dist = torch.sum(im2col_patch.abs_(), dim = 1)
        #print ("l1_dist.shape = ", dist.shape)
    else:
        dist = torch.sum(im2col_patch.pow_(2), dim = 1)
        #print ("l2_dist.shape = ", dist.shape)
    return dist # in shape [N, k*k,H,W]

def im2parity(labels, k = 3, r = 1):
    """
    args:
       labels : segmentation labels, [N, C=1, H, W] (Note: in PyTorch is [N C H W], while in TensorFlow is [N H W C];
       k : filter size, corresponding to window in size [k, k]
       r: equivalent to rate in dilated (a.k.a. Atrous) convolutions;
    """
    _, C, _, _ = labels.size()[:]
    assert (C == 1) 
    #print ("[****] im2parity: N = {}, C={}, H = {}, W = {}".format(N, C,H,W))
    im2col_patch = im2col(labels, k, r, is5D=False) # in shape [N, k*k, H, W]
    """ torch supports broadcasting 
        > see: https://pytorch.org/docs/stable/notes/broadcasting.html#broadcasting-semantics
    """
    im2col_tile =  labels # already in shape [N, 1, H, W]
    # Computes element-wise equality
    parity = torch.eq(im2col_patch, im2col_tile) # in shape [N,k*k, H, W]
    #print ("[***] parity.shape = ", parity.shape)
    return parity.float()


def dist_loss(parity, distance, labels, ignore_label = 255, k = 3, alpha = 0.5, beta = 2.0):
    """
    To implement this equation (1):
               |- max(||e_i - e_j || - alpha , 0), if I_i = I_j (i.e., has same label)
     l_{i,j} = |
               |- max(beta - ||e_i - e_j||, 0), o.w. 
    
    args:
       parity: [N, k*k, H, W], indicate whether I_i == I_j or not;
       distance: [N, k*k, H, W], indicaate the distance of embedding e_i and e_j; 
       labels : segmentation labels, [N, C=1, H, W] (Note: in PyTorch is [N C H W], while in TensorFlow is [N H W C];

       k : filter size, corresponding to window in size [k, k]
       alpha: hyperparametrs in equ(1);
       beta: hyperparametrs in equ(1); alpha <= beta
    """

    label2col_patch = im2col(labels, k, r = 1, is5D = False) # in shape [N, k*k, H, W]
    
    #with torch.no_grad():
    with torch.set_grad_enabled(False):
        ignore = 1.0 - torch.eq(label2col_patch, ignore_label).float() # in shape [N, k*k, H, W]
    
    # Or a simple way to implement ReLU(x) = max(x, 0):
    #  ReLU(x) = 0.5*(|x| + x)
    
    dist_same_label = ignore*F.relu_(distance-alpha) # in shape [N,k*k, H,W]
    dist_diff_label = ignore*F.relu_(beta-distance) # in shape [N,k*k, H,W]
    # to computer : loss = parity * dist_same + (1- parity)*dist_diff, with parity has 0, 1 elements;
    loss = parity*(dist_same_label)+(1.0-parity)*dist_diff_label
    loss = torch.sum(loss, axis=1) # in shape [N,H,W]
    #print ("loss = ", loss)
    loss_avg = torch.mean(loss)
    
    return loss_avg

def get_embed_mask(x_embed, k = 3, r= 1, hardness_lambda = 0.1, isL2Dist = False ):
    """
    args:
       x_embed: embedding feature, [N, C, H, W], e.g., C = F = 64
       args_dict : parameters 
    """

    #print ('Calculating mask:')
    dist = im2dist(x_embed, k = k, r = 1, isL2Dist = isL2Dist) #[N, k*k,H,W]
    mask = torch.exp_(-hardness_lambda*dist) # [N,k*k,H,W]
    mask_sum = torch.sum(mask, dim=1, keepdim= True) #[N,1,H,W]
    return mask_sum # [N,1,H,W]



##############################
""" embedding network loss """ 
##############################
def get_embed_losses(x_embed, labels, args_dict = None):
    """
    args:
       x_embed: embedding feature, [N, C, H, W], e.g., C = F = 64
       labels : segmentation labels, [N, C=1, H, W] (Note: in PyTorch is [N C H W], while in TensorFlow is [N H W C];
       args_dict : parameters 
    """
    if args_dict is None:
        args_dict = {
                'alpha': 0.5, 
                'beta' : 2.0, 
                'norm' :"L1",
                'r1_weight'  : 1.0, 
                'r2_weight'  : 1.0,
                'r5_weight'  : 1.0, 
                'ignore_label' : 255
                }

    #print ('Calculating Losses:')
    """ parameters """
    isL2Dist = args_dict['norm'] == 'L2'
    alpha = args_dict['alpha']
    beta = args_dict['beta']
    ignore_label = args_dict['ignore_label']
    r1_weight = args_dict['r1_weight']
    r2_weight = args_dict['r2_weight']
    r5_weight = args_dict['r5_weight']

    """ embedding features to distance """
    # apply in2dist to the embedding;
    dist_r1 = im2dist(x_embed, k = 3, r = 1, isL2Dist = isL2Dist)
    dist_r2 = im2dist(x_embed, k = 3, r = 2, isL2Dist = isL2Dist)
    dist_r5 = im2dist(x_embed, k = 3, r = 5, isL2Dist = isL2Dist)

    """ parity: at different scales, i.e., r = 1,2,5, 
        corresponding to the rate in dilated (a.k.a. Atrous) convolutions; 
    """
    #All new operations in the torch.set_grad_enabled(False) 
    #block won’t require gradients. 
    with torch.set_grad_enabled(False):
        parity_r1 = im2parity(labels, k = 3, r = 1)
        parity_r2 = im2parity(labels, k = 3, r = 2)
        parity_r5 = im2parity(labels, k = 3, r = 5)
    
    """ loss """ 
    loss_r1 = dist_loss(parity_r1, dist_r1, labels, ignore_label, k=3, alpha=alpha, beta=beta)
    loss_r2 = dist_loss(parity_r2, dist_r2, labels, ignore_label, k=3, alpha=alpha, beta=beta)
    loss_r5 = dist_loss(parity_r5, dist_r5, labels, ignore_label, k=3, alpha=alpha, beta=beta)
    #print ("loss_r1 = {}\nloss_r2 = {}\nloss_r5={}".format(loss_r1, loss_r2, loss_r5))
    
    loss = r1_weight * loss_r1 + r2_weight * loss_r2 + r5_weight*loss_r5
    #print ("loss = {}".format(loss))
    return loss, loss_r1, loss_r2, loss_r5


""" 
embedding network
"""
class embed_net(nn.Module):
    def __init__(self):
        super(embed_net, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        """ conv 1 """
        self.F1 = 64
        self.conv1_0 = nn.Conv2d(in_channels = 3, out_channels = self.F1, 
                               kernel_size = 3, stride=1, padding=1, dilation=1)
        self.conv1_1 = nn.Conv2d(in_channels = self.F1, out_channels = self.F1, 
                               kernel_size = 3, stride=1, padding=1, dilation=1)
        
        """ conv 2 """
        self.F2 = 128
        self.conv2_0 = nn.Conv2d(in_channels = self.F1, out_channels = self.F2, 
                               kernel_size = 3, stride=1, padding=1, dilation=1)
        self.conv2_1 = nn.Conv2d(in_channels = self.F2, out_channels = self.F2, 
                               kernel_size = 3, stride=1, padding=1, dilation=1)
        
        """ conv 3 """
        self.F3 = 256
        self.conv3_0 = nn.Conv2d(in_channels = self.F2, out_channels = self.F3, 
                               kernel_size = 3, stride=1, padding=1, dilation=1)
        self.conv3_1 = nn.Conv2d(in_channels = self.F3, out_channels = self.F3, 
                               kernel_size = 3, stride=1, padding=1, dilation=1)
        self.conv3_2 = nn.Conv2d(in_channels = self.F3, out_channels = self.F3, 
                               kernel_size = 3, stride=1, padding=1, dilation=1)
        
        self.max_pool = nn.MaxPool2d(kernel_size = 3, stride=2, padding=1, dilation= 1)
        
        """ fusion, aka, weighted avg """
        """ 1x1 conv for avg"""
        self.F4 = 64
        self.fused = []
        # channel of fused output
        self.fused_C = self.get_fused_channels(C_x = 3)
        #print ("[***] fused channels :", self.fused_C)
        self.avgConv = nn.Conv2d(in_channels = self.fused_C, out_channels = self.F4, 
                kernel_size = 1, stride=1, padding= 0, dilation=1)
    
    #####################################
    #####################################
    def get_fused_channels(self, C_x = 3):
        fused_C = C_x # append x

        """ conv 1 """
        fused_C += self.F1 # append self.F1
        fused_C += self.F1 # append self.F1
        
        """ conv 2 """
        fused_C += self.F2 # append self.F2
        fused_C += self.F2 # append self.F2

        """ conv 3 """
        fused_C += self.F3 # append self.F3
        
        """ fusion, aka, weighted avg """
        return fused_C
    
    #####################################
    #####################################
    def forward(self, x):
        """ reset self.fused to empty """ 
        self.fused = []
        """ make sure the inputs are normalized to [0, 1] """
        N, C, H, W = x.size()[:]
        
        self.fused.append(x)

        """ conv 1 """
        x = self.relu(self.conv1_0(x))
        self.fused.append(x)
        x = self.relu(self.conv1_1(x))
        self.fused.append(x)
        x = self.max_pool(x)
        #print ("[***]", x.shape)
        
        """ conv 2 """
        x = self.relu(self.conv2_0(x))
        self.fused.append(F.interpolate(x, size = [H,W], mode='bilinear',align_corners = True))
        x = self.relu(self.conv2_1(x))
        self.fused.append(F.interpolate(x, size = [H,W], mode='bilinear',align_corners = True))
        x = self.max_pool(x)

        """ conv 3 """
        x = self.relu(self.conv3_0(x))
        x = self.relu(self.conv3_1(x))
        x = self.relu(self.conv3_2(x))
        self.fused.append(F.interpolate(x, size = [H,W], mode='bilinear',align_corners = True))
        
        """ fusion, aka, weighted avg """
        x = torch.cat(self.fused, dim = 1) #[N, C, H, W]
        #print('[***] fusion shape = ', x.shape)
        """ the embedding features """
        out = self.avgConv(x)

        return out
