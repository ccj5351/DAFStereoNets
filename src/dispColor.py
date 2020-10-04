# !/usr/bin/env python3
# -*-coding:utf-8-*-
# @file:
# @brief:
# @author: Changjiang Cai, ccai1@stevens.edu, caicj5351@gmail.com
# @version: 0.0.1
# @creation date: 08-01-2020
# @last modified: Mon 03 Feb 2020 08:48:02 PM EST

import cv2
import numpy as np
#from src.utils import writeKT15FalseColors # this is numpy fuction, it is SO SLOW !!!
# this is cython fuction, it is SO QUICK !!!
from src.cython import writeKT15FalseColor as KT15FalseClr
from src.cython import writeKT15ErrorLogColor as KT15LogClr

def colormap_jet(img):
    return cv2.cvtColor(cv2.applyColorMap(np.uint8(img), 2), cv2.COLOR_BGR2RGB)

def colormap_jet_batch_image(batch_inp, isGray = False):
    """
        args:
            batch_inp: in shape [N,C,H,W]
    """
    """Convert a Tensor to numpy image."""
    #print ("[??]batch_inp shape = ", batch_inp.shape)
    if not isinstance(batch_inp, np.ndarray ):
        batch_inp = batch_inp.cpu().numpy().transpose((0,2,3,1))# change to [N,H,W,C]
    else:
        batch_inp = batch_inp.transpose((0,2,3,1))# change to [N,H,W,C]
    tmp = []
    if not isGray:
        for i in range(batch_inp.shape[0]):
            tmp.append(colormap_jet(batch_inp[i])[:,:,::-1])
        return np.stack(tmp, axis = 0)
    else:
        for i in range(batch_inp.shape[0]):
            tmp.append(batch_inp[i])
        return np.stack(tmp, axis = 0)


def KT15FalseColorDisp(batch_inp, max_disp = 256):
    """
        args:
        batch_inp: in shape [N,C,H,W]
    """
    """Convert a Tensor to numpy image."""
    #print ("[??]batch_inp shape = ", batch_inp.shape)
    #batch_inp = batch_inp.cpu().numpy().transpose((0,2,3,1))# change to [N,H,W,C]
    batch_inp = batch_inp.cpu().numpy() # in shape [N,C=1,H,W]
    tmp = []
    for i in range(batch_inp.shape[0]):
        tmp.append(KT15FalseClr.writeKT15FalseColor(
            np.squeeze(batch_inp[i],0), 
            max_disp)[:,:,::-1] # RGB to BGR for tensorboard image write
            )
    return np.stack(tmp, axis = 0)

def KT15LogColorDispErr(batch_disp_gt, batch_disp):
    """
        args:
        batch_disp: in shape [N,C,H,W], disparity prediction;
        batch_disp_gt: in shape [N,C,H,W], disparity ground truth;
    """
    """Convert a Tensor to numpy image."""
    #batch_disp = batch_disp.cpu().numpy().transpose((0,2,3,1))# change to [N,H,W,C]
    #batch_disp_gt = batch_disp_gt.cpu().numpy().transpose((0,2,3,1))# change to [N,H,W,C]
    batch_disp = batch_disp.cpu().numpy() # in [N,C=1,H,W]
    #print ('[???] batch_disp shape = ', batch_disp.shape)
    batch_disp_gt = batch_disp_gt.cpu().numpy()#in [N,C=1,H,W]
    tmp = []
    for i in range(batch_disp.shape[0]):
        tmp.append(
            KT15LogClr.writeKT15ErrorDispLogColor(
                np.squeeze(batch_disp[i],0), 
                np.squeeze(batch_disp_gt[i],0))[:,:,::-1]# RGB to BGR for tensorboard image write
            )
    return np.stack(tmp, axis = 0)
