# !/usr/bin/env python3
# -*-coding:utf-8-*-
# @file: virtual_kitti2_labels.py
# @brief:
# @author: Changjiang Cai, ccai1@stevens.edu, caicj5351@gmail.com
# @version: 0.0.1
# @creation date: 15-05-2020
# @last modified: Fri 15 May 2020 02:39:59 AM EDT

import cv2
import numpy as np
import src.pfmutil as pfm
from os.path import join as pjoin
import os

#--------------------------------------------------------------------------------
# Definitions:
# # code is adopated from https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py;
#--------------------------------------------------------------------------------
from collections import namedtuple
# a label and all meta information
Label = namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class

    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).
                    # Do not modify these IDs, since exactly these IDs are expected by the
                    # evaluation server.

    'color'       , # The color of this label
    ] )


"""
#> see: https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-2/
ID: Category r g b
0: Undefined 0 0 0
1: Terrain 210 0 200
2: Sky 90 200 255
3: Tree 0 199 0
4: Vegetation 90 240 0
5: Building 140 140 140
6: Road 100 60 100
7: GuardRail 250 100 255
8: TrafficSign 255 255 0
9: TrafficLight 200 200 0
10: Pole 255 130 0
11: Misc 80 80 80
12: Truck 160 60 60
13: Car 255 127 80
14: Van 0 139 139
"""
virtual_kitti_2_labels = [
    #       name                     id        color
    Label(  'undefined'            ,  0 ,      (  0,  0,  0)   ),
    Label(  'terrain'              ,  1 ,      (210,  0,  200) ),
    Label(  'sky'                  ,  2 ,      ( 90, 200, 255) ),
    Label(  'tree'                 ,  3 ,      (  0, 199, 0)   ),
    Label(  'vegetation'           ,  4 ,      ( 90, 240, 0)   ),
    Label(  'building'             ,  5 ,      (140, 140, 140) ),
    Label(  'road'                 ,  6 ,      (100, 60,  100) ),
    Label(  'guard rail'           ,  7 ,      (250, 100, 255) ),
    Label(  'traffic sign'         ,  8 ,      (255, 255, 0)   ),
    Label(  'traffic light'        ,  9 ,      (200, 200, 0)   ),
    Label(  'pole'                 , 10 ,      (255, 130, 0)   ),
    Label(  'misc'                 , 11 ,      (80,  80, 80)   ),
    Label(  'truck'                , 12 ,      (160, 60, 60)   ),
    Label(  'car'                  , 13 ,      (255, 127, 80)  ),
    Label(  'van'                  , 14 ,      (0, 139, 139)   ),
]


def get_virtual_kitti2_labels():
    # semantic segmentation labels
    
    """
    Returns:
        np.ndarray with dimensions (34, 3)
    """
    n_classes = 15
    virtual_kt2_clr_labels = np.zeros((n_classes, 3), np.uint8)

    for label in virtual_kitti_2_labels:
        if label.id >= 0:
            virtual_kt2_clr_labels[label.id] = label.color

    return virtual_kt2_clr_labels

def encode_color_segmap(mask):
    """Encode segmentation label images as virtual kitti classes

    Args:
        mask (np.ndarray): raw segmentation label image of dimension
          (M, N, 3), in which the virtual KITTI2 classes are encoded as colors.

    Returns:
        (np.ndarray): class map with dimensions (M,N), where the value at
        a given location is the integer denoting the class index.
    """
    mask = mask.astype(int)
    label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
    for ii, label in enumerate(get_virtual_kitti2_labels()):
        label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
    label_mask = label_mask.astype(int)
    return label_mask



# e.g., current_file = "Scene01/15-deg-left/frames/rgb/Camera_0/rgb_00001.jpg"
# e.g., file_path = "/media/ccjData2/datasets/Virtual-KITTI-V2/"
if __name__ == "__main__":
    
    file_path = "/media/ccjData2/datasets/Virtual-KITTI-V2/"
    current_file = "Scene01/15-deg-left/frames/rgb/Camera_0/rgb_00001.jpg"
    A = current_file 
    # e.g., /media/ccjData2/datasets/Virtual-KITTI-V2/vkitti_2.0.3_rgb/Scene01/15-deg-left/frames/rgb/Camera_0/rgb_00001.jpg
    imglname = pjoin(file_path, "vkitti_2.0.3_rgb/" + A) 
    print ("imgl = ", imglname)
    
    """ load current file from the list"""
    left = cv2.imread(imglname)[:,:,::-1].astype(np.float32) # change BRG to RGB via ``::-1`;
    print ('[???] left image shape = ', left.shape)
    #pfm.show_uint8(left)
    imgrname = pjoin(file_path, "vkitti_2.0.3_rgb/" + A[:-22] + 'Camera_1/' + A[-13:])
    print ("imgr = ", imgrname)
    right = cv2.imread(imgrname)[:,:,::-1].astype(np.float32) # change BRG to RGB via ``::-1`;
    #pfm.show_uint8(right)
    print ('[???] right image shape = ', right.shape)
    height, width = left.shape[:2]
    
    depth_png_filename = pjoin(file_path, "vkitti_2.0.3_depth/" + A[:-26] + 'depth/Camera_0/depth_' + A[-9:-4] + ".png")
    print ("depth_l = ", depth_png_filename)
    #NOTE: The depth map in centimeters can be directly loaded
    depth_left = cv2.imread(depth_png_filename, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

    # offset(i.e., distance between stereo views): B = 0.532725 m = 53.2725 cm
    B = 53.2725 # in centimeters;
    f = 725.0087 # in pixels
    # set zero as a invalid disparity value;
    disp_left = np.zeros([height, width], 'float32')
    mask = depth_left > 0
    disp_left[mask] = f*B/ depth_left[mask] # d = fB/z
    pfm.show(disp_left, title="disp_left")
    disp_pfm_filename = pjoin(file_path, "vkitti_2.0.3_disparity/" + A[:-26] + 'disparity/Camera_0/disparity_' + A[-9:-4] + ".pfm")
    print ("disp_pfm_filename = ", disp_pfm_filename)
    tmp_len = disp_pfm_filename.rfind("/")
    tmp_dir = disp_pfm_filename[0:tmp_len]
    print ("tmp_dir = ", tmp_dir)
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    pfm.save(disp_pfm_filename, disp_left)

    #loadding semantic segmantation labels
    seg_filename = pjoin(file_path, "vkitti_2.0.3_classSegmentation/" + A[:-26] + 'classSegmentation/Camera_0/classgt_' + A[-9:-4] + ".png")
    print ("semantic label: {}".format(seg_filename))
    semantic_rgb_label = cv2.imread(seg_filename)[:,:,::-1] # change BRG to RGB via ``::-1`;
    pfm.show_uint8(semantic_rgb_label, title="semantic_rgb_label")
    semantic_label = encode_color_segmap(semantic_rgb_label)
    pfm.show(semantic_label, title="semantic_label")