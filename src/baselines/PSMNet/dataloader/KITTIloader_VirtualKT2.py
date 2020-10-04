import torch.utils.data as data

from PIL import Image
import os
import os.path
import numpy as np
from os.path import join as pjoin

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def is_file_list(file_list):
    return file_list is not None and file_list != ''

def dataloader(filepath, train_file_list = None, val_file_list = None):
    assert is_file_list(train_file_list)
    train_list = get_virtual_kitti2_filelist(file_list_txt=train_file_list)
    imgN = len(train_list)
    print ("[**]img# = {}, train_list={}, ..., {}".format(imgN,train_list[0],train_list[imgN -1]))
    #upzipping the value using zip(*):
    left_train, right_train, disp_train_L = zip(*[load_virtual_kitti2_data(filepath, current_file) for current_file in train_list])
    
    if is_file_list(val_file_list):
        val_list = get_virtual_kitti2_filelist(file_list_txt = val_file_list)
        imgN = len(val_list)
        print ("[**]img# = {}, val_list={}, ..., {}".format(imgN,val_list[0],val_list[imgN-1]))
        left_val, right_val, disp_val_L = zip(*[load_virtual_kitti2_data(filepath, current_file) for current_file in val_list])
        return left_train, right_train, disp_train_L, left_val, right_val, disp_val_L
    else:
        return left_train, right_train, disp_train_L


def dataloader_eval(filepath, val_file_list):
    assert is_file_list(val_file_list)
    val_list = get_virtual_kitti2_filelist(file_list_txt = val_file_list)
    imgN = len(val_list)
    print ("[**]img# = {}, val_list={}, ..., {}".format(imgN, val_list[0],val_list[imgN-1]))
    left_val, right_val, disp_val_L = zip(*[load_virtual_kitti2_data(filepath, current_file) for current_file in val_list])
    return left_val, right_val, disp_val_L


"""
# this function is added by CCJ;
# this function is also used in src/loaddata/dataset.py 
"""
def get_virtual_kitti2_filelist(file_list_txt = './lists/virtual_kitti2_train.list'):
    """
    #> see: https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-2/
    - format: 
            SceneX/Y/frames/rgb/Camera_Z/rgb_%05d.jpg
    - where 
       - X ∈ {01, 02, 06, 18, 20} and represent one of 5 different locations.
       - Y ∈ {15-deg-left, 15-deg-right, 30-deg-left, 30-deg-right, clone, fog, morning, overcast, rain, sunset} and represent the different variations.
       - Z ∈ [0, 1] and represent the left (same as in virtual kitti) or right camera (offset by 0.532725m to the right). 
    - Note that our indexes always start from 0.
    """
    #Xs = ['01', '02', '06', '18', '20']
    Ys = ['15-deg-left', '15-deg-right', '30-deg-left', 
          '30-deg-right', 'clone', 'fog', 'morning', 
          'overcast', 'rain', 'sunset']
    f = open(file_list_txt, 'r')
    file_list = [l.rstrip() for l in f.readlines() if not l.rstrip().startswith('#')]
    file_list_new =  []
    for name in file_list:
        for Y in Ys:
            #e.g., name = 'Scene01/clone/frames/rgb/Camera_0/rgb_00138.jpg'
            # replace 'clone' with Y
            name_new = name[:7] + '/' + Y + name[13:]
            file_list_new.append(name_new)
    print ("[***] Virtual KITTI 2: %s has %d images." %(file_list_txt, len(file_list_new)))
    return file_list_new


"""
# this function is added by CCJ;
"""
def load_virtual_kitti2_data(file_path, current_file):
    # e.g., current_file = "Scene01/15-deg-left/frames/rgb/Camera_0/rgb_00001.jpg"
    # e.g., file_path = "/media/ccjData2/datasets/Virtual-KITTI-V2/"
    A = current_file 
    # e.g., /media/ccjData2/datasets/Virtual-KITTI-V2/vkitti_2.0.3_rgb/Scene01/15-deg-left/frames/rgb/Camera_0/rgb_00001.jpg
    imglname = pjoin(file_path, "vkitti_2.0.3_rgb/" + A) 
    imgrname = pjoin(file_path, "vkitti_2.0.3_rgb/" + A[:-22] + 'Camera_1/' + A[-13:])
    depth_png_filename = pjoin(file_path, "vkitti_2.0.3_depth/" + A[:-26] + 'depth/Camera_0/depth_' + A[-9:-4] + ".png")
    return imglname, imgrname, depth_png_filename