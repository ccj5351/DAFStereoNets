import torch.utils.data as data

from PIL import Image
import os
import os.path
import numpy as np

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def is_file_list(file_list):
    return file_list is not None and file_list != ''

def dataloader(filepath, train_file_list = None, val_file_list = None):

    #left_fold = 'image_2/'
    #right_fold = 'image_3/'
    #disp_L = 'disp_occ_0/'
    #disp_R = 'disp_occ_1/'
    # updated by CCJ: for dataloading
    left_fold = 'image_0/'
    right_fold = 'image_1/'
    disp_L = 'disp_occ_0/'
    disp_R = 'disp_occ_1/'
    

    if is_file_list(train_file_list) and is_file_list(val_file_list):
        f = open(train_file_list, 'r')
        train = [l.rstrip() for l in f.readlines()]
        imgN = len(train)
        print ("[**]img# = {}, train_list={}, ..., {}".format(imgN,train[0],train[imgN -1]))
        
        f = open(val_file_list, 'r')
        val = [l.rstrip() for l in f.readlines()]
        imgN = len(val)
        print ("[**]img# = {}, val_list={}, ..., {}".format(imgN,val[0],val[imgN-1]))
    else:
        image = [img for img in os.listdir(filepath+left_fold) if img.find('_10') > -1]
        train = image[:160]
        val = image[160:]

    left_train = [filepath+left_fold+img for img in train]
    right_train = [filepath+right_fold+img for img in train]
    disp_train_L = [filepath+disp_L+img for img in train]
    #disp_train_R = [filepath+disp_R+img for img in train]

    left_val = [filepath+left_fold+img for img in val]
    right_val = [filepath+right_fold+img for img in val]
    disp_val_L = [filepath+disp_L+img for img in val]
    #disp_val_R = [filepath+disp_R+img for img in val]

    return left_train, right_train, disp_train_L, left_val, right_val, disp_val_L
