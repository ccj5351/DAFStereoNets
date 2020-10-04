""" the code are adapted from GANet CVPR2019 Paper code:
    > see the code at: https://github.com/feihuzhang/GANet/tree/ab6782aff8b21cf2a70f3f839fc4030d24b29a1c/dataloader
"""
from .dataset import DatasetFromList
from PIL import Image
import numpy as np

def get_training_set(data_path, 
        train_list, 
        crop_size=[256,256], 
        kitti2012=False, 
        kitti2015=False, 
        shift=0,
        is_semantic=True,
        is_kt12_gray = True
        ):
    return DatasetFromList(data_path, train_list,
            crop_size, True, 
            kitti2012, kitti2015, shift, is_semantic, is_kt12_gray)


def get_valid_set(data_path,
        test_list, 
        crop_size=[256,256], 
        kitti2012=False, 
        kitti2015=False
        ):
    return DatasetFromList(data_path, test_list, 
            crop_size, False, 
            kitti2012, kitti2015)

def load_test_data(leftname, rightname):
    left = Image.open(leftname)
    right = Image.open(rightname)
    size = np.shape(left)
    height = size[0]
    width = size[1]
    left = np.asarray(left)
    right = np.asarray(right)
    
    #loading gray images
    if left.ndim == 2 or (left.ndim == 3 and left.shape[2] != 3):
        left = np.stack([left, left, left], axis=2)
        right = np.stack([right, right, right], axis=2)
        print ("left shape: ", left.shape, ", right shape: ", right.shape)
    
    temp_data = np.zeros([6, height, width], 'float32')
    r = left[:, :, 0]
    g = left[:, :, 1]
    b = left[:, :, 2]
    temp_data[0, :, :] = (r - np.mean(r[:])) / np.std(r[:])
    temp_data[1, :, :] = (g - np.mean(g[:])) / np.std(g[:])
    temp_data[2, :, :] = (b - np.mean(b[:])) / np.std(b[:])
    r = right[:, :, 0]
    g = right[:, :, 1]
    b = right[:, :, 2]	
    #r,g,b,_ = right.split()
    temp_data[3, :, :] = (r - np.mean(r[:])) / np.std(r[:])
    temp_data[4, :, :] = (g - np.mean(g[:])) / np.std(g[:])
    temp_data[5, :, :] = (b - np.mean(b[:])) / np.std(b[:])
    return temp_data

def test_transform(temp_data, crop_height, crop_width):
    import torch
    _, h, w=np.shape(temp_data)

    if h <= crop_height and w <= crop_width:
        temp = temp_data
        temp_data = np.zeros([6, crop_height, crop_width], 'float32')
        temp_data[:, crop_height - h: crop_height, crop_width - w: crop_width] = temp
    elif h > crop_height and w > crop_width:
        start_x = max((w - crop_width) // 2, 0)
        start_y = max((h - crop_height)// 2, 0)
        temp_data = temp_data[:, start_y: start_y + crop_height, start_x: start_x + crop_width]
    else:
        raise ValueError("crop size is not correct!!")

    
    left = np.ones([1, 3,crop_height,crop_width],'float32')
    left[0, :, :, :] = temp_data[0: 3, :, :]
    right = np.ones([1, 3, crop_height, crop_width], 'float32')
    right[0, :, :, :] = temp_data[3: 6, :, :]
    return torch.from_numpy(left).float(), torch.from_numpy(right).float(), h, w
