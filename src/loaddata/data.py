""" the code are adapted from GANet CVPR2019 Paper code:
    > see the code at: https://github.com/feihuzhang/GANet/tree/ab6782aff8b21cf2a70f3f839fc4030d24b29a1c/dataloader
"""
from .dataset import DatasetFromList, normalize_rgb_via_mean_std, Normalize
from PIL import Image
import numpy as np
import src.pfmutil as pfm


def get_training_set(data_path, 
        train_list, 
        crop_size=[256,256], 
        #left_right=False, 
        kitti2012=False, 
        kitti2015=False,
        virtual_kitti2 = False, 
        shift=0,
        is_semantic=True,
        #is_kt12_gray = True
        kt12_image_mode = 'rgb',
        is_data_augment = False
        ):
    return DatasetFromList(data_path, train_list,
            crop_size, True, 
            kitti2012, kitti2015, virtual_kitti2,
            shift, is_semantic, 
            #is_kt12_gray,
            kt12_image_mode,
            is_data_augment
            )


#def get_valid_set(data_path,
#        test_list, 
#        crop_size=[256,256], 
#        #left_right=False, 
#        kitti2012=False, 
#        kitti2015=False
#        ):
#    return DatasetFromList(data_path, test_list, 
#            crop_size, False, 
#            kitti2012, kitti2015)
__imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                    'std': [0.229, 0.224, 0.225]}

def load_test_data(leftname, rightname, is_data_augment = False ):
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

    #NOTE:
    # if is_data_augment == True ===> Set is_own_mean_std = False;
    # if is_data_augment == False ===> Set is_own_mean_std = True;
    if not is_data_augment: # using its own mean and std for normalization;
        #left 
        r = left[:, :, 0]
        g = left[:, :, 1]
        b = left[:, :, 2]
        temp_data[0, :, :] = (r - np.mean(r[:])) / np.std(r[:])
        temp_data[1, :, :] = (g - np.mean(g[:])) / np.std(g[:])
        temp_data[2, :, :] = (b - np.mean(b[:])) / np.std(b[:])
        #right
        r = right[:, :, 0]
        g = right[:, :, 1]
        b = right[:, :, 2]
        temp_data[3, :, :] = (r - np.mean(r[:])) / np.std(r[:])
        temp_data[4, :, :] = (g - np.mean(g[:])) / np.std(g[:])
        temp_data[5, :, :] = (b - np.mean(b[:])) / np.std(b[:])
    else: # using ImageNet mean and std for normalization;
        mean = __imagenet_stats["mean"]
        std = __imagenet_stats["std"]
        #left 
        r = left[:, :, 0]/255.0
        g = left[:, :, 1]/255.0
        b = left[:, :, 2]/255.0
        temp_data[0, :, :] = (r - mean[0]) / std[0]
        temp_data[1, :, :] = (g - mean[1]) / std[1]
        temp_data[2, :, :] = (b - mean[2]) / std[2]
        #right
        r = right[:, :, 0]/255.0
        g = right[:, :, 1]/255.0
        b = right[:, :, 2]/255.0
        temp_data[3, :, :] = (r - mean[0]) / std[0]
        temp_data[4, :, :] = (g - mean[1]) / std[1]
        temp_data[5, :, :] = (b - mean[2]) / std[2]
    """
    norm_left = np.zeros([height, width, 3], 'float32')
    norm_right = np.zeros([height, width, 3], 'float32')
    margin_tmp = np.zeros([10, width, 3], 'float32')
    for i in range(0,3):
        norm_left[:,:,i] = temp_data[i, :, :]
    
    for i in range(0,3):
        norm_right[:,:,i] = temp_data[i+3, :, :]
    pfm.show(np.concatenate([norm_left, margin_tmp, norm_right], axis=0), title="normalized left right")
    """
    return temp_data

def test_transform(temp_data, crop_height, crop_width):
    import torch
    _, h, w=np.shape(temp_data)

    if h <= crop_height and w <= crop_width:
        temp = temp_data
        temp_data = np.zeros([6, crop_height, crop_width], 'float32')
        temp_data[:, crop_height - h: crop_height, crop_width - w: crop_width] = temp
        #temp_data[:, crop_height - h: crop_height, 0: w] = temp #top-right padding, was found worse than top-left padding pattern on embed+dispnetc!!
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