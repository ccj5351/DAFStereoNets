"""
# the code is adapted from GANet paper code:
# > see the code at: https://github.com/feihuzhang/GANet/tree/ab6782aff8b21cf2a70f3f839fc4030d24b29a1c/dataloader
"""
import torch
import torch.utils.data as data
#import skimage
#import skimage.io
#import skimage.transform

from PIL import Image
import numpy as np
import random
from struct import unpack
import re
import sys


from os.path import join as pjoin
import src.pfmutil as pfm
import cv2
from .virtual_kitti2_labels import encode_color_segmap
import random

def train_transform(temp_data, crop_height, crop_width, shift=0):
    _, h, w = np.shape(temp_data)
    
    if h > crop_height and w <= crop_width:
        temp = temp_data
        temp_data = np.zeros([8, h+shift, crop_width + shift], 'float32')
        temp_data[6:7,:,:] = 1000
        temp_data[:, shift: h + shift, crop_width + shift - w: crop_width + shift] = temp
        _, h, w = np.shape(temp_data)
   
    if h <= crop_height and w <= crop_width:
        temp = temp_data
        temp_data = np.zeros([8, crop_height + shift, crop_width + shift], 'float32')
        temp_data[6: 7, :, :] = 1000
        temp_data[:, crop_height + shift - h: crop_height + shift, crop_width + shift - w: crop_width + shift] = temp
        _, h, w = np.shape(temp_data)
    
    if shift > 0:
        start_x = random.randint(0, w - crop_width)
        shift_x = random.randint(-shift, shift)
        if shift_x + start_x < 0 or shift_x + start_x + crop_width > w:
            shift_x = 0
        start_y = random.randint(0, h - crop_height)
        left = temp_data[0: 3, start_y: start_y + crop_height, start_x + shift_x: start_x + shift_x + crop_width]
        right = temp_data[3: 6, start_y: start_y + crop_height, start_x: start_x + crop_width]
        target = temp_data[6: 7, start_y: start_y + crop_height, start_x + shift_x : start_x+shift_x + crop_width]
        target = target - shift_x
        return left, right, target
    
    if h <= crop_height and w <= crop_width:
        temp = temp_data
        temp_data = np.zeros([8, crop_height, crop_width], 'float32')
        temp_data[:, crop_height - h: crop_height, crop_width - w: crop_width] = temp
    
    else:
        start_x = random.randint(0, w - crop_width)
        #print ("[???]", w, crop_width, h, crop_height)
        start_y = random.randint(0, h - crop_height)
        temp_data = temp_data[:, start_y: start_y + crop_height, start_x: start_x + crop_width]
    
    left = temp_data[0: 3, :, :]
    right = temp_data[3: 6, :, :]
    target = temp_data[6: 7, :, :]
    left_rgb = temp_data[8:11,:,:]
    right_rgb = temp_data[11:14,:,:]
    semantic_label = temp_data[14:15,:,:] # if is_semantic == False, will be all zeros 
    
    # added by CCJ on 2020/05/23;
    left = torch.from_numpy(left)
    right = torch.from_numpy(right)
    target = torch.from_numpy(target)
    left_rgb = torch.from_numpy(left_rgb)
    right_rgb = torch.from_numpy(right_rgb)
    semantic_label = torch.from_numpy(semantic_label)

    return left, right, target, left_rgb, right_rgb, semantic_label

def train_transform_augmentation(temp_data, crop_height, crop_width, scale = [0.5, 1.15], method='nearest'):
    temp_data = temp_data.transpose((1,2,0))
    #print ("[???] temp_data shape = ", temp_data.shape)
    h, w = temp_data.shape[0:2]
    #print ("[???] h = %d, w = %d" %(h, w))

    temp_scale = scale[0] + (scale[1] - scale[0]) * random.random()
    new_crop_h = int(crop_height * temp_scale)
    new_crop_w = int(crop_width * temp_scale)
    assert new_crop_h < h and new_crop_w < w
    #print ("[???] crop_h = %d, crop_w = %d" %(crop_height, crop_width))
    #print ("[???] new crop_h = %d, new crop_w = %d" %(new_crop_h, new_crop_w))

    start_x = random.randint(0, w - new_crop_w)
    start_y = random.randint(0, h - new_crop_h)

    temp_data = temp_data[start_y: start_y + new_crop_h, start_x: start_x + new_crop_w, :]
    temp_out_data = np.zeros([crop_height, crop_width, 12], 'float32')
    
    # resize images;
    # 0:3 left image,  3:6 right image
    temp_out_data[:,:,0:6] = resize_images(temp_data[:,:,0:6], # in [H,W,C] format
                        crop_height, crop_width, method)
    # 8:11 left_rgb image,  11:14 right_rgb image for tensorboard visulization;
    temp_out_data[:,:,6:12] = resize_images(temp_data[:,:,8:14], # in [H,W,C] format
                        crop_height, crop_width, method)
    
    # resize disparity
    target = resize_disparity(temp_data[:,:,6], crop_height, crop_width, method)[None,...]
    semantic_label = resize_images(temp_data[:,:,14], crop_height, crop_width, "nearest")[None,...]
    assert target.shape[0] == 1 and semantic_label.shape[0] == 1
    
    if len(semantic_label.shape) == 2:
        semantic_label = semantic_label[None,:,:]
    
    # added by CCJ on 2020/05/23;
    left = torch.from_numpy(temp_out_data[:,:,0:3]) #[H, W, 3]
    right = torch.from_numpy(temp_out_data[:,:,3:6])
    target = torch.from_numpy(target) #[1, H, W]
    left_rgb = torch.from_numpy(temp_out_data[:,:,6:9].transpose(2,0,1)) #[H, W, 3]
    right_rgb = torch.from_numpy(temp_out_data[:,:,9:12].transpose(2,0,1)) #[H, W, 3]
    semantic_label = torch.from_numpy(semantic_label) # [1, H, W]
    
    my_color_trans = RandomPhotometric(
                    noise_stddev=0.0,
                    min_contrast=-0.3,
                    max_contrast=0.3,
                    brightness_stddev=0.02,
                    min_color=0.9,
                    max_color=1.1,
                    min_gamma=0.7,
                    max_gamma=1.5)

    # will change [H,W,3] to [3, H, W] 
    left, right = my_color_trans([left, right])
    my_normalizer = Normalize(**__imagenet_stats)
    left, right = my_normalizer([left, right])
    #print ("[???] ", left.shape, right.shape, target.shape, left_rgb.shape, right_rgb.shape, semantic_label.shape)
    return left, right, target, left_rgb, right_rgb, semantic_label


#added by CCJ:
# this code is adapted from PSMNet;

__imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                    'std': [0.229, 0.224, 0.225]}


# this code is adapted from HD^3;
class Normalize(object):
    """Given mean: (R, G, B) and std: (R, G, B),
    will normalize each channel of the torch.*Tensor, i.e.
    channel = (channel - mean) / std
    """

    def __init__(self, mean, std):
        self.mean = torch.FloatTensor(mean)
        self.std = torch.FloatTensor(std)

    def __call__(self, img_list):
        for img in img_list:
            for t, m, s in zip(img, self.mean, self.std):
                t.sub_(m).div_(s)
        return img_list

# this code is adapted from HD^3;
class RandomPhotometric(object):
    """Applies photometric augmentations to a list of image tensors.
    Each image in the list is augmented in the same way.

    Args:
        ims: list of 3-channel images normalized to [0, 1].

    Returns:
        normalized images with photometric augmentations. Has the same
        shape as the input.
    """

    def __init__(self,
                 noise_stddev=0.0,
                 min_contrast=0.0,
                 max_contrast=0.0,
                 brightness_stddev=0.0,
                 min_color=1.0,
                 max_color=1.0,
                 min_gamma=1.0,
                 max_gamma=1.0):
        self.noise_stddev = noise_stddev
        self.min_contrast = min_contrast
        self.max_contrast = max_contrast
        self.brightness_stddev = brightness_stddev
        self.min_color = min_color
        self.max_color = max_color
        self.min_gamma = min_gamma
        self.max_gamma = max_gamma

    def __call__(self, ims):
        """
         ims: list of images in size [H,W,3]; 
         return:
            list of images in size [3, H, W];
        """
        contrast = np.random.uniform(self.min_contrast, self.max_contrast)
        gamma = np.random.uniform(self.min_gamma, self.max_gamma)
        gamma_inv = 1.0 / gamma
        color = torch.from_numpy(
            np.random.uniform(self.min_color, self.max_color, (3))).float()
        if self.noise_stddev > 0.0:
            noise = np.random.normal(scale=self.noise_stddev)
        else:
            noise = 0
        if self.brightness_stddev > 0.0:
            brightness = np.random.normal(scale=self.brightness_stddev)
        else:
            brightness = 0

        out = []
        for im in ims:
            #im_re = im.permute(1, 2, 0)
            im_re = im
            im_re = (im_re * (contrast + 1.0) + brightness) * color
            im_re = torch.clamp(im_re, min=0.0, max=1.0)
            im_re = torch.pow(im_re, gamma_inv)
            im_re += noise

            im = im_re.permute(2, 0, 1)
            out.append(im)

        return out


def resize_images(
    imgs, #[H,W,C]
    des_height, des_width, method = "bilinear"):
    #assert len(imgs.shape) == 3
    src_height, src_width = imgs.shape[0:2]
    if src_width == des_width and src_height == des_height:
        return imgs
    if method == 'bilinear':
        out = cv2.resize(imgs, (des_width, des_height), interpolation=cv2.INTER_LINEAR)
    elif method == 'nearest':
        out = cv2.resize(imgs, (des_width, des_height), interpolation=cv2.INTER_NEAREST)
    else:
        raise Exception('Invalid resize images method!')
    return out


def resize_disparity(
    disp_label, #[H, W, 1]
    des_height, des_width, method = "bilinear"):
    assert len(disp_label.shape) == 2
    src_height = disp_label.shape[0]
    src_width = disp_label.shape[1]
    if src_width == des_width and src_height == des_height:
        return disp_label
    
    ratio_width = float(des_width) / float(src_width)
    if method == 'bilinear':
        out = cv2.resize(disp_label, (des_width, des_height), interpolation=cv2.INTER_LINEAR)
    elif method == 'nearest':
        out = cv2.resize(disp_label, (des_width, des_height), interpolation=cv2.INTER_NEAREST)
    else:
        raise Exception('Invalid resize disparity method!')
    out = out*ratio_width # remember to adjust the disparity value;
    #if len(out.shape) == 2:
    #    out = out[:,:,None]
    #print ("[???] out shape = ", out.shape)
    return out


def valid_transform(temp_data, crop_height, crop_width):
    _, h, w = np.shape(temp_data)
    if h <= crop_height and w <= crop_width:
        temp = temp_data
        temp_data = np.zeros([8,crop_height,crop_width], 'float32')
        temp_data[6: 7, :, :] = 1000
        temp_data[:, crop_height - h: crop_height, crop_width - w: crop_width] = temp
    else:
        start_x = (w-crop_width)/2
        start_y = (h-crop_height)/2
        temp_data = temp_data[:, start_y: start_y + crop_height, start_x: start_x + crop_width]
   
    left = temp_data[0: 3, :, :]
    right = temp_data[3: 6, :, :]
    target = temp_data[6: 7, :, :]
  #  sign=np.ones([1,1,1],'float32')*-1
    # added by CCJ on 2020/05/23;
    left = torch.from_numpy(left)
    right = torch.from_numpy(right)
    target = torch.from_numpy(target)
    return left, right, target

""" added by CCJ """
def normalize_rgb_via_mean_std(img, is_mean_std = True):
    assert (np.shape(img)[2] == 3)
    if is_mean_std:
        # left : R G B
        r = img[:, :, 0]
        g = img[:, :, 1]
        b = img[:, :, 2]

        r = (r-np.mean(r)) / np.std(r)
        g = (g-np.mean(g)) / np.std(g)
        b = (b-np.mean(b)) / np.std(b)
    else:
        r = img[:, :, 0]/255.0
        g = img[:, :, 1]/255.0
        b = img[:, :, 2]/255.0
    
    return r,g,b

""" added by CCJ """
def normalize_gray_via_mean_std(img, is_mean_std = True):
    assert img.ndim == 2
    if is_mean_std:
        return (img - np.mean(img)) / np.std(img)
    else:
        return img/255.0

# load sf (scene flow) data
def load_sfdata(data_path, current_file, self_guassian_normalize = True):
    A = current_file
    #filename = pjoin(data_path, 'frames_finalpass', A)
    filename = pjoin(data_path, A)
    #print ("[****] limg: {}".format(filename))
    """ if using Pillow Image, the loaded image will have RGBA for PNG in Scene Flow dataset """
    #left  = Image.open(filename)
    #left.show()
    #left  = np.asarray(Image.open(filename), dtype=np.float32, order='C')
    left = cv2.imread(filename)[:,:,::-1].astype(np.float32) # change BRG to RGB via `::-1`;
    #print ('[???] left image shape = ', left.shape)
    #pfm.show_uint8(left)
    #filename = pjoin(data_path, 'frames_finalpass/' + A[:-13] + 'right/' + A[len(A)-8:])
    filename = pjoin(data_path, A[:-13] + 'right/' + A[len(A)-8:])
    #print ("[****] rimg: {}".format(filename))
    right = cv2.imread(filename)[:,:,::-1].astype(np.float32)
    #pfm.show_uint8(right)
    #filename = pjoin(data_path, 'disparity/' + A[0:-4] + '.pfm')
    #print ("[???]", current_file, type(current_file))
    pos = A.find('/')
    tmp_len = len('frames_finalpass')
    filename = pjoin(data_path, A[0:pos] + '/disparity' + A[pos+1+tmp_len:-4] + '.pfm')
    #print ("[****] ldisp: {}".format(filename))
    disp_left = pfm.readPFM(filename)
    disp_left[disp_left == np.inf] = 0 # set zero as a invalid disparity value;
    #print ('[***] disp_left shape = ', disp_left.shape)
    #pfm.show(disp_left)
    #print ("[???] ",data_path +  'disparity/' + A[0:-13] + 'right/' + A[len(A)-8:-4] + '.pfm' )
    filename = pjoin(data_path, A[0:pos] + '/disparity' + A[pos+1+tmp_len:-13] + 'right/' + A[len(A)-8:-4] + '.pfm')
    #print ("[****] rdisp: {}".format(filename))
    disp_right = pfm.readPFM(filename)
    #pfm.show(disp_right)
    height,width = left.shape[:2]
    
    temp_data = np.zeros([8+6+1, height, width], 'float32')
    temp_data[0],temp_data[1],temp_data[2] = normalize_rgb_via_mean_std(left, is_mean_std=self_guassian_normalize)
    temp_data[3],temp_data[4],temp_data[5] = normalize_rgb_via_mean_std(right, is_mean_std=self_guassian_normalize)
    #temp_data[6:7, :, :] = width * 2
    temp_data[6, :, :] = disp_left
    temp_data[7, :, :] = disp_right
    # save for tensorboard visualization
    temp_data[8, :, :] = left[:,:,0]/255.0 #R 
    temp_data[9, :, :] = left[:,:,1]/255.0 #G
    temp_data[10, :, :] = left[:,:,2]/255.0 #B
    temp_data[11, :, :] = right[:,:,0]/255.0 #R 
    temp_data[12, :, :] = right[:,:,1]/255.0 #G
    temp_data[13, :, :] = right[:,:,2]/255.0 #B

    return temp_data


def load_kitti2012_data(file_path, current_file, self_guassian_normalize = True):
    """ load current file from the list"""
    filename = pjoin(file_path, 'colored_0/' + current_file)
    #print ("limg: {}".format(filename))
    left = np.asarray(Image.open(filename),dtype=np.float32, order="C")
    #filename = pjoin(file_path, 'image_1/' + current_file)
    filename = pjoin(file_path, 'colored_1/' + current_file)
    #print ("rimg: {}".format(filename))
    right = np.asarray(Image.open(filename), dtype=np.float32, order='C')
    
    filename = pjoin(file_path, 'disp_occ_pfm/' + current_file[0:-4]+ '.pfm')
    #print ("ldisp: {}".format(filename))
    disp_left = pfm.readPFM(filename)
    height, width = disp_left.shape[:2]
    #disp_left[disp_left == np.inf] = width*2 #set 2*width as a invalid disparity value;
    disp_left[disp_left == np.inf] = 0 # set zero as a invalid disparity value;

    temp_data = np.zeros([8+6+1, height, width], 'float32')
    
    temp_data[0],temp_data[1],temp_data[2] = normalize_rgb_via_mean_std(left, is_mean_std=self_guassian_normalize)
    temp_data[3],temp_data[4],temp_data[5] = normalize_rgb_via_mean_std(right, is_mean_std=self_guassian_normalize)
    temp_data[6, :, :] = disp_left
    # save for tensorboard visualization
    temp_data[8, :, :] = left[:,:,0]/255.0 #R 
    temp_data[9, :, :] = left[:,:,1]/255.0 #G
    temp_data[10, :, :] = left[:,:,2]/255.0 #B
    temp_data[11, :, :] = right[:,:,0]/255.0 #R 
    temp_data[12, :, :] = right[:,:,1]/255.0 #G
    temp_data[13, :, :] = right[:,:,2]/255.0 #B
    return temp_data

def load_kitti2012_gray_data(file_path, current_file, self_guassian_normalize = True):
    """ load current file from the list"""
    imglname = pjoin(file_path, 'image_0/' + current_file)
    left = np.asarray(Image.open(imglname), dtype=np.float32, order="C")
    #print ("left shape = ", left.shape)
    imgrname = pjoin(file_path, 'image_1/' + current_file)
    right = np.asarray(Image.open(imgrname), dtype=np.float32, order='C')
    #print ("right shape = ", right.shape)
    displname = pjoin(file_path, 'disp_occ_pfm/' + current_file[0:-4]+ '.pfm')
    disp_left = pfm.readPFM(displname)
    #print ("disp shape = ", disp_left.shape)
    #print ("limg: {}, rimg: {}, ldisp: {}".format(imglname, imgrname, displname))
    height, width = disp_left.shape[:2]
    #disp_left[disp_left == np.inf] = width*2 #set 2*width as a invalid disparity value;
    disp_left[disp_left == np.inf] = 0 # set zero as a invalid disparity value;

    temp_data = np.zeros([8+6+1, height, width], 'float32')

    tmp_l = normalize_gray_via_mean_std(left, is_mean_std=self_guassian_normalize) 
    # copy left gray image to each of the three ones
    temp_data[0] = tmp_l
    temp_data[1] = tmp_l
    temp_data[2] = tmp_l
    # copy right gray image to each of the three ones
    tmp_r = normalize_gray_via_mean_std(right, is_mean_std=self_guassian_normalize) 
    temp_data[3] = tmp_r
    temp_data[4] = tmp_r
    temp_data[5] = tmp_r
    
    temp_data[6, :, :] = disp_left
    # save for tensorboard visualization
    rgb_lname = pjoin(file_path, 'colored_0/' + current_file)
    left_rgb = np.asarray(Image.open(rgb_lname), dtype=np.float32, order="C")
    rgb_rname = pjoin(file_path, 'colored_1/' + current_file)
    right_rgb = np.asarray(Image.open(rgb_rname), dtype=np.float32, order="C")
    # save for tensorboard visualization
    temp_data[8, :, :] = left_rgb[:,:,0]/255.0 #R 
    temp_data[9, :, :] = left_rgb[:,:,1]/255.0 #G
    temp_data[10, :, :] = left_rgb[:,:,2]/255.0 #B
    temp_data[11, :, :] = right_rgb[:,:,0]/255.0 #R 
    temp_data[12, :, :] = right_rgb[:,:,1]/255.0 #G
    temp_data[13, :, :] = right_rgb[:,:,2]/255.0 #B
    return temp_data


def load_kitti2015_data(file_path, current_file, is_semantic = True, self_guassian_normalize = True):
    """ load current file from the list"""
    filename = pjoin(file_path, 'image_0/' + current_file)
    #print ("limg: {}".format(filename))
    left = np.asarray(Image.open(filename), dtype=np.float32, order="C")
    filename = pjoin(file_path, 'image_1/' + current_file)
    #print ("rimg: {}".format(filename))
    right = np.asarray(Image.open(filename), dtype=np.float32, order="C")
    filename = file_path + 'disp_occ_0_pfm/' + current_file[0:-4] + '.pfm'
    #print ("ldisp: {}".format(filename))

    disp_left = pfm.readPFM(filename)
    height, width = disp_left.shape[:2]
    #disp_left[disp_left == np.inf] = width*2 # set 2*width as a invalid disparity value;
    disp_left[disp_left == np.inf] = 0 # set zero as a invalid disparity value;

    temp_data = np.zeros([8+6+1, height, width], 'float32')
    temp_data[0],temp_data[1],temp_data[2] = normalize_rgb_via_mean_std(left, is_mean_std=self_guassian_normalize)
    temp_data[3],temp_data[4],temp_data[5] = normalize_rgb_via_mean_std(right, is_mean_std=self_guassian_normalize)
    
    temp_data[6, :, :] = disp_left
    # save for tensorboard visualization
    temp_data[8, :, :] = left[:,:,0]/255.0 #R 
    temp_data[9, :, :] = left[:,:,1]/255.0 #G
    temp_data[10, :, :] = left[:,:,2]/255.0 #B
    temp_data[11, :, :] = right[:,:,0]/255.0 #R 
    temp_data[12, :, :] = right[:,:,1]/255.0 #G
    temp_data[13, :, :] = right[:,:,2]/255.0 #B
    # semantic segmentaion label
    if is_semantic:
        # uint8 gray png image
        filename = pjoin(file_path, '../data_semantics/training/semantic/' + current_file)
        #print ("semantic label: {}".format(filename))
        semantic_label = np.asarray(Image.open(filename), dtype=np.float32, order="C")
        #pfm.show(semantic_label)
        temp_data[14,:,:] = semantic_label
    
    return temp_data


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

def load_virtual_kitti2_data(file_path, current_file, is_semantic = True, self_guassian_normalize = True):
    # e.g., current_file = "Scene01/15-deg-left/frames/rgb/Camera_0/rgb_00001.jpg"
    # e.g., file_path = "/media/ccjData2/datasets/Virtual-KITTI-V2/"
    A = current_file 
    # e.g., /media/ccjData2/datasets/Virtual-KITTI-V2/vkitti_2.0.3_rgb/Scene01/15-deg-left/frames/rgb/Camera_0/rgb_00001.jpg
    imglname = pjoin(file_path, "vkitti_2.0.3_rgb/" + A) 
    
    """ load current file from the list"""
    left = cv2.imread(imglname)[:,:,::-1].astype(np.float32) # change BRG to RGB via ``::-1`;
    #pfm.show_uint8(left, title='left image')
    imgrname = pjoin(file_path, "vkitti_2.0.3_rgb/" + A[:-22] + 'Camera_1/' + A[-13:])
    right = cv2.imread(imgrname)[:,:,::-1].astype(np.float32) # change BRG to RGB via ``::-1`;
    #pfm.show_uint8(right, title= 'right image')
    height, width = left.shape[:2]
    
    depth_png_filename = pjoin(file_path, "vkitti_2.0.3_depth/" + A[:-26] + 'depth/Camera_0/depth_' + A[-9:-4] + ".png")
    #print ("imgl = ", imglname, ", imgr = ", imgrname, ", depth_left = ", depth_png_filename)
    #NOTE: The depth map in centimeters can be directly loaded
    depth_left = cv2.imread(depth_png_filename, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

    # Intrinsi: f_x = f_y = 725.0087 
    # offset(i.e., distance between stereo views): B = 0.532725 m = 53.2725 cm
    B = 53.2725 # in centimeters;
    f = 725.0087 # in pixels
    # set zero as a invalid disparity value;
    disp_left = np.zeros([height, width], 'float32')
    mask = depth_left > 0
    #pfm.show_uint8(mask*255, title='mask')

    disp_left[mask] = f*B/ depth_left[mask] # d = fB/z
    #pfm.show(disp_left, title='disp_left')
    
    #NOTE: saving disparity file if needed!
    #disp_pfm_filename = pjoin(file_path, "vkitti_2.0.3_disparity/" + A[:-26] + 'disparity/Camera_0/disparity_' + A[-9:-4] + ".pfm")
    #print ("disp_pfm_filename = ", disp_pfm_filename)
    #tmp_len = disp_pfm_filename.rfind("/")
    #tmp_dir = disp_pfm_filename[0:tmp_len]
    #print ("tmp_dir = ", tmp_dir)
    #if not os.path.exists(tmp_dir):
    #    os.makedirs(tmp_dir)
    #pfm.save(disp_pfm_filename, disp_left)

    temp_data = np.zeros([8+6+1, height, width], 'float32')
    temp_data[0],temp_data[1],temp_data[2] = normalize_rgb_via_mean_std(left, is_mean_std=self_guassian_normalize)
    temp_data[3],temp_data[4],temp_data[5] = normalize_rgb_via_mean_std(right, is_mean_std=self_guassian_normalize)
    
    temp_data[6, :, :] = disp_left
    # save for tensorboard visualization
    temp_data[8, :, :] = left[:,:,0]/255.0 #R 
    temp_data[9, :, :] = left[:,:,1]/255.0 #G
    temp_data[10, :, :] = left[:,:,2]/255.0 #B
    temp_data[11, :, :] = right[:,:,0]/255.0 #R 
    temp_data[12, :, :] = right[:,:,1]/255.0 #G
    temp_data[13, :, :] = right[:,:,2]/255.0 #B
    # semantic segmentaion label
    if is_semantic:
        #loadding semantic segmantation labels
        seg_filename = pjoin(file_path, "vkitti_2.0.3_classSegmentation/" + A[:-26] + 'classSegmentation/Camera_0/classgt_' + A[-9:-4] + ".png")
        #print ("semantic label: {}".format(seg_filename))
        semantic_rgb_label = cv2.imread(seg_filename)[:,:,::-1] # change BRG to RGB via ``::-1`;
        #pfm.show_uint8(semantic_rgb_label, title="semantic_rgb_label")
        semantic_label = encode_color_segmap(semantic_rgb_label)
        #pfm.show(semantic_label, title="semantic_label")
        temp_data[14,:,:] = semantic_label.astype(np.float32)
    
    return temp_data

class DatasetFromList(data.Dataset): 
    def __init__(self, data_path, 
            file_list_txt, 
            crop_size=[256, 512], 
            training=True, 
            kitti2012=False, kitti2015=False, 
            virtual_kitti2 = False, 
            shift=0, 
            is_semantic=True,
            kt12_image_mode = 'rgb',
            is_data_augment = False
            ):
        super(DatasetFromList, self).__init__()
        #self.image_filenames = [join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)]
        f = open(file_list_txt, 'r')
        self.data_path = data_path
        if virtual_kitti2:
            self.file_list = get_virtual_kitti2_filelist(file_list_txt)
        else:
            self.file_list = [l.rstrip() for l in f.readlines() if not l.rstrip().startswith('#')]

        print ("[***] img# = {}, file_list = {}, ..., {}".format(len(self.file_list), 
            self.file_list[0], self.file_list[len(self.file_list) - 1]))
        self.training = training
        self.crop_height = crop_size[0]
        self.crop_width = crop_size[1]
        self.kitti2012 = kitti2012
        self.kitti2015 = kitti2015
        self.virtual_kitti2 = virtual_kitti2
        self.shift = shift
        self.is_semantic = is_semantic
        self.kt12_image_mode = str(kt12_image_mode).lower()
        #print (self.kt12_image_mode)
        assert self.kt12_image_mode in ['gray', 'rgb', 'gray2rgb']
        self.is_data_augment = is_data_augment
        self.my_kwargs = {
            "crop_height": self.crop_height,
            "crop_width": self.crop_width,
            }
        
        if is_data_augment:
            self.self_guassian_normalize = False
            self.train_trans = train_transform_augmentation
            self.my_kwargs.update({
                'scale':  [0.5, 1.15],
                #'scale':  [0.9, 1.15],
                #'scale':  [1.0, 1.0],
                'method': 'nearest'
            })
        else:
            self.self_guassian_normalize = True
            self.train_trans = train_transform
            self.my_kwargs.update({
                'shift':  self.shift
            })

    def __getitem__(self, index):
    #    print self.file_list[index]
        if self.kitti2012 and self.kt12_image_mode in ['gray', 'gray2rgb']: #load kitti2012 gray dataset
            temp_data = load_kitti2012_gray_data(self.data_path, self.file_list[index], self.self_guassian_normalize)

        elif self.kitti2012 and self.kt12_image_mode == 'rgb' : #load kitti2012 color dataset
            temp_data = load_kitti2012_data(self.data_path, self.file_list[index], self.self_guassian_normalize)
        
        elif self.kitti2015: #load kitti2015 dataset
            temp_data = load_kitti2015_data(self.data_path, self.file_list[index], self.is_semantic, self.self_guassian_normalize)
        
        elif self.virtual_kitti2: #load virtual kitti 2 dataset
            temp_data = load_virtual_kitti2_data(self.data_path, self.file_list[index], self.is_semantic, self.self_guassian_normalize)
        
        else: #load scene flow dataset
            temp_data = load_sfdata(self.data_path, self.file_list[index], self.self_guassian_normalize)
        
        if self.training:
            input1, input2, target, input1_rgb, input2_rgb, semantic_label = self.train_trans(temp_data, 
                    **self.my_kwargs)
            return input1, input2, target, input1_rgb, input2_rgb, semantic_label
        else:
            input1, input2, target = valid_transform(temp_data, self.crop_height, self.crop_width)
            return input1, input2, target

    def __len__(self):
        return len(self.file_list)