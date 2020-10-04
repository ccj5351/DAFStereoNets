"""
# the code is adapted from GANet paper code:
# > see the code at: https://github.com/feihuzhang/GANet/tree/ab6782aff8b21cf2a70f3f839fc4030d24b29a1c/dataloader
"""
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
        start_y = random.randint(0, h - crop_height)
        temp_data = temp_data[:, start_y: start_y + crop_height, start_x: start_x + crop_width]
    
    left = temp_data[0: 3, :, :]
    right = temp_data[3: 6, :, :]
    target = temp_data[6: 7, :, :]
    left_rgb = temp_data[8:11,:,:]
    right_rgb = temp_data[11:14,:,:]
    semantic_label = temp_data[14:15,:,:] # if is_semantic == False, will be all zeros 
    return left, right, target,left_rgb,right_rgb, semantic_label

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
    return left, right, target

""" added by CCJ """
def normalize_rgb_via_mean_std(img):
    assert (np.shape(img)[2] == 3)
    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]

    # left : R G B
    r = (r-np.mean(r)) / np.std(r)
    g = (g-np.mean(g)) / np.std(g)
    b = (b-np.mean(b)) / np.std(b)
    return r,g,b

# load sf (scene flow) data
def load_sfdata(data_path, current_file):
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
    temp_data[0],temp_data[1],temp_data[2] = normalize_rgb_via_mean_std(left)
    temp_data[3],temp_data[4],temp_data[5] = normalize_rgb_via_mean_std(right)
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


def load_kitti2012_data(file_path, current_file):
    """ load current file from the list"""
    filename = pjoin(file_path, 'colored_0/' + current_file)
    #filename = pjoin(file_path, 'image_0/' + current_file)
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
    
    temp_data[0],temp_data[1],temp_data[2] = normalize_rgb_via_mean_std(left)
    temp_data[3],temp_data[4],temp_data[5] = normalize_rgb_via_mean_std(right)
    temp_data[6, :, :] = disp_left
    # save for tensorboard visualization
    temp_data[8, :, :] = left[:,:,0]/255.0 #R 
    temp_data[9, :, :] = left[:,:,1]/255.0 #G
    temp_data[10, :, :] = left[:,:,2]/255.0 #B
    temp_data[11, :, :] = right[:,:,0]/255.0 #R 
    temp_data[12, :, :] = right[:,:,1]/255.0 #G
    temp_data[13, :, :] = right[:,:,2]/255.0 #B
    return temp_data


def load_kitti2015_data(file_path, current_file, is_semantic = True):
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
    temp_data[0],temp_data[1],temp_data[2] = normalize_rgb_via_mean_std(left)
    temp_data[3],temp_data[4],temp_data[5] = normalize_rgb_via_mean_std(right)
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



class DatasetFromList(data.Dataset): 
    def __init__(self, data_path, file_list, crop_size=[256, 256], 
            training=True, kitti2012=False, kitti2015=False, shift=0, is_semantic=True):
        super(DatasetFromList, self).__init__()
        #self.image_filenames = [join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)]
        f = open(file_list, 'r')
        self.data_path = data_path
        self.file_list = [l.rstrip() for l in f.readlines()]
        print ("[***] img# = {}, file_list = {}, ..., {}".format(len(self.file_list), 
            self.file_list[0], self.file_list[len(self.file_list) - 1]))
        self.training = training
        self.crop_height = crop_size[0]
        self.crop_width = crop_size[1]
        self.kitti2012 = kitti2012
        self.kitti2015 = kitti2015
        self.shift = shift
        self.is_semantic = is_semantic

    def __getitem__(self, index):
    #    print self.file_list[index]
        if self.kitti2012: #load kitti2012 dataset
            temp_data = load_kitti2012_data(self.data_path, self.file_list[index])
        elif self.kitti2015: #load kitti2015 dataset
            temp_data = load_kitti2015_data(self.data_path, self.file_list[index], self.is_semantic)
        else: #load scene flow dataset
            temp_data = load_sfdata(self.data_path, self.file_list[index])
        
        if self.training:
            input1, input2, target, input1_rgb, input2_rgb, semantic_label = train_transform(temp_data, 
                    self.crop_height, self.crop_width, self.shift)
            return input1, input2, target, input1_rgb, input2_rgb, semantic_label
        else:
            input1, input2, target = valid_transform(temp_data, self.crop_height, self.crop_width)
            return input1, input2, target

    def __len__(self):
        return len(self.file_list)
