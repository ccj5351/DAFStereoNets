import os
import torch
import torch.utils.data as data
import torch
import torchvision.transforms as transforms
import random
from PIL import Image, ImageOps
import numpy as np

from . import preprocess
import cv2

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def default_loader(path):
    return Image.open(path).convert('RGB')


def disparity_loader(path):
    return Image.open(path)

def virtual_kt_2_disparity_from_png_depth(depth_png_filename):
    depth = cv2.imread(depth_png_filename, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    height, width = depth.shape[:2]
    # Intrinsi: f_x = f_y = 725.0087 
    # offset(i.e., distance between stereo views): B = 0.532725 m = 53.2725 cm
    B = 53.2725 # in centimeters;
    f = 725.0087 # in pixels
    # set zero as a invalid disparity value;
    disparity = np.zeros([height, width], 'float32')
    mask = depth > 0
    #pfm.show_uint8(mask*255, title='mask')
    disparity[mask] = f*B/ depth[mask] # d = fB/z
    return disparity

class myImageFloder(data.Dataset):
    def __init__(self, left, right, left_disparity, training, 
            loader = default_loader, 
            dploader = disparity_loader,
            virtual_kitti2 = False # boolean flag; 
            ):

        self.left = left
        self.right = right
        self.disp_L = left_disparity
        self.loader = loader
        
        if virtual_kitti2:
            self.dploader = virtual_kt_2_disparity_from_png_depth
        else:
            self.dploader = dploader
        self.training = training
        self.virtual_kitti2 = virtual_kitti2

    def __getitem__(self, index):
        left = self.left[index]
        right = self.right[index]
        disp_L = self.disp_L[index]

        left_img = self.loader(left)
        right_img = self.loader(right)
        dataL = self.dploader(disp_L)

        if self.training:
            w, h = left_img.size
            th, tw = 256, 512

            x1 = random.randint(0, w - tw)
            y1 = random.randint(0, h - th)

            left_img = left_img.crop((x1, y1, x1 + tw, y1 + th))
            # image in format "CHW";
            left_img_rgb = np.asarray(left_img, dtype=np.float32, order="C").transpose(2,0,1)/255.0 # added by CCJ;
            
            right_img = right_img.crop((x1, y1, x1 + tw, y1 + th))
            if not self.virtual_kitti2:
                dataL = np.ascontiguousarray(dataL, dtype=np.float32)/256
            dataL = dataL[y1:y1 + th, x1:x1 + tw]

            processed = preprocess.get_transform(augment=False)
            left_img = processed(left_img)
            right_img = processed(right_img)

            return left_img, right_img, dataL, left_img_rgb
        else:
            w, h = left_img.size

            left_img = left_img.crop((w-1232, h-368, w, h))
            right_img = right_img.crop((w-1232, h-368, w, h))
            w1, h1 = left_img.size

            if self.virtual_kitti2:
                dataL = dataL[h-368:h, w-1232:w]
            else:
                # left, top, right, bottom 
                dataL = dataL.crop((w-1232, h-368, w, h))
                dataL = np.ascontiguousarray(dataL, dtype=np.float32)/256

            processed = preprocess.get_transform(augment=False)
            left_img = processed(left_img)
            right_img = processed(right_img)

            return left_img, right_img, dataL

    def __len__(self):
        return len(self.left)
