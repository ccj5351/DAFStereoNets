# !/usr/bin/env python3
# -*-coding:utf-8-*-
# @file: pascal_voc_loader.py
# @brief:
# @author: Changjiang Cai, ccai1@stevens.edu, caicj5351@gmail.com
# @version: 0.0.1
# @creation date: 23-09-2019
# @last modified: Fri 01 Nov 2019 03:36:15 PM EDT

""" coded adapted from https://github.com/meetshah1995/pytorch-semseg/blob/master/ptsemseg/loader/pascal_voc_loader.py """

import os
from os.path import join as pjoin
import collections
import json
import torch
import numpy as np
import scipy.misc as m
import scipy.io as io
import matplotlib.pyplot as plt
import glob

from PIL import Image
# see https://pypi.org/project/tqdm/
from tqdm import tqdm
from torch.utils import data
from torchvision import transforms


""" added by CCJ: """
def pascal_voc_color_map(N=256, normalized=False):
    """ see source code func colorize() at https://gist.github.com/jimfleming/c1adfdb0f526465c99409cc143dea97b"""
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap



def get_pascal_labels():
    """Load the mapping that associates pascal classes with label colors
    Returns:
    np.ndarray with dimensions (21, 3)
    """
    return np.asarray(
            [
                [0, 0, 0], # 0: background
                [128, 0, 0], #1: aeroplane
                [0, 128, 0], # 2: bicycle
                [128, 128, 0], # 3: bird
                [0, 0, 128], #4:'boat'    
                [128, 0, 128], #5: 'bottle'
                [0, 128, 128], #6: 'bus'
                [128, 128, 128], #7: 'car'
                [64, 0, 0], #8: 'cat'
                [192, 0, 0], #9: 'chair'
                [64, 128, 0],#10: 'cow'
                [192, 128, 0],#11: 'diningtable'
                [64, 0, 128], #12: 'dog'
                [192, 0, 128], #13:#'horse'
                [64, 128, 128],#14: 'motorbike'
                [192, 128, 128],#15:'person'
                [0, 64, 0],#16:'pottedplant'
                [128, 64, 0],#17:sheep
                [0, 192, 0],#18:sofa
                [128, 192, 0],#19:train
                [0, 64, 128],#20:'tv/monitor'
            ]
            )

def pascal_voc_batch_label2RGB(batch_label_mask):
        """Decode segmentation class labels into a color image

        Args:
            batch_label_mask (np.ndarray): an (N, H, W) or (N,H,W,1), or (N,1,H,W) 
                                           array of integer values denoting the class 
                                           label at each spatial location.

        Returns:
            (np.ndarray, optional): the resulting decoded color image.
        """
        #print ("[???] batch_label_mask shape = ", batch_label_mask.shape )
        if batch_label_mask.shape[-1] == 1:
            channel_dim = -1
        elif batch_label_mask.shape[1] == 1:
            channel_dim = 1

        indices = np.squeeze(batch_label_mask, axis=channel_dim).astype(np.int32) # in shape [N, H, W]
        N, H, W = indices.shape[:]
        #print ("[???] indices.shape = ", indices.shape)
        
        batch_label_rgb = np.zeros((N,H,W,3))
        #get pascal voc colormap
        cmap = pascal_voc_color_map(N=256, normalized=False)# in shape [256, 3]
        #print ('[???] camp shape = ', cmap.shape)
        for j in range(N):
            # gather
            batch_label_rgb[j,:,:,0] = np.take(cmap[:,0], indices[j]) / 255 # Red
            batch_label_rgb[j,:,:,1] = np.take(cmap[:,1], indices[j]) / 255 # Green
            batch_label_rgb[j,:,:,2] = np.take(cmap[:,2], indices[j]) / 255 # Blue
        
        return batch_label_rgb
        

class pascalVOCLoader(data.Dataset):
    """Data loader for the Pascal VOC semantic segmentation dataset.

    Annotations from both the original VOC data (which consist of RGB images
    in which colours map to specific classes) and the SBD (Berkely) dataset(
    > see: http://home.bharathh.info/pubs/codes/SBD/download.html) (where 
    annotations are stored as .mat files) are converted into a common
    `label_mask` format.  Under this format, each mask is an (M,N) array of
    integer values from 0 to 21, where 0 represents the background class.

    The label masks are stored in a new folder, called `pre_encoded`, which
    is added as a subdirectory of the `SegmentationClass` folder in the
    original Pascal VOC data layout.

    A total of five data splits are provided for working with the VOC data:
        train: The original VOC 2012 training data - 1464 images
        val: The original VOC 2012 validation data - 1449 images
        trainval: The combination of `train` and `val` - 2913 images
        train_aug: The unique images present in both the train split and
                   training images from SBD: - 8829 images (the unique members
                   of the result of combining lists of length 1464 and 8498)
        train_aug_val: The original VOC 2012 validation data minus the images
                   present in `train_aug` (This is done with the same logic as
                   the validation set used in FCN PAMI paper, but with VOC 2012
                   rather than VOC 2011) - 904 images
    """

    def __init__(
        self,
        root,
        sbd_path=None,
        split="train_aug",
        is_transform=False,
        img_size=161, # 512
        augmentations=None,
        #img_norm=True,
        test_mode=False,
    ):
        self.root = root
        self.sbd_path = sbd_path
        self.split = split
        self.is_transform = is_transform
        self.augmentations = augmentations
        #self.img_norm = img_norm
        self.test_mode = test_mode
        self.n_classes = 21
        self.mean = np.array([104.00699, 116.66877, 122.67892])
        self.files = collections.defaultdict(list)
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)

        if not self.test_mode:
            for split in ["train", "val", "trainval"]:
                path = pjoin(self.root, "ImageSets/Segmentation", split + ".txt")
                file_list = tuple(open(path, "r"))
                file_list = [id_.rstrip() for id_ in file_list]
                self.files[split] = file_list
            self.setup_annotations()

        self.tf = transforms.Compose(
            [
                #ToTensor(): Converts a PIL Image or numpy.ndarray (H x W x C) 
                #            in the range [0, 255] to a torch.FloatTensor of 
                #            shape (C x H x W) in the range [0.0, 1.0] if the 
                #            PIL Image belongs to one of the modes (L, LA, P, I, 
                #            F, RGB, YCbCr, RGBA, CMYK, 1) or if the numpy.ndarray 
                #            has dtype = np.uint8
                transforms.ToTensor(),
                # > see: https://discuss.pytorch.org/t/how-to-preprocess-input-for-pre-trained-networks/683
                # All pretrained torchvision models have the same preprocessing, 
                # which is to normalize using the following mean/std values:
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], #imagenet mean
                    std=[0.229, 0.224, 0.225]), # imagenet std
            ]
        )

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        im_name = self.files[self.split][index]
        im_path = pjoin(self.root, "JPEGImages", im_name + ".jpg")
        lbl_path = pjoin(self.root, "SegmentationClass/pre_encoded", im_name + ".png")
        im = Image.open(im_path)
        lbl = Image.open(lbl_path)
        #print ("[***] img size : ", im.size)
        #print ("[***] lable size : ", lbl.size)
        if self.augmentations is not None:
            im, lbl = self.augmentations(im, lbl)
        if self.is_transform:
            im, lbl = self.transform(im, lbl)

        # change [H, W] to [C = 1, H, W]
        lbl = torch.unsqueeze(lbl, 0)
        #print ("[***] expend lable shape : ", lbl.shape)
        return im, lbl

    def transform(self, img, lbl):
        if self.img_size == ("same", "same"):
            pass
        else:
            img = img.resize((self.img_size[0], self.img_size[1]))# uint8 with RGB mode
            lbl = lbl.resize((self.img_size[0], self.img_size[1]))
        img = self.tf(img) # in shape (C x H x W), in the range [0.0, 1.0];
        #lbl = torch.from_numpy(np.array(lbl)).long()
        lbl = np.array(lbl)
        #lbl[lbl == 255] = 0 # disabled it, updated on 2019/11/01;
        lbl = torch.from_numpy(lbl).float()
        return img, lbl
    """
    pascol_labels = [
      'background', #0
      'aeroplane', #1
      'bicycle', #2
      'bird', #3
      'boat', #4
      'bottle', #5
      'bus', #6
      'car', #7
      'cat', #8
      'chair', #9
      'cow', #10
      'diningtable', #11
      'dog', #12
      'horse', #13
      'motorbike', #14
      'person', #15
      'pottedplant', #16
      'sheep', #17
      'sofa', #18
      'train', #19
      'tv/monitor', #20
      "void/unlabelled", #255
      ] 
    """

    def get_pascal_labels(self):
        """Load the mapping that associates pascal classes with label colors

        Returns:
            np.ndarray with dimensions (21, 3)
        """
        return np.asarray(
            [
                [0, 0, 0], # 0: background
                [128, 0, 0], #1: aeroplane
                [0, 128, 0], # 2: bicycle
                [128, 128, 0], # 3: bird
                [0, 0, 128], #4:'boat'    
                [128, 0, 128], #5: 'bottle'
                [0, 128, 128], #6: 'bus'
                [128, 128, 128], #7: 'car'
                [64, 0, 0], #8: 'cat'
                [192, 0, 0], #9: 'chair'
                [64, 128, 0],#10: 'cow'
                [192, 128, 0],#11: 'diningtable'
                [64, 0, 128], #12: 'dog'
                [192, 0, 128], #13:#'horse'
                [64, 128, 128],#14: 'motorbike'
                [192, 128, 128],#15:'person'
                [0, 64, 0],#16:'pottedplant'
                [128, 64, 0],#17:sheep
                [0, 192, 0],#18:sofa
                [128, 192, 0],#19:train
                [0, 64, 128],#20:'tv/monitor'
            ]
        )

    def encode_segmap(self, mask):
        """Encode segmentation label images as pascal classes

        Args:
            mask (np.ndarray): raw segmentation label image of dimension
              (M, N, 3), in which the Pascal classes are encoded as colors.

        Returns:
            (np.ndarray): class map with dimensions (M,N), where the value at
            a given location is the integer denoting the class index.
        """
        mask = mask.astype(int)
        label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
        for ii, label in enumerate(self.get_pascal_labels()):
            label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
        label_mask = label_mask.astype(int)
        return label_mask

    def decode_segmap(self, label_mask, plot=False):
        """Decode segmentation class labels into a color image

        Args:
            label_mask (np.ndarray): an (M,N) array of integer values denoting
              the class label at each spatial location.
            plot (bool, optional): whether to show the resulting color image
              in a figure.

        Returns:
            (np.ndarray, optional): the resulting decoded color image.
        """
        label_colours = self.get_pascal_labels()
        r = label_mask.copy()
        g = label_mask.copy()
        b = label_mask.copy()
        for ll in range(0, self.n_classes):
            r[label_mask == ll] = label_colours[ll, 0]
            g[label_mask == ll] = label_colours[ll, 1]
            b[label_mask == ll] = label_colours[ll, 2]
        rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        if plot:
            plt.imshow(rgb)
            plt.show()
        else:
            return rgb

    def setup_annotations(self):
        """Sets up Berkley annotations by adding image indices to the
        `train_aug` split and pre-encode all segmentation labels into the
        common label_mask format (if this has not already been done). This
        function also defines the `train_aug` and `train_aug_val` data splits
        according to the description in the class docstring
        """
        sbd_path = self.sbd_path
        target_path = pjoin(self.root, "SegmentationClass/pre_encoded")
        if not os.path.exists(target_path):
            os.makedirs(target_path)
        #print ('sbd_path = ', sbd_path)
        path = pjoin(sbd_path, "dataset/train.txt")
        #print ('path = ', path)
        sbd_train_list = tuple(open(path, "r"))
        sbd_train_list = [id_.rstrip() for id_ in sbd_train_list]
        train_aug = self.files["train"] + sbd_train_list

        # keep unique elements (stable)
        train_aug = [train_aug[i] for i in sorted(np.unique(train_aug, return_index=True)[1])]
        self.files["train_aug"] = train_aug
        set_diff = set(self.files["val"]) - set(train_aug)  # remove overlap
        self.files["train_aug_val"] = list(set_diff)

        pre_encoded = glob.glob(pjoin(target_path, "*.png"))
        expected = np.unique(self.files["train_aug"] + self.files["val"]).size

        if len(pre_encoded) != expected:
            print("Pre-encoding segmentation masks...")
            for ii in tqdm(sbd_train_list):
                lbl_path = pjoin(sbd_path, "dataset/cls", ii + ".mat")
                data = io.loadmat(lbl_path)
                lbl = data["GTcls"][0]["Segmentation"][0].astype(np.int32)
                lbl = m.toimage(lbl, high=lbl.max(), low=lbl.min())
                m.imsave(pjoin(target_path, ii + ".png"), lbl)

            for ii in tqdm(self.files["trainval"]):
                fname = ii + ".png"
                lbl_path = pjoin(self.root, "SegmentationClass", fname)
                lbl = self.encode_segmap(m.imread(lbl_path))
                lbl = m.toimage(lbl, high=lbl.max(), low=lbl.min())
                m.imsave(pjoin(target_path, fname), lbl)

        assert expected == 9733, "unexpected dataset sizes"


# Leave code for debugging purposes
if __name__ == "__main__":
    #dummy main
    import src.augmentations as aug
    from six.moves import input
    local_path = '/media/ccjData2/datasets/pascal_voc_seg/VOCdevkit/VOC2012/'
    bs = 4
    prob = 0.5
    sbd_path = '/media/ccjData2/datasets/pascal_voc_seg/SBD_benchmark_RELEASE/'
    augs = aug.Compose([aug.RandomRotate(10), aug.RandomHorizontallyFlip(prob)])
    dst = pascalVOCLoader(root=local_path, sbd_path = sbd_path, is_transform=True, augmentations=augs)
    trainloader = data.DataLoader(dst, batch_size=bs)
    for i, data in enumerate(trainloader):
        imgs, labels = data
        imgs = imgs.numpy()[:, ::-1, :, :]
        imgs = np.transpose(imgs, [0,2,3,1])
        f, axarr = plt.subplots(bs, 2)
        for j in range(bs):
            axarr[j][0].imshow(imgs[j])
            axarr[j][1].imshow(dst.decode_segmap(labels.numpy()[j]))
        plt.show()
        a = input()
        if a == 'ex':
            break
        else:
            plt.close()
