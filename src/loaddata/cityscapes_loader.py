# !/usr/bin/env python3
# -*-coding:utf-8-*-
# @file: cityscapes_loader.py
# @brief:
# @author: Changjiang Cai, ccai1@stevens.edu, caicj5351@gmail.com
# @version: 0.0.1
# @creation date: 25-01-2020
# @last modified: Sun 26 Jan 2020 01:59:45 AM EST

import json
import os
from collections import namedtuple
import zipfile

import torch
from torchvision.datasets.utils import verify_str_arg, iterable_to_str
from torchvision.datasets.vision import VisionDataset
from PIL import Image
from torchvision import transforms
from torch.utils import data
import numpy as np
import matplotlib.pyplot as plt
# Based on https://github.com/mcordts/cityscapesScripts
CityscapesClass = namedtuple('CityscapesClass', 
        ['name', 'id', 'train_id', 'category', 'category_id',
            'has_instances', 'ignore_in_eval', 'color'])

cityscapes_labels = [
    #                name, id, trainId, category, catId, hasInstances, ignoreInEval color
    CityscapesClass('unlabeled', 0, 255, 'void', 0, False, True, (0, 0, 0)),
    CityscapesClass('ego vehicle', 1, 255, 'void', 0, False, True, (0, 0, 0)),
    CityscapesClass('rectification border', 2, 255, 'void', 0, False, True, (0, 0, 0)),
    CityscapesClass('out of roi', 3, 255, 'void', 0, False, True, (0, 0, 0)),
    CityscapesClass('static', 4, 255, 'void', 0, False, True, (0, 0, 0)),
    CityscapesClass('dynamic', 5, 255, 'void', 0, False, True, (111, 74, 0)),
    CityscapesClass('ground', 6, 255, 'void', 0, False, True, (81, 0, 81)),
    CityscapesClass('road', 7, 0, 'flat', 1, False, False, (128, 64, 128)),
    CityscapesClass('sidewalk', 8, 1, 'flat', 1, False, False, (244, 35, 232)),
    CityscapesClass('parking', 9, 255, 'flat', 1, False, True, (250, 170, 160)),
    CityscapesClass('rail track', 10, 255, 'flat', 1, False, True, (230, 150, 140)),
    CityscapesClass('building', 11, 2, 'construction', 2, False, False, (70, 70, 70)),
    CityscapesClass('wall', 12, 3, 'construction', 2, False, False, (102, 102, 156)),
    CityscapesClass('fence', 13, 4, 'construction', 2, False, False, (190, 153, 153)),
    CityscapesClass('guard rail', 14, 255, 'construction', 2, False, True, (180, 165, 180)),
    CityscapesClass('bridge', 15, 255, 'construction', 2, False, True, (150, 100, 100)),
    CityscapesClass('tunnel', 16, 255, 'construction', 2, False, True, (150, 120, 90)),
    CityscapesClass('pole', 17, 5, 'object', 3, False, False, (153, 153, 153)),
    CityscapesClass('polegroup', 18, 255, 'object', 3, False, True, (153, 153, 153)),
    CityscapesClass('traffic light', 19, 6, 'object', 3, False, False, (250, 170, 30)),
    CityscapesClass('traffic sign', 20, 7, 'object', 3, False, False, (220, 220, 0)),
    CityscapesClass('vegetation', 21, 8, 'nature', 4, False, False, (107, 142, 35)),
    CityscapesClass('terrain', 22, 9, 'nature', 4, False, False, (152, 251, 152)),
    CityscapesClass('sky', 23, 10, 'sky', 5, False, False, (70, 130, 180)),
    CityscapesClass('person', 24, 11, 'human', 6, True, False, (220, 20, 60)),
    CityscapesClass('rider', 25, 12, 'human', 6, True, False, (255, 0, 0)),
    CityscapesClass('car', 26, 13, 'vehicle', 7, True, False, (0, 0, 142)),
    CityscapesClass('truck', 27, 14, 'vehicle', 7, True, False, (0, 0, 70)),
    CityscapesClass('bus', 28, 15, 'vehicle', 7, True, False, (0, 60, 100)),
    CityscapesClass('caravan', 29, 255, 'vehicle', 7, True, True, (0, 0, 90)),
    CityscapesClass('trailer', 30, 255, 'vehicle', 7, True, True, (0, 0, 110)),
    CityscapesClass('train', 31, 16, 'vehicle', 7, True, False, (0, 80, 100)),
    CityscapesClass('motorcycle', 32, 17, 'vehicle', 7, True, False, (0, 0, 230)),
    CityscapesClass('bicycle', 33, 18, 'vehicle', 7, True, False, (119, 11, 32)),
    CityscapesClass('license plate', -1, -1, 'vehicle', 7, False, True, (0, 0, 142)),
]


def get_cityscapes_labels():
        """Load the mapping that associates cityscapes classes with label colors

        Returns:
            np.ndarray with dimensions (34, 3)
        """
        n_classes = len(cityscapes_labels) - 1 # remove the id=-1;
        assert (n_classes == 34)
        cs_clr_labels = np.zeros((n_classes, 3), np.uint8)
        for label in cityscapes_labels:
            if label.id >= 0:
                cs_clr_labels[label.id] = label.color

        return cs_clr_labels

def encode_cityscapes_segmap(mask):
    """Encode segmentation label images as cityscapes classes

    Args:
        mask (np.ndarray): raw segmentation label image of dimension
          (M, N, 3), in which the cityscapes classes are encoded as colors.

    Returns:
        (np.ndarray): class map with dimensions (M,N), where the value at
        a given location is the integer denoting the class index.
    """
    mask = mask.astype(int)
    label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
    for ii, label in enumerate(get_cityscapes_labels()):
        label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
    label_mask = label_mask.astype(int)
    return label_mask

def decode_cityscapes_segmap(label_mask, plot=False):
    """Decode segmentation class labels into a color image

    Args:
        label_mask (np.ndarray): an (M,N) array of integer values denoting
          the class label at each spatial location.
        plot (bool, optional): whether to show the resulting color image
          in a figure.

    Returns:
        (np.ndarray, optional): the resulting decoded color image.
    """
    label_colors = get_cityscapes_labels()
    #print ("[???] label_colors shape = ", label_colors.shape)
    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    n_classes = 34
    for ll in range(0, n_classes):
        r[label_mask == ll] = label_colors[ll, 0]
        g[label_mask == ll] = label_colors[ll, 1]
        b[label_mask == ll] = label_colors[ll, 2]
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    #print ('[???] rgb shape = ', rgb.shape)
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    if plot:
        plt.imshow(rgb)
        plt.show()
    else:
        return rgb

def cityscape_batch_label2RGB(batch_label_mask):
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
        #get cityscape colormap
        cmap = get_cityscapes_labels() # in shape (34, 3)
        #print ('[???] camp shape = ', cmap.shape)
        """ deal with invalid label values due to data augmentation """
        #NOTE: due to data augmentation to label and image, e.g., random rotation,
        # so somewhere will have invalid label values (i.e., label > num_classes = 34),
        # e.g., after augmentation, label = 250, which is in valid, due to 250 > 34;
        # so we have to take this case into consideration !!!
        indices[indices >= 34] = 0
        for j in range(N):
            
            #NOTE:assert already verified!!!
            #tmp_max = np.amax(indices[j])
            #assert tmp_max < 34, 'batch_idx = {}, max_value_index = {}, max_val = {}'.format(
            #    j, np.unravel_index(np.argmax(indices[j], axis=None), indices[j].shape), tmp_max)
            
            # gather
            batch_label_rgb[j,:,:,0] = np.take(cmap[:,0], indices[j]) / 255 # Red
            batch_label_rgb[j,:,:,1] = np.take(cmap[:,1], indices[j]) / 255 # Green
            batch_label_rgb[j,:,:,2] = np.take(cmap[:,2], indices[j]) / 255 # Blue
        
        return batch_label_rgb


# > this code is adopted from TORCHVISION.DATASETS.CITYSCAPES, at 
# https://pytorch.org/docs/stable/_modules/torchvision/datasets/cityscapes.html;

class CityscapesLoader(data.Dataset):
    """`Cityscapes <http://www.cityscapes-dataset.com/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory ``leftImg8bit``
            and ``gtFine`` or ``gtCoarse`` are located.
        split (string, optional): The image split to use, ``train``, ``test`` or ``val`` if mode="gtFine"
            otherwise ``train``, ``train_extra`` or ``val``
        mode (string, optional): The quality mode to use, ``gtFine`` or ``gtCoarse``
        target_type (string): Type of target to use, ``instance``, ``semantic``, ``polygon``
            or ``color``. 

    Examples:

        Get semantic segmentation target

        .. code-block:: python

            dataset = Cityscapes('./data/cityscapes', split='train', mode='fine',
                                 target_type='semantic')

            img, smnt = dataset[0]
        

        Validate on the "coarse" set

        .. code-block:: python

            dataset = Cityscapes('./data/cityscapes', split='val', mode='coarse',
                                 target_type='semantic')

            img, smnt = dataset[0]
    """

    
    def __init__(self, 
                 root, 
                 split='train', 
                 mode='fine', 
                 target_type='semantic',
                 is_transform= True,
                 img_size=161, # 512
                 augmentations=None,
                 ):
        super(CityscapesLoader, self).__init__()
        self.root = root
        self.mode = 'gtFine' if mode == 'fine' else 'gtCoarse'
        self.images_dir = os.path.join(self.root, 'leftImg8bit', split)
        self.targets_dir = os.path.join(self.root, self.mode, split)
        
        self.split = split
        self.images = []
        self.targets = []
        
        self.augmentations = augmentations
        self.is_transform = is_transform
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.tf = transforms.Compose(
            [
                #ToTensor(): Converts a PIL Image or numpy.ndarray (H x W x C) 
                #            in the range [0, 255] to a torch.FloatTensor of 
                #            shape (C x H x W) in the range [0.0, 1.0] if the 
                #            PIL Image belongs to one of the modes (L, LA, P, I, 
                #            F, RGB, YCbCr, RGBA, CMYK, 1) or if the numpy.ndarray 
                #            has dtype = np.uint8
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )        

        verify_str_arg(mode, "mode", ("fine", "coarse"))
        if mode == "fine":
            valid_modes = ("train", "test", "val")
        else:
            valid_modes = ("train", "train_extra", "val")
        msg = ("Unknown value '{}' for argument split if mode is '{}'. "
               "Valid values are {{{}}}.")
        msg = msg.format(split, mode, iterable_to_str(valid_modes))
        verify_str_arg(split, "split", valid_modes, msg)

        #if not isinstance(target_type, list):
        #    self.target_type = [target_type]
        #else:
        #    self.target_type = target_type
        #[verify_str_arg(value, "target_type",
        #                ("instance", "semantic", "polygon", "color")) for value in self.target_type]
        self.target_type = target_type
        
        for city in os.listdir(self.images_dir):
            img_dir = os.path.join(self.images_dir, city)
            target_dir = os.path.join(self.targets_dir, city)
            for file_name in os.listdir(img_dir):
                target_name = '{}_{}'.format(file_name.split('_leftImg8bit')[0],
                        self._get_target_suffix(self.mode, self.target_type))
                self.targets.append(os.path.join(target_dir, target_name))
                self.images.append(os.path.join(img_dir, file_name))
    

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a tuple of all target types if target_type is a list with more
            than one item. Otherwise target is a json object if target_type="polygon", else the image segmentation.
        """
        #print ("[???] loading ", self.images[index], ';', self.targets[index])
        image = Image.open(self.images[index]).convert('RGB')

        if self.target_type  == 'polygon':
            target = self._load_json(self.targets[index])
        else:
            target = Image.open(self.targets[index])
        
        if self.augmentations is not None:
            image, target = self.augmentations(image, target)
        if self.is_transform:
            image, target = self.my_transform(image, target)

        # change [H, W] to [C = 1, H, W]
        #print ("[***] expend lable shape : ", target.shape)
        target = torch.unsqueeze(target, 0)
        return image, target


    # added by CCJ:
    def my_transform(self, img, lbl):
        if self.img_size == ("same", "same"):
            pass
        else:
            img = img.resize((self.img_size[0], self.img_size[1]))# uint8 with RGB mode
            lbl = lbl.resize((self.img_size[0], self.img_size[1]))
        
        # do transform: totensor, and normalization
        img = self.tf(img) # in shape (C x H x W), in the range [0.0, 1.0];
        lbl = torch.from_numpy(np.array(lbl)).float()
        return img, lbl
    
    def __len__(self):
        return len(self.images)

    def extra_repr(self):
        lines = ["Split: {split}", "Mode: {mode}", "Type: {target_type}"]
        return '\n'.join(lines).format(**self.__dict__)

    def _load_json(self, path):
        with open(path, 'r') as file:
            data = json.load(file)
        return data

    def _get_target_suffix(self, mode, target_type):
        if target_type == 'instance':
            return '{}_instanceIds.png'.format(mode)
        elif target_type == 'semantic':
            return '{}_labelIds.png'.format(mode)
        elif target_type == 'color':
            return '{}_color.png'.format(mode)
        else:
            return '{}_polygons.json'.format(mode)

# Leave code for debugging purposes
if __name__ == "__main__":
    #dummy main
    import src.augmentations as aug
    from six.moves import input
    bs = 2
    prob = 0.5
    augs = aug.Compose([aug.RandomRotate(10), aug.RandomHorizontallyFlip(prob)])
    #augs = None
    dst = CityscapesLoader(
                 root = '/media/ccjData2/datasets/cityscapes/', 
                 split='train', 
                 mode='fine', 
                 target_type ='semantic',
                 is_transform= True,
                 img_size= 161, # 512
                 augmentations= augs)
    trainloader = data.DataLoader(dst, batch_size=bs)
    for i, data in enumerate(trainloader):
        imgs, labels = data
        imgs = imgs.numpy()[:, ::-1, :, :]
        imgs = np.transpose(imgs, [0,2,3,1])

        #print ('imgs, labels: ', imgs.shape, labels.shape)
        if 0:
            f, axarr = plt.subplots(bs, 2)
            for j in range(bs):
                axarr[j][0].imshow(imgs[j])
                axarr[j][1].imshow(decode_cityscapes_segmap(
                    np.squeeze(labels.numpy()[j], axis = 0)
                    ))
            plt.show()
            a = input()
            if a == 'ex':
                break
            else:
                plt.close()
        if 1:
            print ('{}/{}'.format(i, len(trainloader)))
            batch_label_rgb = cityscape_batch_label2RGB(labels.numpy())
