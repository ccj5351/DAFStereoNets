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

#added by CCJ:
def get_imgs_from_files(
        file_txt_path # e.g., == "./filenames/kitti15_train.txt"
        ):
    #print (file_txt_path)
    with open(file_txt_path) as f:
        # each line has this format: 
        # colored_0/000000_10.png colored_1/000000_10.png disp_occ/000000_10.png
        lines = f.readlines()
        # now get the first element;
        imgs = [l.rstrip().split(' ')[0] for l in lines if not l.startswith('#')]
        # now extract the image name without file extension, i.e., we get '000000_10' here;
        imgs = [i[i.rfind('/')+1: i.rfind('.')] for i in imgs]
    return imgs

def dataloader(filepath, file_txt_path = None):
    #left_fold = 'image_2/'
    #right_fold = 'image_3/'
    #updated by CCJ for datadir:
    left_fold = 'image_0/'
    right_fold = 'image_1/'

    if file_txt_path is not None and file_txt_path != '':
        image = get_imgs_from_files(file_txt_path)
        image = [n + '.png' for n in image]

    else:
        image = [img for img in os.listdir(
            filepath+left_fold) if img.find('_10') > -1]
    image = sorted(image)
    imgN = len(image)
    print ("[**]Img# = {}, img_list={}, ..., {}".format(imgN, image[0], image[imgN -1]))
    left_test = [filepath+left_fold+img for img in image]
    right_test = [filepath+right_fold+img for img in image]
    return left_test, right_test