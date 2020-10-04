import torch.utils.data as data

from PIL import Image
import os
import os.path
from os.path import join as pjoin
import random

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def dataloader(filepath, fraction = 1.0):
    #classes = [d for d in os.listdir(filepath) if os.path.isdir(os.path.join(filepath, d))]
    #print ('classes = ', classes)
    #image = [img for img in classes if img.find('frames_cleanpass') > -1]
    #disp  = [dsp for dsp in classes if dsp.find('disparity') > -1]

    #monkaa_path = filepath + [x for x in image if 'monkaa' in x][0]
    #monkaa_disp = filepath + [x for x in disp if 'monkaa' in x][0]
    # updated by CCJ for different data load:

    monkaa_path = pjoin(filepath, 'monkaa/frames_finalpass')
    print('monkaa_path = ', monkaa_path)
    monkaa_disp = pjoin(filepath, 'monkaa/disparity')

    monkaa_dir = os.listdir(monkaa_path)

    all_left_img = []
    all_right_img = []
    all_left_disp = []
    test_left_img = []
    test_right_img = []
    test_left_disp = []

    print('monkaa_dir = ', monkaa_dir)
    for dd in monkaa_dir:
        for im in os.listdir(monkaa_path+'/'+dd+'/left/'):
            if is_image_file(monkaa_path+'/'+dd+'/left/'+im):
                all_left_img.append(monkaa_path+'/'+dd+'/left/'+im)
                all_left_disp.append(monkaa_disp+'/'+dd +
                                     '/left/'+im.split(".")[0]+'.pfm')

        for im in os.listdir(monkaa_path+'/'+dd+'/right/'):
            if is_image_file(monkaa_path+'/'+dd+'/right/'+im):
                all_right_img.append(monkaa_path+'/'+dd+'/right/'+im)

    assert (len(all_left_img) == len(all_left_disp))
    assert (len(all_left_img) == len(all_right_img))
    #flying_path = filepath + [x for x in image if x == 'frames_cleanpass'][0]
    #flying_disp = filepath + [x for x in disp if x == 'frames_disparity'][0]
    flying_path = pjoin(filepath, 'flyingthings3d/frames_finalpass')
    flying_disp = pjoin(filepath, 'flyingthings3d/disparity')
    flying_dir = flying_path+'/TRAIN/'
    subdir = ['A', 'B', 'C']

    for ss in subdir:
        flying = os.listdir(flying_dir+ss)

        for ff in flying:
            imm_l = os.listdir(flying_dir+ss+'/'+ff+'/left/')
            for im in imm_l:
                if is_image_file(flying_dir+ss+'/'+ff+'/left/'+im):
                    all_left_img.append(flying_dir+ss+'/'+ff+'/left/'+im)
                    all_left_disp.append(
                        flying_disp+'/TRAIN/'+ss+'/'+ff+'/left/'+im.split(".")[0]+'.pfm')

                if is_image_file(flying_dir+ss+'/'+ff+'/right/'+im):
                    all_right_img.append(flying_dir+ss+'/'+ff+'/right/'+im)

    flying_dir = flying_path+'/TEST/'

    subdir = ['A', 'B', 'C']

    for ss in subdir:
        flying = os.listdir(flying_dir+ss)

        for ff in flying:
            imm_l = os.listdir(flying_dir+ss+'/'+ff+'/left/')
            for im in imm_l:
                if is_image_file(flying_dir+ss+'/'+ff+'/left/'+im):
                    test_left_img.append(flying_dir+ss+'/'+ff+'/left/'+im)
                    test_left_disp.append(
                        flying_disp+'/TEST/'+ss+'/'+ff+'/left/'+im.split(".")[0]+'.pfm')

                if is_image_file(flying_dir+ss+'/'+ff+'/right/'+im):
                    test_right_img.append(flying_dir+ss+'/'+ff+'/right/'+im)

    assert (len(all_left_img) == len(all_left_disp))
    assert (len(all_left_img) == len(all_right_img))

    #driving_dir = filepath + [x for x in image if 'driving' in x][0] + '/'
    #driving_disp = filepath + [x for x in disp if 'driving' in x][0]
    driving_dir = pjoin(filepath, 'driving/frames_finalpass')
    driving_disp = pjoin(filepath, 'driving/disparity')

    subdir1 = ['35mm_focallength', '15mm_focallength']
    subdir2 = ['scene_backwards', 'scene_forwards']
    subdir3 = ['fast', 'slow']

    for i in subdir1:
        for j in subdir2:
            for k in subdir3:
                imm_l = os.listdir(driving_dir + '/' + i+'/'+j+'/'+k+'/left/')
                for im in imm_l:
                    if is_image_file(driving_dir + '/' + i+'/'+j+'/'+k+'/left/'+im):
                        all_left_img.append(
                            driving_dir + '/'+i+'/'+j+'/'+k+'/left/'+im)
                        all_left_disp.append(
                            driving_disp+'/'+i+'/'+j+'/'+k+'/left/'+im.split(".")[0]+'.pfm')

                    if is_image_file(driving_dir + '/'+i+'/'+j+'/'+k+'/right/'+im):
                        all_right_img.append(
                            driving_dir + '/'+i+'/'+j+'/'+k+'/right/'+im)

    assert (len(all_left_img) == len(all_left_disp))
    assert (len(all_left_img) == len(all_right_img))
    if 0:
        tmp = all_left_img
        N = len(tmp)
        print('Num imgs = {}, all_left_img = {}, {}, ..., {}'.format(
            N, tmp[0], tmp[random.randint(0, N)], tmp[N-1]))

        tmp = all_right_img
        N = len(tmp)
        print('Num imgs = {}, all_right_img = {}, {}, ..., {}'.format(
            N, tmp[0], tmp[random.randint(0, N)], tmp[N-1]))

        tmp = all_left_disp
        N = len(tmp)
        print('Num imgs = {}, all_left_disp = {}, {}, ..., {}'.format(
            N, tmp[0], tmp[random.randint(0, N)], tmp[N-1]))
    N = len(all_left_img)
    F = int(N * fraction)
    return all_left_img[0:F], all_right_img[0:F], all_left_disp[0:F], test_left_img, test_right_img, test_left_disp
