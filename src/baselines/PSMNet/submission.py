from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F

#import skimage
#import skimage.io
#import skimage.transform
#updated by CCJ:
from PIL import Image

import numpy as np
import time
import math
from utils import preprocess
from utils import pfmutil as pfm
from models import *
from os.path import join as pjoin
import cv2
from dispColor import colormap_jet_batch_image,KT15FalseColorDisp,KT15LogColorDispErr
from cython import writeKT15FalseColor as KT15FalseClr
from cython import writeKT15ErrorLogColor as KT15LogClr
from utils import pfmutil as pfm
from datetime import datetime

# 2012 data /media/jiaren/ImageNet/data_scene_flow_2012/testing/

parser = argparse.ArgumentParser(description='PSMNet')
parser.add_argument('--KITTI', default='2015',
                    help='KITTI version')
parser.add_argument('--datapath', default='/media/ccjData2/datasets/KITTI-2015/testing/',
                    help='select model')
parser.add_argument('--loadmodel', default=None,
                    help='loading model')
parser.add_argument('--model', default='stackhourglass',
                    help='select model')
parser.add_argument('--maxdisp', type=int, default=192,
                    help='maxium disparity')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--file_txt_path', type = str, default=None,
                    help='img list specefied for loading')
parser.add_argument('--result_dir', type = str, default='./results',
                    help='result dir')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

kitti2015 = False
kitti2012 = False
kitti_vkt2 = False


if args.KITTI == '2015':
    print ("processing KT15!")
    data_type_str= "kt15"
    from dataloader import KITTI_submission_loader as DA
    test_left_img, test_right_img = DA.dataloader(args.datapath, args.file_txt_path)
    test_left_disp = None
    kitti2015 = True
elif args.KITTI == '2012':
    print ("processing KT12!")
    data_type_str= "kt12"
    from dataloader import KITTI_submission_loader2012 as DA
    test_left_img, test_right_img = DA.dataloader(args.datapath, args.file_txt_path)
    test_left_disp = None
    kitti2012 = True

# added by CCJ on 2020/05/22:
elif args.KITTI == 'virtual_kt_2':
    print ("processing Virtual KT 2!")
    data_type_str= "virtual_kt2" 
    from dataloader import KITTIloader_VirtualKT2 as DA
    kitti_vkt2 = True
    test_left_img, test_right_img, test_left_disp = DA.dataloader_eval(args.datapath, args.file_txt_path)
else:
    raise Exception("No suitable KITTI found ...")


if args.model == 'stackhourglass':
    model = stackhourglass(args.maxdisp)
elif args.model == 'basic':
    model = basic(args.maxdisp)
else:
    print('no model')

model = nn.DataParallel(model, device_ids=[0])
model.cuda()

if args.loadmodel is not None:
    state_dict = torch.load(args.loadmodel)
    print ('[***] Loading model ', args.loadmodel)
    model.load_state_dict(state_dict['state_dict'])

print('Number of model parameters: {}'.format(
    sum([p.data.nelement() for p in model.parameters()])))


def test(imgL, imgR):
    model.eval()

    if args.cuda:
        imgL = torch.FloatTensor(imgL).cuda()
        imgR = torch.FloatTensor(imgR).cuda()

    #imgL, imgR = Variable(imgL), Variable(imgR)

    with torch.no_grad():
        output = model(imgL, imgR)
    output = torch.squeeze(output)
    pred_disp = output.data.cpu().numpy()
    return pred_disp

def main():
    
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
        print('mkdir %s' % args.result_dir)
                        
    processed = preprocess.get_transform(augment=False)

    imgN = len(test_left_img)
    avg_err = 0
    avg_rate1 = 0
    avg_rate3 = 0

    for inx in range(0, imgN):
    #for inx in range(2):
        #imgL_o = (skimage.io.imread(test_left_img[inx]).astype('float32'))
        #imgR_o = (skimage.io.imread(test_right_img[inx]).astype('float32'))
        #updated by CCJ: to remove the astype('float32');
        #ToTensor(): Converts a PIL Image or numpy.ndarray (H x W x C) 
        #            in the range [0, 255] to a torch.FloatTensor of 
        #            shape (C x H x W) in the range [0.0, 1.0] if the 
        #            PIL Image belongs to one of the modes (L, LA, P, I, 
        #            F, RGB, YCbCr, RGBA, CMYK, 1) or if the numpy.ndarray 
        #            has dtype = np.uint8
        
        imgL_o = np.asarray(Image.open(test_left_img[inx]))
        imgR_o = np.asarray(Image.open(test_right_img[inx]))
        imgL = processed(imgL_o).numpy()
        imgR = processed(imgR_o).numpy()
        imgL = np.reshape(imgL, [1, 3, imgL.shape[1], imgL.shape[2]])
        imgR = np.reshape(imgR, [1, 3, imgR.shape[1], imgR.shape[2]])

        # pad to (384, 1248)
        top_pad = 384-imgL.shape[2]
        right_pad = 1248-imgL.shape[3]
        imgL = np.lib.pad(imgL, ((0, 0), (0, 0), (top_pad, 0),
                                 (0, right_pad)), mode='constant', constant_values=0)
        imgR = np.lib.pad(imgR, ((0, 0), (0, 0), (top_pad, 0),
                                 (0, right_pad)), mode='constant', constant_values=0)
        
        savename = pjoin(args.result_dir, test_left_img[inx].split('/')[-1][:-4]+".pfm")
        
        if test_left_disp is not None:
            depth_png_filename = test_left_disp[inx]
            if os.path.isfile(depth_png_filename):
                #NOTE: The depth map in centimeters can be directly loaded
                depth_left = cv2.imread(depth_png_filename, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
                height, width = depth_left.shape[:2]
                # Intrinsi: f_x = f_y = 725.0087 
                # offset(i.e., distance between stereo views): B = 0.532725 m = 53.2725 cm
                B = 53.2725 # in centimeters;
                f = 725.0087 # in pixels
                # set zero as a invalid disparity value;
                dispGT = np.zeros([height, width], 'float32')
                mask = depth_left > 0
                dispGT[mask] = f*B/ depth_left[mask] # d = fB/z
                #pfm.show(dispGT, title='dispGT')
                savename = pjoin(args.result_dir, '%04d.pfm'%(inx))

        start_time = time.time()
        pred_disp = test(imgL, imgR)

        pred_disp = pred_disp[top_pad:, :-right_pad]
        # save pfm float disparity
        if any([kitti2015, kitti2012, inx % 250 == 0]):
            pfm.save(savename, pred_disp)
            print('Processing %d/%d, time = %.2f' % (inx, imgN, time.time()-start_time))
            print ("saved {}".format(savename))
            if 0:
                # save uint16 disparity
                img = Image.fromarray((img*255).astype('uint16'))
                #skimage.io.imsave('./results/' + test_left_img[inx].split(
                #    '/')[-1], (img*256).astype('uint16'))
                img.save(tmp_name + '.png')
        
        if kitti_vkt2 and dispGT is not None:
            error, rate1, rate3 = get_epe_rate(dispGT, pred_disp, threshold=1.0, threshold2=3.0)
            avg_err += error
            avg_rate1 += rate1
            avg_rate3 += rate3
            if inx % 250 == 0:
                print("===> Frame {}: ".format(inx) + test_left_img[inx] + " ==> EPE Error: {:.4f}, Bad-{:.1f} Error: {:.4f}, Bad-{:.1f} Error: {:.4f}".format(
                    error, 1.0, rate1, 3.0, rate3))
                """ disp color """
                tmp_dir = pjoin(args.result_dir, "dispColor")
                if not os.path.exists(tmp_dir):
                    os.makedirs(tmp_dir)
                tmp_dispname = pjoin(tmp_dir, '%04d.png'%(inx))
                cv2.imwrite(tmp_dispname, KT15FalseClr.writeKT15FalseColor(np.ascontiguousarray(pred_disp)).astype(np.uint8)[:,:,::-1])
                if inx < 1:
                    print ('savded ', tmp_dispname)
                
                """ err-disp """
                tmp_dir = pjoin(args.result_dir, "errDispColor")
                if not os.path.exists(tmp_dir):
                    os.makedirs(tmp_dir)
                tmp_errdispname = pjoin(tmp_dir, '%04d.png'%(inx))
                cv2.imwrite(tmp_errdispname, 
                        KT15LogClr.writeKT15ErrorDispLogColor(np.ascontiguousarray(pred_disp), np.ascontiguousarray(dispGT)).astype(np.uint8)[:,:,::-1])
                if inx < 1:
                    print ('savded ', tmp_errdispname)
    
    #average error and rate:
    if dispGT is not None:
        avg_err /= imgN
        avg_rate1 /= imgN
        avg_rate3 /= imgN
        print("===> Total {} Frames ==> AVG EPE Error: {:.4f}, AVG Bad-{:.1f} Error: {:.4f}, AVG Bad-{:.1f} Error: {:.4f}".format(
            imgN, avg_err, 1.0, avg_rate1, 3.0, avg_rate3))
    
        """ save as csv file, Excel file format """
        csv_file = os.path.join(args.result_dir, 'bad-err.csv')
        print ("write ", csv_file, "\n")
        timeStamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        messg = timeStamp + ',{},bad-1.0,{:.4f},bad-3.0,{:.4f},epe,{:.4f},fileDir={},for log,{:.3f}(epe); {:.3f}%(bad1); {:.3f}%(bad3)\n'.format(
                data_type_str, avg_rate1, avg_rate3, avg_err, 
                args.result_dir, 
                avg_err, avg_rate1*100.0, avg_rate3*100.0)
        
        with open( csv_file, 'w') as fwrite:
            fwrite.write(messg)

def get_epe_rate(disp, prediction, threshold = 1.0, threshold2 = 3.0):
    #mask = np.logical_and(disp >= 0.001, disp <= max_disp)
    mask = disp >= 0.001
    error = np.mean(np.abs(prediction[mask] - disp[mask]))
    rate = np.sum(np.abs(prediction[mask] - disp[mask]) > threshold) / np.sum(mask)
    rate2 = np.sum(np.abs(prediction[mask] - disp[mask]) > threshold2) / np.sum(mask)
    #print(" ==> EPE Error: {:.4f}, Error Rate: {:.4f}".format(error, rate))
    return error, rate, rate2

if __name__ == '__main__':
    main()
