from __future__ import print_function
import argparse
import skimage
import skimage.io
import skimage.transform
from PIL import Image
from math import log10
#from GCNet.modules.GCNet import L1Loss
import sys
import shutil
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

import numpy as np
from src.cython import writeKT15FalseColor as KT15FalseClr
from src.cython import writeKT15ErrorLogColor as KT15LogClr
from os.path import join as pjoin
import src.pfmutil as pfm
import cv2
#from src.dispColor import colormap_jet_batch_image,KT15FalseColorDisp,KT15LogColorDispErr
from datetime import datetime
from src.loaddata.dataset import get_virtual_kitti2_filelist


# Training settings
parser = argparse.ArgumentParser(description='PyTorch GANet Example')
parser.add_argument('--crop_height', type=int, required=True, help="crop height")
parser.add_argument('--crop_width', type=int, required=True, help="crop width")
parser.add_argument('--max_disp', type=int, default=192, help="max disp")
parser.add_argument('--resume', type=str, default='', help="resume from saved model")
parser.add_argument('--cuda', type=bool, default=True, help='use cuda?')
parser.add_argument('--data_path', type=str, required=True, help="data root")
parser.add_argument('--test_list', type=str, required=True, help="training list")
#parser.add_argument('--save_path', type=str, default='./result/', help="location to save result")
parser.add_argument('--model', type=str, default='GANet_deep', help="model to train")

# added by CCJ:
parser.add_argument('--resultDir', type=str, default= "./results")
parser.add_argument('--kitti2012', type=int, default=0, help='kitti dataset? Default=False')
parser.add_argument('--kitti2015', type=int, default=0, help='kitti 2015? Default=False')
parser.add_argument('--virtual_kitti2', type=int, default=0, help='virtual_kitti2? Default=False')

#parser.add_argument('--threshold', type=float, default=3.0, help="threshold of error rates")

opt = parser.parse_args()


print(opt)
if opt.model == 'GANet11':
    from .models.GANet11 import GANet
elif opt.model == 'GANet_deep':
    from .models.GANet_deep import GANet
else:
    raise Exception("No suitable model found ...")
    
cuda = opt.cuda
#cuda = True
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

#torch.manual_seed(opt.seed)
#if cuda:
#    torch.cuda.manual_seed(opt.seed)
#print('===> Loading datasets')


print('===> Building model ', opt.model)
model = GANet(opt.max_disp)

if cuda:
    model = torch.nn.DataParallel(model).cuda()

if opt.resume:
    if os.path.isfile(opt.resume):
        print("=> loading checkpoint '{}'".format(opt.resume))
        checkpoint = torch.load(opt.resume)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
       
    else:
        print("=> no checkpoint found at '{}'".format(opt.resume))


def test_transform(temp_data, crop_height, crop_width):
    _, h, w=np.shape(temp_data)

    if h <= crop_height and w <= crop_width:
        temp = temp_data
        temp_data = np.zeros([6, crop_height, crop_width], 'float32')
        temp_data[:, crop_height - h: crop_height, crop_width - w: crop_width] = temp
    else:
        start_x = int((w - crop_width) / 2)
        start_y = int((h - crop_height) / 2)
        temp_data = temp_data[:, start_y: start_y + crop_height, start_x: start_x + crop_width]
    left = np.ones([1, 3,crop_height,crop_width],'float32')
    left[0, :, :, :] = temp_data[0: 3, :, :]
    right = np.ones([1, 3, crop_height, crop_width], 'float32')
    right[0, :, :, :] = temp_data[3: 6, :, :]
    return torch.from_numpy(left).float(), torch.from_numpy(right).float(), h, w

def load_data(leftname, rightname):
    left = Image.open(leftname)
    right = Image.open(rightname)
    size = np.shape(left)
    height = size[0]
    width = size[1]
    temp_data = np.zeros([6, height, width], 'float32')
    left = np.asarray(left)
    right = np.asarray(right)
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

def test(leftname, rightname, savename):
  #  count=0
    
    input1, input2, height, width = test_transform(load_data(leftname, rightname), opt.crop_height, opt.crop_width)

    
    input1 = Variable(input1, requires_grad = False)
    input2 = Variable(input2, requires_grad = False)

    model.eval()
    if cuda:
        input1 = input1.cuda()
        input2 = input2.cuda()
    with torch.no_grad():
        prediction = model(input1, input2)
     
    temp = prediction.cpu()
    temp = temp.detach().numpy()
    if height <= opt.crop_height and width <= opt.crop_width:
        temp = temp[0, opt.crop_height - height: opt.crop_height, opt.crop_width - width: opt.crop_width]
    else:
        temp = temp[0, :, :]
    #skimage.io.imsave(savename, (temp * 256).astype('uint16'))
    return temp

#added by CCJ:
def get_epe_rate2(disp, prediction, max_disp = 192, threshold = 1.0):
    mask = np.logical_and(disp >= 0.001, disp <= max_disp)
    error = np.mean(np.abs(prediction[mask] - disp[mask]))
    rate = np.sum(np.abs(prediction[mask] - disp[mask]) > threshold) / np.sum(mask)
    #print(" ==> EPE Error: {:.4f}, Error Rate: {:.4f}".format(error, rate))
    return error, rate

def get_epe_rate(disp, prediction, threshold = 1.0, threshold2 = 3.0):
    #mask = np.logical_and(disp >= 0.001, disp <= max_disp)
    mask = disp >= 0.001
    error = np.mean(np.abs(prediction[mask] - disp[mask]))
    rate = np.sum(np.abs(prediction[mask] - disp[mask]) > threshold) / np.sum(mask)
    rate2 = np.sum(np.abs(prediction[mask] - disp[mask]) > threshold2) / np.sum(mask)
    #print(" ==> EPE Error: {:.4f}, Error Rate: {:.4f}".format(error, rate))
    return error, rate, rate2

   
if __name__ == "__main__":
    file_path = opt.data_path
    file_list_txt = opt.test_list

    f = open(file_list_txt, 'r')
    if opt.virtual_kitti2:
        filelist = get_virtual_kitti2_filelist(file_list_txt)
    else:
        filelist = [l.rstrip() for l in f.readlines() if not l.rstrip().startswith('#')]

    avg_err = 0
    avg_rate1 = 0
    avg_rate3 = 0
    
    if not os.path.exists(opt.resultDir):
        os.makedirs(opt.resultDir)
        print ('makedirs {}'.format(opt.resultDir))
    
    img_num = len(filelist)
    print ("[***]To test %d imgs" % img_num)
    for index in range(img_num):
        current_file = filelist[index]

        # updated by CCJ for image path:
        if opt.kitti2015:
            leftname = pjoin(file_path, 'image_0/' + current_file)
            if index < 1:
                print ("limg: {}".format(leftname))
            rightname = pjoin(file_path, 'image_1/' + current_file)
            dispname = pjoin(file_path, 'disp_occ_0_pfm/' + current_file[0:-4] + '.pfm')
            if os.path.isfile(dispname):
                dispGT = pfm.readPFM(dispname)
                dispGT[dispGT == np.inf] = .0
            else:
                dispGT= None
            savename = pjoin(opt.resultDir, current_file[0:-4] + '.pfm')
                
        elif opt.kitti2012: # KITTI 2012
            #leftname = pjoin(file_path, 'image_0/' + current_file)
            leftname =  pjoin(file_path, 'colored_0/' + current_file)
            rightname = pjoin(file_path, 'colored_1/' + current_file)
            dispname = pjoin(file_path, 'disp_occ_0_pfm/' + current_file[0:-4] + '.pfm')
            if os.path.isfile(dispname):
                dispGT = pfm.readPFM(dispname)
                dispGT[dispGT == np.inf] = .0
            else:
                dispGT= None
            savename = pjoin(opt.resultDir, current_file[0:-4] + '.pfm')
        
        elif opt.virtual_kitti2:
            data_type_str= "virtual_kt2" 
            A = current_file 
            # e.g., /media/ccjData2/datasets/Virtual-KITTI-V2/vkitti_2.0.3_rgb/Scene01/15-deg-left/frames/rgb/Camera_0/rgb_00001.jpg
            leftname = pjoin(file_path, "vkitti_2.0.3_rgb/" + A) 
            rightname = pjoin(file_path, "vkitti_2.0.3_rgb/" + A[:-22] + 'Camera_1/' + A[-13:])
            #load depth GT and change it to disparity GT: 
            depth_png_filename = pjoin(file_path, "vkitti_2.0.3_depth/" + A[:-26] + 'depth/Camera_0/depth_' + A[-9:-4] + ".png")
            #print ("imgl = ", leftname, ", imgr = ", rightname, ", depth_left = ", depth_png_filename)
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
            savename = pjoin(opt.resultDir, '%04d.pfm'%(index))

        else:
            A = current_file
            leftname = pjoin(file_path, A)
            rightname = pjoin(file_path, A[:-13] + 'right/' + A[len(A)-8:]) 
            # check disparity GT exists or not!!!
            pos = A.find('/')
            tmp_len = len('frames_finalpass')
            dispname = pjoin(file_path, A[0:pos] + '/disparity' + A[pos+1+tmp_len:-4] + '.pfm')
            #print ("[****] ldisp: {}".format(dispname))
            if os.path.isfile(dispname):
                dispGT = pfm.readPFM(dispname)
                dispGT[dispGT == np.inf] = .0
            else:
                dispGT= None
            savename = pjoin(opt.resultDir, str(index) + '.pfm')

        disp = test(leftname, rightname, savename)
        
        if any([opt.kitti2015, opt.kitti2012, index % 250 == 0]):
            pfm.save(savename, disp)
            """ disp """
            tmp_dir = pjoin(opt.resultDir, "dispColor")
            if not os.path.exists(tmp_dir):
                os.makedirs(tmp_dir)
            
            # save kt15 color
            if opt.kitti2015 or opt.kitti2012:
                tmp_dispname = pjoin(tmp_dir, current_file[0:-4] + '.png')
            else:
                tmp_dispname = pjoin(tmp_dir, '%04d.png'%(index))
            cv2.imwrite(tmp_dispname, 
                    KT15FalseClr.writeKT15FalseColor(np.ascontiguousarray(disp)).astype(np.uint8)[:,:,::-1])
            print ("save ", tmp_dispname)
            if dispGT is not None: #If KT benchmark submission, then No dispGT;
                """ err-disp """
                tmp_dir = pjoin(opt.resultDir, "errDispColor")
                if not os.path.exists(tmp_dir):
                    os.makedirs(tmp_dir)
                
                if opt.kitti2015 or opt.kitti2012:
                    tmp_errdispname = pjoin(tmp_dir, current_file[0:-4]  + '.png')
                else:
                    tmp_errdispname = pjoin(tmp_dir, '%04d.png'%(index))

                cv2.imwrite(tmp_errdispname, 
                        KT15LogClr.writeKT15ErrorDispLogColor(np.ascontiguousarray(disp), np.ascontiguousarray(dispGT)).astype(np.uint8)[:,:,::-1])
                print ("save ", tmp_errdispname)
        
        error, rate1, rate3 = get_epe_rate(dispGT, disp, threshold=1.0, threshold2=3.0)
        avg_err += error
        avg_rate1 += rate1
        avg_rate3 += rate3
        
        if index % 250 == 0:
            message_info = "===> Frame {}: ".format(index) + current_file + " ==> EPE Error: {:.4f}, Bad-{:.1f} Error: {:.4f}, Bad-{:.1f} Error: {:.4f}".format(
                error, 1.0, rate1, 3.0, rate3)
            print (message_info)
            #sys.stdout.flush()
    
    # end of test data loop
    if dispGT is not None:
        avg_err /= img_num
        avg_rate1 /= img_num
        avg_rate3 /= img_num
        print("===> Total {} Frames ==> AVG EPE Error: {:.4f}, AVG Bad-{:.1f} Error: {:.4f}, AVG Bad-{:.1f} Error: {:.4f}".format(
            img_num, avg_err, 1.0, avg_rate1, 3.0, avg_rate3))
        
        """ save as csv file, Excel file format """
        csv_file = os.path.join(opt.resultDir, 'bad-err.csv')
        print ("write ", csv_file, "\n")
        timeStamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        messg = timeStamp + ',{},bad-1.0,{:.4f},bad-3.0,{:.4f},epe,{:.4f},fileDir={},for log,{:.3f}(epe); {:.3f}%(bad1); {:.3f}%(bad3)\n'.format(
            data_type_str, avg_rate1, avg_rate3, avg_err, 
            opt.resultDir, 
            avg_err, avg_rate1*100.0, avg_rate3*100.0)
            
        with open( csv_file, 'w') as fwrite:
            fwrite.write(messg)
    
    print ('{} testing finished!'.format(opt.model))