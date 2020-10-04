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
import numpy as np
import time
import math
#from dataloader import KITTIloader2015 as ls
from dataloader import KITTILoader as DA
from os.path import join as pjoin

from models import *
import copy
from torch.utils.tensorboard import SummaryWriter
from dispColor import colormap_jet_batch_image,KT15FalseColorDisp,KT15LogColorDispErr
from utils import pfmutil as pfm

parser = argparse.ArgumentParser(description='PSMNet')
parser.add_argument('--maxdisp', type=int, default=192,
                    help='maxium disparity')
parser.add_argument('--batch_size', type=int, default=6, 
                    help='maxium disparity')
parser.add_argument('--model', default='stackhourglass',
                    help='select model')
parser.add_argument('--datatype', default='2015')
parser.add_argument('--datapath', default='/media/jiaren/ImageNet/data_scene_flow_2015/training/',
                    help='datapath')
parser.add_argument('--epochs', type=int, default=300,
                    help='number of epochs to train')
parser.add_argument('--loadmodel', default='./trained/submission_model.tar',
                    help='load model')
parser.add_argument('--savemodel', default='./',
                    help='save model')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--kt15_train_list', type=str, 
        default='/home/ccj/atten-stereo/lists/kitti2015_train170.list', help="training list")
parser.add_argument('--kt15_val_list', type=str, 
        default='/home/ccj/atten-stereo/lists/kitti2015_val30.list', help="validation list")
parser.add_argument('--kt12_train_list', type=str, 
        default='/home/ccj/atten-stereo/lists/kitti2012_train170.list', help="training list")
parser.add_argument('--kt12_val_list', type=str, 
        default='/home/ccj/atten-stereo/lists/kitti2012_val24.list', help="validation list")

parser.add_argument('--vkt2_train_list', type=str, 
        default='/home/ccj/atten-stereo/lists/virtual_kitti2_wo_scene06_fixed_train.list', help="training list")
parser.add_argument('--vkt2_val_list', type=str, 
        default='/home/ccj/atten-stereo/lists/virtual_kitti2_wo_scene06_fixed_test.list', help="validation list")

#newly added by CCJ on 2020/05/23;
parser.add_argument('--train_logdir', dest='train_logdir',  default='./logs/tmp', help='log dir')
parser.add_argument('--resultDir', type=str, default= "./results")
parser.add_argument('--log_summary_step', type=int, default=200, help='every 200 steps to build training summary')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if args.datatype == '2015':
    print ("processing KT15!")
    from dataloader import KITTIloader2015 as ls
    train_file_list = args.kt15_train_list
    val_file_list = args.kt15_val_list
    virtual_kitti2 = False

elif args.datatype == '2012':
    print ("processing KT12!")
    from dataloader import KITTIloader2012 as ls
    train_file_list = args.kt12_train_list
    val_file_list = args.kt12_val_list
    virtual_kitti2 = False

# added by CCJ on 2020/05/22:
elif args.datatype == 'virtual_kt_2':
    print ("processing Virtual KT 2!")
    from dataloader import KITTIloader_VirtualKT2 as ls
    train_file_list = args.vkt2_train_list
    val_file_list = args.vkt2_val_list
    virtual_kitti2 = True

else:
    raise Exception("No suitable KITTI found ...")


print ('[??] args.datapath = ', args.datapath)
all_left_img, all_right_img, all_left_disp, test_left_img, \
        test_right_img, test_left_disp = ls.dataloader(
                args.datapath, train_file_list, val_file_list)

TrainImgLoader = torch.utils.data.DataLoader(
    DA.myImageFloder(all_left_img, all_right_img, all_left_disp, training=True, virtual_kitti2=virtual_kitti2),
    batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=False)

TestImgLoader = torch.utils.data.DataLoader(
    DA.myImageFloder(test_left_img, test_right_img, test_left_disp, training=False, virtual_kitti2=virtual_kitti2),
    batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=False)

if args.model == 'stackhourglass':
    model = stackhourglass(args.maxdisp)
elif args.model == 'basic':
    model = basic(args.maxdisp)
else:
    print('no model')

if args.cuda:
    model = nn.DataParallel(model)
    model.cuda()

if args.loadmodel is not None:
    state_dict = torch.load(args.loadmodel)
    model.load_state_dict(state_dict['state_dict'])
    print ('[***]Successfully loaded model ', args.loadmodel)

#added by CCJ;
log_summary_step = args.log_summary_step

print('Number of PSMNet model parameters: {}'.format(
    sum([p.data.nelement() for p in model.parameters()])))

optimizer = optim.Adam(model.parameters(), lr=0.1, betas=(0.9, 0.999))

#added by CCJ:
writer = SummaryWriter(args.train_logdir)
def build_train_summaries(imgl, imgr, disp, disp_gt, global_step, loss, epe_err, is_KT15Color = False):
    """ loss and epe error """
    writer.add_scalar(tag = 'train_loss', scalar_value = loss, global_step = global_step)
    writer.add_scalar(tag = 'train_err', scalar_value = epe_err, global_step = global_step)
    
    """ Add batched image data to summary:
        Note: add_images(img_tensor): img_tensor could be torch.Tensor, numpy.array, or string/blobname;
        so we could use torch.Tensor or numpy.array !!!
    """
    writer.add_images(tag='train_imgl',img_tensor=imgl, global_step = global_step, dataformats='NCHW')
    if imgr is not None:
        writer.add_images(tag='train_imgr',img_tensor=imgr, global_step = global_step, dataformats='NCHW')
    
    with torch.set_grad_enabled(False):
        if is_KT15Color:
            disp_tmp = KT15FalseColorDisp(disp)
            disp_gt_tmp = KT15FalseColorDisp(disp_gt)
        else:
            disp_tmp = colormap_jet_batch_image(disp)
            disp_gt_tmp = colormap_jet_batch_image(disp_gt)

        writer.add_images(tag='train_disp', img_tensor=disp_tmp, global_step = global_step, dataformats='NHWC')
        writer.add_images(tag='train_dispGT',img_tensor=disp_gt_tmp, global_step = global_step, dataformats='NHWC')
        writer.add_images(tag='train_dispErr',img_tensor=KT15LogColorDispErr(disp, disp_gt), 
                                global_step = global_step, dataformats='NHWC')

def train(imgL, imgR, disp_L):
    model.train()
    #imgL = Variable(torch.FloatTensor(imgL))
    #imgR = Variable(torch.FloatTensor(imgR))
    #disp_L = Variable(torch.FloatTensor(disp_L))
    #updated by CCJ:
    imgL = torch.FloatTensor(imgL)
    imgR = torch.FloatTensor(imgR)
    disp_L = torch.FloatTensor(disp_L)

    if args.cuda:
        imgL, imgR, disp_true = imgL.cuda(), imgR.cuda(), disp_L.cuda()

    # ---------
    mask = (disp_true > 0)
    mask.detach_()
    # ----

    optimizer.zero_grad()

    if args.model == 'stackhourglass':
        output1, output2, output3 = model(imgL, imgR)
        output1 = torch.squeeze(output1, 1)
        output2 = torch.squeeze(output2, 1)
        output3 = torch.squeeze(output3, 1)
        disp = output3
        # updated by CCJ due to the deprecated warning;
        # change size_average=True ==> reduction='mean';
        loss = 0.5*F.smooth_l1_loss(output1[mask], disp_true[mask], 
                #size_average=True
                reduction='mean'
                ) + 0.7*F.smooth_l1_loss(output2[mask], disp_true[mask], 
                reduction='mean') + F.smooth_l1_loss(output3[mask], disp_true[mask], 
                reduction='mean')
        
    elif args.model == 'basic':
        output = model(imgL, imgR)
        output = torch.squeeze(output3, 1)
        disp = output
        loss = F.smooth_l1_loss(
            output3[mask], disp_true[mask], reduction='mean')

    loss.backward()
    optimizer.step()
    # MAE error
    error = torch.mean(torch.abs(disp[mask] - disp_true[mask]))
    #return loss.data[0]
    #updated by CCJ:
    return disp, loss.item(), error.item()


def test(imgL, imgR, disp_true):
    model.eval()
    imgL = Variable(torch.FloatTensor(imgL))
    imgR = Variable(torch.FloatTensor(imgR))
    if args.cuda:
        imgL, imgR = imgL.cuda(), imgR.cuda()

    with torch.no_grad():
        output3 = model(imgL, imgR)

    pred_disp = output3.cpu().numpy()
    disp_true = disp_true.cpu().numpy()
    
    error = 0
    rate1 = 0
    rate3 = 0

    if virtual_kitti2: # added by CCJ;
        error, rate1, rate3 = get_epe_rate(disp_true, pred_disp, threshold=1.0, threshold2=3.0)
    else:
        # for KT15 and KT12 evaluation;
        #computing 3-px error#
        true_disp = disp_true
        index = np.argwhere(true_disp > 0)
        disp_true[index[0][:], index[1][:], index[2][:]] = np.abs(
            true_disp[index[0][:], index[1][:], index[2][:]]-pred_disp[index[0][:], index[1][:], index[2][:]])
        correct = (disp_true[index[0][:], index[1][:], index[2][:]] < 3) | (
            disp_true[index[0][:], index[1][:], index[2][:]] < true_disp[index[0][:], index[1][:], index[2][:]]*0.05)
        torch.cuda.empty_cache()
        rate3 = 1-(float(torch.sum(correct))/float(len(index[0])))
    
    return error, rate1, rate3

#added by CCJ, used to evaluate Virtual KITTI 2 only;
def get_epe_rate2(disp, prediction, max_disp = 192, threshold = 1.0, threshold2 = 3.0):
    mask = np.logical_and(disp >= 0.001, disp <= max_disp)
    error = np.mean(np.abs(prediction[mask] - disp[mask]))
    rate = np.sum(np.abs(prediction[mask] - disp[mask]) > threshold) / np.sum(mask)
    rate2 = np.sum(np.abs(prediction[mask] - disp[mask]) > threshold2) / np.sum(mask)
    #print(" ==> EPE Error: {:.4f}, Error Rate: {:.4f}".format(error, rate))
    return error, rate, rate2

#added by CCJ, used to evaluate Virtual KITTI 2 only;
def get_epe_rate(disp, prediction, threshold = 1.0, threshold2 = 3.0):
    #mask = np.logical_and(disp >= 0.001, disp <= max_disp)
    mask = (disp >= 0.001)
    #pfm.show(disp[0], title="dispGT")
    #pfm.show(prediction[0], title="disp")
    #print (disp.shape, mask.shape)
    #print (np.sum(mask))
    error = np.mean(np.abs(prediction[mask] - disp[mask]))
    rate = np.sum(np.abs(prediction[mask] - disp[mask]) > threshold) / np.sum(mask)
    rate2 = np.sum(np.abs(prediction[mask] - disp[mask]) > threshold2) / np.sum(mask)
    #print(" ==> EPE Error: {:.4f}, Error Rate: {:.4f}".format(error, rate))
    return error, rate, rate2


def adjust_learning_rate(optimizer, epoch):
    if epoch <= 200:
        lr = 0.001
    else:
        lr = 0.0001
    print('lr = %f' % lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
    
    #newly added by CCJ:
    best_err = 100
    best_epo = 0
    best_model_wts = copy.deepcopy(model.state_dict())
    
    start_full_time = time.time()
    train_loader_len = len(TrainImgLoader)
    if args.epochs <= 20:
        save_epoch_step = 1
    elif 20 < args.epochs < 50:
        save_epoch_step = 5
    else:
        save_epoch_step = 25
    print ("[***] save_epoch_step = ", save_epoch_step)
    for epoch in range(1, args.epochs+1):
        total_train_loss = 0
        #total_test_loss = 0
        total_test_epe = 0
        total_test_rate1 = 0
        total_test_rate3 = 0

        adjust_learning_rate(optimizer, epoch)
        """ running log loss """
        log_running_loss = 0.0
        log_running_err  = 0.0

        ## training ##
        for batch_idx, (imgL_crop, imgR_crop, disp_crop_L, left_rgb) in enumerate(TrainImgLoader):
            start_time = time.time()
            disp, loss, error = train(imgL_crop, imgR_crop, disp_crop_L)
            #print('Iter %d training loss = %.3f , time = %.2f' %
            #      (batch_idx, loss, time.time() - start_time))
            total_train_loss += loss
            
            #added by CCJ:
            # save summary for tensorboard visualization
            log_running_loss += loss
            log_running_err += error
            # epoch - 1: here argument `epoch` is starting from 1, instead of 0 (zer0);
            train_global_step = (epoch-1)*train_loader_len + batch_idx
            
            if batch_idx % log_summary_step == (log_summary_step - 1):
                build_train_summaries(
                    torch.FloatTensor(left_rgb),
                    None, #right_rgb,
                    # in the latest versions of PyTorch you can add a new axis by indexing with None 
                    # > see: https://discuss.pytorch.org/t/what-is-the-difference-between-view-and-unsqueeze/1155;
                    #torch.unsqueeze(disp0, dim=1) ==> disp0[:,None]
                    disp[:,None], 
                    disp_crop_L[:,None],
                    train_global_step, 
                    log_running_loss/log_summary_step, 
                    log_running_err/log_summary_step, 
                    is_KT15Color = False
                    #is_KT15Color = True
                    )
                # reset to zeros
                log_running_loss = 0.0
                log_running_err = 0.0

            # print the loss info
            if batch_idx % 10 == 9:
                time_elapsed = time.time() - start_time 
                est_left_time = time_elapsed * \
                        (train_loader_len - 1 - batch_idx)/3600.0
                print('===> Epoch[%d](Iter [%5d/%5d]): Step %d, loss = %.3f, time = %.2f/step, EST = %.2f Hrs' % (
                    epoch, batch_idx, train_loader_len, train_global_step, loss, time_elapsed, est_left_time))

        print('epoch %d / %d total training loss = %.3f' %
              (epoch, args.epochs, total_train_loss/train_loader_len))

        ## Test ##
        for batch_idx, (imgL, imgR, disp_L) in enumerate(TestImgLoader):
            #test_loss = test(imgL, imgR, disp_L)
            #print('Iter %d 3-px error in val = %.3f' %(batch_idx, test_loss*100))
            #total_test_loss += test_loss
            test_epe, test_rate1, test_rate3 = test(imgL, imgR, disp_L)
            total_test_epe += test_epe
            total_test_rate1 += test_rate1
            total_test_rate3 += test_rate3

        #print('epoch %d / %d total 3-px error in val = %.3f' %
        #      (epoch, args.epochs, total_test_loss/len(TestImgLoader)*100))
        test_imgs_num = len(TestImgLoader)
        print('epoch {} / {} Validation Avg EPE Error: {:.4f}, Bad-{:.1f} Error: {:.4f}, Bad-{:.1f} Error: {:.4f}'.format(
            epoch, args.epochs, total_test_epe/test_imgs_num, 
            1.0, total_test_rate1/test_imgs_num, 3.0, total_test_rate3/test_imgs_num))
        
        """ validation epe error """
        writer.add_scalar(tag = 'valid_bad3', scalar_value = total_test_rate3/test_imgs_num, global_step = train_global_step)
        if total_test_epe > 0:
            writer.add_scalar(tag = 'valid_epe', scalar_value = total_test_epe/test_imgs_num, global_step = train_global_step)
        if total_test_rate1 > 0:
            writer.add_scalar(tag = 'valid_bad1', scalar_value = total_test_rate1/test_imgs_num, global_step = train_global_step)
        
        #if total_test_loss/len(TestImgLoader)*100 < best_err:
        if total_test_rate3/test_imgs_num*100 < best_err:
            best_err = total_test_rate3/test_imgs_num*100
            best_epo = epoch
            #newly added by CCJ:
            best_model_wts = copy.deepcopy(model.state_dict())
        print('Current Best epoch %d total test bad-3.0 error = %.3f' % (best_epo, best_err))

        # SAVE
        if epoch % save_epoch_step == 0:
            savefilename = pjoin(args.savemodel, "model_epoch_%05d.tar" % epoch)
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'train_loss': total_train_loss/len(TrainImgLoader),
                'test_epe': total_test_epe/test_imgs_num,
                'test_bad1': total_test_rate1/test_imgs_num*100,
                'test_bad3': total_test_rate3/test_imgs_num*100,
            }, savefilename)
    
    """ saving the best model weights """
    # load best model weights
    model.load_state_dict(best_model_wts)
    savefilename = pjoin(args.savemodel, "best_model_epoch_%05d_valerr_%.4f.tar" % (best_epo, best_err))
    torch.save({
        'epoch': best_epo,
        'state_dict': model.state_dict(),
        'train_loss': -1,
        'test_epe': total_test_epe/test_imgs_num,
        'test_bad1': total_test_rate1/test_imgs_num*100,
        'test_bad3': best_err,
    }, savefilename)
    print ("[***] best model saved at ", savefilename)

    print('full finetune time = %.2f HR' %
          ((time.time() - start_full_time)/3600))
    print('Final best_epo = ', best_epo)
    print('Final best_bad3 = ', best_err)


if __name__ == '__main__':
    main()
