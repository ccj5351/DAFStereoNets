# !/usr/bin/env python3
# -*-coding:utf-8-*-
# @file: main_GCNet.py
# @brief:
# @author: Changjiang Cai, ccai1@stevens.edu, caicj5351@gmail.com
# @version: 0.0.1
# @creation date: 07-01-2020
# @last modified: Mon 20 Jan 2020 03:09:30 AM EST

from __future__ import print_function
from PIL import Image
import matplotlib.pyplot as plt
import sys
import shutil
import os
from os.path import join as pjoin
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import cv2
from torch.utils.data import DataLoader
import torch.nn.functional as F

from .loaddata.data import get_training_set, get_valid_set, load_test_data, test_transform

from torch.utils.tensorboard import SummaryWriter
from src.dispColor import colormap_jet_batch_image,KT15FalseColorDisp,KT15LogColorDispErr

#from src.utils import writeKT15FalseColors # this is numpy fuction, it is SO SLOW !!!
# this is cython fuction, it is SO QUICK !!!
from src.cython import writeKT15FalseColor as KT15FalseClr
from src.cython import writeKT15ErrorLogColor as KT15LogClr
import numpy as np
import src.pfmutil as pfm
import time
from .models.loss import valid_accu3, MyLoss2
from .models.dispnet import DispNet, scale_pyramid


""" train and test DispNet """
class MyDispNet(object):
    def __init__(self, args):
        self.args = args
        self.model_name = args.model_name
        if str(self.model_name).lower() == 'dispnets': # DispNetS
            self.is_corr = False
        elif str(self.model_name).lower() == 'dispnetc': # DispNetC
            self.is_corr = True
        else:
            raise Exception("No suitable model name found ... Should be DispNetS or DispNetC")

        self.lr = args.lr
        self.kitti2012  = args.kitti2012
        self.kitti2015  = args.kitti2015
        self.checkpoint_dir = args.checkpoint_dir
        self.log_summary_step = args.log_summary_step
        self.isTestingMode = (str(args.mode).lower() == 'test')
        self.cuda = args.cuda
        self.weights1To6 = [6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
        # if KT fine-tuning, just use the loss of the last output disp1
        if self.kitti2012 or self.kitti2015:
            self.weights1To6 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        self.weights1To6 = [i/np.sum(self.weights1To6) for i in self.weights1To6]
        print("[***] weights1To6 : ", self.weights1To6)
        
        if not self.isTestingMode: # training mode
            print('===> Loading datasets')
            train_set = get_training_set(args.data_path, args.training_list, 
                    [args.crop_height, args.crop_width], 
                    args.kitti2012, args.kitti2015, args.shift, False# is_semantic
                    )
            
            self.training_data_loader = DataLoader(dataset=train_set, 
                    num_workers=args.threads, batch_size=args.batchSize, 
                    shuffle=True, drop_last=True)
            
            self.train_loader_len = len(self.training_data_loader)
            self.criterion = MyLoss2(thresh=3, alpha=2)

        
        print('===> Building {} Model'.format(self.model_name))
        # due to TWO consecutive downsampling, so here e.g., maxdisp=48,
        # actually means 4*maxdisp=192 in the original input image pair;
        # that is why we pass args.max_disp//4 to construct DispNet;
        self.model = DispNet(
                is_corr = self.is_corr,
                maxdisp = args.max_disp//4,
                corr_func_type = args.corr_func,
                is_bn = (str(args.is_bn).lower() == 'true'),
                is_relu = (str(args.is_relu).lower() == 'true'),
                )
        if self.cuda:
            self.model = torch.nn.DataParallel(self.model).cuda()
        
        if not self.isTestingMode: # training mode
            """ We need to set requires_grad == False to freeze the parameters 
                so that the gradients are not computed in backward();
                Parameters of newly constructed modules have requires_grad=True by default;
            """
            # updated for the cases where some subnetwork was forzen!!!
            params_to_update = [p for p in self.model.parameters() if p.requires_grad]
            if 0:
                print ('[****] params_to_update = ')
                for p in params_to_update:
                    print (type(p.data), p.size())

            print('[***]Number of model parameters: {}'.format(sum([p.data.nelement() for p in self.model.parameters()])))
            
            #NOTE: added by CCJ on Jan. 12, 2020; 
            """ If you need to move a model to GPU via .cuda(), please do so 
                before constructing optimizers for it. 
                Parameters of a model after .cuda() will be different 
                objects with those before the call. 
            """
            self.optimizer= optim.Adam(params_to_update, lr = args.lr, betas=(0.9,0.999))
            self.writer = SummaryWriter(args.train_logdir)
        

        if self.isTestingMode:
            assert os.path.isfile(args.resume) == True, "Model Test but NO checkpoint found at {}".format(args.resume)
        if args.resume:
            if os.path.isfile(args.resume):
                print("[***] => loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(args.resume)
                self.model.load_state_dict(checkpoint['state_dict'], strict=False)
                if not self.isTestingMode: # training mode
                    self.optimizer.load_state_dict(checkpoint['optimizer'])
            else:
                print("=> no checkpoint found at {}".format(args.resume))
        
    
    def save_checkpoint(self, epoch, state_dict, is_best=False):
        saved_checkpts = pjoin(self.checkpoint_dir, self.model_name)
        if not os.path.exists(saved_checkpts):
            os.makedirs(saved_checkpts)
            print ('makedirs {}'.format(saved_checkpts))
        
        filename = pjoin(saved_checkpts, "model_epoch_%05d.tar" % epoch)
        torch.save(state_dict, filename)
        print ('Saved checkpoint at %s' % filename) 
        if is_best:
            best_fname = pjoin(saved_checkpts, 'model_best.tar')
            shutil.copyfile(filename, best_fname)

    def adjust_learning_rate(self, epoch):
        if epoch <= 200:
            self.lr = self.args.lr
        else:
            self.lr = self.args.lr * 0.1
        
        print('learning rate = ', self.lr)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr

    def load_checkpts(self, saved_checkpts = ''):
        print(" [*] Reading checkpoint %s" % saved_checkpts)
        
        checkpoint = None
        if saved_checkpts and saved_checkpts != '':
            try: #Exception Handling
                f = open(saved_checkpts, 'rb')
            except IsADirectoryError as error:
                print (error)
            else:
                checkpoint = torch.load(saved_checkpts)
        return checkpoint

    def build_train_summaries(self, imgl, imgr, disp, disp_gt, global_step, loss, epe_err, 
            epe_err1 = None, epe_err2 = None, is_KT15Color = False):
            """ loss and epe error """
            self.writer.add_scalar(tag = 'train_loss', scalar_value = loss, global_step = global_step)
            self.writer.add_scalar(tag = 'train_err_full', scalar_value = epe_err, global_step = global_step)
            if epe_err1 is not None:
                self.writer.add_scalar(tag = 'train_err1_half', scalar_value = epe_err1, global_step = global_step)
            if epe_err2 is not None:
                self.writer.add_scalar(tag = 'train_err2_quarter', scalar_value = epe_err2, global_step = global_step)
            
            """ Add batched image data to summary:
                Note: add_images(img_tensor): img_tensor could be torch.Tensor, numpy.array, or string/blobname;
                so we could use torch.Tensor or numpy.array !!!
            """
            self.writer.add_images(tag='train_imgl',img_tensor=imgl, global_step = global_step, dataformats='NCHW')
            if imgr is not None:
                self.writer.add_images(tag='train_imgr',img_tensor=imgr, global_step = global_step, dataformats='NCHW')
            
            with torch.set_grad_enabled(False):
                if is_KT15Color:
                    disp_tmp = KT15FalseColorDisp(disp)
                    disp_gt_tmp = KT15FalseColorDisp(disp_gt)
                else:
                    disp_tmp = colormap_jet_batch_image(disp)
                    disp_gt_tmp = colormap_jet_batch_image(disp_gt)

                self.writer.add_images(tag='train_disp', img_tensor=disp_tmp, global_step = global_step, dataformats='NHWC')
                self.writer.add_images(tag='train_dispGT',img_tensor=disp_gt_tmp, global_step = global_step, dataformats='NHWC')
                self.writer.add_images(tag='train_dispErr',img_tensor=KT15LogColorDispErr(disp, disp_gt), 
                                       global_step = global_step, dataformats='NHWC')
    


    #---------------------
    #---- Training -------
    #---------------------
    def train(self, epoch):
        """Set up TensorBoard """
        epoch_loss = 0
        epoch_epe = 0
        epoch_accu3 = 0
        valid_iteration = 0

        # setting to train mode;
        self.model.train()
        self.adjust_learning_rate(epoch)

        """ running log loss """
        log_running_loss = 0.0
        log_running_err  = 0.0
        log_running_err1  = 0.0
        log_running_err2  = 0.0
        
        for iteration, batch_data in enumerate(self.training_data_loader):
            start = time.time()
            #print (" [***] iteration = %d" % iteration)
            input1 = batch_data[0].float() # False by default;
            #print ("[???] input1 require_grad = ", input1.requires_grad) # False
            input2 = batch_data[1].float()
            target = batch_data[2].float()
            left_rgb = batch_data[3].float()
            #right_rgb = batch_data[4].float()
            
            if self.cuda:
                input1 = input1.cuda()
                input2 = input2.cuda()
                target = target.cuda()

            target = torch.squeeze(target,1)
            N,H,W = target.size()[:]
            # valid pixels: 0 < disparity < max_disp
            mask = (target - args.max_disp)*target < 0
            mask.detach_()

            num_scales = 6
            #NOTE: Updated by CCJ on 2020/01/17, adding arg `is_value_scaled=True`;
            #      1) not only down-sample target in space for target1, 
            #      2) but also divide the disparity value by 2, due to down-sampled image width;
            target_pyramid = scale_pyramid(target.view(N,1,H,W), num_scales, is_value_scaled=True)
            target_pyramid = [ torch.squeeze(t, 1) for t in target_pyramid ]
            mask_pyramid = []
            for i in range(num_scales):
                ratio = 2**(i+1)
                tmp_target = target_pyramid[i]
                tmp_mask = (tmp_target - args.max_disp//ratio)*tmp_target < 0
                tmp_mask.detach_()
                mask_pyramid.append(tmp_mask)

            valid_disp = target[mask].size()[0]
            
            if valid_disp > 0:
                self.optimizer.zero_grad()
                # including disp0, disp1, ..., disp6
                #NOTE: disp0 is in original image size, disp1 is in half size, and so on!!!
                disp_results = self.model(input1, input2)
                disp0 = disp_results[0] # in original size
                disp1 = disp_results[1] # in half size 
                disp2 = disp_results[2] # in quarter size 
                target1 = target_pyramid[0] # in half size
                mask1 = mask_pyramid[0]
                target2 = target_pyramid[1] # in quarter size
                mask2 = mask_pyramid[1]
                
                loss = F.smooth_l1_loss(disp0[mask], target[mask], reduction='mean')
                for i in range(1,6):
                    loss += self.weights1To6[i]*F.l1_loss(disp_results[i+1][mask_pyramid[i]], 
                                    target_pyramid[i][mask_pyramid[i]], reduction='mean')

                if self.kitti2012 or self.kitti2015:
                    loss = 0.4*loss + 0.6*self.criterion(disp0[mask], target[mask])
                
                loss.backward()
                self.optimizer.step()
                # MAE error
                error = torch.mean(torch.abs(disp0[mask] - target[mask]))
                # in 1/2 size
                error1 = torch.mean(torch.abs(disp1[mask1] - target1[mask1]))
                # in 1/4 size
                error2 = torch.mean(torch.abs(disp2[mask2] - target2[mask2]))

                # accu3, in original size
                accu = valid_accu3(target[mask], disp0[mask])

                epoch_loss += loss.item()
                epoch_epe += error.item()
                epoch_accu3 += accu.item() 
                valid_iteration += 1
                
                # epoch - 1: here argument `epoch` is starting from 1, instead of 0 (zer0);
                train_global_step = (epoch-1)*self.train_loader_len + iteration      
                print("===> Epoch[{}]({}/{}): Step {}, Loss: {:.3f}, EPE: {:.2f}, Acu3: {:.2f}; {:.2f} s/step".format(
                                epoch, iteration, self.train_loader_len, train_global_step,
                                loss.item(), error.item(), accu.item(), time.time() -start))
                sys.stdout.flush()

                # save summary for tensorboard visualization
                log_running_loss += loss.item()
                log_running_err += error.item()
                log_running_err1 += error1.item()
                log_running_err2 += error2.item()
                
                if iteration % self.log_summary_step == (self.log_summary_step - 1):
                    self.build_train_summaries( 
                          F.interpolate(left_rgb, size=[H//2, W//2], mode='bilinear', align_corners = True), 
                          #left_rgb,
                          None, #right_rgb,
                          # in the latest versions of PyTorch you can add a new axis by indexing with None 
                          # > see: https://discuss.pytorch.org/t/what-is-the-difference-between-view-and-unsqueeze/1155;
                          #torch.unsqueeze(disp0, dim=1) ==> disp0[:,None]
                          disp1[:,None], target1[:,None],
                          train_global_step, 
                          log_running_loss/self.log_summary_step, 
                          log_running_err /self.log_summary_step, 
                          log_running_err1/self.log_summary_step, 
                          log_running_err2/self.log_summary_step, 
                          is_KT15Color = False
                          #is_KT15Color = True
                          )
                    # reset to zeros
                    log_running_loss = 0.0
                    log_running_err  = 0.0
                    log_running_err1 = 0.0
                    log_running_err2 = 0.0

        
        # end of data_loader
        # save the checkpoints
        avg_loss = epoch_loss / valid_iteration
        avg_err = epoch_epe / valid_iteration
        avg_accu = epoch_accu3 / valid_iteration
        print("===> Epoch {} Complete: Avg. Loss: {:.4f}, Avg. EPE Error: {:.4f}, Accu3: {:.4f})".format(
                  epoch, avg_loss, avg_err, avg_accu))

        is_best = False
        model_state_dict = {
                        'epoch': epoch,
                        'state_dict': self.model.state_dict(),
                        'optimizer' : self.optimizer.state_dict(),
                        'loss': avg_loss,
                        'epe_err': avg_err, 
                        'accu3': avg_accu
                    }

        if self.kitti2012 or self.kitti2015:
            #if epoch % 50 == 0 and epoch >= 300:
            #if epoch % 50 == 0:
            if epoch % 25 == 0:
                self.save_checkpoint(epoch, model_state_dict, is_best)
        else:
            #if epoch >= 7:
            #    self.save_checkpoint(epoch, model_state_dict, is_best)
            self.save_checkpoint(epoch, model_state_dict, is_best)
        # avg loss
        return avg_loss, avg_err, avg_accu


    #---------------------
    #---- Test ----- -----
    #---------------------
    def test(self):
        self.model.eval()
        file_path = self.args.data_path
        file_list = self.args.test_list
        f = open(file_list, 'r')
        filelist = [l.rstrip() for l in f.readlines()]
        crop_width = self.args.crop_width
        crop_height = self.args.crop_height

        if not os.path.exists(self.args.resultDir):
            os.makedirs(self.args.resultDir)
            print ('makedirs {}'.format(self.args.resultDir))
        
        avg_err = 0
        avg_rate = 0
        for index in range(len(filelist)):
        #for index in range(1):
            current_file = filelist[index]
            if self.kitti2015:
                leftname = pjoin(file_path, 'image_0/' + current_file)
                if index % 20 == 0:
                    print ("limg: {}".format(leftname))
                rightname = pjoin(file_path, 'image_1/' + current_file)
                dispname = pjoin(file_path, 'disp_occ_0_pfm/' + current_file[0:-4] + '.pfm')
                if os.path.isfile(dispname):
                    dispGT = pfm.readPFM(dispname)
                    dispGT[dispGT == np.inf] = .0
                else:
                    dispGT= None
                savename = pjoin(self.args.resultDir, current_file[0:-4] + '.pfm')
                
            elif self.kitti2012:
                #leftname = pjoin(file_path, 'image_0/' + current_file)
                leftname =  pjoin(file_path, 'colored_0/' + current_file)
                rightname = pjoin(file_path, 'colored_1/' + current_file)
                dispname = pjoin(file_path, 'disp_occ_0_pfm/' + current_file[0:-4] + '.pfm')
                if os.path.isfile(dispname):
                    dispGT = pfm.readPFM(dispname)
                    dispGT[dispGT == np.inf] = .0
                else:
                    dispGT= None
                savename = pjoin(self.args.resultDir, current_file[0:-4] + '.pfm')
                #disp = Image.open(dispname)
                #disp = np.asarray(disp) / 256.0

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
                savename = pjoin(self.args.resultDir, str(index) + '.pfm')

            input1, input2, height, width = test_transform(
                    load_test_data(leftname, rightname), crop_height, crop_width)
                
            if self.cuda:
                input1 = input1.cuda()
                input2 = input2.cuda()
            with torch.no_grad():
                prediction1, prediction = self.model(input1, input2)
            
            disp = prediction.cpu().detach().numpy() # in full size
            disp1 = prediction1.cpu().detach().numpy() # in half size
            if height <= crop_height and width <= crop_width:
                disp = disp[0, crop_height - height: crop_height, crop_width-width: crop_width]
                disp1 = disp1[0, crop_height//2 - height//2: crop_height//2, crop_width//2-width//2: crop_width//2]
            else:
                disp = disp[0, :, :]
                disp1 = disp1[0,:,:]
            
            #skimage.io.imsave(savename, (disp * 256).astype('uint16'))
            if self.kitti2015 or self.kitti2012 or index %50 == 0:
                pfm.save(savename, disp)
                #print ('saved ', savename)
                if 0 and dispGT is not None:
                    left = np.asarray(Image.open(leftname))[:,:,:3].astype(np.float32) 
                    right = np.asarray(Image.open(rightname))[:,:,:3].astype(np.float32)
                    #print ("[???]", left.shape)
                    pfm.save(savename[:-4] + '-iml.pfm', left)
                    pfm.save(savename[:-4] + '-imr.pfm', right)
                    pfm.save(savename[:-4] + '-gt.pfm', dispGT.astype(np.float32))
                    pfm.save(savename[:-4] + '-half.pfm', disp1)
            
            if dispGT is not None:
                error, rate = get_epe_rate(dispGT, disp, self.args.max_disp, 
                        self.args.threshold)
                avg_err += error
                avg_rate += rate
                if index % 20 == 0:
                    print("===> Frame {}: ".format(index) + leftname + " ==> EPE Error: {:.4f}, Bad-{:.1f} Error: {:.4f}".format(
                        error, self.args.threshold, rate))
            
            # save kt15 color
            if self.kitti2015:
                """ disp """
                tmp_dir = pjoin(self.args.resultDir, "dispColor")
                if not os.path.exists(tmp_dir):
                    os.makedirs(tmp_dir)
                tmp_dispname = pjoin(tmp_dir, current_file[0:-4] + '.png')
                cv2.imwrite(tmp_dispname, 
                        KT15FalseClr.writeKT15FalseColor(np.ascontiguousarray(disp)).astype(np.uint8)[:,:,::-1])
                if index % 20 == 0:
                    print ('saved ', tmp_dispname)
                if dispGT is not None: #If KT benchmark submission, then No dispGT;
                    """ err-disp """
                    tmp_dir = pjoin(self.args.resultDir, "errDispColor")
                    if not os.path.exists(tmp_dir):
                        os.makedirs(tmp_dir)
                    tmp_errdispname = pjoin(tmp_dir, current_file[0:-4]  + '.png')
                    cv2.imwrite(tmp_errdispname, 
                            KT15LogClr.writeKT15ErrorDispLogColor(np.ascontiguousarray(disp), np.ascontiguousarray(dispGT)).astype(np.uint8)[:,:,::-1])
                    if index % 20 == 0:
                        print ('saved ', tmp_errdispname)

        if dispGT is not None:
            avg_err = avg_err / len(filelist)
            avg_rate = avg_rate / len(filelist)
            print("===> Total {} Frames ==> AVG EPE Error: {:.4f}, AVG Bad-{:.1f} Error: {:.4f}".format(
                len(filelist), avg_err, self.args.threshold, avg_rate))




def get_epe_rate(disp, prediction, max_disp = 192, threshold = 3.0):
    mask = np.logical_and(disp >= 0.001, disp <= max_disp)
    error = np.mean(np.abs(prediction[mask] - disp[mask]))
    rate = np.sum(np.abs(prediction[mask] - disp[mask]) > threshold) / np.sum(mask)
    #print(" ==> EPE Error: {:.4f}, Error Rate: {:.4f}".format(error, rate))
    return error, rate



def main(args):
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    
    #----------------------------
    # some initilization thing 
    #---------------------------
    cuda = args.cuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")
    torch.manual_seed(args.seed)
    if cuda:
        torch.cuda.manual_seed(args.seed)
    
    myNet = MyDispNet(args)
    """ for debugging """
    if args.mode == 'debug':
        myNet.model.train()
        import gc
        crop_h = 256
        crop_w = 512
        #x_l = torch.randn((1, 3, crop_h, crop_w), requires_grad=True)
        #x_r = torch.randn((1, 3, crop_h, crop_w), requires_grad=True)
        x_l = torch.randn((1, 3, crop_h, crop_w)).cuda()
        x_r = torch.randn((1, 3, crop_h, crop_w)).cuda()
        y = torch.randn((1, crop_h, crop_w)).cuda()
        z = torch.randn((1, 1, crop_h//3, crop_w//3)).cuda()

        
        from pytorch_memlab import profile, MemReporter
        # pass in a model to automatically infer the tensor names
        # You can better understand the memory layout for more complicated module
        if 1:
            reporter = MemReporter(myNet.model)
            disp = myNet.model(x_l, x_r)
            loss = F.smooth_l1_loss(disp, y, reduction='mean')
            reporter.report(verbose=True)
            print('========= before backward =========')
            loss.backward()
            reporter.report(verbose=True)

        # generate prof which can be loaded by Google chrome trace at chrome://tracing/
        if 1:
            with torch.autograd.profiler.profile(use_cuda=True) as prof:
                myNet.model(x_l, x_r)
            print(prof)
            prof.export_chrome_trace('./results/tmp/prof.out')
    
    if args.mode == 'train':
        print('strat training !!!')
        for epoch in range(1 + args.startEpoch, args.startEpoch + args.nEpochs + 1):
            print ("[**] do training at epoch %d/%d" % (epoch, args.startEpoch + args.nEpochs))

            with torch.autograd.set_detect_anomaly(True):
                avg_loss, avg_err, avg_accu = myNet.train(epoch)
        # save the last epoch always!!
        myNet.save_checkpoint(args.nEpochs + args.startEpoch,
            {
                'epoch': args.nEpochs + args.startEpoch,
                'state_dict': myNet.model.state_dict(),
                'optimizer' : myNet.optimizer.state_dict(),
                'loss': avg_loss,
                'epe_err': avg_err, 
                'accu3': avg_accu
            }, 
            is_best = False)
        print('done training !!!')
    
    if args.mode == 'test': 
        print('strat testing !!!')
        myNet.test()


if __name__ == '__main__':
    
    import argparse
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch GANet Example')
    parser.add_argument('--crop_height', type=int, required=True, help="crop height")
    parser.add_argument('--max_disp', type=int, default=192, help="max disp")
    parser.add_argument('--crop_width', type=int, required=True, help="crop width")
    parser.add_argument('--resume', type=str, default='', help="resume from saved model")
    parser.add_argument('--batchSize', type=int, default=1, help='training batch size')
    parser.add_argument('--log_summary_step', type=int, default=200, help='every 200 steps to build training summary')
    parser.add_argument('--nEpochs', type=int, default=400, help='number of epochs to train for')
    parser.add_argument('--startEpoch', type=int, default=0, help='starting point, used for fine-tuning')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate. Default=0.001')
    parser.add_argument('--cuda', type=int, default=1, help='use cuda? Default=True')
    parser.add_argument('--threads', type=int, default=1, help='number of threads for data loader to use')
    parser.add_argument('--seed',  type=int, default=123, help='random seed to use. Default=123')
    parser.add_argument('--shift', type=int, default=0, help='random shift of left image. Default=0')
    parser.add_argument('--kitti2012', type=int, default=0, help='kitti 2012 dataset? Default=False')
    parser.add_argument('--kitti2015', type=int, default=0, help='kitti 2015? Default=False')
    parser.add_argument('--data_path', type=str, default='/data/ccjData', help="data root")
    parser.add_argument('--training_list', type=str, default='./lists/sceneflow_train.list', help="training list")
    parser.add_argument('--test_list', type=str, default='./lists/sceneflow_test_select.list', help="evaluation list")
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint/', help="location to save models")
    parser.add_argument('--train_logdir', dest='train_logdir',  default='./logs/tmp', help='log dir')
    """Arguments related to run mode"""
    parser.add_argument('--model_name', type=str, default='DispNetC', help="model name")
    parser.add_argument('--mode', dest='mode', type = str, default='train', help='train, test')
    parser.add_argument('--resultDir', type=str, default= "./results")
    parser.add_argument('--is_bn', dest='is_bn', type = str, default='true', help='using BN or not')
    parser.add_argument('--is_relu', dest='is_relu', type = str, default='true', help='using ReLU or not')
    parser.add_argument('--corr_func', dest='corr_func', type = str, default='correlation1D_map_V1', help='corr1D function type')
    parser.add_argument('--threshold', type=float, default=3.0, help="threshold of error rates")

    args = parser.parse_args()
    print('[***] args = ', args)
    main(args)
