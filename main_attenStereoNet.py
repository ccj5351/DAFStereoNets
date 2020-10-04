# !/usr/bin/env python3
# -*-coding:utf-8-*-
# @file: train_attenStereoNet.py
# @brief:
# @author: Changjiang Cai, ccai1@stevens.edu, caicj5351@gmail.com
# @version: 0.0.1
# @creation date: 17-10-2019
# @last modified: Sat 08 Aug 2020 02:00:13 AM EDT

from __future__ import print_function
from math import log10
import math

from src.baselines.GANet.libs.GANet.modules.GANet import MyLoss2
import sys
import shutil
import os
from os.path import join as pjoin
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import cv2

from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F

from src.loaddata.data import get_training_set, load_test_data, test_transform
from src.loaddata.dataset import get_virtual_kitti2_filelist

from torch.utils.tensorboard import SummaryWriter

#from src.utils import writeKT15FalseColors # this is numpy fuction, it is SO SLOW !!!
# this is cython fuction, it is SO QUICK !!!
from src.cython import writeKT15FalseColor as KT15FalseClr
from src.cython import writeKT15ErrorLogColor as KT15LogClr
#combine them to the following:
from src.dispColor import colormap_jet_batch_image,KT15FalseColorDisp,KT15LogColorDispErr

import numpy as np
import src.pfmutil as pfm
from src.modules.embednetwork import get_embed_losses
import time
import json
from datetime import datetime
import random

""" compressed embeddings to k (e.g., k = 3) dimensions by PCA for visualization """
def pca_embedding(embedding, k = 3, isChanelLast = True):
    #print( "[***] embedding shape = ", embedding.shape)
    embedding = embedding.permute(0,2,3,1)
    embedding = embedding.contiguous()
    N, H, W, C = embedding.size()[:]
    #print( "[***] permute embedding shape = ", embedding.shape)
    embedding = embedding.view(-1, C)
    u, s, _ = torch.svd(embedding)
    #print( "[***] u shape = ", u.shape, " s shape = ", s.shape)

    # first k singular values;
    s_k = s[0:k] # in shape [k,k]
    u_k = u[:,0:k] # in shape [-1, k]
    output = torch.mm(u_k, torch.diag(s_k)).view(N,H,W,k)
    if not isChanelLast:
        output = output.permute(0, 3, 1, 2)
    return output



""" ASN: Attension Stereomatching Network """
class attenStereoNet(object):
    def __init__(self, args):
        self.args = args
        self.max_disp = args.max_disp
        self.model_name = args.model_name
        #self.isFreezeEmbed = (str(args.isFreezeEmbed) == 'true')
        #self.is_embed = str(args.is_embed).lower() == 'true'
        self.lr = args.lr
        self.kitti2012  = args.kitti2012
        self.kitti2015  = args.kitti2015
        self.virtual_kitti2 = args.virtual_kitti2
        self.checkpoint_dir = args.checkpoint_dir
        self.log_summary_step = args.log_summary_step
        self.isTestingMode = (str(args.mode).lower() == 'test')
        self.is_semantic = (str(args.is_semantic).lower() == 'true')
        self.cost_filter_grad = (str(args.cost_filter_grad).lower() == 'true')
        self.is_quarter_size_cost_volume_gcnet = str(args.is_quarter_size_cost_volume_gcnet).lower() == 'true'
        # newly added for lr schedule, especially for DFN+PSM;
        #self.is_fixed_lr = str(args.is_fixed_lr).lower() == 'true'
        self.lr_adjust_epo_thred = args.lr_adjust_epo_thred
        self.lr_scheduler = str(args.lr_scheduler).lower()
        self.lr_epoch_steps = [int(i) for i in str(args.lr_epoch_steps).split("-")] if args.lr_epoch_steps is not "" else []

        #self.is_kt12_gray = (str(args.is_kt12_gray).lower() == 'true')
        self.kt12_image_mode = str(args.kt12_image_mode).lower()
        self.is_data_augment = str(args.is_data_augment).lower() == 'true'
        #print ("[***] is_fixed_lr = ", self.is_fixed_lr)
        print ("[***] is_data_augment = ", self.is_data_augment)
        # I find complicated data_augment is not helpful here;
        assert self.is_data_augment == False
        
        if self.kitti2012:
            self.is_semantic = False
            #update args for saving it to a json file;
            args.is_semantic = 'false'
            print("[***]processing kitti2012 {} images, and maunally setting self.is_semantic = {}".format(
                self.kt12_image_mode, self.is_semantic))
        

        if self.model_name == 'ASN-Embed-GANet-Deep':
            from src.modules.attenStereoNet_embed_ganet_deep import AttenStereoNet
        elif self.model_name == 'ASN-Embed-GANet11': # i.e., GANet-11
            from src.modules.attenStereoNet_embed_ganet11 import AttenStereoNet
        elif self.model_name == 'ASN-Embed-PSM':
            from src.modules.attenStereoNet_embed_psm import AttenStereoNet
        elif self.model_name == 'ASN-Embed-GCNet':
            from src.modules.attenStereoNet_embed_gcnet import AttenStereoNet
        #elif self.model_name == 'ASN-Embed-DispNetC-V0':
        #    from src.modules.attenStereoNet_embed_dispnetc_v0 import AttenStereoNet
        elif self.model_name == 'ASN-Embed-DispNetC':
            from src.modules.attenStereoNet_embed_dispnetc import AttenStereoNet
        elif self.model_name == 'ASN-DFN-DispNetC':
            from src.modules.attenStereoNet_dfn_dispnetc import AttenStereoNet
        elif self.model_name == 'ASN-DFN-PSM':
            from src.modules.attenStereoNet_dfn_psm import AttenStereoNet
        elif self.model_name == 'ASN-DFN-GCNet':
            from src.modules.attenStereoNet_dfn_gcnet import AttenStereoNet
        elif self.model_name == 'ASN-DFN-GANet-Deep':
            from src.modules.attenStereoNet_dfn_ganet_deep import AttenStereoNet
        elif self.model_name == 'ASN-PAC-GANet-Deep':
            from src.modules.attenStereoNet_pac_ganet_deep import AttenStereoNet
        elif self.model_name == 'ASN-PAC-PSM':
            from src.modules.attenStereoNet_pac_psm import AttenStereoNet
        elif self.model_name == 'ASN-PAC-GCNet':
            from src.modules.attenStereoNet_pac_gcnet import AttenStereoNet
        elif self.model_name == 'ASN-PAC-DispNetC':
            from src.modules.attenStereoNet_pac_dispnetc import AttenStereoNet
        elif self.model_name == 'ASN-SGA-PSM':
            from src.modules.attenStereoNet_sga_psm import AttenStereoNet
        elif self.model_name == 'ASN-SGA-GCNet':
            from src.modules.attenStereoNet_sga_gcnet import AttenStereoNet
        elif self.model_name == 'ASN-SGA-DispNetC':
            from src.modules.attenStereoNet_sga_dispnetc import AttenStereoNet
        else:
            raise Exception("No suitable model found ...")
        
        self.cuda = args.cuda
        
        if not self.isTestingMode: # training mode
            print('===> Loading datasets')
            train_set = get_training_set(args.data_path, args.training_list, 
                    [args.crop_height, args.crop_width], 
                    args.kitti2012, args.kitti2015, args.virtual_kitti2,
                    args.shift, 
                    self.is_semantic, 
                    #self.is_kt12_gray
                    self.kt12_image_mode,
                    self.is_data_augment
                    )
            
            self.training_data_loader = DataLoader(dataset=train_set, 
                    num_workers=args.threads, batch_size=args.batchSize, 
                    shuffle=True, drop_last=True)
            
            self.train_loader_len = len(self.training_data_loader)
            self.criterion = MyLoss2(thresh=3, alpha=2)
    
            if not os.path.exists(self.checkpoint_dir):
                os.makedirs(self.checkpoint_dir)
    

        
        print('===> Building model')
        if self.model_name.find('GCNet') != -1:# including `GCNet`
            my_kwargs = {
                'is_kendall_version': str(args.is_kendall_version).lower() == 'true',
                'is_quarter_size_cost_volume_gcnet': self.is_quarter_size_cost_volume_gcnet,
            }
        else:
            my_kwargs = {}
        
        if self.model_name in ['ASN-DFN-PSM','ASN-DFN-DispNetC', 'ASN-DFN-GANet-Deep', 'ASN-DFN-GCNet']:
            print ('[!!!] loading ASN-DFN-X net')
            self.is_dfn = str(args.is_dfn).lower() == 'true'
            self.dfn_kernel_size = args.dfn_kernel_size
            self.is_embed = False
            self.isFreezeEmbed = False
            self.is_semantic = False
            my_kwargs.update({
                'maxdisp':  args.max_disp, 
                'kernel_size': args.dfn_kernel_size,
                'crop_img_h': args.crop_height,
                'crop_img_w': args.crop_width,
                'isDFN': self.is_dfn,
                'dilation': args.dilation,
                'cost_filter_grad': self.cost_filter_grad
                })
            
        elif self.model_name in ['ASN-SGA-PSM', 'ASN-SGA-DispNetC', 'ASN-SGA-GCNet']:
            print ('[!!!] loading ASN-SGA-X net')
            self.is_sga_guide_from_img = str(args.is_sga_guide_from_img).lower() == 'true'
            #self.is_quarter_size = str(args.is_quarter_size).lower() == 'true'
            self.downsample_scale = args.sga_downsample_scale
            self.is_lga = str(args.is_lga).lower() == 'true'
            if self.is_sga_guide_from_img:
                self.is_embed = False
                self.isFreezeEmbed = False
                self.is_semantic = False
            else:
                self.is_embed = True
                self.isFreezeEmbed = False
                self.is_semantic = True

            self.is_dfn = False
            self.is_pac = False
            my_kwargs.update({
                    'maxdisp': args.max_disp, 
                    'is_sga_guide_from_img': self.is_sga_guide_from_img,
                    #'is_quarter_size': self.is_quarter_size, 
                    'downsample_scale': self.downsample_scale,
                    'is_lga': self.is_lga,
                    'cost_filter_grad': self.cost_filter_grad
                })
        
        elif self.model_name in ['ASN-PAC-PSM', 'ASN-PAC-DispNetC', 'ASN-PAC-GANet-Deep', 'ASN-PAC-GCNet']:
            if self.model_name == 'ASN-PAC-GANet-Deep':
                self.pac_in_channels = 64
                self.pac_out_channels = 64
                print ('[!!!] loading ASN-PAC-GANet-Deep')
            elif self.model_name == 'ASN-PAC-PSM':
                self.pac_in_channels = 64
                self.pac_out_channels = 64
                print ('[!!!] loading ASN-PAC-PSM net')
            elif self.model_name == 'ASN-PAC-GCNet':
                self.pac_in_channels = 64
                self.pac_out_channels = 64
                print ('[!!!] loading ASN-PAC-GCNet')
            elif self.model_name == 'ASN-PAC-DispNetC':
                self.pac_in_channels = self.max_disp // 4
                self.pac_out_channels = self.max_disp // 4
                print ('[!!!] loading ASN-PAC-DispNetC net')
            
            self.is_pac = str(args.is_pac).lower() == 'true'

            self.pac_kernel_size = args.pac_kernel_size
            self.is_embed = str(args.is_embed).lower() == 'true'
            self.isFreezeEmbed = (str(args.isFreezeEmbed).lower() == 'true')
            if not self.is_embed:
                self.is_semantic = False
            self.is_dfn = False
            my_kwargs.update({
                    'maxdisp': args.max_disp,
                    'kernel_size': args.pac_kernel_size,
                    'isPAC': self.is_pac,
                    'isEmbed': self.is_embed,
                    'pac_in_channels': self.pac_in_channels,
                    'pac_out_channels': self.pac_out_channels,
                    'dilation': args.dilation,
                    'cost_filter_grad': self.cost_filter_grad,
                    'native_impl': str(args.pac_native_imple).lower() == 'true'
                })
        
        else: # embedding bilateral filtering (EBF);
            print ('[!!!] loading ', self.model_name)
            self.is_dfn = False
            self.is_pac = False
            self.isFreezeEmbed = (str(args.isFreezeEmbed) == 'true')
            self.is_embed = str(args.is_embed).lower() == 'true'
            if not self.is_embed:
                self.is_semantic = False
            my_kwargs.update({
                    'maxdisp': args.max_disp,
                    'sigma_s': args.bilateral_sigma_s,
                    'sigma_v': args.bilateral_sigma_v,
                    'isEmbed': self.is_embed,
                    'dilation': args.dilation,
                    'cost_filter_grad': self.cost_filter_grad,
                })
        
        #----------------
        # get the model
        #----------------
        self.model = AttenStereoNet(**my_kwargs)
        
        print('[***]Number of {} parameters: {}'.format(
            self.model_name,
            sum([p.data.nelement() for p in self.model.parameters()])))
        
        #print('[***]where, number of {} sga_costAgg.get_g_from_img parameters: {}'.format(
        #    self.model_name,
        #    sum([p.data.nelement() for n,p in self.model.named_parameters() if 'sga_costAgg.get_g_from_img' in n])))
        
        #for i, (n, p) in enumerate(self.model.named_parameters()):
        #    print (i, "  layer ", n, "has # param : ", p.data.nelement())
        #sys.exit()
        


        if not self.isTestingMode: # training mode
            """ We need to set requires_grad == False to freeze the parameters 
                so that the gradients are not computed in backward();
                
                Parameters of newly constructed modules have requires_grad=True by default;
            """
            if self.is_embed and self.isFreezeEmbed:
                print("Freeze EmbeddingNet Module during training!!!")
                for param in self.model.embednet.parameters():
                    param.requires_grad = False
           
            #Freeze Bilateral Filter
            if self.is_embed:
                # for some cases, no bifilter attribute exists;
                if hasattr(self.model, 'bifilter'):
                    if isinstance(self.model.bifilter, torch.nn.Module):
                        #print ("Freeze Bilateral Filter Module during training!!!")
                        #for param in self.model.bifilter.parameters():
                        #    param.requires_grad = False
                        #    print ("Freeze Bilateral Filter Module during training!!!")
                        for name, param in self.model.bifilter.named_parameters():
                            param.requires_grad = False
                            print ("[***]During training, Freeze Bilateral Filter Module: ", name)

            
            # updated for the cases where some subnetwork was forzen!!!
            params_to_update = [p for p in self.model.parameters() if p.requires_grad]
            if 0:
                print ('[****] params_to_update = ')
                for p in params_to_update:
                    print (type(p.data), p.size())
            
            self.optimizer= optim.Adam(params_to_update, lr = args.lr, betas=(0.9,0.999))
            self.writer = SummaryWriter(args.train_logdir)
            
            # saving settings into a json file
            tmp_dir = pjoin(self.checkpoint_dir, self.model_name)
            if not os.path.exists(tmp_dir):
                os.makedirs(tmp_dir)
                print ('makedirs {}'.format(tmp_dir))
            json_file = pjoin(tmp_dir, 'train_args.json')
            with open(json_file, 'wt') as f_json:
                # 't' refers to the text mode. 
                # There is no difference between 'r' and 'rt' 
                # or 'w' and 'wt' since text mode is the default.
                json.dump(vars(args), f_json, indent = 4)
                json.dump(my_kwargs, f_json, indent = 4)
            print ("[***] Saving args to json file ", json_file)
        

        if self.is_embed and os.path.isfile(args.saved_embednet_checkpoint):
            """ loading pre-trained embedding network model"""
            print ('[**] For pretrained embedding net model loading: saved_embednet_checkpoint = ', 
                args.saved_embednet_checkpoint)
            embed_checkpoint = self.load_checkpts(args.saved_embednet_checkpoint)
            if embed_checkpoint is not None:
                self.model.embednet.load_state_dict(embed_checkpoint['model_state_dict'])
            else:
                print("Embednet saved checkpoint load failed ... neglected\n Start Training ...")


        
        if self.cuda:
            self.model = torch.nn.DataParallel(self.model).cuda()

        if self.isTestingMode:
            assert os.path.isfile(args.resume) == True, "Model Test but NO checkpoint found at {}".format(args.resume)
        if args.resume:
            if os.path.isfile(args.resume):
                print("[***] => loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(args.resume)
                """debug DFN saved checkpoint """
                #n = 0
                #for k,v in checkpoint['state_dict'].items():
                #    print ('idx = %d' %n, k, v.shape)
                #    n += 1
                #sys.exit()
                self.model.load_state_dict(checkpoint['state_dict'], strict=False)
                if not self.isTestingMode and hasattr(checkpoint, 'optimizer'):
                    self.optimizer.load_state_dict(checkpoint['optimizer'])
            else:
                print("=> no checkpoint found at {}".format(args.resume))
        
        #print ("[***] {} weights inilization done!".format(self.model_name))

    
    def save_checkpoint(self, epoch, state_dict, is_best=False):
        saved_checkpts = pjoin(self.checkpoint_dir, self.model_name)
        if not os.path.exists(saved_checkpts):
            os.makedirs(saved_checkpts)
            print ('makedirs {}'.format(saved_checkpts))
        
        #./checkpoint/sceneflow
        filename = pjoin(saved_checkpts, "model_epoch_%05d.tar" % epoch)
        torch.save(state_dict, filename)
        print ('Saved checkpoint at %s' % filename) 
        if is_best:
            best_fname = pjoin(saved_checkpts, 'model_best.tar')
            shutil.copyfile(filename, best_fname)

    def adjust_learning_rate(self, epoch):
        #if epoch <= 300:
        #lr_adjust_epo = 300
        #if  self.model_name.find('PSM') != -1: # PSMNet
        #    lr_adjust_epo = 200
        #else:
        #    lr_adjust_epo = 300

        #if lr_epoch_steps is None:
        #    lr_epoch_steps = [self.lr_adjust_epo_thred]
        
        old_lr = self.lr
        print ("decrease lr by 10 at these epochs: ", self.lr_epoch_steps)
        if epoch in self.lr_epoch_steps:
            self.lr *= 0.1
            print ("[!!!]Epo={}, adjust lr from {} to {}".format(epoch, old_lr, self.lr))
        
        #if epoch <= self.lr_adjust_epo_thred:
        #    self.lr = self.args.lr
        #else:
        #    self.lr = self.args.lr * 0.1
        print('[***]learning rate = ', self.lr, ' and epo = ', epoch)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr
    
    # Newly added for DFN-PSM case, fine-tuning on Virtual KITTI 2;
    # This function keeps the learning rate at 0.001 for the first ten epochs
    # and decreases it exponentially after that.
    def fine_tuning_scheduler(self, epoch, lr_min = 1.0e-4):
        #lr_adjust_epo = 2
        if epoch <= self.lr_adjust_epo_thred:
            self.lr = self.args.lr
        else:
            self.lr = self.args.lr * math.exp(0.1 * (self.lr_adjust_epo_thred - epoch))
        
        self.lr = max(self.lr, lr_min)

        print('learning rate = ', self.lr, ' and epo = ', epoch)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr


    def verify_lr_scheduler(self, nEpochs):
        # Add some assertion:
        # fine-tuning on small dataset, so we need large epochs, like nEpochs = 400; 
        if nEpochs > 200:
            return self.lr_adjust_epo_thred >= 100
        elif 50 < nEpochs <= 200:
            return self.lr_adjust_epo_thred >= 60
        # fine-tuning on large dataset, so we need small epochs, like nEpochs = 20;
        elif 10 < nEpochs <= 50:
            return self.lr_adjust_epo_thred <= 8
        # training from scratch on scene flow dataset, typically we set nEpochs = 10
        elif 1 < nEpochs <= 10:
            return self.lr_adjust_epo_thred > 8
    
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

    def build_train_summaries(self, imgl, imgr, disp0, disp1, disp2, disp_gt, global_step, loss, 
            err0, err1, err2, embed = None, embed_loss = None, dfn_filter = None, dfn_bias = None, is_KT15Color = False):
            """ loss and epe error """
            self.writer.add_scalar(tag = 'train_loss', scalar_value = loss, global_step = global_step)
            if self.is_semantic and embed_loss is not None:
                self.writer.add_scalar(tag = 'train_lossEmbed', scalar_value = embed_loss, global_step = global_step)
            if err0 is not None:
                self.writer.add_scalar(tag = 'train_err0', scalar_value = err0, global_step = global_step)
            if err1 is not None:
                self.writer.add_scalar(tag = 'train_err1', scalar_value = err1, global_step = global_step)
            if err2 is not None:
                self.writer.add_scalar(tag = 'train_err2', scalar_value = err2, global_step = global_step)
            """ learning rate """
            self.writer.add_scalar(tag = 'train_lr', scalar_value = self.lr, global_step = global_step)
            """ Add batched image data to summary:
                Note: add_images(img_tensor): img_tensor could be torch.Tensor, numpy.array, or string/blobname;
                so we could use torch.Tensor or numpy.array !!!
            """
            self.writer.add_images(tag='train_imgl',img_tensor=imgl, global_step = global_step, dataformats='NCHW')
            if imgr is not None:
                self.writer.add_images(tag='train_imgr',img_tensor=imgr, global_step = global_step, dataformats='NCHW')
            
            def show_dfn(batch_dfn_filter, dfn_kernel_size = 9):
                """Convert a Tensor to numpy image."""
                #batch_dfn_filter = batch_dfn_filter.cpu().numpy().transpose((0,2,3,1))# change to [N,H,W,C]
                batch_dfn_filter = batch_dfn_filter.cpu().numpy() # change to [N,C,H,W]
                N, C, H, W = batch_dfn_filter.shape[:]
                res = np.zeros([N,1, H, W])
                assert (dfn_kernel_size **2 == C)
                k_half = (dfn_kernel_size -1)//2
                #print ("[***]k_half = ", k_half)
                for i in range(k_half, H - k_half, dfn_kernel_size): # along height
                    for j in range(k_half, W - k_half, dfn_kernel_size): # along width
                        tmp_idx = 0
                        for ki in range(-k_half, k_half+1): # along kernel_height
                            for kj in range(-k_half, k_half+1): # along kernel_width
                                res[:,0,i+ki,j+kj] = batch_dfn_filter[:,tmp_idx,i,j]
                                tmp_idx += 1
                return res

            
            with torch.set_grad_enabled(False):
                if is_KT15Color:
                    my_disp_clr_func = KT15FalseColorDisp
                else:
                    my_disp_clr_func = colormap_jet_batch_image
                if disp0 is not None:
                    self.writer.add_images(tag='train_disp0',img_tensor=my_disp_clr_func(disp0), global_step = global_step, dataformats='NHWC')
                if disp1 is not None:
                    self.writer.add_images(tag='train_disp1',img_tensor=my_disp_clr_func(disp1), global_step = global_step, dataformats='NHWC')
                if disp2 is not None: 
                    self.writer.add_images(tag='train_disp2', img_tensor=my_disp_clr_func(disp2), global_step = global_step, dataformats='NHWC')
                    self.writer.add_images(tag='train_dispGT',img_tensor=my_disp_clr_func(disp_gt), global_step = global_step, dataformats='NHWC')
                    self.writer.add_images(tag='train_dispErr',img_tensor=KT15LogColorDispErr(disp2, disp_gt), global_step = global_step, dataformats='NHWC')
                if embed is not None:
                    embed_pca = pca_embedding(embed, k=3,isChanelLast=False)
                    self.writer.add_images(tag='train_embed_pca3',img_tensor= embed_pca, global_step = global_step, dataformats='NCHW')
                if dfn_filter is not None:
                    #dfn_filter_pca = pca_embedding(dfn_filter, k=3,isChanelLast=False)
                    dfn_filter_vis = show_dfn(dfn_filter, dfn_kernel_size = self.dfn_kernel_size)
                    self.writer.add_images(tag='train_vis_dfn_filter',img_tensor= dfn_filter_vis, global_step = global_step, dataformats='NCHW')
                if dfn_bias is not None:
                    self.writer.add_images(tag='train_vis_dfn_bias',img_tensor= dfn_bias, global_step = global_step, dataformats='NCHW')
    

    #---------------------
    #---- Training -------
    #---------------------
    def train(self, 
            epoch,# epoch idx 
            nEpochs = 400 # total # of epoches for training
            ):
        """Set up TensorBoard """
        epoch_loss = 0
        epoch_error0 = 0
        epoch_error1 = 0
        epoch_error2 = 0
        valid_iteration = 0

        #for iteration, batch_data in enumerate(self.training_data_loader):
        #    print (" [***] iteration = %d/%d" % (iteration, self.train_loader_len))
        #    input1 = batch_data[0].float() # False by default;
        #    input2 = batch_data[1].float()
        #    target = batch_data[2].float()
        #    left_rgb = batch_data[3].float()
        #sys.exit()
        
        # setting to train mode;
        self.model.train()
        # Add some assertion:
        assert self.verify_lr_scheduler(self.args.startEpoch + nEpochs), "lr_scheduler=%f is not CORRECT !!!" %(self.lr_adjust_epo_thred)
        
        # 1) piecewise: lr = 1e-3  if epoch <= lr_adjust_epo_thred else 1e-4
        # 2) exponential: lr = 1e-3  if epoch <= lr_adjust_epo_thred else 1e-3 * math.exp(0.1 * ( lr_adjust_epo_thred - epoch))
        # 3) constant: lr = 1e-3, i.e., constant learning rate;
        if self.lr_scheduler == "piecewise":
            self.adjust_learning_rate(epoch)
        elif self.lr_scheduler == "exponential":    
            self.fine_tuning_scheduler(epoch)
        elif self.lr_scheduler == "constant":
            print("fixed learning rate!")
        else:
            raise Exception("No suitable lr_scheduler type found ...")

        """ running log loss """
        log_running_loss = 0.0
        log_running_embed_loss = 0.0
        log_running_err0 = 0.0
        log_running_err1 = 0.0
        log_running_err2 = 0.0
        
        
        for iteration, batch_data in enumerate(self.training_data_loader):
            start = time.time()
            #print (" [***] iteration = %d" % iteration)
            input1 = batch_data[0].float() # False by default;
            #print ("[???] input1 require_grad = ", input1.requires_grad) # False
            input2 = batch_data[1].float()
            target = batch_data[2].float()
            
            left_rgb = batch_data[3].float()
            #right_rgb = batch_data[4].float()
            semantic_label=batch_data[5].float()
            
            # from GANet
            #input1, input2, target = Variable(batch_data[0], requires_grad=True), Variable(batch_data[1], requires_grad=True), Variable(batch_data[2], requires_grad=False)
            
            if self.cuda:
                input1 = input1.cuda()
                input2 = input2.cuda()
                target = target.cuda()
                semantic_label = semantic_label.cuda()

            target = torch.squeeze(target,1)
            #mask = target < self.max_disp
            # valid pixels: 0 < disparity < max_disp
            mask = (target - self.max_disp)*target < 0
            mask.detach_()
            valid_disp = target[mask].size()[0]
            
            if valid_disp > 0:
                self.optimizer.zero_grad()
                
                if self.model_name == 'ASN-Embed-GANet-Deep':
                    disp0, disp1, disp2, embed = self.model(input1, input2)
                    loss0 = F.smooth_l1_loss(disp0[mask], target[mask], reduction='mean')
                    loss1 = F.smooth_l1_loss(disp1[mask], target[mask], reduction='mean')
                    if self.kitti2012 or self.kitti2015:
                        loss2 = self.criterion(disp2[mask], target[mask])
                    else:
                        loss2 = F.smooth_l1_loss(disp2[mask], target[mask], reduction='mean')
                    
                    loss = 0.2*loss0 + 0.6*loss1 + loss2
                    
                    if self.is_semantic:
                        semantic_label = F.interpolate(semantic_label, [semantic_label.size()[2]//3, semantic_label.size()[3]//3], 
                                mode='bilinear', align_corners=False)
                        embed_loss, _, _, _ = get_embed_losses(embed, semantic_label, args_dict = None)
                        loss += self.args.embed_loss_weight * embed_loss


                elif self.model_name == 'ASN-PAC-GANet-Deep':
                    disp0, disp1, disp2, pac_guide_fea = self.model(input1, input2)
                    loss0 = F.smooth_l1_loss(disp0[mask], target[mask], reduction='mean')
                    loss1 = F.smooth_l1_loss(disp1[mask], target[mask], reduction='mean')
                    if self.kitti2012 or self.kitti2015:
                        loss2 = self.criterion(disp2[mask], target[mask])
                    else:
                        loss2 = F.smooth_l1_loss(disp2[mask], target[mask], reduction='mean')
                    
                    loss = 0.2*loss0 + 0.6*loss1 + loss2
                    
                    if self.is_semantic:
                        embed = pac_guide_fea
                        semantic_label = F.interpolate(semantic_label, [semantic_label.size()[2]//3, semantic_label.size()[3]//3], 
                                mode='bilinear', align_corners=False)
                        embed_loss, _, _, _ = get_embed_losses(embed, semantic_label, args_dict = None)
                        loss += self.args.embed_loss_weight * embed_loss
                    
                elif self.model_name == 'ASN-Embed-GANet11':

                    disp1, disp2, embed = self.model(input1, input2)
                    disp0 = (disp1 + disp2)/2
                    loss0 = F.smooth_l1_loss(disp0[mask], target[mask], reduction='mean')
                    
                    loss1 = F.smooth_l1_loss(disp1[mask], target[mask], reduction='mean')
                    if self.kitti2012 or self.kitti2015:
                        loss2 = self.criterion(disp2[mask], target[mask])
                    else:
                        loss2 = F.smooth_l1_loss(disp2[mask], target[mask], reduction='mean')
                    
                    #loss = 0*loss0 + 0.4*loss1 + 1.2*loss2
                    loss = 0.4*loss1 + 1.2*loss2
                    
                    if self.is_semantic:
                        semantic_label = F.interpolate(semantic_label, [semantic_label.size()[2]//3, semantic_label.size()[3]//3], 
                                mode='bilinear', align_corners=False)
                        embed_loss, _, _, _ = get_embed_losses(embed, semantic_label, args_dict = None)
                        loss += self.args.embed_loss_weight * embed_loss
                
                elif self.model_name in ['ASN-Embed-PSM']:
                    # disp in shape [N, H, W]
                    disp0, disp1, disp2, embed = self.model(input1, input2)
                    loss0 = F.smooth_l1_loss(disp0[mask], target[mask], reduction='mean') 
                    loss1 = F.smooth_l1_loss(disp1[mask], target[mask], reduction='mean')
                    loss2 = F.smooth_l1_loss(disp2[mask], target[mask], reduction='mean')
                    loss = 0.5*loss0 + 0.7*loss1 + loss2
                    if self.is_semantic:
                        semantic_label = F.interpolate(semantic_label, [semantic_label.size()[2]//4, 
                            semantic_label.size()[3]//4], mode='bilinear', align_corners=False)
                        embed_loss, _, _, _ = get_embed_losses(embed, semantic_label, args_dict = None)
                        loss += self.args.embed_loss_weight * embed_loss
                
                elif self.model_name in ['ASN-Embed-GCNet']:
                    # disp in shape [N, H, W]
                    disp2, embed = self.model(input1, input2)
                    loss = F.smooth_l1_loss(disp2[mask], target[mask], reduction='mean')
                    if self.kitti2012 or self.kitti2015:
                        loss = 0.4*loss + 0.6*self.criterion(disp2[mask], target[mask])
                    if self.is_quarter_size_cost_volume_gcnet:
                        tmp_scale = 4
                    else:
                        tmp_scale = 2
                    if self.is_semantic:
                        semantic_label = F.interpolate(semantic_label, 
                            [semantic_label.size()[2]//tmp_scale, semantic_label.size()[3]//tmp_scale], 
                             mode='bilinear', align_corners=False)
                        embed_loss, _, _, _ = get_embed_losses(embed, semantic_label, args_dict = None)
                        loss += self.args.embed_loss_weight * embed_loss
                
                elif self.model_name == 'ASN-PAC-GCNet':
                    disp2, pac_guide_fea = self.model(input1, input2)
                    loss = F.smooth_l1_loss(disp2[mask], target[mask], reduction='mean')

                    if self.kitti2012 or self.kitti2015:
                        loss = 0.4*loss + 0.6*self.criterion(disp2[mask], target[mask])
                    
                    if self.is_quarter_size_cost_volume_gcnet:
                        tmp_scale = 4
                    else:
                        tmp_scale = 2
                    if self.is_semantic:
                        embed = pac_guide_fea
                        semantic_label = F.interpolate(semantic_label, [semantic_label.size()[2]//tmp_scale, semantic_label.size()[3]//tmp_scale], 
                                mode='bilinear', align_corners=False)
                        embed_loss, _, _, _ = get_embed_losses(embed, semantic_label, args_dict = None)
                        loss += self.args.embed_loss_weight * embed_loss
                
                elif self.model_name == 'ASN-SGA-GCNet':
                    disp2, g_in = self.model(input1, input2)
                    loss = F.smooth_l1_loss(disp2[mask], target[mask], reduction='mean')

                    if self.kitti2012 or self.kitti2015:
                        loss = 0.4*loss + 0.6*self.criterion(disp2[mask], target[mask])
                    
                    if self.is_quarter_size_cost_volume_gcnet:
                        tmp_scale = 4
                    else:
                        tmp_scale = 2
                    if self.is_semantic:
                        embed = g_in
                        semantic_label = F.interpolate(semantic_label, [semantic_label.size()[2]//tmp_scale, semantic_label.size()[3]//tmp_scale], 
                                mode='bilinear', align_corners=False)
                        embed_loss, _, _, _ = get_embed_losses(embed, semantic_label, args_dict = None)
                        loss += self.args.embed_loss_weight * embed_loss
                
                elif self.model_name in ['ASN-DFN-GCNet']:
                    # disp in shape [N, H, W]
                    disp2, dfn_filter, dfn_bias = self.model(input1, input2)
                    loss = F.smooth_l1_loss(disp2[mask], target[mask], reduction='mean')
                    if self.kitti2012 or self.kitti2015:
                        loss = 0.4*loss + 0.6*self.criterion(disp2[mask], target[mask])

                elif self.model_name in ['ASN-Embed-DispNetC']:
                    # NOTE: three outputs at the same scale: H x W 
                    disp0, disp1, disp2, embed = self.model(input1, input2)

                    loss0 = F.smooth_l1_loss(disp0[mask], target[mask], reduction='mean') 
                    loss1 = F.smooth_l1_loss(disp1[mask], target[mask], reduction='mean')
                    loss2 = F.smooth_l1_loss(disp2[mask], target[mask], reduction='mean')
                    loss = 0.5*loss0 + 0.7*loss1 + loss2
                    
                    if self.kitti2012 or self.kitti2015:
                        loss = 0.4*loss + 0.6*self.criterion(disp2[mask], target[mask])
                    if self.is_semantic:
                        semantic_label = F.interpolate(semantic_label, [semantic_label.size()[2]//4, 
                            semantic_label.size()[3]//4], mode='bilinear', align_corners=False)
                        embed_loss, _, _, _ = get_embed_losses(embed, semantic_label, args_dict = None)
                        loss += self.args.embed_loss_weight * embed_loss

                elif self.model_name in ['ASN-DFN-DispNetC']:
                    # NOTE: three outputs at the same scale: H x W 
                    disp0, disp1, disp2, dfn_filter, dfn_bias = self.model(input1, input2)
                    loss0 = F.smooth_l1_loss(disp0[mask], target[mask], reduction='mean') 
                    loss1 = F.smooth_l1_loss(disp1[mask], target[mask], reduction='mean')
                    loss2 = F.smooth_l1_loss(disp2[mask], target[mask], reduction='mean')
                    loss = 0.5*loss0 + 0.7*loss1 + loss2
                    
                    if self.kitti2012 or self.kitti2015:
                        loss = 0.4*loss + 0.6*self.criterion(disp2[mask], target[mask])
                
                elif self.model_name in ['ASN-DFN-PSM']:
                    # disp in shape [N, H, W]
                    disp0, disp1, disp2, dfn_filter, dfn_bias = self.model(input1, input2)
                    loss0 = F.smooth_l1_loss(disp0[mask], target[mask], reduction='mean') 
                    loss1 = F.smooth_l1_loss(disp1[mask], target[mask], reduction='mean')
                    loss2 = F.smooth_l1_loss(disp2[mask], target[mask], reduction='mean')
                    loss = 0.5*loss0 + 0.7*loss1 + loss2
                
                elif self.model_name in ['ASN-DFN-GANet-Deep']:
                    # disp in shape [N, H, W]
                    disp0, disp1, disp2, dfn_filter, dfn_bias = self.model(input1, input2)
                    loss0 = F.smooth_l1_loss(disp0[mask], target[mask], reduction='mean')
                    loss1 = F.smooth_l1_loss(disp1[mask], target[mask], reduction='mean')
                    if self.kitti2012 or self.kitti2015:
                        loss2 = self.criterion(disp2[mask], target[mask])
                    else:
                        loss2 = F.smooth_l1_loss(disp2[mask], target[mask], reduction='mean')
                    
                    loss = 0.2*loss0 + 0.6*loss1 + loss2
                    #print ('[???] finished DFN-GANet-Deep 1 Iteration , disp0 and loss devices = ', disp0.get_device(), loss.get_device())
                
                elif self.model_name in ['ASN-SGA-PSM']:
                    # disp in shape [N, H, W]
                    disp0, disp1, disp2, g_in = self.model(input1, input2)
                    loss0 = F.smooth_l1_loss(disp0[mask], target[mask], reduction='mean') 
                    loss1 = F.smooth_l1_loss(disp1[mask], target[mask], reduction='mean')
                    loss2 = F.smooth_l1_loss(disp2[mask], target[mask], reduction='mean')
                    loss = 0.5*loss0 + 0.7*loss1 + loss2
                    if self.is_semantic:
                        embed = g_in
                        semantic_label = F.interpolate(semantic_label, [semantic_label.size()[2]//4, 
                            semantic_label.size()[3]//4], mode='bilinear', align_corners=False)
                        embed_loss, _, _, _ = get_embed_losses(embed, semantic_label, args_dict = None)
                        loss += self.args.embed_loss_weight * embed_loss
                
                elif self.model_name in ['ASN-SGA-DispNetC']:
                    # NOTE: three outputs at the same scale: H x W 
                    disp0, disp1, disp2, g_in = self.model(input1, input2)
                    loss0 = F.smooth_l1_loss(disp0[mask], target[mask], reduction='mean') 
                    loss1 = F.smooth_l1_loss(disp1[mask], target[mask], reduction='mean')
                    loss2 = F.smooth_l1_loss(disp2[mask], target[mask], reduction='mean')
                    loss = 0.5*loss0 + 0.7*loss1 + loss2
                    
                    if self.kitti2012 or self.kitti2015:
                        loss = 0.4*loss + 0.6*self.criterion(disp2[mask], target[mask])
                    if self.is_semantic:
                        embed = g_in
                        semantic_label = F.interpolate(semantic_label, [semantic_label.size()[2]//4, 
                            semantic_label.size()[3]//4], mode='bilinear', align_corners=False)
                        embed_loss, _, _, _ = get_embed_losses(embed, semantic_label, args_dict = None)
                        loss += self.args.embed_loss_weight * embed_loss
                
                elif self.model_name in ['ASN-PAC-DispNetC']:
                    # NOTE: three outputs at the same scale: H x W 
                    disp0, disp1, disp2, pac_guide_fea = self.model(input1, input2)
                    loss0 = F.smooth_l1_loss(disp0[mask], target[mask], reduction='mean') 
                    loss1 = F.smooth_l1_loss(disp1[mask], target[mask], reduction='mean')
                    loss2 = F.smooth_l1_loss(disp2[mask], target[mask], reduction='mean')
                    loss = 0.5*loss0 + 0.7*loss1 + loss2
                    
                    if self.kitti2012 or self.kitti2015:
                        loss = 0.4*loss + 0.6*self.criterion(disp2[mask], target[mask])
                    if self.is_semantic:
                        embed = pac_guide_fea
                        semantic_label = F.interpolate(semantic_label, [semantic_label.size()[2]//4, 
                            semantic_label.size()[3]//4], mode='bilinear', align_corners=False)
                        embed_loss, _, _, _ = get_embed_losses(embed, semantic_label, args_dict = None)
                        loss += self.args.embed_loss_weight * embed_loss

                elif self.model_name in ['ASN-PAC-PSM']:
                    # disp in shape [N, H, W]
                    disp0, disp1, disp2, pac_guide_fea = self.model(input1, input2)
                    loss0 = F.smooth_l1_loss(disp0[mask], target[mask], reduction='mean') 
                    loss1 = F.smooth_l1_loss(disp1[mask], target[mask], reduction='mean')
                    loss2 = F.smooth_l1_loss(disp2[mask], target[mask], reduction='mean')
                    loss = 0.5*loss0 + 0.7*loss1 + loss2
                    
                    if self.is_semantic:
                        embed = pac_guide_fea
                        semantic_label = F.interpolate(semantic_label, [semantic_label.size()[2]//4, 
                            semantic_label.size()[3]//4], mode='bilinear', align_corners=False)
                        embed_loss, _, _, _ = get_embed_losses(embed, semantic_label, args_dict = None)
                        loss += self.args.embed_loss_weight * embed_loss
                else:
                    raise Exception("No suitable model found ...")
                    
                loss.backward()
                #print ('[???] finished DFN-GANet-Deep 1 Iteration , loss backward = ', loss.get_device())

                self.optimizer.step()
                #print ('[???] finished DFN-GANet-Deep 1 Iteration , optimizer.step() = ', loss.get_device())
                
                # MAE error
                if  (self.model_name.find('GCNet') != -1): # GCNet
                    error2 = torch.mean(torch.abs(disp2[mask] - target[mask]))
                    epoch_error2 += error2.item()
                    
                    # epoch - 1: here argument `epoch` is starting from 1, instead of 0 (zer0);
                    train_global_step = (epoch-1)*self.train_loader_len + iteration
                    message_info = "===> Epoch[{}]({}/{}): Step {}, Loss: {:.3f} - LossEmbed: {:.2f}; EPE: {:.2f}; {:.2f} s/step".format(
                                epoch, iteration, self.train_loader_len, train_global_step,
                                loss.item(),  embed_loss.item() if self.is_semantic else -1.0,
                                error2.item(), time.time() - start )
                    #sys.stdout.flush()
                    # save summary for tensorboard visualization
                    log_running_err2 += error2.item()
                    log_running_loss += loss.item()
                
                else:
                    error0 = torch.mean(torch.abs(disp0[mask] - target[mask])) 
                    error1 = torch.mean(torch.abs(disp1[mask] - target[mask]))
                    error2 = torch.mean(torch.abs(disp2[mask] - target[mask]))
                
                    epoch_error0 += error0.item()
                    epoch_error1 += error1.item()
                    epoch_error2 += error2.item()
                
                    # save summary for tensorboard visualization
                    log_running_err0 += error0.item()
                    log_running_err1 += error1.item()
                    log_running_err2 += error2.item()
                    log_running_loss += loss.item()
                
                    # epoch - 1: here argument `epoch` is starting from 1, instead of 0 (zer0);
                    train_global_step = (epoch-1)*self.train_loader_len + iteration      
                    message_info = "===> Epoch[{}]({}/{}): Step {}, Loss: {:.3f} - Loss0/1/2/embed: ({:.2f} {:.2f} {:.2f} {:.2f}); EPE: ({:.2f} {:.2f} {:.2f}); {:.2f} s/step".format(
                            epoch, iteration, self.train_loader_len, train_global_step,
                            loss.item(), loss0.item(), loss1.item(), loss2.item(), 
                            embed_loss.item() if self.is_semantic else -1.0,
                            error0.item(), error1.item(), error2.item(), time.time() -start)
                    #sys.stdout.flush()

                #----------------------
                print (message_info)
                epoch_loss += loss.item()
                valid_iteration += 1
                if self.is_semantic:
                    log_running_embed_loss += embed_loss.item()
                
                if iteration % self.log_summary_step == (self.log_summary_step - 1):
                    
                    #NOTE: For tensorboard visulization, we could just show half size version, i.e., [H/2, W/2],
                    # for saving the disk space;
                    # disp0 in size [N, H, W]
                    # in the latest versions of PyTorch you can add a new axis by indexing with None 
                    # in size [N, 1, H/2, W/2]

                    with torch.set_grad_enabled(False):
                        H, W = disp2.size()[-2:]
                        left_rgb_vis = F.interpolate(left_rgb, size=[H//2, W//2], mode='bilinear', align_corners = True)
                        #NOTE: In the latest versions of PyTorch you can add a new axis by indexing with None 
                        # > see: https://discuss.pytorch.org/t/what-is-the-difference-between-view-and-unsqueeze/1155;
                        #torch.unsqueeze(disp0, dim=1) ==> disp0[:,None]
                        
                        if (self.model_name.find('GCNet') != -1): # GCNet
                            disp0_vis = None
                            disp1_vis = None
                            log_running_err0 = None
                            log_running_err1 = None
                        else:
                            disp0_vis = F.interpolate(disp0[:,None,...], size=[H//2, W//2], mode='bilinear', align_corners = True)
                            disp1_vis = F.interpolate(disp1[:,None,...], size=[H//2, W//2], mode='bilinear', align_corners = True)
                            log_running_err0 /= self.log_summary_step
                            log_running_err1 /= self.log_summary_step
                        
                        
                        log_running_err2 /= self.log_summary_step 
                        log_running_loss /= self.log_summary_step
                        
                        disp2_vis = F.interpolate(disp2[:,None,...], size=[H//2, W//2], mode='bilinear', align_corners = True)
                        target_vis = F.interpolate(target[:,None,...], size=[H//2, W//2], mode='bilinear', align_corners = True)
                        self.build_train_summaries( 
                              #left_rgb, 
                              left_rgb_vis, 
                              None, #right_rgb,
                              #disp0[:,None], disp1[:,None], 
                              #disp2[:,None], target[:,None],
                              disp0_vis, disp1_vis, disp2_vis, target_vis,
                              train_global_step, 
                              log_running_loss, log_running_err0, log_running_err1, log_running_err2,
                              # upsample embed to original image size for tensorboard visulization;
                              #F.interpolate(embed, [input1.size()[2], input1.size()[3]], mode='bilinear', align_corners=False) if self.is_embed else None,
                              # upsample embed to half image size for tensorboard visulization;
                              F.interpolate(embed, [H//2, W//2], mode='bilinear', align_corners=False) if self.is_embed else None,
                              log_running_embed_loss/self.log_summary_step,
                              # upsample dfn to half image size for tensorboard visulization;
                              F.interpolate(dfn_filter, [H//2, W//2], mode='bilinear', align_corners=False) if self.is_dfn else None,
                              F.interpolate(dfn_bias, [H//2, W//2], mode='bilinear', align_corners=False) if self.is_dfn else None
                              )
                    
                    # reset to zeros
                    log_running_loss = 0.0
                    log_running_embed_loss = 0.0
                    log_running_err0 = 0.0
                    log_running_err1 = 0.0
                    log_running_err2 = 0.0

        
        # end of data_loader
        # save the checkpoints
        avg_loss = epoch_loss / valid_iteration
        avg_err0 = epoch_error0/valid_iteration
        avg_err1 = epoch_error1/valid_iteration
        avg_err2 = epoch_error2/valid_iteration
        print("===> Epoch {} Complete: Avg. Loss: {:.4f}, Avg. EPE Error: ({:.4f} {:.4f} {:.4f})".format(
                  epoch, avg_loss, avg_err0, avg_err1, avg_err2 ))
        

        is_best = False
        model_state_dict = {
                        'epoch': epoch,
                        'state_dict': self.model.state_dict(),
                        'optimizer' : self.optimizer.state_dict(),
                        'loss': avg_loss,
                        'err0': avg_err0, 
                        'err1': avg_err1, 
                        'err2': avg_err2,
                    }
        #if nEpochs > 500:
        if nEpochs > 900:
            save_epo_step = 50
        elif 300 < nEpochs <= 900:
            save_epo_step = 25
        elif 200 < nEpochs <= 300:
            save_epo_step = 20
        elif 100 <= nEpochs <= 200:
            save_epo_step = 10
        elif 50 <= nEpochs < 100:
            save_epo_step = 5
        else:
            save_epo_step = 25
        if self.kitti2012 or self.kitti2015:
            #if epoch % 50 == 0 and epoch >= 300:
            #if epoch % 50 == 0:
            if epoch % save_epo_step == 0:
                self.save_checkpoint(epoch, model_state_dict, is_best)
        else:
            #if epoch >= 7:
            #    self.save_checkpoint(epoch, model_state_dict, is_best)
            self.save_checkpoint(epoch, model_state_dict, is_best)
        # avg loss
        return avg_loss, avg_err0, avg_err1, avg_err2
       
    # see the GPU runtime counting code from https://github.com/sacmehta/ESPNet/issues/57;
    def computeInferenceTime(self, imgH = 384, imgW = 1280):
            input1 = torch.randn(1, 3, imgH, imgW)
            input2 = torch.randn(1, 3, imgH, imgW)
            if self.cuda:
                input1 = input1.cuda()
                input2 = input2.cuda()

            self.model.eval()
            i = 0
            time_spent = []
            while i < 50:
                #if i % 10 == 0:
                #    print ("i = ", i)
                start_time = time.time()
                with torch.no_grad():
                    _, _ = self.model(input1, input2)

                if self.cuda:
                    torch.cuda.synchronize()  # wait for cuda to finish (cuda is asynchronous!)
                if i > 5:
                    time_spent.append(time.time() - start_time)
                i += 1
            print('Model {}: Avg execution time (ms): {:.3f}'.format(self.model_name, 1000.0*np.mean(time_spent)))


    #---------------------
    #---- Test ----- -----
    #---------------------
    def test(self):
        self.model.eval()
        file_path = self.args.data_path
        file_list_txt = self.args.test_list
        f = open(file_list_txt, 'r')
        if self.virtual_kitti2:
            filelist = get_virtual_kitti2_filelist(file_list_txt)
            random.seed(30)
            random.shuffle(filelist)
        else:
            filelist = [l.rstrip() for l in f.readlines() if not l.rstrip().startswith('#')]

        crop_width = self.args.crop_width
        crop_height = self.args.crop_height
        batch_in_image = str(self.args.batch_in_image).lower() == 'true'
        #print ("[???] batch_in_image = ", batch_in_image)

        if not os.path.exists(self.args.resultDir):
            os.makedirs(self.args.resultDir)
            print ('makedirs {}'.format(self.args.resultDir))
        
        avg_err = 0
        avg_rate1 = 0
        avg_rate3 = 0
        
        saved_image_name_list = []
        saved_image_idx_list = []
        err_list = []
        rate1_list = []
        rate3_list = []

        
        if self.virtual_kitti2:
            dict_avg_err = {}
            Xs = ['01', '02', '06', '18', '20']
            Ys = ['15-deg-left', '15-deg-right', '30-deg-left', 
                  '30-deg-right', 'clone', 'fog', 'morning', 
                  'overcast', 'rain', 'sunset']
            for tmpx in Xs:
                for tmpy in Ys:
                    dict_avg_err[ 'Scene' + tmpx + '/' + tmpy] = [.0, .0, .0, .0] # epe, rate1, rate3, number_counting;


        for index in range(len(filelist)):
            current_file = filelist[index]
            if self.kitti2015:
                data_type_str= "kt15" 
                leftname = pjoin(file_path, 'image_0/' + current_file)
                if index < 1:
                    print ("limg: {}".format(leftname))
                rightname = pjoin(file_path, 'image_1/' + current_file)
                dispname = pjoin(file_path, 'disp_occ_0_pfm/' + current_file[0:-4] + '.pfm')
                dispGT=pfm.readPFM(dispname)
                dispGT[dispGT == np.inf] = .0
                savename = pjoin(self.args.resultDir, current_file[0:-4] + '.pfm')
                
            elif self.kitti2012:
                data_type_str= "kt12" 
                #leftname = pjoin(file_path, 'image_0/' + current_file)
                leftname =  pjoin(file_path, 'colored_0/' + current_file)
                rightname = pjoin(file_path, 'colored_1/' + current_file)
                dispname = pjoin(file_path, 'disp_occ_pfm/' + current_file[0:-4] + '.pfm')
                dispGT=pfm.readPFM(dispname)
                dispGT[dispGT == np.inf] = .0
                savename = pjoin(self.args.resultDir, current_file[0:-4] + '.pfm')
                #disp = Image.open(dispname)
                #disp = np.asarray(disp) / 256.0
            
            elif self.virtual_kitti2:
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
                savename = pjoin(self.args.resultDir, '%04d.pfm'%(index))


            else:
                data_type_str= "scene_flow" 
                A = current_file
                leftname = pjoin(file_path, A)
                rightname = pjoin(file_path, A[:-13] + 'right/' + A[len(A)-8:]) 
                savename = pjoin(self.args.resultDir, '%04d.pfm'%(index))

            #print ("[???] crop_height = %d, crop_width = %d" % (crop_height, crop_width))
            input1, input2, height, width = test_transform(
                    load_test_data(leftname, rightname, 
                          #is_data_augment=self.is_data_augment
                          is_data_augment= False
                          ), crop_height, crop_width)
            if self.cuda:
                input1 = input1.cuda()
                input2 = input2.cuda()
            
            with torch.no_grad():
                if batch_in_image == False: # process the whole image
                    prediction, _ = self.model(input1, input2)
                    disp = prediction.cpu().detach().numpy()
                else:
                    #print ("[***] Processing the image to several small batches along image height !!!")
                    #print ("[???] encoder_ds = ", self.args.encoder_ds)
                    #print ("[???] input1 = ", input1.size())
                    disp = test_small_height_batch( self.model,
                            input1, input2,
                            self.args.batch_h,
                            self.args.encoder_ds, # e.g., multiple of 64, due to SGA module having 1/64 size Feature!
                           )

            if height <= crop_height and width <= crop_width:
                disp = disp[0, crop_height - height: crop_height, crop_width-width: crop_width]
                #disp = disp[0, crop_height - height: crop_height, 0: width] # top-right padding, was found worse than top-left padding pattern on embed+dispnetc!!
            else:
                disp = disp[0, :, :]
            
            #skimage.io.imsave(savename, (disp * 256).astype('uint16'))
            #if any([self.kitti2015, self.kitti2012, index % 100 == 0]):
            if any([self.kitti2015, self.kitti2012, index % 5 == 0]):
                pfm.save(savename, disp)
                #skimage.io.imsave(savename, (disp * 256).astype('uint16'))
                #print ('saved ', savename)
                if 0 and dispGT is not None:
                    left = np.asarray(Image.open(leftname))[:,:,:3].astype(np.float32) 
                    right = np.asarray(Image.open(rightname))[:,:,:3].astype(np.float32)
                    #print ("[???]", left.shape)
                    pfm.save(savename[:-4] + '-iml.pfm', left)
                    pfm.save(savename[:-4] + '-imr.pfm', right)
                    pfm.save(savename[:-4] + '-gt.pfm', dispGT.astype(np.float32))
            
            #error, rate1, rate3 = get_epe_rate2(dispGT, disp, self.args.max_disp, threshold=1.0, threshold2=3.0)
            error, rate1, rate3 = get_epe_rate(dispGT, disp, threshold=1.0, threshold2=3.0)
            avg_err += error
            avg_rate1 += rate1
            avg_rate3 += rate3
            

            if self.virtual_kitti2:
                tmp_key = current_file.split('/')[0] + '/' + current_file.split('/')[1]
                #print ("[???] tmp_key = ", tmp_key)
                dict_avg_err[tmp_key][0] += error
                dict_avg_err[tmp_key][1] += rate1
                dict_avg_err[tmp_key][2] += rate3
                dict_avg_err[tmp_key][3] += 1.0

            if index % 100 == 0:
                print("===> Frame {}: ".format(index) + leftname + " ==> EPE Error: {:.4f}, Bad-{:.1f} Error: {:.4f}, Bad-{:.1f} Error: {:.4f}".format(
                    error, 1.0, rate1, 3.0, rate3))

            # save kt15 color
            #if any([self.kitti2015, self.kitti2012, index%100==0]):
            if any([self.kitti2015, self.kitti2012, index%5==0]):
                """ save the list """
                saved_image_name_list.append(current_file)
                saved_image_idx_list.append(index)
                err_list.append(error)
                rate1_list.append(rate1)
                rate3_list.append(rate3)
                
                """ disp color """
                tmp_dir = pjoin(self.args.resultDir, "dispColor")
                if not os.path.exists(tmp_dir):
                    os.makedirs(tmp_dir)
                if self.kitti2015 or self.kitti2012:
                    tmp_dispname = pjoin(tmp_dir, current_file[0:-4] + '.png')
                else:
                    tmp_dispname = pjoin(tmp_dir, '%04d.png'%(index))
                cv2.imwrite(tmp_dispname, 
                        KT15FalseClr.writeKT15FalseColor(np.ascontiguousarray(disp)).astype(np.uint8)[:,:,::-1])
                if index < 1:
                    print ('savded ', tmp_dispname)
                """ err-disp """

                tmp_dir = pjoin(self.args.resultDir, "errDispColor")
                if not os.path.exists(tmp_dir):
                    os.makedirs(tmp_dir)
                if self.kitti2015 or self.kitti2012:
                    tmp_errdispname = pjoin(tmp_dir, current_file[0:-4]  + '.png')
                else:
                    tmp_errdispname = pjoin(tmp_dir, '%04d.png'%(index))
                cv2.imwrite(tmp_errdispname, 
                        KT15LogClr.writeKT15ErrorDispLogColor(np.ascontiguousarray(disp), np.ascontiguousarray(dispGT)).astype(np.uint8)[:,:,::-1])
                if index < 1:
                    print ('savded ', tmp_errdispname)
        
        #average error and rate:
        if dispGT is not None:
            avg_err /= len(filelist)
            avg_rate1 /= len(filelist)
            avg_rate3 /= len(filelist)
            print("===> Total {} Frames ==> AVG EPE Error: {:.4f}, AVG Bad-{:.1f} Error: {:.4f}, AVG Bad-{:.1f} Error: {:.4f}".format(
                len(filelist), avg_err, 1.0, avg_rate1, 3.0, avg_rate3))
            if self.virtual_kitti2:
                print ("====> Virtual KT 2 Category result:")
                # for verification
                tmp_avg_err = .0
                tmp_avg_rate1 = .0
                tmp_avg_rate3 = .0
                tmp_sum_N = .0
                for key, val in dict_avg_err.items():
                    tmp_N = val[3]
                    if tmp_N > 0:
                        tmp_err = val[0]
                        tmp_rate1 = val[1]
                        tmp_rate3 = val[2]
                        
                        tmp_avg_err += tmp_err
                        tmp_avg_rate1 += tmp_rate1
                        tmp_avg_rate3 += tmp_rate3
                        tmp_sum_N += tmp_N

                        print("=====> Category {}: total {} Frames ==> AVG EPE Error: {:.4f}, AVG Bad-{:.1f} Error: {:.4f}, AVG Bad-{:.1f} Error: {:.4f}".format(
                            key, int(tmp_N), tmp_err/tmp_N, 1.0, tmp_rate1/tmp_N, 3.0, tmp_rate3/tmp_N))
                print("====> To Sum: Total {} Frames ==> AVG EPE Error: {:.4f}, AVG Bad-{:.1f} Error: {:.4f}, AVG Bad-{:.1f} Error: {:.4f}".format(
                    int(tmp_sum_N), tmp_avg_err/tmp_sum_N, 1.0, tmp_avg_rate1/tmp_sum_N, 3.0, tmp_avg_rate3/tmp_sum_N))

            """ save as csv file, Excel file format """
            csv_file = os.path.join(self.args.resultDir, 'bad-err.csv')
            print ("write ", csv_file, "\n")
            timeStamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
            messg = timeStamp + ',{},bad-1.0,{:.4f},bad-3.0,{:.4f},epe,{:.4f},fileDir={},for log,{:.3f}(epe); {:.3f}%(bad1); {:.3f}%(bad3)\n'.format(
                   data_type_str, avg_rate1, avg_rate3, avg_err, 
                   self.args.resultDir, 
                   avg_err, avg_rate1*100.0, avg_rate3*100.0)
            
            with open( csv_file, 'w') as fwrite:
                fwrite.write(messg)
           
            """ save error for individual image """
            with open( csv_file, 'a') as fwrite:
                for i in range(len(saved_image_idx_list)):
                    messg = 'image index,{},image name,{},bad-1.0,{:.4f},bad-3.0,{:.4f},epe,{:.4f},{:.3f}(epe); {:.3f}%(bad1); {:.3f}%(bad3)\n'.format(
                           saved_image_idx_list[i], saved_image_name_list[i], 
                           rate1_list[i]*100.0, rate3_list[i]*100.0, err_list[i],
                           err_list[i], rate1_list[i]*100.0, rate3_list[i]*100.0,
                           rate1_list[i]*100.0)
                    fwrite.write(messg)

            # saving virtual KT 2 evaluation a json file
            if self.virtual_kitti2:
                json_file = pjoin(self.args.resultDir, 'vkt2-err.json')
                mysave_dict_avg_err = {'model_name': self.model_name}
                for key, val in dict_avg_err.items():
                    tmp_N = val[3]
                    if tmp_N > 0:
                        mysave_dict_avg_err[key] = val
                
                with open(json_file, 'wt') as f_json:
                    # 't' refers to the text mode. 
                    # There is no difference between 'r' and 'rt' 
                    # or 'w' and 'wt' since text mode is the default.
                    json.dump(mysave_dict_avg_err, f_json, indent = 4)
                print ("[***] Saving vkt2 errors to json file ", json_file)

def get_epe_rate2(disp, prediction, max_disp = 192, threshold = 1.0, threshold2 = 3.0):
    mask = np.logical_and(disp >= 0.001, disp <= max_disp)
    error = np.mean(np.abs(prediction[mask] - disp[mask]))
    rate = np.sum(np.abs(prediction[mask] - disp[mask]) > threshold) / np.sum(mask)
    rate2 = np.sum(np.abs(prediction[mask] - disp[mask]) > threshold2) / np.sum(mask)
    #print(" ==> EPE Error: {:.4f}, Error Rate: {:.4f}".format(error, rate))
    return error, rate, rate2

def get_epe_rate(disp, prediction, threshold = 1.0, threshold2 = 3.0):
    #mask = np.logical_and(disp >= 0.001, disp <= max_disp)
    mask = disp >= 0.001
    error = np.mean(np.abs(prediction[mask] - disp[mask]))
    rate = np.sum(np.abs(prediction[mask] - disp[mask]) > threshold) / np.sum(mask)
    rate2 = np.sum(np.abs(prediction[mask] - disp[mask]) > threshold2) / np.sum(mask)
    #print(" ==> EPE Error: {:.4f}, Error Rate: {:.4f}".format(error, rate))
    return error, rate, rate2

#-----------------------------------
#---- Test Small Portion  ----------
#-----------------------------------
def test_small_height_batch(
        model,
        input1,
        input2,
        batch_h = 256,
        encoder_ds = 64, # e.g., multiple of 64, due to SGA module having 1/64 size Feature!
        ):
    h = input1.size()[2]
    assert batch_h % encoder_ds == 0, "batch_h %d is NOT multiple of %d"%(batch_h, encoder_ds)
    # to avoid sharp edge between adjacent batches coming from the batching processing during test;
    overlap_board = encoder_ds // 2
    #print ("[??] overlap_board = %d" % overlap_board)
    num_iters_one_img = int(math.ceil(float(h)/ float(batch_h)))
    #print ("[??] num_iters_one_img = %d" % num_iters_one_img)
    disp_list = []
    for i in range(0, num_iters_one_img):
        # get the current batch;
        # to avoid sharp edge between adjacent batches coming from the batching processing during test;
        
        #--------------------
        # the first batch_h
        #--------------------
        if i == 0:
            idx_end = batch_h + 2*overlap_board
            valid_idx_end = batch_h
            idx_begin = 0
            valid_idx_begin = 0

        #--------------------
        # the last batch_h
        #--------------------
        elif i == num_iters_one_img - 1:
            idx_end = h
            idx_begin = idx_end - batch_h - 2*overlap_board
            valid_idx_begin = -(h - i*batch_h)
            valid_idx_end = None
        else:
            idx_end = (i+1)*batch_h + overlap_board
            valid_idx_end = - overlap_board
            idx_begin = i*batch_h - overlap_board 
            valid_idx_begin = overlap_board
       
        left_image  = input1[:,:, idx_begin:idx_end, :]
        right_image = input2[:,:, idx_begin:idx_end, :]
        #print ("[???] left_image size = ", left_image.size())
        #print ("[???] right_image size = ", right_image.size())
        prediction, _ = model(left_image, right_image)
        #print ("[???] prediction: ", prediction.size())
        disp = prediction.cpu().detach().numpy()[:, valid_idx_begin:valid_idx_end,:]
        #print ("[???] disp: ", disp.shape)
        disp_list.append(disp)
    disp_out = np.concatenate(disp_list, axis = 1)
    return disp_out

def main(args):
    #----------------------------
    # some initilization thing 
    #---------------------------
    cuda = args.cuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")
    torch.manual_seed(args.seed)
    if cuda:
        torch.cuda.manual_seed(args.seed)
    
    myAttenStereoNet = attenStereoNet(args)
    """ for debugging """
    if args.mode == 'debug':
        myAttenStereoNet.model.train()
        import gc
        crop_h = 240
        crop_w = 576
        #x_l = torch.randn((1, 3, crop_h, crop_w), requires_grad=True)
        #x_r = torch.randn((1, 3, crop_h, crop_w), requires_grad=True)
        x_l = torch.randn((1, 3, crop_h, crop_w)).cuda()
        x_r = torch.randn((1, 3, crop_h, crop_w)).cuda()
        y = torch.randn((1, crop_h, crop_w)).cuda()
        z = torch.randn((1, 1, crop_h//3, crop_w//3)).cuda()

        
        from pytorch_memlab import profile, MemReporter
        # pass in a model to automatically infer the tensor names
        # You can better understand the memory layout for more complicated module
        if 0:
            reporter = MemReporter(myAttenStereoNet.model)
            disp0, disp1, disp2, embed = myAttenStereoNet.model(x_l, x_r)
            loss0 = F.smooth_l1_loss(disp0, y, reduction='mean')
            loss1 = F.smooth_l1_loss(disp1, y, reduction='mean')
            loss2 = F.smooth_l1_loss(disp2, y, reduction='mean')
            loss = 0.2*loss0 + 0.6*loss1 + loss2
            embed_loss, _, _, _ = get_embed_losses(embed, z, args_dict = None)
            loss += 0.006 * embed_loss
            reporter.report(verbose=True)
            print('========= before backward =========')
            loss.backward()
            reporter.report(verbose=True)

        # generate prof which can be loaded by Google chrome trace at chrome://tracing/
        if 1:
            with torch.autograd.profiler.profile(use_cuda=True) as prof:
                myAttenStereoNet.model(x_l, x_r)
            print(prof)
            prof.export_chrome_trace('./results/tmp/prof.out')
        
        """ not work ??? """
        #gpu_profile(frame=sys._getframe(), event='line', arg=None)
        #from src.gpu_profile import gpu_profile, mem_report
        #mem_report()
    
    if args.mode == 'train':
        print('strat training !!!')
        for epoch in range(1 + args.startEpoch, args.startEpoch + args.nEpochs + 1):
            print ("[**] do training at epoch %d/%d" % (epoch, args.startEpoch + args.nEpochs))

            with torch.autograd.set_detect_anomaly(True):
                avg_loss, avg_err0, avg_err1, avg_err2 = myAttenStereoNet.train(epoch, args.nEpochs)
        # save the last epoch always!!
        myAttenStereoNet.save_checkpoint(args.nEpochs + args.startEpoch,
            {
                'epoch': args.nEpochs + args.startEpoch,
                'state_dict': myAttenStereoNet.model.state_dict(),
                'optimizer' : myAttenStereoNet.optimizer.state_dict(),
                'loss': avg_loss,
                'err0': avg_err0, 
                'err1': avg_err1, 
                'err2': avg_err2,
            }, 

            is_best = False)
        print('done training !!!')
    
    if args.mode == 'test': 
        print('strat testing !!!')
        myAttenStereoNet.test()
    if args.mode == 'inferencetime':
        print('strat computing GPU Inference Time !!!')
        myAttenStereoNet.computeInferenceTime(
                imgH = args.crop_height, 
                imgW = args.crop_width
                )
        


def set_trace_gpu():
    import os
    from src.gpu_profile import gpu_profile
    os.environ['CUDA_VISIBLE_DEVICES']='1'
    os.environ['GPU_DEBUG']='2'
    sys.settrace(gpu_profile)
    #gpu_profile(frame=sys._getframe(), event='line', arg=None)



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
    parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
    parser.add_argument('--nEpochs', type=int, default=400, help='number of epochs to train for')
    parser.add_argument('--startEpoch', type=int, default=0, help='starting point, used for fine-tuning')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate. Default=0.001')
    parser.add_argument('--cuda', type=int, default=1, help='use cuda? Default=True')
    parser.add_argument('--threads', type=int, default=1, help='number of threads for data loader to use')
    parser.add_argument('--seed',  type=int, default=123, help='random seed to use. Default=123')
    parser.add_argument('--shift', type=int, default=0, help='random shift of left image. Default=0')
    parser.add_argument('--kitti2012', type=int, default=0, help='kitti 2012 dataset? Default=False')
    parser.add_argument('--kitti2015', type=int, default=0, help='kitti 2015? Default=False')
    parser.add_argument('--virtual_kitti2', type=int, default=0, help='virtual_kitti2? Default=False')
    parser.add_argument('--data_path', type=str, default='/data/ccjData', help="data root")
    parser.add_argument('--training_list', type=str, default='./lists/sceneflow_train.list', help="training list")
    #parser.add_argument('--val_list', type=str, default='./lists/sceneflow_test_select.list', help="validation list")
    parser.add_argument('--test_list', type=str, default='./lists/sceneflow_test_select.list', help="evaluation list")
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint/', help="location to save models")
    parser.add_argument('--saved_embednet_checkpoint', type=str, 
                        default=None,
                        #default='./checkpoints/pascalvoc-embednet-epo30/vgg-like-embed/best_model_epoch_00030_valloss_1.2961.tar', 
                        help="the saved embedding network model")
    parser.add_argument('--model_name', type=str, default='ASN-Embed-SGA', help="model name")
    parser.add_argument('--train_logdir', dest='train_logdir',  default='./logs/tmp', help='log dir')
    """Arguments related to run mode"""
    parser.add_argument('--mode', dest='mode', type = str, default='train', help='train, test')
    parser.add_argument('--sigma_s', dest='bilateral_sigma_s', type=float, default= 0.7, help='bilateral_sigma_s')
    parser.add_argument('--sigma_v', dest='bilateral_sigma_v', type=float, default= 0.1, help='bilateral_sigma_v')
    parser.add_argument('--is_embed', type=str, default= 'true', help='flag to use embedding or not')
    parser.add_argument('--isFreezeEmbed', type=str, default= 'true', help='flag to freeze the embed net or not')
    parser.add_argument('--is_semantic', type=str, default= "true", help='flag to use semantic loss to train the embed net or not')
    parser.add_argument('--resultDir', type=str, default= "./results")
    """ arguments related to weights to combine different loss items"""
    parser.add_argument('--embed_loss_weight', type=float, default= 0.6, help='weight for embedding loss')
    # enable torch.set_grad_enabled() Ture or False for cost volume filtering via embedding;
    parser.add_argument('--cost_filter_grad', type=str, default= "true", help='flag to torch.set_grad_enabled(True) to cost volume filtering or not')
    parser.add_argument('--dilation', type=int, default= 1, help='im2col dilation')
    """ dynamic filter network (dfn) """
    parser.add_argument('--is_dfn', type=str, default= 'true', help='flag to use dynamic filter network or not')
    parser.add_argument('--dfn_kernel_size', dest='dfn_kernel_size', type=int, default= 9, help='dyn filter size')
    """ pixel-adaptive convolution (pac) network """
    parser.add_argument('--is_pac', type=str, default= 'true', help='flag to use pac network or not')
    parser.add_argument('--pac_native_imple', type=str, default= 'true', help='flag to implement of PAC in fiter version or native version')
    parser.add_argument('--pac_kernel_size', dest='pac_kernel_size', type=int, default= 9, help='pac kernel filter size')
    """ for SGA module"""
    parser.add_argument('--is_sga_guide_from_img', type=str, default= 'true', help='flag to use  sga guide from input img')
    #parser.add_argument('--is_quarter_size', type=str, default= 'true', help='flag to generate quarter_size feature, otherwise 1/3 of the size')
    parser.add_argument('--sga_downsample_scale', type=int, default= 4, help='flag to generate 1/4, 1/3, or 1/2 image size feature')
    parser.add_argument('--is_lga', type=str, default= 'false', help='flag to generate LGA(local guided aggregation) weigths or not')

    """ newly added for processing one image in small batch parts due to limited GPU memory """
    parser.add_argument('--batch_in_image', type=str, default= 'false', help='flag of batch_in_image')
    parser.add_argument('--batch_h', type=int, default=256, help='batch_h if batch_in_image True')
    parser.add_argument('--encoder_ds', type=int, default=64, help='encoder-decoder downsampling scale') 

    """ added for GCNet """
    parser.add_argument('--is_kendall_version_gcnet', 
                dest = 'is_kendall_version', type=str, default= 'false', 
                help = "excatly following the structure in Kendall's GCNet paper")
    parser.add_argument('--is_quarter_size_cost_volume_gcnet', 
                dest = 'is_quarter_size_cost_volume_gcnet', type=str, default= 'false', 
                help = "cost volume in quarter image size, i.e., [D/4, H/4, W/4]")
    
    """ added for DFN+PSM debugging """
    #parser.add_argument('--is_fixed_lr', dest = 'is_fixed_lr', type=str, default= 'false', 
    #            help = "fix the learning rate or adjust it dynamically")
    parser.add_argument('--lr_adjust_epo_thred', dest = 'lr_adjust_epo_thred', type = int, default= 10, 
                help = "apply exponential lr scheduling after this specific epoch")
    # learning rate scheduler:
    # 1) piecewise: lr *= 0.1  if epoch in lr_epoch_steps;
    # 2) exponential: lr = 1e-3  if epoch <= lr_adjust_epo_thred else 1e-3 * math.exp(0.1 * ( lr_adjust_epo_thred - epoch))
    # 3) constant: lr = 1e-3, i.e., constant learning rate;
    parser.add_argument('--lr_scheduler', dest = 'lr_scheduler', type=str, default= 'constant', 
                help = "Learning rate scheduler, can be 'piecewise', 'exponential', or 'constant'")
    parser.add_argument('--lr_epoch_steps', default='', type=str, help='decrease lr by 10 at these epochs')

    """ added for KT2012 gray images loadding """
    #parser.add_argument('--is_kt12_gray', type=str, default= "false", help='flag to load kt2012 gray images or not')
    parser.add_argument('--kt12_image_mode', type=str, default= "rgb", help='flag to load kt2012 gray images, rgb, or gray2rgb')
    parser.add_argument('--is_data_augment', type=str, default= "false", help='flag to use data_augment, including random scale crop, color change etc')

    args = parser.parse_args()
    print('[***] args = ', args)
    main(args)
