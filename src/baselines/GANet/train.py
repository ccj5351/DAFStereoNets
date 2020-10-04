from __future__ import print_function
import argparse
from math import log10

from .libs.GANet.modules.GANet import MyLoss2, valid_accu3
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
#from models.GANet_deep import GANet
import torch.nn.functional as F

#from .dataloader.data import get_training_set, get_valid_set
#NOTE:Updated on 2020/05/17;
# using global dataloading function
from src.loaddata.data import get_training_set, load_test_data, test_transform
from src.loaddata.dataset import get_virtual_kitti2_filelist

#added by CCJ:
from torch.utils.tensorboard import SummaryWriter
from src.dispColor import colormap_jet_batch_image,KT15FalseColorDisp,KT15LogColorDispErr
import time
from os.path import join as pjoin

# Training settings
parser = argparse.ArgumentParser(description='PyTorch GANet Example')
parser.add_argument('--crop_height', type=int, required=True, help="crop height")
parser.add_argument('--max_disp', type=int, default=192, help="max disp")
parser.add_argument('--crop_width', type=int, required=True, help="crop width")
parser.add_argument('--resume', type=str, default='', help="resume from saved model")
parser.add_argument('--left_right', type=int, default=0, help="use right view for training. Default=False")
parser.add_argument('--batchSize', type=int, default=1, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--nEpochs', type=int, default=2048, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate. Default=0.001')
parser.add_argument('--cuda', type=int, default=1, help='use cuda? Default=True')
parser.add_argument('--threads', type=int, default=1, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--shift', type=int, default=0, help='random shift of left image. Default=0')
parser.add_argument('--kitti2012', dest='kitti', type=int, default=0, help='kitti2012 dataset? Default=False')
parser.add_argument('--kitti2015', type=int, default=0, help='kitti 2015? Default=False')
parser.add_argument('--virtual_kitti2', type=int, default=0, help='virtual_kitti2? Default=False')
parser.add_argument('--data_path', type=str, default='/ssd1/zhangfeihu/data/stereo/', help="data root")
parser.add_argument('--training_list', type=str, default='./lists/sceneflow_train.list', help="training list")
#parser.add_argument('--val_list', type=str, default='./lists/sceneflow_test_select.list', help="validation list")
parser.add_argument('--val_list', type=str, default= None, help="validation list")
#parser.add_argument('--save_checkpoint', dest='save_path', type=str, default='./checkpoint/', help="location to save models")
parser.add_argument('--model', type=str, default='GANet_deep', help="model to train")

# added by CCJ:
parser.add_argument('--train_logdir', dest='train_logdir',  default='./logs/tmp', help='log dir')
parser.add_argument('--checkpoint_dir', dest='save_path', type=str, default='./checkpoint/', help="location to save models")
parser.add_argument('--log_summary_step', type=int, default=200, help='every 200 steps to build training summary')
#parser.add_argument('--resultDir', type=str, default= "./results")



opt = parser.parse_args()

print(opt)
if opt.model == 'GANet11':
    from .models.GANet11 import GANet
elif opt.model == 'GANet_deep':
    from .models.GANet_deep import GANet

elif opt.model == 'GANet_deep_dfn':
    from .models.GANet_deep_dfn import GANet
else:
    raise Exception("No suitable model found ...")
    
cuda = opt.cuda
#cuda = True
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)

print('===> Loading training datasets')
train_set = get_training_set(
    opt.data_path, opt.training_list, [opt.crop_height, opt.crop_width], 
    opt.kitti, 
    opt.kitti2015, 
    opt.virtual_kitti2,
    opt.shift,
    kt12_image_mode = 'rgb',
    is_data_augment = False
    )
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True, drop_last=True)

#if opt.val_list is not None and opt.val_list != '':
#    print('===> Loading validation datasets')
#    test_set = get_valid_set(
#        opt.data_path, 
#        opt.val_list, 
#        [576,960], 
#        #opt.left_right, 
#        opt.kitti, opt.kitti2015)
#    testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)

print('===> Building model ', opt.model)
model = GANet(opt.max_disp)
print('[***]Number of {} model parameters: {}'.format(opt.model, sum([p.data.nelement() for p in model.parameters()])))
#sys.exit()

criterion = MyLoss2(thresh=3, alpha=2)
if cuda:
    model = torch.nn.DataParallel(model).cuda()
optimizer=optim.Adam(model.parameters(), lr=opt.lr,betas=(0.9,0.999))


if opt.resume:
    if os.path.isfile(opt.resume):
        print("=> loading checkpoint '{}'".format(opt.resume))
        checkpoint = torch.load(opt.resume)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
#        optimizer.load_state_dict(checkpoint['optimizer'])
    else:
        print("=> no checkpoint found at '{}'".format(opt.resume))

writer = SummaryWriter(opt.train_logdir)

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

def train(epoch):
    epoch_loss = 0
    epoch_error0 = 0
    epoch_error1 = 0
    epoch_error2 = 0
    epoch_accu3 = 0
    valid_iteration = 0
    model.train()
    
    log_summary_step = opt.log_summary_step
    """ running log loss """
    log_running_loss = 0.0
    log_running_err  = 0.0
    
    if opt.kitti == 1:
        accu_thred = 3.0
    elif opt.kitti2015 == 1:
        accu_thred = 3.0
    else:
        accu_thred = 1.0
    
    
    for iteration, batch in enumerate(training_data_loader):
        start = time.time()
        input1, input2, target = Variable(batch[0], requires_grad=True), Variable(batch[1], requires_grad=True), Variable(batch[2], requires_grad=False)
        left_rgb = batch[3].float()
        #right_rgb = batch[4].float()
        if cuda:
            input1 = input1.cuda()
            input2 = input2.cuda()
            target = target.cuda()

        target = torch.squeeze(target,1)
        mask = target < opt.max_disp
        mask.detach_()
        valid = target[mask].size()[0]
        if valid > 0:
            optimizer.zero_grad()
            
            if opt.model == 'GANet11':
                disp1, disp2 = model(input1, input2)
                disp0 = (disp1 + disp2)/2.
                if opt.kitti or opt.kitti2015:
                    loss = 0.4 * F.smooth_l1_loss(disp1[mask], target[mask], reduction='mean') + 1.2 * criterion(disp2[mask], target[mask])
                else:
                    loss = 0.4 * F.smooth_l1_loss(disp1[mask], target[mask], reduction='mean') + 1.2 * F.smooth_l1_loss(disp2[mask], target[mask], reduction='mean')
            elif opt.model in ['GANet_deep', 'GANet_deep_dfn']:
                disp0, disp1, disp2 = model(input1, input2)
                if opt.kitti or opt.kitti2015:
                    loss = 0.2 * F.smooth_l1_loss(disp0[mask], target[mask], reduction='mean') + 0.6 * F.smooth_l1_loss(disp1[mask], target[mask], reduction='mean') +  criterion(disp2[mask], target[mask])
                else:
                    loss = 0.2 * F.smooth_l1_loss(disp0[mask], target[mask], reduction='mean') + 0.6 * F.smooth_l1_loss(disp1[mask], target[mask], reduction='mean') +  F.smooth_l1_loss(disp2[mask], target[mask], reduction='mean')
            else:
                raise Exception("No suitable model found ...")
                
            loss.backward()
            optimizer.step()
            # MAE error
            error0 = torch.mean(torch.abs(disp0[mask] - target[mask])) 
            error1 = torch.mean(torch.abs(disp1[mask] - target[mask]))
            error2 = torch.mean(torch.abs(disp2[mask] - target[mask]))

            #added by CCJ:
            accu = valid_accu3(target[mask], disp2[mask], thred = accu_thred)

            epoch_loss += loss.item()
            valid_iteration += 1
            epoch_error0 += error0.item()
            epoch_error1 += error1.item()
            epoch_error2 += error2.item()      
            epoch_accu3 += accu.item() 

            # epoch - 1: here argument `epoch` is starting from 1, instead of 0 (zer0);
            train_global_step = (epoch-1)* len(training_data_loader) + iteration   
            print("===> Epoch[{}]({}/{}): Loss: {:.4f}, Error: ({:.4f} {:.4f} {:.4f}), Accu{:.1f}: {:.4f}, {:.2f} s/step".format(
                epoch, iteration, len(training_data_loader), 
                loss.item(), error0.item(), error1.item(), error2.item(),
                accu_thred, accu.item(), time.time() -start))
            sys.stdout.flush()


            # save summary for tensorboard visualization
            log_running_loss += loss.item()
            log_running_err += error2.item()
            if iteration % log_summary_step == (log_summary_step - 1):
                build_train_summaries(left_rgb, None, 
                    disp2[:,None],
                    target[:,None], 
                    train_global_step, 
                    log_running_loss/log_summary_step, 
                    log_running_err/log_summary_step, 
                    is_KT15Color = False)
                # reset to zeros
                log_running_loss = 0.0
                log_running_err = 0.0
    
    # end of data_loader
    avg_loss = epoch_loss / valid_iteration
    avg_err0 = epoch_error0 / valid_iteration
    avg_err1 = epoch_error1 / valid_iteration
    avg_err2 = epoch_error2 / valid_iteration
    avg_accu = epoch_accu3 / valid_iteration
    print("===> Epoch {} Complete: Avg. Loss: {:.4f}, Avg. Error: ({:.4f} {:.4f} {:.4f}), Accu{:.1f}: {:.4f}".format(
        epoch, avg_loss, avg_err0, avg_err1, avg_err2, accu_thred, avg_accu))

def val(testing_data_loader):
    epoch_error2 = 0

    valid_iteration = 0
    model.eval()
    for iteration, batch in enumerate(testing_data_loader):
        input1, input2, target = Variable(batch[0],requires_grad=False), Variable(batch[1], requires_grad=False), Variable(batch[2], requires_grad=False)
        if cuda:
            input1 = input1.cuda()
            input2 = input2.cuda()
            target = target.cuda()
        target = torch.squeeze(target, 1)
        mask = target < opt.max_disp
        mask.detach_()
        valid = target[mask].size()[0]
        if valid>0:
            with torch.no_grad():
                disp2 = model(input1,input2)
                error2 = torch.mean(torch.abs(disp2[mask] - target[mask]))
                valid_iteration += 1
                epoch_error2 += error2.item()      
                print("===> Test({}/{}): Error: ({:.4f})".format(iteration, len(testing_data_loader), error2.item()))

    print("===> Test: Avg. Error: ({:.4f})".format(epoch_error2 / valid_iteration))
    return epoch_error2 / valid_iteration

def save_checkpoint(save_path, epoch, state, is_best):
    filename = pjoin(save_path, "model_epoch_%05d.tar" % epoch)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print ('makedirs {}'.format(save_path))
    
    torch.save(state, filename)
    if is_best:
        tmp_filename = pjoin(save_path, "model_epoch_best.tar")
        shutil.copyfile(filename, tmp_filename)
    print("Checkpoint saved to {}".format(filename))

def adjust_learning_rate(optimizer, epoch):
    if epoch <= 400:
       lr = opt.lr
    else:
       lr = opt.lr*0.1
    print('learning rate = ', lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__ == '__main__':
    error=100
    for epoch in range(1, opt.nEpochs + 1):
#        if opt.kitti or opt.kitti2015:
        adjust_learning_rate(optimizer, epoch)
        train(epoch)
        is_best = False
#        loss=val()
#        if loss < error:
#            error=loss
#            is_best = True
        if opt.kitti or opt.kitti2015:
            #if epoch%50 == 0 and epoch >= 300:
            if epoch%50 == 0:
                save_checkpoint(opt.save_path, epoch,
                        { 'epoch': epoch,
                          'state_dict': model.state_dict(),
                          'optimizer' : optimizer.state_dict(),}, is_best)
        else:
            #if epoch>=8:
            #if epoch>=3:
            save_checkpoint(opt.save_path, epoch,
                    { 'epoch': epoch,
                      'state_dict': model.state_dict(),
                      'optimizer' : optimizer.state_dict(),}, is_best)

    # save the last epoch always!!
    save_checkpoint(opt.save_path, opt.nEpochs,{
            'epoch': opt.nEpochs,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
        }, is_best)