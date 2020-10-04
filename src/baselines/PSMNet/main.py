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
import numpy as np
import time
import math
import sys

from dataloader import listflowfile as lt
from dataloader import SceneFlowLoader as DA
from models import *

parser = argparse.ArgumentParser(description='PSMNet')
parser.add_argument('--maxdisp', type=int, default=192,
                    help='maxium disparity')
parser.add_argument('--model', default='stackhourglass',
                    help='select model')
parser.add_argument('--datapath', default='dataset/',
                    help='datapath')
parser.add_argument('--epochs', type=int, default=10,
                    help='number of epochs to train')
parser.add_argument('--fraction', type=float, default=1.0,
                    help='use a fraction of total # of train data')
parser.add_argument('--batch_size', type=int, default=6, # for 24GB memory
                    help='number of epochs to train')
parser.add_argument('--loadmodel', default=None,
                    help='load model')
parser.add_argument('--savemodel', default='./',
                    help='save model')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

# set gpu id used
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

all_left_img, all_right_img, all_left_disp, test_left_img, \
        test_right_img, test_left_disp = lt.dataloader(
                args.datapath, args.fraction)


TrainImgLoader = torch.utils.data.DataLoader(
    DA.myImageFloder(all_left_img, all_right_img, all_left_disp, True),
    batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=False)

train_loader_len = len(TrainImgLoader)

TestImgLoader = torch.utils.data.DataLoader(
    DA.myImageFloder(test_left_img, test_right_img, test_left_disp, False),
    batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=False)

test_loader_len = len(TestImgLoader)

if args.model == 'stackhourglass':
    model = stackhourglass(args.maxdisp)
elif args.model == 'basic':
    model = basic(args.maxdisp)
else:
    print('no model')

print('[***]Number of PSMNet model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
#sys.exit()

if args.cuda:
    model = nn.DataParallel(model)
    model.cuda()

if args.loadmodel is not None and args.loadmodel != '':
    state_dict = torch.load(args.loadmodel)
    model.load_state_dict(state_dict['state_dict'])

print('Number of PSMNet model parameters: {}'.format(
    sum([p.data.nelement() for p in model.parameters()])))
if 0:
    print('Including:\n1) number of Feature Extraction module parameters: {}'.format(
        sum(
            [p.data.nelement() for n, p in model.named_parameters() if 'feature_extraction' in n]
            )))

    print('2) number of Other modules parameters: {}'.format(
        sum(
            [p.data.nelement() for n, p in model.named_parameters() if 'feature_extraction' not in n]
            )))
    for i, (n, p) in enumerate(model.named_parameters()):
            print (i, "  layer ", n, "has # param : ", p.data.nelement())
    sys.exit()

optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))


def train(imgL, imgR, disp_L):
    model.train()
    #imgL = Variable(torch.FloatTensor(imgL))
    #imgR = Variable(torch.FloatTensor(imgR))
    #disp_L = Variable(torch.FloatTensor(disp_L))
    imgL = torch.FloatTensor(imgL)
    imgR = torch.FloatTensor(imgR)
    disp_L = torch.FloatTensor(disp_L)

    if args.cuda:
        imgL, imgR, disp_true = imgL.cuda(), imgR.cuda(), disp_L.cuda()

   # ---------
    mask = disp_true < args.maxdisp
    mask.detach_()
    # ----
    optimizer.zero_grad()

    if args.model == 'stackhourglass':
        output1, output2, output3 = model(imgL, imgR)
        #print ("[???] disp1 shape = ", output1.shape)
        #output1 = torch.squeeze(output1, 1) # 
        #print ("[???] squeezed disp1 shape = ", output1.shape)
        #output2 = torch.squeeze(output2, 1)
        #output3 = torch.squeeze(output3, 1)
        loss = 0.5*F.smooth_l1_loss(output1[mask], disp_true[mask],
                                    # updated by CCJ due to deprecated warning:
                                    # size_average=True ==> reduction='mean';
                                    reduction='mean') + 0.7*F.smooth_l1_loss(output2[mask], disp_true[mask],
                                                                             reduction='mean') + F.smooth_l1_loss(output3[mask], disp_true[mask],
                                                                                                                  reduction='mean')
    elif args.model == 'basic':
        output = model(imgL, imgR)
        output = torch.squeeze(output, 1)
        loss = F.smooth_l1_loss(output[mask], disp_true[mask],
                                # size_average=True
                                reduction='mean'
                                )

    loss.backward()
    optimizer.step()

    # return loss.data[0]
    # updated by CCJ:
    return loss.item()


def test(imgL, imgR, disp_true):
    model.eval()
    imgL = Variable(torch.FloatTensor(imgL))
    imgR = Variable(torch.FloatTensor(imgR))
    if args.cuda:
        imgL, imgR = imgL.cuda(), imgR.cuda()

    # ---------
    mask = disp_true < 192
    # ----

    with torch.no_grad():
        output3 = model(imgL, imgR)

    output = torch.squeeze(output3.data.cpu(), 1)[:, 4:, :]

    if len(disp_true[mask]) == 0:
        loss = 0
    else:
        # end-point-error
        loss = torch.mean(torch.abs(output[mask]-disp_true[mask]))

    return loss


def adjust_learning_rate(optimizer, epoch):
    lr = 0.001
    print('learning rate = %f' %lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():

    start_full_time = time.time()
    for epoch in range(1, args.epochs+1):
        print('This is %d-th epoch' % (epoch))
        total_train_loss = 0
        adjust_learning_rate(optimizer, epoch)

        ## training ##

        for batch_idx, (imgL_crop, imgR_crop, disp_crop_L) in enumerate(TrainImgLoader):
            start_time = time.time()
            loss = train(imgL_crop, imgR_crop, disp_crop_L)
            # print the loss info
            if batch_idx % 10 == 9:
                time_elapsed = time.time() - start_time
                est_left_time = time_elapsed * \
                    (train_loader_len - 1 - batch_idx)/3600.0
                print('Iter [%d/%d] training loss = %.3f , time = %.2f, EST = %.2f Hrs' % (
                    batch_idx, train_loader_len, loss, time_elapsed, est_left_time))
            total_train_loss += loss
        print('epoch %d total training loss = %.3f' %
              (epoch, total_train_loss / train_loader_len))

        # SAVE
        savefilename = args.savemodel+'/checkpoint_'+str(epoch)+'.tar'
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'train_loss': total_train_loss/len(TrainImgLoader),
        }, savefilename)

    print('full training time = %.2f HR' %
          ((time.time() - start_full_time)/3600.0))

    # ------------- TEST ------------------------------------------------------------
    total_test_loss = 0
    for batch_idx, (imgL, imgR, disp_L) in enumerate(TestImgLoader):
        test_loss = test(imgL, imgR, disp_L)
        print('Iter %d test loss = %.3f' % (batch_idx, test_loss))
        total_test_loss += test_loss

    print('total test loss = %.3f' % (total_test_loss/len(TestImgLoader)))
    # ----------------------------------------------------------------------------------
    # SAVE test information
    savefilename = args.savemodel+'testinformation.tar'
    torch.save({
        'test_loss': total_test_loss/len(TestImgLoader),
    }, savefilename)


if __name__ == '__main__':
    main()
