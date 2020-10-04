# !/usr/bin/env python3
# -*-coding:utf-8-*-
# @file: loss.py
# @brief:
# @author: Changjiang Cai, ccai1@stevens.edu, caicj5351@gmail.com
# @version: 0.0.1
# @creation date: 07-01-2020
# @last modified: Mon 13 Jan 2020 01:57:12 AM EST
import torch

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd import Variable

""" accuracy with threshold = 3 """
def valid_accu3(y_true_valid, y_pred_valid, thred = 3.0): 
    #epsilon = 0.0000000001
    right_guess = torch.le(torch.abs(y_true_valid - y_pred_valid), thred).float()
    accu3 = torch.mean(right_guess)
    return accu3

""" this loss function is adapoted from GANet, CVPR2019 """
class MyLoss2Function(Function):
    @staticmethod
    def forward(ctx, input1, input2, thresh=1, alpha=2):
        ctx.thresh = thresh
        ctx.alpha = alpha
        ctx.diff = input1 - input2
        temp=torch.abs(ctx.diff)
        temp[temp < ctx.thresh] = temp[temp < ctx.thresh] ** 2 / ctx.thresh
        tag = (temp <= ctx.thresh + ctx.alpha) & (temp >= ctx.thresh)
        temp[tag]=temp[tag] * 2 - (temp[tag] - ctx.thresh) ** 2 /(2.0 * ctx.alpha) - ctx.thresh
        temp[temp > ctx.thresh + ctx.alpha] += (ctx.alpha / 2.0)
        
        return torch.mean(temp)
    
    @staticmethod
    def backward(ctx, gradOutput):
        scale = torch.abs(ctx.diff)
        scale[scale > ctx.thresh + ctx.alpha] = 1
        tag = (scale <= ctx.thresh + ctx.alpha) & (scale >= ctx.thresh)
        scale[tag] = 2 - (scale[tag] - ctx.thresh) / ctx.alpha
        tag = scale < ctx.thresh
        scale[tag] = 2*scale[tag] / ctx.thresh
        ctx.diff[ctx.diff > 0] = 1.0
        ctx.diff[ctx.diff < 0] = -1.0
        ctx.diff = ctx.diff * scale * gradOutput / scale.numel()
        #return ctx.diff, Variable(torch.Tensor([0]))
        return ctx.diff, Variable(torch.Tensor([0])), None, None

""" this loss function is adapoted from GANet, CVPR2019 """
class MyLoss2(nn.Module):
    def __init__(self, thresh=1, alpha=2):
        super(MyLoss2, self).__init__()
        self.thresh = thresh
        self.alpha = alpha
    def forward(self, input1, input2):
        result = MyLoss2Function.apply(input1, input2, self.thresh, self.alpha)
        return result


