from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math
from .submodule import *


class hourglass(nn.Module):
    def __init__(self, inplanes):
        super(hourglass, self).__init__()

        self.conv1 = nn.Sequential(convbn_3d(inplanes, inplanes*2, kernel_size=3, stride=2, pad=1),
                                   nn.ReLU(inplace=True))

        self.conv2 = convbn_3d(inplanes*2, inplanes*2,
                               kernel_size=3, stride=1, pad=1)

        self.conv3 = nn.Sequential(convbn_3d(inplanes*2, inplanes*2, kernel_size=3, stride=2, pad=1),
                                   nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(convbn_3d(inplanes*2, inplanes*2, kernel_size=3, stride=1, pad=1),
                                   nn.ReLU(inplace=True))

        self.conv5 = nn.Sequential(nn.ConvTranspose3d(inplanes*2, inplanes*2, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
                                   nn.BatchNorm3d(inplanes*2))  # +conv2

        self.conv6 = nn.Sequential(nn.ConvTranspose3d(inplanes*2, inplanes, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
                                   nn.BatchNorm3d(inplanes))  # +x

    def forward(self, x, presqu, postsqu):

        out = self.conv1(x)  # in:1/4 out:1/8
        pre = self.conv2(out)  # in:1/8 out:1/8
        if postsqu is not None:
            pre = F.relu(pre + postsqu, inplace=True)
        else:
            pre = F.relu(pre, inplace=True)

        out = self.conv3(pre)  # in:1/8 out:1/16
        out = self.conv4(out)  # in:1/16 out:1/16

        if presqu is not None:
            post = F.relu(self.conv5(out)+presqu,
                          inplace=True)  # in:1/16 out:1/8
        else:
            post = F.relu(self.conv5(out)+pre, inplace=True)

        out = self.conv6(post)  # in:1/8 out:1/4

        return out, pre, post


class PSMNet(nn.Module):
    def __init__(self, maxdisp):
        super(PSMNet, self).__init__()
        self.maxdisp = maxdisp

        self.feature_extraction = feature_extraction()

        self.dres0 = nn.Sequential(convbn_3d(64, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.dres1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1))

        self.dres2 = hourglass(32)

        self.dres3 = hourglass(32)

        self.dres4 = hourglass(32)

        self.classif1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif2 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif3 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * \
                    m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()


    #-------------------
    #NOTE: this function is the original version from PSMNet code ??? TOO SLOW??;
    #-------------------
    def get_costVolume(self, x, y):
        """
        args: 
            x : left feature,  in size [N,C,H/4,W/4]
            y : right feature, in size [N,C,H/4,W/4]
        return:
            cost: cost volume in size [N,2C,D/4,H/4,W/4]
        """
        # matching
        assert(x.is_contiguous() == True)
        N0, C0, H0, W0 = x.size()[:]
        cost = Variable(torch.FloatTensor(N0, C0*2, self.maxdisp//4, H0, W0).zero_()).cuda()
        #cost = torch.tensor((), dtype=torch.float32).new_zeros((N0,2*C0,self.maxdisp//4,H0,W0),requires_grad=True).cuda()
        
        for i in range(self.maxdisp//4):
            if i > 0:
                cost[:, :C0, i, :, i:] = x[:, :, :, i:]
                cost[:, C0:, i, :, i:] = y[:, :, :, :-i]
            else:
                cost[:, :C0, i, :, :] = x
                cost[:, C0:, i, :, :] = y
        
        return cost.contiguous()

    #NOTE: faster!! But consume memory than the above one???
    def cost_volume_faster(self, x, y):
        """
        args:
            x : left feature,  in size [N,C,H/4,W/4]
            y : right feature, in size [N,C,H/4,W/4]
        return:
            cost: cost volume in size [N,2C,D/4,H/4,W/4]
        """
        N0, C0, H0, W0 = x.size()[:]
        cv_list = []
        # Pads the input tensor boundaries with zero.
        # padding = (padding_left, padding_right, padding_top, padding_bottom) 
        # along the [H, W] dim; 
        y_pad = nn.ZeroPad2d((self.maxdisp//4, 0, 0, 0))(y)

        for d in reversed(range(self.maxdisp//4)):
            x_slice = x
            #Note added by CCJ:
            #Note that you don’t need to use torch.narrow or select, 
            #but instead basic indexing will do it for you.
            y_slice = y_pad[:,:,:,d:d+W0]
            xy_temp = torch.cat((x_slice, y_slice), 1)
            cv_list.append(xy_temp)
        
        #Stacks a list of rank-R tensors into one rank-(R+1) tensor.
        cv = torch.stack(cv_list, 2)
        #assert(cv.is_contiguous() == True)
        #print ("[???] cv shape = ", cv.shape)
        return cv

    def forward(self, left, right):

        x = self.feature_extraction(left) # left feature
        y = self.feature_extraction(right) # right feature

        # matching
        cost = self.get_costVolume(x, y)
        #cost = self.cost_volume_faster(x, y)

        cost0 = self.dres0(cost)
        cost0 = self.dres1(cost0) + cost0

        out1, pre1, post1 = self.dres2(cost0, None, None)
        out1 = out1+cost0

        out2, pre2, post2 = self.dres3(out1, pre1, post1)
        out2 = out2+cost0

        out3, pre3, post3 = self.dres4(out2, pre1, post2)
        out3 = out3+cost0

        cost1 = self.classif1(out1)
        cost2 = self.classif2(out2) + cost1
        cost3 = self.classif3(out3) + cost2

        if self.training:
            # updated by CCJ: due to deprecated warning!
            #cost1 = F.upsample(cost1, [self.maxdisp,left.size()[2],left.size()[3]], mode='trilinear')
            #cost2 = F.upsample(cost2, [self.maxdisp,left.size()[2],left.size()[3]], mode='trilinear')
            cost1 = F.interpolate(cost1, [self.maxdisp, left.size()[2], left.size()[3]],
                                  mode='trilinear', align_corners=True)
            cost2 = F.interpolate(cost2, [self.maxdisp, left.size()[2], left.size()[3]],
                                  mode='trilinear', align_corners=True)

            cost1 = torch.squeeze(cost1, 1)
            pred1 = F.softmax(cost1, dim=1)
            pred1 = disparityregression(self.maxdisp)(pred1)

            cost2 = torch.squeeze(cost2, 1)
            pred2 = F.softmax(cost2, dim=1)
            pred2 = disparityregression(self.maxdisp)(pred2)

        #cost3 = F.upsample(cost3, [self.maxdisp,left.size()[2],left.size()[3]], mode='trilinear')
        cost3 = F.interpolate(cost3, [self.maxdisp, left.size()[2], left.size()[3]],
                              mode='trilinear', align_corners=True)
        cost3 = torch.squeeze(cost3, 1)
        pred3 = F.softmax(cost3, dim=1)
        # For your information: This formulation 'softmax(c)' learned "similarity"
        # while 'softmax(-c)' learned 'matching cost' as mentioned in the paper.
        # However, 'c' or '-c' do not affect the performance because feature-based cost volume provided flexibility.
        pred3 = disparityregression(self.maxdisp)(pred3)

        if self.training:
            return pred1, pred2, pred3
        else:
            return pred3
