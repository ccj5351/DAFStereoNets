from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math
#from .psmnet_submodule import *
from .psmnet_submodule import PSMNet, disparityregression

from .embednetwork import embed_net
from .bilateral import bilateralFilter
#from src.net_init import net_init_v0
from src.modules.cost_volume import cost_volume_faster

"""
our network
"""
# adapted from PSMNet:
# updated: using Python Inheritance:
#class AttenStereoNet(nn.Module):
class AttenStereoNet(PSMNet):
    def __init__(self, maxdisp=192, sigma_s = 0.7, # 1.7: 13 x 13; 0.3 : 3 x 3;
            sigma_v = 0.1, isEmbed = True, 
            dilation = 1,
            cost_filter_grad = False
            ):
        super(AttenStereoNet, self).__init__(maxdisp = maxdisp)
        #self.maxdisp = maxdisp
        self.isEmbed = isEmbed # True of False
        self.sigma_s = sigma_s
        self.sigma_v = sigma_v
        self.dilation = dilation
        self.cost_filter_grad = cost_filter_grad
        """ embedding network """
        if self.isEmbed:
            print(' Enable Embedding Network!!!')
            self.embednet = embed_net()
            #the module layer:
            self.bifilter = bilateralFilter(sigma_s, sigma_v, isCUDA = True, dilation = self.dilation)
            # the function version
            #self.bifilter = bilateralFilter
        else:
            self.embednet = None
            self.bifilter = None

        """ the followind initilization is omitted due to inheritance from PSMNet """

    def forward(self, left, right):

        x = self.feature_extraction(left) # left feature
        y = self.feature_extraction(right) # right feature

        # matching volume, in size [N,2C,D/4, H/4, W/4];
        cost = cost_volume_faster(x, y, self.maxdisp//4)
        
        if not self.isEmbed:
            embed = None
        else:
            # downscale x to [N,C,H/4, W/4] then fed into embeddingnet,
            # because the cost volume generated below is in shape [N,C,D/4, H/4, W/4]
            left_scale = F.interpolate(left, [left.size()[2]//4, left.size()[3]//4], 
                    mode='bilinear', align_corners=True)
            #print ('[???] left shape', left.shape)
            #print ('[???] left_scale shape', left_scale.shape)
            """ embed shape [2, 64, 64, 128]"""
            embed = self.embednet(left_scale)
            #print ('[???] embed shape', embed.shape)
            
            """ cost shape [2, 64, 36, 64, 128]"""
            N, C, D, H, W = cost.size()[:]
            #print ('[???] cost shape', cost.shape)
            
            # NOTE: this might be the memory consuming!!!
            # NO sure this torch.no_grad() will distory the training or not !!!!
            #with torch.set_grad_enabled(False):
            #with torch.set_grad_enabled(True):
            with torch.set_grad_enabled(self.cost_filter_grad):
                for d in range(0,D):
                #for d in range(0,1):
                    #print ('bilateral filtering cost volume slice %d/%d' %(d+1, D))
                    # apply bilateral filter to cost volume [N,C,H,W];
                    cv_d_slice = cost[:,:,d,:,:].contiguous()
                    #print ('[???] cv_d_slice shape', cv_d_slice.shape)
                    cost[:,:,d,:,:] = self.bifilter(embed, cv_d_slice)
             
        cost = cost.contiguous()

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
            return pred1, pred2, pred3, embed
        else:
            return pred3, embed
