from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math


from ..baselines.GCNet.models.gcnet import GCNet
from .embednetwork import embed_net
from .bilateral import bilateralFilter
from src.net_init import net_init_v0
from src.modules.cost_volume import cost_volume_faster

"""
our network
"""
# adapted from GCNet:
# updated: using Python Inheritance:
#class AttenStereoNet(nn.Module):
class AttenStereoNet(GCNet):
    def __init__(self, 
            maxdisp=192, 
            sigma_s = 0.7, # 1.7: 13 x 13; 0.3 : 3 x 3;
            sigma_v = 0.1, 
            isEmbed = True, 
            dilation = 1,
            cost_filter_grad = False,
            is_kendall_version = True, # excatly following the structure in Kendall's GCNet paper;
            is_quarter_size_cost_volume_gcnet = False # cost volume in quarter image size, i.e., [D/4, H/4, W/4]
            ):
        
        super(AttenStereoNet, self).__init__(
            maxdisp = maxdisp,
            #newly added arguments:
            is_kendall_version = is_kendall_version,
            is_quarter_size_cost_volume_gcnet = is_quarter_size_cost_volume_gcnet
            )

        self.isEmbed = isEmbed # True of False
        self.sigma_s = sigma_s
        self.sigma_v = sigma_v
        self.dilation = dilation
        self.cost_filter_grad = cost_filter_grad
        """ embedding network """
        if self.isEmbed:
            print(' Enable Embedding Network!!!')
            self.embednet = embed_net()
            self.bifilter = bilateralFilter(sigma_s, sigma_v, isCUDA = True, dilation = self.dilation)
        else:
            self.embednet = None
            self.bifilter = None

        """ the followind initilization is omitted due to inheritance from GCNet """
        net_init_v0(self)
        print ("[***] attenStereoNet_embed_gcnet weights inilization done!")

    
    def forward(self, imgLeft, imgRight):
        N, C, H, W = imgLeft.size()[:]
        assert C == 3, 'should be RGB images as input'
        #print ("[???] imgLeft shape: ", imgLeft.shape)
        
        #NOTE: newly added for EBF, try to reduece cost volume size for GPU memory limitation;
        # add one downsample operation:
        if self.is_quarter_size_cost_volume_gcnet:
            img_ds_scale = 2
            imgl = F.interpolate(imgLeft,  [H//2, W//2], mode='bilinear', align_corners=True)
            imgr = F.interpolate(imgRight, [H//2, W//2], mode='bilinear', align_corners=True)
        else:
            img_ds_scale = 1
            imgl = imgLeft
            imgr = imgRight

        # feature extraction; 
        f_imgl = self.feature_extraction(imgl)
        f_imgr = self.feature_extraction(imgr)

        # cost volume
        cv = cost_volume_faster(f_imgl, f_imgr, d = self.maxdisp//(2*img_ds_scale))
        #print ("[???] cv shape: ", cv.shape)
         
        
        if not self.isEmbed:
            embed = None
        else:
            # downscale x to [N,C,H/2, W/2] then fed into embeddingnet,
            # because the cost volume generated below is in shape [N,C,D/2, H/2, W/2]
            left_scale = F.interpolate(imgLeft, 
                    [imgLeft.size()[2]//(2*img_ds_scale), imgLeft.size()[3]//(2*img_ds_scale)], 
                    mode='bilinear', align_corners=True)
            #print ('[???] left_scale shape', left_scale.shape)
            """ embed shape [2, 64, 64, 128]"""
            embed = self.embednet(left_scale)
            #print ('[???] embed shape', embed.shape)
            
            D = cv.size()[2]
            #print ('[???] cost shape', cost.shape)
            
            # NOTE: this might be the memory consuming!!!
            # NO sure this torch.no_grad() will distory the training or not !!!!
            #with torch.set_grad_enabled(False):
            #with torch.set_grad_enabled(True):
            with torch.set_grad_enabled(self.cost_filter_grad):
                for d in range(0,D):
                    #print ('bilateral filtering cost volume slice %d/%d' %(d+1, D))
                    cv_d_slice = cv[:,:,d,:,:].contiguous()
                    #print ('[???] cv_d_slice shape', cv_d_slice.shape)
                    cv[:,:,d,:,:] = self.bifilter(embed, cv_d_slice)
             
        cv = cv.contiguous()
        # cost volume aggregation
        if self.is_kendall_version:
            out = self.cost_aggregation_kendall(cv)
        else:
            out = self.cost_aggregation(cv)
        
        out = out.view(N, self.maxdisp//img_ds_scale, H//img_ds_scale, W//img_ds_scale)
        #NOTE: This is right!!! Updated on 04/12/2020;
        # We should upsample the cost volume (now in quarter size) to full size before the soft-argmin operation;
        # which can gaurantee that the regressed disparity range should be in [0, D) (instead of in [0, D/4));
        if self.is_quarter_size_cost_volume_gcnet:
            # corresponding to the first downsampling at the beginning to the input image pair;
            out = out[:,None,...] # add channel C first, i.e., chang [N,D,H,W] to [N,C=1,D,H,W];
            out = F.interpolate(out, [self.maxdisp, H, W], mode='trilinear', align_corners=True) # in size [N,C=1,D,H,W];
            out = torch.squeeze(out, 1) # in size [N,D,H,W]
        prob = F.softmax(out, 1)
        #disp = self.disparityregression(prob, maxdisp=self.maxdisp//img_ds_scale)
        #NOTE: This is right!!! Updated on 04/12/2020;
        disp = self.disparityregression(prob, maxdisp=self.maxdisp)

        #if self.is_quarter_size_cost_volume_gcnet:
        #    # NOTE: newly added for SGA: upsampling operation, 
        #    # corresponding to the first downsampling at the beginning to the input image pair;
        #    disp = F.interpolate(disp[:,None,...], [H, W], mode='bilinear', align_corners=True)
        #    disp = torch.squeeze(disp,1) # [N,H,W]
        return disp, embed