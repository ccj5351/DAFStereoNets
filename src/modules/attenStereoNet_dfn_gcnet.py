from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math


from ..baselines.GCNet.models.gcnet import GCNet
from .dfn import filterGenerator, DynamicFilterLayerOneChannel, DynamicFilterLayer
#from .embednetwork import embed_net
#from .bilateral import bilateralFilter
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
            kernel_size = 5,
            crop_img_h = 256,
            crop_img_w = 512,
            isDFN = True, 
            dilation = 2,
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

        self.isDFN = isDFN # True of False
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.cost_filter_grad = cost_filter_grad
        """ dynamic filter network """
        if self.isDFN:
            print(' Enable Dynamic Filter Network!!!')
            self.dfn_generator = filterGenerator(
                F = 32, 
                dynamic_filter_size=(kernel_size, kernel_size), 
                in_channels = 3
                )
            #the module layer:
            self.dfn_layer = DynamicFilterLayer(kernel_size, dilation)
        else:
            print ('[!!!] No dfn_generator and dfn_layer!!')
            self.dfn_generator = None
            self.dfn_layer = None

        """ the followind initilization is omitted due to inheritance from GCNet """
        net_init_v0(self)
        print ("[***] attenStereoNet_dfn_gcnet weights inilization done!")

    
    def forward(self, imgLeft, imgRight):
        N, C, H, W = imgLeft.size()[:]
        assert C == 3, 'should be RGB images as input'
        #print ("[???] imgLeft shape: ", imgLeft.shape)

        if self.is_quarter_size_cost_volume_gcnet:
            img_ds_scale = 2
            imgl = F.interpolate(imgLeft,  [H//2, W//2], mode='bilinear', align_corners=True) #in size [N, 3, H/2, W/2];
            imgr = F.interpolate(imgRight, [H//2, W//2], mode='bilinear', align_corners=True) #in size [N, 3, H/2, W/2];
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
        
        if not self.isDFN:
            dfn_filter = None
            dfn_bias = None
        else:
            # downscale x to [N,C,H/2, W/2] then fed into embeddingnet,
            # because the cost volume generated below is in shape [N,C,D/2, H/2, W/2]
            left_scale = F.interpolate(imgLeft, 
                    [imgLeft.size()[2]//(2*img_ds_scale), imgLeft.size()[3]//(2*img_ds_scale)], 
                    mode='bilinear', align_corners=True)
            #print ('[???] left shape', left.shape)
            #print ('[???] left_scale shape', left_scale.shape)
            dfn_filter, dfn_bias = self.dfn_generator(left_scale)
            D = cv.size()[2]
            #print ('[???] cost size = ', cost.size())
            
            # NOTE: this might be the memory consuming!!!
            # NO sure this torch.no_grad() will distory the training or not !!!!
            #with torch.set_grad_enabled(False):
            #with torch.set_grad_enabled(True):
            with torch.set_grad_enabled(self.cost_filter_grad):
                for d in range(0, D):
                #for d in range(0,1):
                    #print ('DFN filtering cost volume slice %d/%d' %(d+1, D))
                    # apply DFN filter to cost volume [N,C,H,W];
                    cv_d_slice = cv[:,:,d,:,:].contiguous()
                    #print ('[???] cv_d_slice shape', cv_d_slice.shape)
                    cv[:,:,d,:,:] = self.dfn_layer(cv_d_slice, dfn_filter, dfn_bias)

        cv = cv.contiguous()
        # cost volume aggregation
        if self.is_kendall_version:
            out = self.cost_aggregation_kendall(cv)
        else:
            out = self.cost_aggregation(cv)

        out = out.view(N, self.maxdisp// img_ds_scale, H // img_ds_scale, W // img_ds_scale)
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
        #    # NOTE: newly added for PAC: upsampling operation, 
        #    # corresponding to the first downsampling at the beginning to the input image pair;
        #    disp = F.interpolate(disp[:,None,...], [H, W], mode='bilinear', align_corners=True)
        #    disp = torch.squeeze(disp,1) # [N,H,W]

        if self.training:
            return disp, dfn_filter, dfn_bias
        else:
            return disp, [dfn_filter, dfn_bias]