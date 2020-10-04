import numpy as np
import re
import cv2
import sys
import os

""" check the filename path can be loaded"""
def data_load_check(data_path, filenames_file):
    print ('load {} and {}'.format(data_path, filenames_file))
    with open(filenames_file) as f:
        lines = f.readlines()
        split_line = [l.rstrip().split(' ') for l in lines if not l.startswith('#')]
    i = 0
    n = len(split_line)
    for j in split_line:
        l_image_path = os.path.join(data_path, j[0])
        r_image_path = os.path.join(data_path, j[1])
        l_gt_path    = os.path.join(data_path, j[2])
        r_gt_path    = os.path.join(data_path, j[3])
        l_img = cv2.imread(l_image_path)
        r_img = cv2.imread(r_image_path)
        l_gt = load_pfm(l_gt_path)
        #r_gt = load_pfm(r_gt_path)
        #print ("processing {:>5d} / {:>5d} ".format(i, n))
        print ("processing {:0>5d} / {:0>5d} ".format(i, n))
        i += 1
    
    print ('all {} data have been loaded!'.format(len(split_line)))


def pascal_voc_2012_color_map(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap


def colormap_jet(img):
    return cv2.cvtColor(cv2.applyColorMap(np.uint8(img), 2), cv2.COLOR_BGR2RGB)

def writeKT15FalseColors(disp, max_val = -1.0):
    clr_map = np.array(
            [[0,0,0,114],[0,0,1,185],[1,0,0,114],[1,0,1,174],
             [0,1,0,114],[0,1,1,185],[1,1,0,114],[1,1,1, 1.0e-16],])
    #print (clr_map.shape)
    sum_ = np.sum(clr_map[:,3])
    weights = sum_ / clr_map[:, 3] #relative weights
    cumsum = np.zeros((8), dtype=np.float32)
    for i in range(0,7):
        cumsum[i+1] = cumsum[i] + clr_map[i,3]/sum_
    if max_val <= 0:
        max_val = disp.max()
    print (max_val)
    print (weights, '\n', cumsum)
    rst = np.clip(disp/ max_val, .0, 1.0)
    #print (rst.shape)
    h = int(rst.shape[0])
    w = int(rst.shape[1])
    #print (h,w, type(h), type(w))
    disp_out = np.zeros([h, w, 3], dtype=np.float32)
    for i in range(0, h):
        for j in range(0, w):
            val = rst[i,j]
            # find bin;
            for k in range(0,7):
                if val < cumsum[k+1]:
                    #print (i,j,k, val, cumsum[k+1])
                    break
            # compute red/green/blue values
            tmp_w = 1.0 - (val-cumsum[k]) * weights[k];

            # r, g, b
            disp_out[i,j,:] = (tmp_w*clr_map[k,:3] + (1.0-tmp_w)*clr_map[k+1,:3])*255.0
            if i == 200 and j == 300:
                print (i,j,k, val, cumsum[k+1])
                print (disp_out[200,300,:])

    return disp_out.astype(np.uint8)

if __name__ == "__main__":    

    from pfmutil import readPFM, show_uint8, show
    import time
    if 0:
        disp = readPFM('/data/ccjData/datasets/KITTI-2015/training/disp_occ_0_pfm/000073_10.pfm')
        disp[disp == np.inf] = .0
        print (disp.shape)
        since = time.time()
        rst = writeKT15FalseColors(disp)
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}s'.format(time_elapsed))
        show_uint8(rst)
    
    if 1:
        from cython import writeKT15FalseColor as KT15FalseClr
        from cython import writeKT15ErrorLogColor as KT15LogClr
        since = time.time()
        for i in range(5,6):
            disp_gt = readPFM('/data/ccjData/datasets/KITTI-2015/training/disp_occ_0_pfm/%06d_10.pfm' %i)
            disp_gt[disp_gt == np.inf] = .0
            #disp = readPFM('/media/ccjData2/research-projects/GCNet/results/gcnet-F8-RMSp-sfF3k-epo31-4dsConv-k5-testKT15/disp-epo-030/%06d_10.pfm' %i)
            disp = readPFM('/home/ccj/atten-stereo/results/ganet-sfepo10-kt15epo100/val-30/%06d_10.pfm' %i)
            disp[disp == np.inf] = .0
            rst_disp = KT15FalseClr.writeKT15FalseColor(disp).astype(np.uint8)
            rst_disp_gt = KT15FalseClr.writeKT15FalseColor(disp_gt).astype(np.uint8)
            show_uint8(rst_disp, title = 'disp_kt15_false_color')
            show_uint8(rst_disp_gt, title = 'disp_gt_kt15_false_color')
            err = np.abs(disp_gt - disp)
            rst_err = KT15LogClr.writeKT15ErrorDispLogColor(disp, disp_gt)
            show_uint8(rst_err, title = 'err_kt15_log_color')
            show(err, title = 'err gray color')

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}s'.format(time_elapsed))


# see the GPU runtime counting code from https://github.com/sacmehta/ESPNet/issues/57;
def computeTime(model, device='cuda'):
    inputs = torch.randn(1, 3, 512, 1024)
    if device == 'cuda':
        model = model.cuda()
        inputs = inputs.cuda()

    model.eval()
    i = 0
    time_spent = []
    while i < 100:
        start_time = time.time()
        with torch.no_grad():
            _ = model(inputs)

        if device == 'cuda':
            torch.cuda.synchronize()  # wait for cuda to finish (cuda is asynchronous!)
        if i != 0:
            time_spent.append(time.time() - start_time)
        i += 1
    print('Avg execution time (ms): {:.3f}'.format(np.mean(time_spent)))