# !/usr/bin/env python3
# -*-coding:utf-8-*-
# @file: eval_stereo.py
# @brief:
# @author: Changjiang Cai, ccai1@stevens.edu, caicj5351@gmail.com
# @version: 0.0.1
# @creation date: 25-09-2018
# @last modified: Mon 20 Apr 2020 02:42:32 AM EDT

from datetime import datetime

import os
import sys
import numpy as np

import src.cpp.lib.libevaluate_stereo_training as kt_eval
import src.cpp.lib.libeth3d_2_view_evaluate as eth3d_eval
import src.cpp.lib.libevaldisp_mbv3 as mbv3_eval
import argparse  # argument parser.

from os import listdir
from os.path import isfile, join

mbV3_weights = {'Adirondack': 0.08, 'ArtL':0.08, 'Jadeplant': 0.08, 'Motorcycle': 0.08, 'MotorcycleE':0.08, 'Piano':0.08,'PianoL':0.04,'Pipes':0.08,'Playroom':0.04,'Playtable':0.04,'PlaytableP':0.08,'Recycle':0.08,'Shelves':0.04,'Teddy':0.08,'Vintage':0.04,}

mbV3_training_dataset = ['Adirondack','ArtL','Jadeplant','Motorcycle','MotorcycleE','Piano','PianoL','Pipes','Playroom','Playtable','PlaytableP','Recycle','Shelves','Teddy','Vintage',]

eth3d_training_dataset = [ 
        # fold 1 : 14 images;
        'delivery_area_1s','electro_1l','electro_2s' , 'facade_1s','forest_1s' , 'forest_2s' , 'playground_1l' , 'playground_1s' , 'playground_2l' , 'playground_3l' , 'playground_3s','terrains_1l' , 'terrains_1s', 'terrains_2l',
        # fold 2: 13 images;
        'terrains_2s' , 'playground_2s' , 'delivery_area_3l' , 'delivery_area_1l' , 'delivery_area_3s' , 'terrace_1s' , 'terrace_2s','electro_2l' , 'delivery_area_2l' , 'delivery_area_2s' , 'electro_1s','electro_3s' ,'electro_3l',]

""" Updated (2018/11/17/Saturday): for rounddisp = isint or not;"""
def _read_rounddisp_from_calibF_files(
        FullResolution_ground_truth_path, # always use full resolution gt for evaluation;
        img_names):
    n = len(img_names)
    rounddisps = np.zeros((n,), dtype = np.int32)
    for i in range(0, n):
        with open(os.path.join(FullResolution_ground_truth_path, img_names[i] + "/calib.txt")) as f:
            lines = f.readlines()
            for k in range(0, len(lines)):
                line =  lines[k].strip("\n").split("=")
                if line[0] == "isint":
                    rounddisps[i] = np.int32(line[1])
    return rounddisps

def _read_ndisp_from_calib_files(ground_truth_path, img_names):
    n = len(img_names)
    ndisps = np.zeros((n,), dtype = np.int32)
    for i in range(0, n):
        with open(os.path.join(ground_truth_path, img_names[i] + "/calib.txt")) as f:
            lines = f.readlines()
            for k in range(0, len(lines)):
                line =  lines[k].strip("\n").split("=")
                if line[0] == "ndisp":
                    ndisps[i] = np.int32(line[1])
    return ndisps

""" for calcualting the averaged error for middlebury V3 dataset"""
def _generate_weigths_mb_v3(img_names):
    n = len(img_names)
    weights = np.zeros((n,), dtype = np.float)
    for i in range(0, len(img_names)):
        weights[i] = mbV3_weights[img_names[i]]
    return weights


def get_imgs_from_dir(result_dir):
    #print ('result_dir = ', result_dir)
    onlyfiles = [f for f in listdir(result_dir) if isfile(join(result_dir, f)) and '.pfm' in f]
    onlyfiles.sort()
    #print (onlyfiles)
    return onlyfiles

def get_imgs_from_files(
        file_txt_path # e.g., == "./filenames/kitti15_train.txt"
        ):
    print (file_txt_path)
    with open(file_txt_path) as f:
        # each line has this format: 
        # colored_0/000000_10.png colored_1/000000_10.png disp_occ/000000_10.png
        lines = f.readlines()
        # now get the first element;
        imgs = [l.rstrip().split(' ')[0] for l in lines if not l.startswith('#')]
        # now extract the image name without file extension, i.e., we get '000000_10' here;
        imgs = [i[i.rfind('/')+1: i.rfind('.')] for i in imgs]
    return imgs

def get_imgs_from_files_2(
        file_txt_path # e.g., == "./filenames/kitti15_train.txt"
        ):
    with open(file_txt_path) as f:
        # each line has this format: 
        # 000000_10.png,000001_10.png,...
        imgs = f.read().rstrip().split(',')
        # now extract the image name without file extension, i.e., we get '000000_10' here;
        imgs = [ i[0:i.rfind('.')] for i in imgs]
    return imgs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='disp eval')
    parser.add_argument('--dataroot', dest='my_root', default = '/home/ccj/')
    parser.add_argument('--ktimgdir', dest='kt_img_dir', default = '/home/ccj/PKLS/datasets/KITTI-2015/training/')
    parser.add_argument('--resultdir', dest='result_dir', default = '/home/ccj/CBMV-MP/results/')
    parser.add_argument('--dataset', dest='dataset', default = 'kt15', help = 'mbv3h, mbv3f, mbv3q, eth3d, kt15, kt12, or sf')
    parser.add_argument('--file', dest='file', default = '')
    parser.add_argument('--first_n', dest='first_n', type=int,default = -1)
    parser.add_argument('--mbv3gt', dest='mbv3gt', type=str, default = '') 
    parser.add_argument('--mbv3badthresh', dest='mbv3badthresh', type=float, default = 0)
    args = parser.parse_args()
    eval_types = [#('rf', '_rf_disp0PKLS'), 
                  #('cbmv_net' , '_lep_disp0PKLS'),
                  #('cbmv_net' , ''),
                  ('atten-stero' , ''), 
                  ]
    """ evaluate kt 15 """
    img_list = get_imgs_from_dir(args.result_dir)
    img_list_no_file_extension = [i[0:i.rfind('.')] for i in img_list]
    if args.first_n > 0:
        N = len(img_list_no_file_extension)
        img_list_no_file_extension = img_list_no_file_extension[0:max(args.first_n,N)]
    if args.file != '':
        print ('get image list from file {}'.format(args.file))
        #img_list_no_file_extension = get_imgs_from_files_2(args.file)
        img_list_no_file_extension = get_imgs_from_files(args.file)
        #print (img_list_no_file_extension)

    print ('Py3: to test {} images : {} ... {} '.format(len(img_list_no_file_extension), 
            img_list_no_file_extension[0], img_list_no_file_extension[-1]))

    timeStamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    eval_type = eval_types[0]
    if args.dataset.lower() == 'kt15':
        post_score = kt_eval.evaluate_training_kt2015_from_img_list( args.kt_img_dir, args.result_dir, eval_type[0], eval_type[1], timeStamp, img_list_no_file_extension)
        print('* {:>7s} = {:>6.3f}%(noc), {:>6.3f}%(all)'.format(eval_type[0], post_score[4]*100.0, post_score[11]*100.0))
        print('* {:>7s} = {:>6.3f}(noc-mae), {:>6.3f}(all-mae)'.format(eval_type[0], post_score[16], post_score[22]))
        print('* {:>7s} = {:>6.3f}(noc-rmse), {:>6.3f}(all-rmse)'.format(eval_type[0], post_score[19], post_score[25]))

        """ save as csv file, Excel file format """
        csv_file = os.path.join(args.result_dir, 'kt-err.csv')
        print ("write ", csv_file, "\n")
        # updated for mae and rmse metric
        messg = timeStamp + ',{:>7s},kt15,bad-3.0-noc,{:>6.3f},bad-3.0-all,{:>6.3f},mae-noc,{:>6.3f},mae-all,{:>6.3f},rmse-noc,{:>6.3f},rmse-all,{:>6.3f},fileDir={},for log,{:>6.3f}%(noc); {:>6.3f}%(all)\n'.format(
            eval_type[0], post_score[4]*100.0, post_score[11]*100.0, 
            post_score[16], post_score[22], post_score[19], post_score[25],
            args.result_dir,
            post_score[4]*100.0, post_score[11]*100.0
            )
        with open( csv_file, 'a') as fwrite:
            fwrite.write(messg)


    elif args.dataset.lower() == 'kt12':
        print ("Evaluating KT12:")
        # updated on 2020/04/20 by CCJ: using evaluate from img_list:
        #post_score = kt_eval.evaluate_training(args.kt_img_dir, args.result_dir, eval_type[0], eval_type[1], timeStamp, 0, 194, 0)
        post_score = kt_eval.evaluate_training_from_img_list(args.kt_img_dir, args.result_dir, eval_type[0], eval_type[1], timeStamp, img_list_no_file_extension)
        print('* {:>7s} = {:>6.3f}%(bad3-noc), {:>6.3f}%(bad3-all)'.format(eval_type[0], post_score[0]*100.0, post_score[1]*100.0 ))
        print('* {:>7s} = {:>6.3f}(mae-noc),   {:>6.3f}(mae-all)'.format(eval_type[0], post_score[2], post_score[3]))
        print('* {:>7s} = {:>6.3f}(rmse-noc),  {:>6.3f}(rmse-all)'.format(eval_type[0], post_score[4], post_score[5]))

        """ save as csv file, Excel file format """
        csv_file = os.path.join(args.result_dir, 'kt-err.csv')
        print ("write ", csv_file, "\n")
        messg = timeStamp + ',{:>7s},kt12,bad-3.0-noc,{:>6.3f},bad-3.0-all,{:>6.3f},mae-noc,{:>6.3f},mae-all,{:>6.3f},rmse-noc,{:>6.3f},rmse-all,{:>6.3f},fileDir={},for log,{:>6.3f}%(noc); {:>6.3f}%(all)\n'.format(
                eval_type[0], post_score[0]*100.0, post_score[1]*100.0, 
                post_score[2], post_score[3], post_score[4], post_score[5],
                args.result_dir,
                post_score[0]*100.0, post_score[1]*100.0
                )
        with open( csv_file, 'a') as fwrite:
            fwrite.write(messg)

    elif args.dataset.lower() in ['mbv3h', 'mbv3f', 'mbv3q']: 
        #ground_truth_path = '/media/ccjData2/datasets/MiddleBury/MiddEval3/trainingF/'#Here we use trainingF;
        ground_truth_path = os.path.join(args.my_root, 'datasets/MiddleBury/MiddEval3/trainingF/')#Here we use trainingF;
        if args.mbv3gt != '':
            ground_truth_path = args.mbv3gt  #Here we use one of trainingF/H/Q;
        #train_data_path = '/media/ccjData2/datasets/MiddleBury/MiddEval3/training' + args.dataset.upper()[-1]
        train_data_path = os.path.join(args.my_root, 'datasets/MiddleBury/MiddEval3/training' + args.dataset.upper()[-1])
        ndisps = _read_ndisp_from_calib_files(train_data_path, mbV3_training_dataset)
        rounddisps = _read_rounddisp_from_calibF_files(ground_truth_path, mbV3_training_dataset)
        badthresh = 2.0
        if args.mbv3badthresh > 0: 
            badthresh = args.mbv3badthresh
        #print ("ndisps = {}, type = {}".format(ndisps, type(ndisps)))
        #print ("rounddisps = {}, type = {}".format(rounddisps, type(rounddisps)))
        PyMetrics = mbv3_eval.evaluate_mbv3(args.result_dir, ground_truth_path, badthresh,eval_type[1], ndisps.astype(np.int32), rounddisps.astype(np.int32), mbV3_training_dataset)
        n = len(mbV3_training_dataset)
        if 0:
            for i in range(0, n):
                print ("* {:>7s} = {:>12s}: {:>5.2f}%(noc-bad{:>2.1f}F), {:>5.2f}%(all-bad{:>2.1f}F), {:>5.2f}(noc-mae), {:>5.2f}(all-mae), {:>5.2f}(noc-rmse), {:>5.2f}(all-rmse)\n".format(
                        eval_type[0], 
                        mbV3_training_dataset[i], 
                        PyMetrics[6*i+2]*100.0, badthresh, 
                        PyMetrics[6*i]*100.0, badthresh,
                        PyMetrics[6*i+3], PyMetrics[6*i+1],
                        PyMetrics[6*i+5], PyMetrics[6*i+4],
                        ))
        weights = _generate_weigths_mb_v3(mbV3_training_dataset)
        bad_noc = np.zeros((n,), np.float32)
        bad_all = np.zeros((n,), np.float32)
        mae_noc = np.zeros((n,), np.float32)
        mae_all = np.zeros((n,), np.float32)
        rmse_noc = np.zeros((n,), np.float32)
        rmse_all = np.zeros((n,), np.float32)

        for i in range(0, n):
            bad_all[i] = PyMetrics[6*i]
            bad_noc[i] = PyMetrics[6*i+2]
            # newly added for mae and rmse error metric on 2019/08/31;
            mae_all[i] =  PyMetrics[6*i + 1]
            mae_noc[i] =  PyMetrics[6*i + 3]
            rmse_all[i] = PyMetrics[6*i + 4]
            rmse_noc[i] = PyMetrics[6*i + 5]

        avg_bad_all = sum(bad_all * weights)
        avg_bad_noc = sum(bad_noc * weights) 
        avg_mae_all = sum(mae_all * weights)
        avg_mae_noc = sum(mae_noc * weights) 
        avg_rmse_all = sum(rmse_all * weights)
        avg_rmse_noc = sum(rmse_noc * weights)

        print ("avg_bad_all = {:>5.2f}%, avg_bad_noc = {:>5.2f}%\navg_all_mae = {:>5.2f}, avg_noc_mae = {:>5.2f}\navg_all_rmse = {:>5.2f}, avg_noc_rmse) = {:>5.2f}".format(
            avg_bad_all*100.0, avg_bad_noc*100.0, avg_mae_all, avg_mae_noc, avg_rmse_all, avg_rmse_noc))
                
        """ save as csv file, Excel file format """
        #csv_file = './results/eth-7-hard-img-tuing-err.csv'
        csv_file = os.path.join(args.result_dir, 'mbv3-err.csv')
        print ("write ", csv_file, "\n")
        messg = timeStamp + ',' + eval_type[0] + ','
        for i in range(0, n):
            messg += '{:>5.2f},{:>5.2f},'.format(bad_noc[i]*100.0, bad_all[i]*100.0)
        messg += 'avg-bad-err,{:>6.3f},{:>6.3f},avg-mae,{:>6.3f},{:>6.3f},avg-rmse,{:>6.3f},{:>6.3f},fileDir={}\n'.format(
                avg_bad_noc*100.0 , avg_bad_all*100.0, avg_mae_noc, avg_mae_all,avg_rmse_noc, avg_rmse_all,
                args.result_dir)
        with open( csv_file, 'a') as fwrite:
            fwrite.write(messg)
    elif args.dataset.lower() == 'eth3d':
        eth3d_training_dataset.sort()
        imgIdxEnd = len(eth3d_training_dataset)
        imgIdxStart = 0
        PyMetrics = np.zeros((imgIdxEnd - imgIdxStart,24), np.float64)
        print ("Evaluating ETH3D:")
        for i in range(imgIdxStart, imgIdxEnd):
            reconstruction_path = os.path.join(args.result_dir, eth3d_training_dataset[i] + eval_type[1] + '.pfm')
            #ground_truth_path = '/media/ccjData2/datasets/ETH3D/two_view_training/' + eth3d_training_dataset[i] + '/disp0GT.pfm'
            #mask_path = '/media/ccjData2/datasets/ETH3D/two_view_training/' + eth3d_training_dataset[i] + '/mask0nocc.png'
            ground_truth_path = os.path.join(args.my_root, 'datasets/ETH3D/two_view_training/' + eth3d_training_dataset[i] + '/disp0GT.pfm')
            mask_path = os.path.join(args.my_root, 'datasets/ETH3D/two_view_training/' + eth3d_training_dataset[i] + '/mask0nocc.png')
            #visualizations_path = os.path.join(args.result_dir, eth3d_training_dataset[i] + eval_type[1]) 
            visualizations_path = "" # if == “”, then will be disabled;
            create_training_visualizations = "" # "" means false, o.w. menas true; 
            res_type = eth3d_eval.eth3d_2_view_evaluate(reconstruction_path, ground_truth_path, mask_path, visualizations_path, create_training_visualizations, PyMetrics[i])
            if 0:
                print("* {:>7s} = {:>12s} : {:>5.2f}%(noc-bad1.0), {:>5.2f}%(all-bad1.0), {:>5.2f}%(noc-mae), {:>5.2f}%(all-mae), {:>5.2f}%(noc-rmse), {:>5.2f}%(all-rmse)\n".format(
                    eval_type[0], eth3d_training_dataset[i], 
                    PyMetrics[i,2]*100.0, PyMetrics[i,14]*100.0,
                    PyMetrics[i,5], PyMetrics[i,17],
                    PyMetrics[i,6], PyMetrics[i,18]
                    ))
            
        avg_noc_bad = np.sum(PyMetrics[:,2])/ (imgIdxEnd - imgIdxStart)
        avg_all_bad = np.sum(PyMetrics[:,14])/ (imgIdxEnd - imgIdxStart)
        avg_noc_mae = np.sum(PyMetrics[:,5])/ (imgIdxEnd - imgIdxStart)
        avg_all_mae = np.sum(PyMetrics[:,17])/ (imgIdxEnd - imgIdxStart)
        avg_noc_rmse = np.sum(PyMetrics[:,6])/ (imgIdxEnd - imgIdxStart)
        avg_all_rmse = np.sum(PyMetrics[:,18])/ (imgIdxEnd - imgIdxStart)
            
        """For ETH3D, the averaged error is just the 'all' result shown in the benchmark webpage"""
        print ("* {:>7s} : avg_noc_bad1.0 = {:>5.2f}%, avg_all_bad1.0 = {:>5.2f}%".format( eval_type[0],  avg_noc_bad*100.0, avg_all_bad*100.0))
        print ("* {:>7s} : avg_noc_mae = {:>5.2f}, avg_all_mae = {:>5.2f}".format( eval_type[0],  avg_noc_mae, avg_all_mae))
        print ("* {:>7s} : avg_noc_rmse = {:>5.2f}, avg_all_rmse = {:>5.2f}".format( eval_type[0], avg_noc_rmse, avg_all_rmse))
            
        """ save as csv file, Excel file format """
        csv_file =  os.path.join(args.result_dir, 'eth3d-err.csv')
        print ("write ", csv_file, "\n")
        messg = timeStamp + ',' + eval_type[0] + ','
        for i in range(imgIdxStart, imgIdxEnd):
            messg += '{:>5.2f},{:>5.2f},'.format(PyMetrics[i,2]*100.0, PyMetrics[i,14]*100.0)
        messg += 'avg-bad-err,{:>6.3f},{:>6.3f},avg-mae,{:>6.3f},{:>6.3f},avg-rmse,{:>6.3f},{:>6.3f},fileDir={}\n'.format(
                avg_noc_bad*100.0, avg_all_bad*100.0, 
                avg_noc_mae, avg_all_mae, 
                avg_noc_rmse, avg_all_rmse, 
                args.result_dir)
        with open( csv_file, 'a') as fwrite:
            fwrite.write(messg)
