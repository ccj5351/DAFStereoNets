# !/usr/bin/env python3
# -*-coding:utf-8-*-
# @file: vkt2_plot.py
# @brief:
# @author: Changjiang Cai, ccai1@stevens.edu, caicj5351@gmail.com
# @version: 0.0.1
# @creation date: 29-07-2020
# @last modified: Wed 29 Jul 2020 12:56:11 PM EDT

import numpy as np
import json
import sys
import os
from os.path import join as pjoin
from datetime import datetime

def get_err_analysis( cates, scenes, model_name, err_json_file = 'vkt2-err.json'):
    # Opening JSON file
    with open(err_json_file) as f:
        dict_errs = json.load(f)
    dict_res = {}

    """ per condition """
    for tmp_cat in cates:
        avg_epe_per_cat =  0.
        avg_r1_per_cat =  0.
        avg_r3_per_cat =  0.
        num_per_cat =  0.
        
        for tmp_s in scenes:
            tmp_key = tmp_s + '/' + tmp_cat
            if tmp_key in dict_errs:
                avg_epe_per_cat += dict_errs[tmp_key][0]
                avg_r1_per_cat += dict_errs[tmp_key][1]
                avg_r3_per_cat += dict_errs[tmp_key][2]
                num_per_cat += dict_errs[tmp_key][3]
        dict_res[tmp_cat] = [
                            avg_epe_per_cat, 
                            avg_r1_per_cat,
                            avg_r3_per_cat,
                            num_per_cat
                            ]
        
            
    """ per scene """
    for tmp_s in scenes:
        avg_epe_per_sce =  0.
        avg_r1_per_sce =  0.
        avg_r3_per_sce =  0.
        num_per_sce =  0.
        
        for tmp_cat in cates:
            tmp_key = tmp_s + '/' + tmp_cat
            if tmp_key in dict_errs:
                avg_epe_per_sce += dict_errs[tmp_key][0]
                avg_r1_per_sce += dict_errs[tmp_key][1]
                avg_r3_per_sce += dict_errs[tmp_key][2]
                num_per_sce += dict_errs[tmp_key][3]
        dict_res[tmp_s] = [
            avg_epe_per_sce, 
            avg_r1_per_sce,
            avg_r3_per_sce,
            num_per_sce,
            ]
    
    avg_epe = .0
    avg_r1 = .0
    avg_r3 = .0
    num_sum = .0
    for tmp_s in scenes:
        avg_epe += dict_res[tmp_s][0]
        avg_r1 += dict_res[tmp_s][1]
        avg_r3 += dict_res[tmp_s][2]
        num_sum += dict_res[tmp_s][3]
    dict_res['avg_epe'] = avg_epe/num_sum
    dict_res['avg_r1']  = avg_r1/num_sum
    dict_res['avg_r3']  = avg_r3/num_sum

    errs_name = ['epe', 'r1(%)', 'r3(%)']
    """ save as csv file, Excel file format """
    csv_file = err_json_file[:-5] + '.csv'
    print ("write ", csv_file, "\n")

    if os.path.exists(csv_file):
        os.remove(csv_file)

    with open( csv_file, 'a') as fwrite:
        for idx in [0, 1, 2]:
            first_row_name = "{:>15s},".format("condition")
            for tmp_s in scenes:
                #first_row_name += "{:>15s},".format(tmp_s+"(epe)")
                first_row_name += "{:>16s},".format(tmp_s+"({})".format(errs_name[idx]))
            first_row_name += "{:>20s},".format("Sum_per_Category")
            print ("----------------------------------------------------------------------------------")
            print ("====> Model {} : {} Error Analysis as below!!!".format(model_name, errs_name[idx]))
            print ("----------------------------------------------------------------------------------")
            timeStamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
            print (timeStamp)
            print (first_row_name)
            fwrite.write("Time Stamp {}, Model {}, {} Error Analysis as below!!!\n".format(
                timeStamp, 
                model_name, 
                errs_name[idx])
                )
            fwrite.write(first_row_name + '\n')
            for tmp_cat in cates:
                cur_row = "{:>15s},".format(tmp_cat)
                for tmp_s in scenes:
                    tmp_key = tmp_s + '/' + tmp_cat
                    if idx == 0:
                        cur_row += "{:>15.4f},".format(dict_errs[tmp_key][idx]/dict_errs[tmp_key][3])
                    else:
                        cur_row += "{:>15.4f},".format(100.0*dict_errs[tmp_key][idx]/dict_errs[tmp_key][3])
                # add last column as summation info
                if idx == 0:
                    cur_row += "{:>15.4f},".format(dict_res[tmp_cat][idx]/dict_res[tmp_cat][3])
                else:
                    cur_row += "{:>15.4f},".format(100.0*dict_res[tmp_cat][idx]/dict_res[tmp_cat][3])
                print (cur_row)
                fwrite.write(cur_row + '\n' )
            
            last_row =  "{:>15s},".format("Sum_Per_Scene")
            for tmp_s in scenes:
                if idx == 0:
                    last_row += "{:>15.4f},".format(dict_res[tmp_s][idx]/dict_res[tmp_s][3])
                else:
                    last_row += "{:>15.4f},".format(100.0*dict_res[tmp_s][idx]/dict_res[tmp_s][3])
            print ("----------------")
            print (last_row)
            fwrite.write(last_row + '\n')
            print ("\n\n")
        
        print('=======> Total: {:>.4f}(epe), {:>.4f}%(bad1), {:>.4f}%(bad3)'.format(
            dict_res['avg_epe'], dict_res['avg_r1']*100.0, dict_res['avg_r3']*100.0))
        total_message = 'In total, epe, {:>.4f}, bad1(%), {:>.4f}, bad3(%), {:>.4f}'.format(
            dict_res['avg_epe'], dict_res['avg_r1']*100.0, dict_res['avg_r3']*100.0)
        fwrite.write(total_message + '\n')
        fwrite.write("----- Model {} EPE Bad-1 Bad-3 error done\n".format(model_name))
    
    os.system('cat {} >> {}'.format(csv_file, './results/vkt2-ablation.csv'))
        
    
#> see: https://stackoverflow.com/questions/52910187/how-to-make-a-polygon-radar-spider-chart-in-python#
# How to make a polygon radar (spider) chart in python

def make_radar_chart(
    my_fig, my_axs,
    plot_title = 'foo', 
    stats = [[2,3,4,4,5]], 
    attribute_labels = ['Siege', 'Initiation', 'Crowd_control', 'Wave_clear', 'Objective_damage'], 
    plot_markers = [0, 0.2, 0.4, 0.6, 0.8, 1.0], 
    y_min = 0,
    y_max = 1.0,
    line_marker = ['o-'], 
    line_labels = ['foo'],
    fig_name = './foo.png'):

    labels = np.array(attribute_labels)
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
    angles = np.concatenate((angles,[angles[0]]))
    
    if my_fig is None:
        my_fig = plt.figure()
        my_axs = my_fig.add_subplot(111, polar=True)
        IS_SUBPLOT = False
    else:
        IS_SUBPLOT = True
    for i in range(len(stats)):
        tmp_stat = stats[i]
        tmp_stat = np.concatenate((tmp_stat,[tmp_stat[0]]))
        my_axs.plot(angles, tmp_stat, line_marker[i], label = line_labels[i] , linewidth=2)
    #ax.fill(angles, stats, alpha=0.25)
    if not IS_SUBPLOT:
        my_axs.set_thetagrids(angles * 180/np.pi, labels)
        plt.yticks(plot_markers)
        my_axs.set_title(plot_title)
        my_axs.grid(True)
        my_axs.set_ylim(y_min, y_max)
        locs = ["upper left", "lower left", "center right"]
        #plt.legend(loc = locs[0])
        #plt.legend(bbox_to_anchor=(0.9, 1.1), bbox_transform= my_fig.transFigure)
        plt.legend(bbox_to_anchor=(1.1, 1.17), loc='upper right')
        my_fig.savefig("%s" % fig_name, bbox_inches='tight', dpi=300)
        #sys.exit()
    else:
        my_axs.set_xticks(angles * 180/np.pi)
        my_axs.set_xticklabels(labels)
        my_axs.set_yticks(plot_markers)
        my_axs.set_title(plot_title)
        #my_axs.grid(True)
        my_axs.set_ylim(y_min, y_max)
        my_axs.legend()
    return my_fig





if __name__ == '__main__':
    
    
    models_name = [
        'dispnetc',
        'sabf-dispnetc',
        'dfn-dispnetc',
        'pac-dispnetc',
        'sga-dispnetc',
        
        'psm',
        'sabf-psm',
        'dfn-psm',
        'pac-psm',
        'sga-psm',

        'gcnet',
        'sabf-gcnet',
        'dfn-gcnet',
        'pac-gcnet',
        'sga-gcnet',

        'ganet',
        'sabf-ganet',
        'dfn-ganet',
        'pac-ganet',
        ]

    dic_models_dirs = {
        'dispnetc': 'dispnetV4-D192-BN-corrV1-sfepo20-vkt2epo20/disp-epo-020',
        'sabf-dispnetc': 'asn-embed-k5-d2-D192-dispnetc-sfepo20-vkt2epo20-embedlossW-0.06/disp-epo-017',
        'dfn-dispnetc': 'asn-dfn-k5-d2-D192-dispnetc-sfepo20-vkt2epo20-woEmbed/disp-epo-018',
        'pac-dispnetc': 'asn-npac-k5-d2-D192-dispnetc-sfepo20-vkt2epo20-woEmbed/disp-epo-019',
        'sga-dispnetc': 'asn-sga-k0-d2-D192-dispnetc-sfepo20-vkt2epo20-woEmbed/disp-epo-018',

        'psm': 'psmnet-D192-sfepo10-vkt2epo20/disp-epo-019',
        'sabf-psm': 'asn-embed-k5-d2-D192-psm-sfepo10-vkt2epo30-embedlossW-0.06-lr-0.001-e-epothrd-22/disp-epo-030',
        'dfn-psm': 'asn-dfn-k5-d2-D192-psm-sfepo10-vkt2epo20-woEmbed-lr-0.001-e-epothrd-2/disp-epo-018',
        'pac-psm': 'asn-npac-k5-d2-D192-psm-sfepo10-vkt2epo20-woEmbed-lr-0.001-e-epothrd-2/disp-epo-031',
        'sga-psm': 'asn-sga-k0-d2-D192-psm-sfepo10-vkt2epo20-woEmbed-lr-0.001-e-epothrd-2/disp-epo-024',

        'gcnet': 'gcnetAKQ-D192-sfepo10-vkt2epo20/disp-epo-010',
        'sabf-gcnet': 'asn-embed-k5-d2-D192-gcnetAKQ-sfepo10-vkt2epo20-embedlossW-0.06-lr-0.001-e-epothrd-2/disp-epo-020',
        'dfn-gcnet': 'asn-dfn-k5-d2-D192-gcnetAKQ-sfepo10-vkt2epo20-woEmbed-lr-0.001-e-epothrd-2/disp-epo-020',
        'pac-gcnet': 'asn-pac-k5-d2-D192-gcnetAKQ-sfepo10-vkt2epo30-woEmbed-lr-0.001-e-epothrd-2/disp-epo-060',
        'sga-gcnet': 'asn-sga-k0-d2-D192-gcnetAKQ-sfepo10-vkt2epo20-woEmbed-lr-0.001-p-eposteps-5-18/disp-epo-015',

        'ganet': 'ganet-deep-D192-sfepo10-vkt2epo20/disp-epo-010',
        'sabf-ganet': 'asn-embed-k5-d2-D192-ganetdeep-sfepo10-vkt2epo20-embedlossW-0.06-lr-0.001-p-eposteps-5-18/disp-epo-011',
        'dfn-ganet': 'asn-dfn-k5-d2-D180-ganetdeep-sfepo10-vkt2epo20-woEmbed-lr-0.001-p-eposteps-5-18/disp-epo-013',
        'pac-ganet': 'asn-pac-k5-d2-D192-ganetdeep-sfepo10-vkt2epo20-woEmbed-lr-0.001-p-eposteps-5-18/disp-epo-020',

    }

    Xs = ['Scene01', 'Scene02', 'Scene06', 'Scene18', 'Scene20']
    Xs_wo_s6 = ['Scene01', 'Scene02', 'Scene18', 'Scene20']
    Xs_s6 = ['Scene06']
    Ys = [
          'clone', 'fog', 'morning', 
          'overcast', 'rain', 'sunset',
          '15-deg-left', '15-deg-right', '30-deg-left', '30-deg-right', 
          ]
    project_root = '/media/ccjData2/atten-stereo'
    baseline = 'dispnetc'
    baseline = 'psm'
    baseline = 'gcnet'
    #baseline = 'ganet'
    if 0: # generating the detailed results;
        for name in [
            baseline,
            'sabf-'+ baseline,
            'dfn-' + baseline,
            'pac-' + baseline,
            'sga-'+ baseline,
            ]:
            if 1: # Only Scene06
                tmp_json_file = pjoin(project_root, 'results/vkt2-ablation-s06/' + dic_models_dirs[name] + '/vkt2-err.json')
                get_err_analysis( cates = Ys, scenes = Xs_s6, model_name = name, err_json_file = tmp_json_file)

            if 1: # other scenes excluding Scene06
                tmp_json_file = pjoin(project_root, 'results/vkt2-ablation/' + dic_models_dirs[name] + '/vkt2-err.json')
                get_err_analysis( cates = Ys, scenes = Xs_wo_s6, model_name=name, err_json_file = tmp_json_file)

    if 0: # new files, copy data by hand;
        for name in ['ours-dispnetc', 'ours-psmnet', 'ours-gcnet', 'ours-ganet']:
            for val in ['vkt2-val-s6', 'vkt2-val-wos6']:
                csv_file = pjoin(project_root, 'fig-plot/' + val + '/' + name + '.csv')
                print ('saving ', csv_file)
                with open( csv_file, 'w') as fwrite:
                    fwrite.write("#column model names,{},+sabf,+dfn,+pac,+sga\n".format(name))
                    fwrite.write("#rows condition names,clone,fog,morning,overcast,rain,sunset,15-deg-left,15-deg-right,30-deg-left,30-deg-right\n")
    
    if 0: # read the data;
        for name1,name2 in [['ours-dispnetc', 'dispnetc'], ['ours-psmnet', 'psmnet'], ['ours-gcnet', 'gcnet'], ['ours-ganet', 'ganet']]:
            for val in ['vkt2-val-s6', 'vkt2-val-wos6']:
                csv_file = pjoin(project_root, 'fig-plot/' + val + '/' + name1 + '.csv')
                print ('reading ', csv_file)
                import csv
                import matplotlib
                import matplotlib.pyplot as plt
                with open( csv_file, 'r') as csvfile:
                    mycsvreader = csv.reader(csvfile, delimiter=',')
                    # This skips the first row of the CSV file.
                    # csvreader.next() also works in Python 2.
                    #next(mycsvreader)
                    tmp_y_dict = {}
                    tmp_y_dict[name2] = []
                    tmp_y_dict['+sabf(ours)'] = []
                    tmp_y_dict['+dfn(ours)'] = []
                    tmp_y_dict['+pac(ours)'] = []
                    tmp_y_dict['+sga(ours)'] = []
                    for row in mycsvreader:
                        if not row[0].startswith("#"):
                            tmp_y_dict[name2].append(float(row[0]))
                            tmp_y_dict['+sabf(ours)'].append(float(row[1]))
                            tmp_y_dict['+dfn(ours)'].append(float(row[2]))
                            tmp_y_dict['+pac(ours)'].append(float(row[3]))
                            if name2 != 'ganet':
                                tmp_y_dict['+sga(ours)'].append(float(row[4]))
                    # Set the x axis label of the current axis.
                    # Set the y axis label of the current axis.
                    plt.figure(figsize=(12,3.5)) # width, height
                    plt.ylabel('EPE')
                    # Set a title
                    x = np.arange(0, 10, step=1)
                    plt.title('Ours VS {} on {}'.format(name2, 'VKT2-Val-WoS6' if val == 'vkt2-val-wos6' else 'VKT2-Val-S6'))
                    plt.plot(x, tmp_y_dict[name2], marker='s', label=name2)
                    print (x, tmp_y_dict[name2])
                    plt.plot(x, tmp_y_dict['+sabf(ours)'], marker=  'D', label='+sabf(ours)')
                    plt.plot(x, tmp_y_dict['+dfn(ours)'], marker='o' , label="+dfn(ours)")
                    plt.plot(x, tmp_y_dict['+pac(ours)'], marker='x', label="+pac(ours)")
                    if name2 != 'ganet':
                        plt.plot(x, tmp_y_dict['+sga(ours)'], marker='>', label="+sga(ours)")
                    #plt.ylim(0.3, 0.9)
                    plt.legend()
                    plt.xticks(x, Ys, rotation=45, horizontalalignment="right")
                    plt.grid(color='gray', which='major', axis='y', linestyle='solid')
                    #plt.show()
                    plt.savefig('./fig-plot/' + val + '-' + name2 + '.png', bbox_inches='tight', dpi=300)
                    #fig = matplotlib.pyplot.gcf()
                    #fig.set_size_inches(18.5, 10.5)
                    #fig.savefig('./fig-plot/test2png.png', dpi=100)
                    #sys.exit()


    if 1: # draw spider chart;
        for val in ['vkt2-val-s6', 'vkt2-val-wos6']:
            for name1,name2, name3 in [['ours-dispnetc', 'dispnetc', 'DispNetC'], 
                                ['ours-psmnet', 'psmnet', 'PSMNet'], 
                                ['ours-gcnet', 'gcnet', 'GCNet'], 
                                ['ours-ganet', 'ganet', 'GANet']
                                ]:
                csv_file = pjoin(project_root, 'fig-plot/' + val + '/' + name1 + '.csv')
                print ('reading ', csv_file)
                import csv
                import matplotlib
                import matplotlib.pyplot as plt
                with open( csv_file, 'r') as csvfile:
                    mycsvreader = csv.reader(csvfile, delimiter=',')
                    # This skips the first row of the CSV file.
                    # csvreader.next() also works in Python 2.
                    #next(mycsvreader)
                    tmp_y_dict = {}
                    tmp_y_dict[name2] = []
                    tmp_y_dict['+sabf(ours)'] = []
                    tmp_y_dict['+dfn(ours)'] = []
                    tmp_y_dict['+pac(ours)'] = []
                    tmp_y_dict['+sga(ours)'] = []
                    for row in mycsvreader:
                        if not row[0].startswith("#"):
                            tmp_y_dict[name2].append(float(row[0]))
                            tmp_y_dict['+sabf(ours)'].append(float(row[1]))
                            tmp_y_dict['+dfn(ours)'].append(float(row[2]))
                            tmp_y_dict['+pac(ours)'].append(float(row[3]))
                            if name2 != 'ganet':
                                tmp_y_dict['+sga(ours)'].append(float(row[4]))
                    
                    my_stats = [tmp_y_dict[name2], tmp_y_dict['+sabf(ours)'], tmp_y_dict['+dfn(ours)'], tmp_y_dict['+pac(ours)']]
                    my_line_markers = ['s-', 'D-', 'o-', 'x-', '>-']
                    #my_labels = [ 'baseline', '+sabf(ours)', '+dfn(ours)', '+pac(ours)', '+sga(ours)']
                    #my_labels = [ name2, '+sabf(ours)', '+dfn(ours)', '+pac(ours)', '+sga(ours)']
                    my_labels = [ name3, '+SABF', '+DFN',  '+PAC',  '+SGA']
                    #print (tmp_y_dict[name2])

                    if name2 != 'ganet':
                        my_stats.append(tmp_y_dict['+sga(ours)'])
                    
                    tmp_errs = tmp_y_dict[name2] + tmp_y_dict['+sabf(ours)'] +  tmp_y_dict['+dfn(ours)'] + tmp_y_dict['+pac(ours)']
                    if name2 != 'ganet':
                        tmp_errs += tmp_y_dict['+sga(ours)']
                    print (tmp_errs)
                    val_min, val_max = min(tmp_errs), max(tmp_errs)
                    print ('min = {}, max = {}'.format(val_min, val_max))
                    val_min =  np.floor(val_min*20)*0.05
                    val_max =  np.ceil(val_max*20)*0.05
                    print ('min = {}, max = {}'.format(val_min, val_max))
                    if 1:
                        make_radar_chart(
                            my_fig = None,
                            my_axs = None,
                            #plot_title = 'Ours VS {}\non {}'.format(name2, 'VKT2-Val-WoS6' if val == 'vkt2-val-wos6' else 'VKT2-Val-S6'),
                            plot_title = '{}\non {}'.format(name3, 'VKT2-Val-WoS6' if val == 'vkt2-val-wos6' else 'VKT2-Val-S6'),
                            stats = my_stats, 
                            attribute_labels = Ys, 
                            y_min = val_min,
                            y_max = val_max,
                            plot_markers = [ np.round(i, 2) for i in np.linspace(val_min, val_max, 5)], 
                            line_marker= my_line_markers, 
                            line_labels= my_labels,
                            fig_name = './fig-plot/spider-' + val + '-' + name2 + '.png')
                        
                        #sys.exit()
                    
                    if 0: # not work well ???
                        fig = plt.figure()
                        for i in range(4):
                            a = fig.add_subplot(1,4,i+1, polar=True) # 1 row, 4 columns;
                            make_radar_chart(
                                my_fig = fig,
                                my_axs = a,
                                plot_title = 'Ours VS {} \n on {}'.format(name2, 'VKT2-Val-WoS6' if val == 'vkt2-val-wos6' else 'VKT2-Val-S6'),
                                stats = my_stats, 
                                attribute_labels = Ys, 
                                y_min = val_min,
                                y_max = val_max,
                                plot_markers = [ np.round(i, 2) for i in np.linspace(val_min, val_max, 5)], 
                                line_marker= my_line_markers, 
                                line_labels= my_labels,
                                fig_name = './fig-plot/spider-' + val + '-' + name2 + '.png')

                        plt.show()
                        sys.exit()