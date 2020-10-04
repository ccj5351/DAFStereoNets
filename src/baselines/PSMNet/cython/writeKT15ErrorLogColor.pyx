# !/usr/bin/env python3
# -*-coding:utf-8-*-
# @file: writeKT15ErrorLogColor.pyx
# @brief:
# @author: Changjiang Cai, ccai1@stevens.edu, caicj5351@gmail.com
# @version: 0.0.1
# @creation date: 28-10-2019
# @last modified: Mon 28 Oct 2019 07:21:09 PM EDT

# cython: infer_types=True
from __future__ import print_function
import numpy as np
cimport cython
from libc.math cimport fabs # fabs() from <cmath>;
#The primitive types supported are tied closely to those in C:
ctypedef float MY_DTYPE # means float in C type

@cython.boundscheck(False)
@cython.wraparound(False)

#For extra speed gains, if you know that the NumPy arrays 
#you are providing are contiguous in memory, you can 
#declare the memoryview as contiguous. 
    
#We give an example on an array that has 3 dimensions. 
#If you want to give Cython the information that the 
#data is C-contiguous you have to declare the memoryview like this:
#cdef int [:,:,::1] a

#The cdef statement is used to declare C variables, either local or module-level:
#NOTE:The code is adopted from the C/C++ code provided by KITTI15 official website;
def writeKT15ErrorDispLogColor(
        MY_DTYPE [:,::1] array_disp, # prediciton
        MY_DTYPE [:,::1] array_disp_gt # ground truth
        ):
    # size: [10,5]
    log_clr_map = np.array(
           [[0,0.0625,49,54,149],
            [0.0625,0.125,69,117,180],
            [0.125,0.25,116,173,209],
            [0.25,0.5,171,217,233],
            [0.5,1,224,243,248],
            [1,2,254,224,144],
            [2,4,253,174,97],
            [4,8,244,109,67],
            [8,16,215,48,39],
            [16,1000000000.0,165,0,38]
           ],dtype=np.float32)
    
    cdef MY_DTYPE[:,::1] log_clr_map_view = log_clr_map
    cdef Py_ssize_t h = array_disp.shape[0]
    cdef Py_ssize_t w = array_disp.shape[1]
    cdef Py_ssize_t i, v,u,v2,u2
    cdef float val_red = .0, val_gre = .0, val_blu = .0, d_err, d_mag, n_err
    disp_err = np.zeros([h, w, 3], dtype=np.float32)
    cdef MY_DTYPE[:,:,::1] disp_err_view = disp_err
    for v in range(1, h-1):
        for u in range(1, w-1):
            if array_disp_gt[v,u] > 0: # if valid
                d_err = fabs(array_disp[v,u] - array_disp_gt[v,u])
                d_mag = fabs(array_disp_gt[v,u])
                n_err = min(d_err / 3.0, 20.0*d_err/d_mag)
                for i in range(0, 10):
                    if (n_err >= log_clr_map_view[i,0]) and (n_err < log_clr_map_view[i,1]):
                        val_red = log_clr_map_view[i,2]
                        val_gre = log_clr_map_view[i,3]
                        val_blu = log_clr_map_view[i,4]
                disp_err_view[v,u,0] = val_red
                disp_err_view[v,u,1] = val_gre
                disp_err_view[v,u,2] = val_blu
    return disp_err
