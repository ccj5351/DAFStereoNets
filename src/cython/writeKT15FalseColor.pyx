# cython: infer_types=True
from __future__ import print_function
import numpy as np
cimport cython
#The primitive types supported are tied closely to those in C:
ctypedef float MY_DTYPE # means float in C type

# cdef means here that this function is a plain C function (so faster).
# To get all the benefits, we type the arguments and the return value.
cdef float clip(float a, float min_value, float max_value):
    return min(max(a, min_value), max_value)


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
def writeKT15FalseColor(MY_DTYPE [:,::1] array_disp, max_disp = -1.0):
    clr_map = np.array(
            [[0,0,0,114],[0,0,1,185],[1,0,0,114],[1,0,1,174],
             [0,1,0,114],[0,1,1,185],[1,1,0,114],[1,1,1,0],], 
            dtype=np.float32)
    
    cdef MY_DTYPE[:,::1] clr_map_view = clr_map
    
    cdef float mySum = .0
    cdef Py_ssize_t i, j, k
    
    for i in range(0,8):
        mySum += clr_map_view[i][3]
    
    weights = np.zeros((8), dtype = np.float32)
    cumsum  = np.zeros((8), dtype = np.float32)
    cdef MY_DTYPE [::1] weights_view = weights
    cdef MY_DTYPE [::1] cumsum_view = cumsum
    
    for i in range(0,7):
        weights_view[i] = mySum / clr_map_view[i,3]
        cumsum_view[i+1] = cumsum_view[i] + clr_map_view[i,3] / mySum
    #print ('weights: ', weights)
    #print ('cumsum: ', cumsum)

    cdef Py_ssize_t h = array_disp.shape[0]
    cdef Py_ssize_t w = array_disp.shape[1]
    cdef float max_val = max_disp if max_disp > 0 else -1.0
    for i in range(h):
        for j in range(w):
            if max_val < array_disp[i,j]:
                max_val = array_disp[i,j]
    #print ("max_val = ", max_val)

    disp_out = np.zeros([h, w, 3], dtype=np.float32)
    cdef MY_DTYPE[:,:,::1] disp_out_view = disp_out
    cdef MY_DTYPE tmp_w
    for i in range(0, h):
        for j in range(0, w):
            # get normalized value
            val = clip(array_disp[i,j]/ max_val, .0, 1.0)
            # find bin;
            for k in range(0,7):
                if val < cumsum_view[k+1]:
                    #print (i,j,k, val, cumsum[k+1])
                    break
            # compute red/green/blue values
            tmp_w = 1.0 - (val-cumsum_view[k]) * weights_view[k]
            # r, g, b
            disp_out_view[i,j, 0] = (tmp_w*clr_map_view[k, 0] + (1.0-tmp_w)*clr_map_view[k+1,0])*255.0
            disp_out_view[i,j, 1] = (tmp_w*clr_map_view[k, 1] + (1.0-tmp_w)*clr_map_view[k+1,1])*255.0
            disp_out_view[i,j, 2] = (tmp_w*clr_map_view[k, 2] + (1.0-tmp_w)*clr_map_view[k+1,2])*255.0
            #if i == 200 and j == 300:
                #print ('disp_out[200,300] = ', disp_out[i,j,:])
                #print (i,j,k, val, cumsum[k+1])
    return disp_out
