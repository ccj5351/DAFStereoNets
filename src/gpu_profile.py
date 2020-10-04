# !/usr/bin/env python3
# -*-coding:utf-8-*-
# @file:
# @brief:
# @author: Changjiang Cai, ccai1@stevens.edu, caicj5351@gmail.com
# @version: 0.0.1
# @creation date: 11-11-2019
# @last modified: Mon 11 Nov 2019 02:35:01 PM EST


#NOTE: code is copied from https://gist.github.com/MInner/8968b3b120c95d3f50b8a22a74bf66bc;

import datetime
import linecache
import os

import py3nvml
import torch

print_tensor_sizes = True
last_tensor_sizes = set()
gpu_profile_fn = f'{datetime.datetime.now():%d-%b-%y-%H:%M:%S}-gpu_mem_prof.txt'
if 'GPU_DEBUG' in os.environ:
    print('profiling gpu usage to ', gpu_profile_fn)

lineno = None
func_name = None
filename = None
module_name = None

#N = 0
def gpu_profile(frame, event, arg):
    # it is _about to_ execute (!)
    global last_tensor_sizes
    global lineno, func_name, filename, module_name

    if event == 'line':
        #global N
        #N += 1
        #if N % 900 == 0:
        #    print ('line')
        try:
            # about _previous_ line (!)
            if lineno is not None:
                py3nvml.nvmlInit()
                handle = py3nvml.nvmlDeviceGetHandleByIndex(int(os.environ['GPU_DEBUG']))
                meminfo = py3nvml.nvmlDeviceGetMemoryInfo(handle)
                line = linecache.getline(filename, lineno)
                where_str = module_name+' '+func_name+':'+str(lineno)

                print('profiling gpu usage to ', gpu_profile_fn)
                with open(gpu_profile_fn, 'a+') as f:
                    f.write(f"{where_str:<50}"
                            f":{meminfo.used/1024**2:<7.1f}Mb "
                            f"{line.rstrip()}\n")

                    if print_tensor_sizes is True:
                        for tensor in get_tensors():
                            if not hasattr(tensor, 'dbg_alloc_where'):
                                tensor.dbg_alloc_where = where_str
                        new_tensor_sizes = {(type(x), tuple(x.size()), x.dbg_alloc_where)
                                            for x in get_tensors()}
                        for t, s, loc in new_tensor_sizes - last_tensor_sizes:
                            f.write(f'+ {loc:<50} {str(s):<20} {str(t):<10}\n')
                        for t, s, loc in last_tensor_sizes - new_tensor_sizes:
                            f.write(f'- {loc:<50} {str(s):<20} {str(t):<10}\n')
                        last_tensor_sizes = new_tensor_sizes
                py3nvml.nvmlShutdown()

            # save details about line _to be_ executed
            lineno = None

            func_name = frame.f_code.co_name
            filename = frame.f_globals["__file__"]
            if (filename.endswith(".pyc") or
                    filename.endswith(".pyo")):
                filename = filename[:-1]
            module_name = frame.f_globals["__name__"]
            lineno = frame.f_lineno

            if 'gmwda-pytorch' not in os.path.dirname(os.path.abspath(filename)):
                lineno = None  # skip current line evaluation

            if ('car_datasets' in filename
                    or '_exec_config' in func_name
                    or 'gpu_profile' in module_name
                    or 'tee_stdout' in module_name):
                lineno = None  # skip current

            return gpu_profile

        except (KeyError, AttributeError):
            print('do nothing!!!')
            pass

    return gpu_profile


def get_tensors(gpu_only=True):
    import gc
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                tensor = obj
            elif hasattr(obj, 'data') and torch.is_tensor(obj.data):
                tensor = obj.data
            else:
                continue

            if tensor.is_cuda:
                yield tensor
        except Exception as e:
            pass


# > see code at: https://gist.github.com/Stonesjtu/368ddf5d9eb56669269ecdf9b0d21cbe;
## MEM utils ##
def mem_report():
    import gc
    '''Report the memory usage of the tensor.storage in pytorch
    Both on CPUs and GPUs are reported'''

    def _mem_report(tensors, mem_type):
        '''Print the selected tensors of type
        There are two major storage types in our major concern:
            - GPU: tensors transferred to CUDA devices
            - CPU: tensors remaining on the system memory (usually unimportant)
        Args:
            - tensors: the tensors of specified type
            - mem_type: 'CPU' or 'GPU' in current implementation '''
        print('Storage on %s' %(mem_type))
        print('-'*LEN)
        total_numel = 0
        total_mem = 0
        visited_data = []
        for tensor in tensors:
            if tensor.is_sparse:
                continue
            # a data_ptr indicates a memory block allocated
            data_ptr = tensor.storage().data_ptr()
            if data_ptr in visited_data:
                continue
            visited_data.append(data_ptr)

            numel = tensor.storage().size()
            total_numel += numel
            element_size = tensor.storage().element_size()
            mem = numel*element_size /1024/1024 # 32bit=4Byte, MByte
            total_mem += mem
            element_type = type(tensor).__name__
            size = tuple(tensor.size())

            print('%s\t\t%s\t\t%.2f' % (
                element_type,
                size,
                mem) )
        print('-'*LEN)
        print('Total Tensors: %d \tUsed Memory Space: %.2f MBytes' % (total_numel, total_mem) )
        print('-'*LEN)

    LEN = 65
    print('='*LEN)
    objects = gc.get_objects()
    print('%s\t%s\t\t\t%s' %('Element type', 'Size', 'Used MEM(MBytes)') )
    tensors = [obj for obj in objects if torch.is_tensor(obj)]
    cuda_tensors = [t for t in tensors if t.is_cuda]
    host_tensors = [t for t in tensors if not t.is_cuda]
    _mem_report(cuda_tensors, 'GPU')
    _mem_report(host_tensors, 'CPU')
    print('='*LEN)