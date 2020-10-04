#export LD_LIBRARY_PATH="/home/feihu/anaconda3/lib:$LD_LIBRARY_PATH"
#export LD_INCLUDE_PATH="/home/feihu/anaconda3/include:$LD_INCLUDE_PATH"
#export CUDA_HOME="/usr/local/cuda-10.0"
#export PATH="/home/feihu/anaconda3/bin:/usr/local/cuda-10.0/bin:$PATH"
#export CPATH="/usr/local/cuda-10.0/include"
#export CUDNN_INCLUDE_DIR="/usr/local/cuda-10.0/include"
#export CUDNN_LIB_DIR="/usr/local/cuda-10.0/lib64"

#export LD_LIBRARY_PATH="/home/zhangfeihu/anaconda3/lib:$LD_LIBRARY_PATH"
#export LD_INCLUDE_PATH="/home/zhangfeihu/anaconda3/include:$LD_INCLUDE_PATH"
#export CUDA_HOME="/home/work/cuda-9.2"
#export PATH="/home/zhangfeihu/anaconda3/bin:/home/work/cuda-9.2/bin:$PATH"
#export CPATH="/home/work/cuda-9.2/include"
#export CUDNN_INCLUDE_DIR="/home/work/cudnn/cudnn_v7/include"
#export CUDNN_LIB_DIR="/home/work/cudnn/cudnn_v7/lib64"
TORCH=$(python3 -c "import os; import torch; print(os.path.dirname(torch.__file__))")
#echo $TORCH
#exit

#MY_PYTHON=python3
MY_PYTHON=python3.7
cd libs/GANet
$MY_PYTHON setup.py clean
rm -rf build
$MY_PYTHON setup.py build
cp -r build/lib* build/lib

cd ../sync_bn
$MY_PYTHON setup.py clean
rm -rf build
$MY_PYTHON setup.py build
cp -r build/lib* build/lib
