#ccj's experiments

#t=8000
#t=1
#echo "Hi, I'm sleeping for $t seconds..."
#sleep ${t}s


#---------------
# utility function
#---------------
function makeDir () {
	dstDir="$1"
	if [ ! -d $dstDir ]; then
		mkdir -p $dstDir
		echo "mkdir $dstDir"
	else
		echo $dstDir exists
	fi
}  


MY_PROJECT_ROOT=/media/ccjData2/atten-stereo
if [ ! -d $MY_PROJECT_ROOT ];then
	MY_PROJECT_ROOT="/home/${USER}/atten-stereo"
	echo "Updated : setting MY_PROJECT_ROOT = ${MY_PROJECT_ROOT}"
fi
if [ ! -d $MY_PROJECT_ROOT ];then
	MY_PROJECT_ROOT="/diskb/ccjData2/atten-stereo"
	echo "Updated : setting MY_PROJECT_ROOT = ${MY_PROJECT_ROOT}"
fi


if [ ! -d $MY_PROJECT_ROOT ];then
	echo "Try 3 times, but cannot find directory: MY_PROJECT_ROOT = ${MY_PROJECT_ROOT}"
	exit
else
	echo "Setting MY_PROJECT_ROOT = ${MY_PROJECT_ROOT}"
fi


DATA_ROOT="/media/ccjData2"
echo "data_root = ${DATA_ROOT}"
if [ ! -d $DATA_ROOT ];then
	DATA_ROOT="/data/ccjData"
	echo "Updated : setting data_root = ${DATA_ROOT}"
fi

if [ ! -d $DATA_ROOT ];then
	DATA_ROOT="/home/${USER}"
	echo "Updated : setting data_root = ${DATA_ROOT}"
fi

KT2012=0
KT2015=1
VIRTUAL_KITTI2=0
KT12_IMAGE_MODE='rgb'
#KT12_IMAGE_MODE='gray'
#KT12_IMAGE_MODE='gray2rgb'

#---------------------------------------#
#-----Common Hyperparameters Here ------#
#---------------------------------------#
IS_DATA_AUGMENT="false"
#IS_DATA_AUGMENT="true"
LR_EPOCH_STEPS=''
#IS_FIXED_LR='false'
#IS_FIXED_LR='true'
LEARNING_RATE=0.001

if [ $KT2012 -eq 1 ]; then
	DATA_PATH="${DATA_ROOT}/datasets/KITTI-2012/training/"
	KT_STR='kt12'
  if [ "$KT12_IMAGE_MODE" = 'gray' ]; then
		KT_STR="${KT_STR}gray"
	elif [ "$KT12_IMAGE_MODE" = 'gray2rgb' ]; then
		KT_STR="${KT_STR}g2rgb"
	fi
  TRAINING_LIST="lists/kitti2012_train164.list"
  TEST_LIST="lists/kitti2012_val30.list"
	#revise parameter settings and run "train.sh" and "predict.sh" for 
	#training, finetuning and prediction/testing. 
	#Note that the “crop_width” and “crop_height” must be multiple of 48, 
	#"max_disp" must be multiple of 12 (default: 192).
	let CROP_HEIGHT=240
	let CROP_WIDTH=624
	let MAX_DISP=192
  NUM_EPOCHS=400
	LR_ADJUST_EPO_THRED=200
	LR_SCHEDULER="piecewise"

elif [ $KT2015 -eq 1 ]; then
	DATA_PATH="${DATA_ROOT}/datasets/KITTI-2015/training/"
	KT_STR='kt15'
	TRAINING_LIST="lists/kitti2015_train170.list"
  #VAL_LIST="lists/kitti2015_val30.list"
  TEST_LIST="lists/kitti2015_val30.list"
	#let CROP_HEIGHT=240-96-48
	#let CROP_WIDTH=576-96*2
	let CROP_HEIGHT=240
	let CROP_WIDTH=576-48
	let MAX_DISP=192
	#let MAX_DISP=180
  NUM_EPOCHS=600
  NUM_EPOCHS_STR=600
	LR_ADJUST_EPO_THRED=300
	LR_SCHEDULER="piecewise"
	
	# new try
  LEARNING_RATE=0.001
	LR_SCHEDULER="piecewise"
	LR_EPOCH_STEPS="300"

elif [ $VIRTUAL_KITTI2 -eq 1 ]; then
	DATA_PATH="${DATA_ROOT}/datasets/Virtual-KITTI-V2/"
	KT_STR='vkt2'
	# no shuffle
	TRAINING_LIST="lists/virtual_kitti2_wo_scene06_fixed_train.list"
  TEST_LIST="lists/virtual_kitti2_wo_scene06_fixed_test.list"
	#TRAINING_LIST="lists/virtual_kitti2_wo_scene06_fixed_train_small.list"
  #TEST_LIST="lists/virtual_kitti2_wo_scene06_fixed_test_small.list"
	# with shuffle
	#TRAINING_LIST="lists/virtual_kitti2_wo_scene06_random_train.list"
  #TEST_LIST="lists/virtual_kitti2_wo_scene06_random_test.list"
	#let CROP_HEIGHT=240-96-48
	#let CROP_WIDTH=576-96*2
	let CROP_HEIGHT=256
	let CROP_WIDTH=512
	let MAX_DISP=192
	#let MAX_DISP=180
  NUM_EPOCHS=20
  NUM_EPOCHS_STR=20
	LR_ADJUST_EPO_THRED=2
	LR_SCHEDULER="exponential"
  
	# new try
  #LEARNING_RATE=0.001
	#LR_SCHEDULER="piecewise"
	#LR_EPOCH_STEPS="5-18"

else
	DATA_PATH="${DATA_ROOT}/datasets/SceneFlowDataset/"
  #TRAINING_LIST="lists/sceneflow_train.list"
  TRAINING_LIST="lists/sceneflow_train_small.lis"
  TEST_LIST="lists/sceneflow_test_select.list"
	let CROP_HEIGHT=240
	let CROP_WIDTH=576+48
	let MAX_DISP=192
  NUM_EPOCHS=10
	LR_ADJUST_EPO_THRED=10
	LR_SCHEDULER="constant"
fi
echo "DATA_PATH=$DATA_PATH"

flag=false
START_EPOCH=0
#NUM_EPOCHS=400
#START_EPOCH=400
#NUM_EPOCHS=400
#NUM_EPOCHS=800
#NUM_EPOCHS=20


#NUM_WORKERS=1
NUM_WORKERS=12

#BATCHSIZE=2
BATCHSIZE=4
LOG_SUMMARY_STEP=40
LOG_SUMMARY_STEP=4
#RESUME='./checkpoints/asn-sga-sf-small-tmp/ASN-Embed-SGA/model_epoch_00032.tar'
#MODEL_NAME='ASN-Embed-SGA'
#RESUME_EMBEDNET='./checkpoints/pascalvoc-embednet-epo30/vgg-like-embed/best_model_epoch_00030_valloss_1.2961.tar'


#---------------------------------------#
#-----Common Hyperparameters Here ------#
#---------------------------------------#
#PANATIVE_IMPLE='true'
PAC_NATIVE_IMPLE='false'
#PAC_NATIVE_IMPLE='false' #just for GCNet, excluding GCNetQ;
OUR_NET_NAME='asn' # attention stereo network
#RESUME_EMBEDNET='./checkpoints/pascalvoc-embednet-epo30/vgg-like-embed/best_model_epoch_00030_valloss_1.2961.tar'
RESUME_EMBEDNET='./checkpoints/saved/city_coarse-embednet-epo30/vgg-like-embed/best_model_epoch_00030_valloss_1.0048.tar'
DILATION=1
COST_FILTER_GRAD='true'
#EMBED_LOSS_WEIGHT=0.006
EMBED_LOSS_WEIGHT=0.06
if [ $KT2012 -eq 1 ]; then
	# no segmentation ground truth for KT12;
	EMBED_LOSS_WEIGHT_STR='no'
else
	EMBED_LOSS_WEIGHT_STR=${EMBED_LOSS_WEIGHT}
fi

BATCH_IN_IMAGE='false'
BATCH_H=256 # only work when BATCH_IN_IMAGE='true', else just a dummy parameter!!!
IS_QUARTER_SIZE_COST_VOLUME_GCNET='true'
#IS_QUARTER_SIZE_COST_VOLUME_GCNET='false'
#IS_KENDALL_VERSION_GCNET='false'
IS_KENDALL_VERSION_GCNET='true'
#newly added for GCNet:
if [ "$IS_KENDALL_VERSION_GCNET" = true ]; then
	GCNET_NAME_STR='gcnetAK'
	TMP_GCNET='gcnetAK'
else
	GCNET_NAME_STR='gcnet'
	TMP_GCNET='gcnet'
fi

if [ "$IS_QUARTER_SIZE_COST_VOLUME_GCNET" = true ]; then
	GCNET_NAME_STR="${GCNET_NAME_STR}Q"
fi
echo "GCNET_NAME_STR=$GCNET_NAME_STR"

#let LR_ADJUST_EPO_THRED=${LR_ADJUST_EPO_THRED}+${START_EPOCH}
let LR_ADJUST_EPO_THRED=${LR_ADJUST_EPO_THRED}
echo "LR_ADJUST_EPO_THRED=${LR_ADJUST_EPO_THRED}"

if [ "$LR_SCHEDULER" = 'constant' ]; then
	LR_STR="-lr-${LEARNING_RATE}-c"
elif [ "$LR_SCHEDULER" = 'piecewise' ]; then
	LR_STR="-lr-${LEARNING_RATE}-p-eposteps-${LR_EPOCH_STEPS}"
elif [ "$LR_SCHEDULER" = 'exponential' ]; then
	LR_STR="-lr-${LEARNING_RATE}-e-epothrd-${LR_ADJUST_EPO_THRED}"
else
	echo "Wrong LR_SCHEDULER type: ${LR_SCHEDULER} !!!"
	exit
fi
echo "LR_STR=${LR_STR}"

#--------------------------#
#-----Task Type Here ------#
#--------------------------#
TASK_TYPE='EMBED_BILATERAL'
#TASK_TYPE='DFN'
#TASK_TYPE='PAC'
#TASK_TYPE='PAC-EMBED'
#TASK_TYPE='SGA'
#TASK_TYPE='SGA-EMBED'
#----initial values ---
IS_DFN='false'
IS_EMBED='false'
IS_PAC='false'
IS_SGA_GUIDE_FROM_IMG='false'
SIGMA_S=0.7 # window 7 x 7
SIGMA_V=0.1
DFN_K_WIDTH=11
PAC_K_WIDTH=11
if [ $TASK_TYPE == 'EMBED_BILATERAL' ]; then
	echo 'TASK_TYPE : EMBED_BILATERAL !!!'
	IS_EMBED='true'
	#IS_FREEZE_EMBED='true'
	IS_FREEZE_EMBED='false'
	#SIGMA_S=3.0 # window 21 x 21
	#SIGMA_S=2.0 # window 15 x 15
	#SIGMA_S=1.4 # window 11 x 11
	#SIGMA_S=1.0 # window 9 x 9
	#SIGMA_S=0.7 # window 7 x 7
	SIGMA_S=0.3 # window 3 x 3
	#SIGMA_S=0.5 # window 5 x 5
	KERNEL_WIDTH_FLOAT=$(echo |awk "{ print $SIGMA_S*3+1}")
	KERNEL_WIDTH_INT=${KERNEL_WIDTH_FLOAT%.*}
	SIGMA_V=0.1
  K_WIDTH=$(( KERNEL_WIDTH_INT*2 + 1))
	#d2 means dilation=2 for im2col;
	#d1 means dilation=1 for im2col;
	MODEL_NAME_STR="${OUR_NET_NAME}-embed-k${K_WIDTH}-d${DILATION}"

elif [ $TASK_TYPE == 'DFN' ]; then
	echo 'TASK_TYPE : DFN !!!'
	IS_DFN='true'
	#DFN_K_WIDTH=11
	DFN_K_WIDTH=5
	K_WIDTH=$DFN_K_WIDTH
	MODEL_NAME_STR="${OUR_NET_NAME}-dfn-k${K_WIDTH}-d${DILATION}"

elif [ $TASK_TYPE == 'SGA' ]; then
	echo 'TASK_TYPE : SGA !!!'
  IS_SGA_GUIDE_FROM_IMG='true'
	SGA_K_WIDTH=0
	K_WIDTH=$SGA_K_WIDTH
	MODEL_NAME_STR="${OUR_NET_NAME}-sga-k${K_WIDTH}-d${DILATION}"

elif [ $TASK_TYPE == 'SGA-EMBED' ]; then
	echo 'TASK_TYPE : SGA-EMBED !!!'
  IS_SGA_GUIDE_FROM_IMG='false'
	DFN_K_WIDTH=0
	K_WIDTH=$DFN_K_WIDTH
	MODEL_NAME_STR="${OUR_NET_NAME}-sga-embed-k${K_WIDTH}-d${DILATION}"

elif [ $TASK_TYPE == 'PAC' ]; then
	echo 'TASK_TYPE : PAC !!!'
	IS_PAC='true'
	PAC_K_WIDTH=5
	K_WIDTH=$PAC_K_WIDTH
  if [ $PAC_NATIVE_IMPLE == 'true' ]; then
    # npac: n means naive implementation of PAC
	  MODEL_NAME_STR="${OUR_NET_NAME}-npac-k${K_WIDTH}-d${DILATION}"
	else
	  MODEL_NAME_STR="${OUR_NET_NAME}-pac-k${K_WIDTH}-d${DILATION}"
	fi

elif [ $TASK_TYPE == 'PAC-EMBED' ]; then
	echo 'TASK_TYPE : PAC W/ Embed as adapting feature !!!'
	IS_PAC='true'
	IS_EMBED='true'
	IS_FREEZE_EMBED='false'
	#PAC_K_WIDTH=7
	PAC_K_WIDTH=5
	#PAC_K_WIDTH=3
	K_WIDTH=$PAC_K_WIDTH
  if [ $PAC_NATIVE_IMPLE == 'true' ]; then
	  MODEL_NAME_STR="${OUR_NET_NAME}-npac-embed-k${K_WIDTH}-d${DILATION}"
	else
	  MODEL_NAME_STR="${OUR_NET_NAME}-pac-embed-k${K_WIDTH}-d${DILATION}"
	fi
fi


#--------------------------#
#-----Model Name Here ------#
#--------------------------#
#MODEL_NAME='ASN-Embed-PSM'
#MODEL_NAME='ASN-Embed-GANet-Deep'
#MODEL_NAME='ASN-Embed-GANet11'
MODEL_NAME='ASN-Embed-DispNetC'
#MODEL_NAME='ASN-Embed-GCNet'

#MODEL_NAME='ASN-DFN-PSM'
#MODEL_NAME='ASN-DFN-PSM-NoDFN'
#MODEL_NAME='ASN-DFN-GCNet'
#MODEL_NAME='ASN-DFN-DispNetC'
#MODEL_NAME='ASN-DFN-GANet-Deep'

#MODEL_NAME='ASN-PAC-PSM'
#MODEL_NAME='ASN-PAC-GCNet'
#MODEL_NAME='ASN-PAC-DispNetC'
#MODEL_NAME='ASN-PAC-GANet-Deep'

#MODEL_NAME='ASN-SGA-PSM'
#MODEL_NAME='ASN-SGA-GCNet'
#MODEL_NAME='ASN-SGA-DispNetC'


if [ $MODEL_NAME == 'ASN-PAC-GANet-Deep' ]; then
	let CROP_HEIGHT=240-96 #must be multiple of 48;
	let CROP_WIDTH=528-96 #must be multiple of 48;
	let MAX_DISP=192 #must be multiple of 12, default: 192;
	#let MAX_DISP=192-120 #must be multiple of 12, default: 192;
  BATCHSIZE=1
  
  #------	for debug
	#let CROP_HEIGHT=240-96 #must be multiple of 48;
	#let CROP_WIDTH=528-96-48 #must be multiple of 48;
	#let MAX_DISP=192-120 #must be multiple of 12, default: 192;
  #BATCHSIZE=6
  
	LOG_SUMMARY_STEP=20
	RESUME='./checkpoints/saved/ganet-pretrained/ganet-deep/sceneflow_epoch_10.pth'
  if [ $TASK_TYPE == 'PAC-EMBED' ]; then
		EXP_NAME="${MODEL_NAME_STR}-D${MAX_DISP}-ganetdeep-sfepo10-${KT_STR}epo${NUM_EPOCHS_STR}-embedlossW-${EMBED_LOSS_WEIGHT_STR}${LR_STR}"
	elif [ $TASK_TYPE == 'PAC' ]; then
		EXP_NAME="${MODEL_NAME_STR}-D${MAX_DISP}-ganetdeep-sfepo10-${KT_STR}epo${NUM_EPOCHS_STR}-woEmbed${LR_STR}"
	fi 

elif [ $MODEL_NAME == 'ASN-DFN-GANet-Deep' ]; then
	let CROP_HEIGHT=240-96 #must be multiple of 48;
	let CROP_WIDTH=528-96-48 #must be multiple of 48;
	#let MAX_DISP=192 #must be multiple of 12, default: 192;
	let MAX_DISP=192-12 #must be multiple of 12, default: 192;
  BATCHSIZE=4
  
  #------	for debug
	#IS_DFN='false'

  LOG_SUMMARY_STEP=20
	RESUME='./checkpoints/saved/ganet-pretrained/ganet-deep/sceneflow_epoch_10.pth'
	EXP_NAME="${MODEL_NAME_STR}-D${MAX_DISP}-ganetdeep-sfepo10-${KT_STR}epo${NUM_EPOCHS_STR}-woEmbed${LR_STR}"

elif [ $MODEL_NAME == 'ASN-Embed-GANet-Deep' ]; then
	let CROP_HEIGHT=240-48 #must be multiple of 48;
	let CROP_WIDTH=528-96-48 #must be multiple of 48;
	let MAX_DISP=192 #must be multiple of 12, default: 192;
	#let MAX_DISP=192-120 #must be multiple of 12, default: 192;
  BATCHSIZE=1

  #------	for debug
	#let CROP_HEIGHT=240-96 #must be multiple of 48;
	#let CROP_WIDTH=528-96-48 #must be multiple of 48;
	#let MAX_DISP=192-120 #must be multiple of 12, default: 192;
  #BATCHSIZE=2
  
	LOG_SUMMARY_STEP=20
	RESUME='./checkpoints/saved/ganet-pretrained/ganet-deep/sceneflow_epoch_10.pth'
  #EXP_NAME="ganet-deep-D${MAX_DISP}-sfepo10-${KT_STR}epo${NUM_EPOCHS_STR}"
  EXP_NAME="${MODEL_NAME_STR}-D${MAX_DISP}-ganetdeep-sfepo10-${KT_STR}epo${NUM_EPOCHS_STR}-embedlossW-${EMBED_LOSS_WEIGHT_STR}${LR_STR}"

elif [ $MODEL_NAME == 'ASN-Embed-GANet11' ]; then
	let CROP_HEIGHT=240-96 #must be multiple of 48;
	let CROP_WIDTH=528-96 #must be multiple of 48;
	#let MAX_DISP=192 #must be multiple of 12, default: 192;
	let MAX_DISP=192-120 #must be multiple of 12, default: 192;
  BATCHSIZE=1
  LOG_SUMMARY_STEP=20
	RESUME='./checkpoints/saved/ganet-pretrained/ganet11/ganet11-D192-sfepo10_epoch_3.pth'
  EXP_NAME="${MODEL_NAME_STR}-D${MAX_DISP}-ganet11-sfepo10-${KT_STR}epo${NUM_EPOCHS_STR}-embedlossW-${EMBED_LOSS_WEIGHT_STR}${LR_STR}"

elif [ $MODEL_NAME == 'ASN-SGA-DispNetC' ]; then
	#let CROP_HEIGHT=256
	#let CROP_WIDTH=512
	#let MAX_DISP=192
	let CROP_HEIGHT=320
	let CROP_WIDTH=768-64-64
	let MAX_DISP=192
  BATCHSIZE=2
  LOG_SUMMARY_STEP=20
	RESUME='./checkpoints/saved/dispnetV4-D192-BN-corrV1-sfepo20/DispNetC/model_epoch_00020.tar'
  EXP_NAME="${MODEL_NAME_STR}-D${MAX_DISP}-dispnetc-sfepo20-${KT_STR}epo${NUM_EPOCHS_STR}-woEmbed${LR_STR}"

elif [ $MODEL_NAME == 'ASN-SGA-PSM' ]; then
	let CROP_HEIGHT=256
	let CROP_WIDTH=512+64
	let MAX_DISP=192
  BATCHSIZE=4
  #LOG_SUMMARY_STEP=20
  LOG_SUMMARY_STEP=150
  EXP_NAME="${MODEL_NAME_STR}-D${MAX_DISP}-psm-sfepo10-${KT_STR}epo${NUM_EPOCHS_STR}-woEmbed${LR_STR}"
	if [ $START_EPOCH -eq 0 ]; then
    RESUME='./checkpoints/saved/psmnet-pretrained/pretrained_sceneflow.tar'
		#RESUME='./checkpoints/saved/asn-embed-k5-d2-psm-sfepo10-kt15epo400-embedlossW-0.006/ASN-Embed-PSM/model_epoch_00400.tar'
	else
		RESUME="./checkpoints/saved/${EXP_NAME}/ASN-SGA-PSM/model_epoch_$(printf "%05d" "$START_EPOCH").tar"
	fi

elif [ $MODEL_NAME == 'ASN-Embed-PSM' ]; then
	let CROP_HEIGHT=256
	let CROP_WIDTH=512
	let MAX_DISP=192
  BATCHSIZE=2
  LOG_SUMMARY_STEP=150
	#EXP_NAME="${MODEL_NAME_STR}-D${MAX_DISP}-psm-sfepo10-${KT_STR}epo${NUM_EPOCHS_STR}-embedlossW-${EMBED_LOSS_WEIGHT_STR}${LR_STR}"
	# updated for kt12:
	#EXP_NAME="${MODEL_NAME_STR}-D${MAX_DISP}-psm-sfepo10-kt15epo400-${KT_STR}epo${NUM_EPOCHS_STR}-embedlossW-${EMBED_LOSS_WEIGHT_STR}${LR_STR}"
	EXP_NAME="${MODEL_NAME_STR}-D${MAX_DISP}-psm-sfepo10-${KT_STR}epo${NUM_EPOCHS_STR}-embedlossW-${EMBED_LOSS_WEIGHT_STR}${LR_STR}"
	if [ $START_EPOCH -eq 0 ]; then
		RESUME='./checkpoints/saved/psmnet-pretrained/pretrained_sceneflow.tar'
		#RESUME='./checkpoints/saved/asn-embed-k5-d2-psm-sfepo10-kt15epo400-embedlossW-0.006/ASN-Embed-PSM/model_epoch_00400.tar'
	else
		RESUME="./checkpoints/saved/${EXP_NAME}/ASN-Embed-PSM/model_epoch_$(printf "%05d" "$START_EPOCH").tar"
	fi


elif [ $MODEL_NAME == 'ASN-Embed-GCNet' ]; then
	let CROP_HEIGHT=256
	let CROP_WIDTH=512
	let MAX_DISP=192
  BATCHSIZE=2
  LOG_SUMMARY_STEP=17
  EXP_NAME="${MODEL_NAME_STR}-D${MAX_DISP}-${GCNET_NAME_STR}-sfepo10-${KT_STR}epo${NUM_EPOCHS_STR}-embedlossW-${EMBED_LOSS_WEIGHT_STR}${LR_STR}"
  #RESUME='./checkpoints/saved/gcnet-D192-sfepo20/model_epoch_00020.tar'
	if [ $START_EPOCH -eq 0 ]; then
		RESUME="./checkpoints/saved/${TMP_GCNET}-D${MAX_DISP}-sfepo10/GCNet/model_epoch_00010.tar"
	else
		RESUME="./checkpoints/saved/${EXP_NAME}/ASN-Embed-GCNet/model_epoch_$(printf "%05d" "$START_EPOCH").tar"
	fi

elif [ $MODEL_NAME == 'ASN-DFN-GCNet' ]; then
	let CROP_HEIGHT=256
	let CROP_WIDTH=512
	let MAX_DISP=192
  BATCHSIZE=3
  LOG_SUMMARY_STEP=80
  
	#---------
	# for debugging multiple GPUs;
	let CROP_HEIGHT=256
	let CROP_WIDTH=512
	let MAX_DISP=192
	let MAX_DISP=192-64
  BATCHSIZE=2

  #RESUME='./checkpoints/saved/gcnet-D192-sfepo20/model_epoch_00020.tar'
  EXP_NAME="${MODEL_NAME_STR}-D${MAX_DISP}-${GCNET_NAME_STR}-sfepo10-${KT_STR}epo${NUM_EPOCHS_STR}-woEmbed${LR_STR}"
	if [ $START_EPOCH -eq 0 ]; then
		RESUME="./checkpoints/saved/${TMP_GCNET}-D${MAX_DISP}-sfepo10/GCNet/model_epoch_00010.tar"
	else
		RESUME="./checkpoints/${EXP_NAME}/ASN-DFN-GCNet/model_epoch_$(printf "%05d" "$START_EPOCH").tar"
	fi

elif [ $MODEL_NAME == 'ASN-PAC-GCNet' ]; then
	let CROP_HEIGHT=256+64
	let CROP_WIDTH=512+128
	let MAX_DISP=192
  BATCHSIZE=5
  LOG_SUMMARY_STEP=20
  EXP_NAME="${MODEL_NAME_STR}-D${MAX_DISP}-${GCNET_NAME_STR}-sfepo10-${KT_STR}epo${NUM_EPOCHS_STR}-woEmbed${LR_STR}"
	if [ $START_EPOCH -eq 0 ]; then
		RESUME="./checkpoints/saved/${TMP_GCNET}-D${MAX_DISP}-sfepo10/GCNet/model_epoch_00010.tar"
		RESUME="./checkpoints/saved/asn-pac-k5-d2-D192-gcnetAKQ-sfepo10-vkt2epo30-woEmbed-lr-0.001-e-epothrd-2/ASN-PAC-GCNet/model_epoch_00030.tar"
	else
		RESUME="./checkpoints/saved/${EXP_NAME}/ASN-PAC-GCNet/model_epoch_$(printf "%05d" "$START_EPOCH").tar"
	fi

elif [ $MODEL_NAME == 'ASN-SGA-GCNet' ]; then
	let CROP_HEIGHT=256
	let CROP_WIDTH=512
	let MAX_DISP=192
  BATCHSIZE=7
  LOG_SUMMARY_STEP=11
  #LOG_SUMMARY_STEP=20
  #EXP_NAME="${MODEL_NAME_STR}-D${MAX_DISP}-gcnet-sfepo20-${KT_STR}epo${NUM_EPOCHS_STR}-woEmbed"
  EXP_NAME="${MODEL_NAME_STR}-D${MAX_DISP}-${GCNET_NAME_STR}-sfepo10-${KT_STR}epo${NUM_EPOCHS_STR}-woEmbed${LR_STR}"
	if [ $START_EPOCH -eq 0 ]; then
		RESUME="./checkpoints/saved/${TMP_GCNET}-D${MAX_DISP}-sfepo10/GCNet/model_epoch_00010.tar"
	else
		RESUME="./checkpoints/saved/${EXP_NAME}/ASN-SGA-GCNet/model_epoch_$(printf "%05d" "$START_EPOCH").tar"
	fi

elif [ $MODEL_NAME == 'ASN-DFN-PSM' ]; then
	let CROP_HEIGHT=256
	let CROP_WIDTH=512
	let MAX_DISP=192
	LOG_SUMMARY_STEP=150
  BATCHSIZE=2
  RESUME='./checkpoints/saved/psmnet-pretrained/pretrained_sceneflow.tar'
  EXP_NAME="${MODEL_NAME_STR}-D${MAX_DISP}-psm-sfepo10-${KT_STR}epo${NUM_EPOCHS_STR}-woEmbed${LR_STR}"
  #newly added lr scheduler for virtual kitti 2 dataset fine-tuning;
	NUM_EPOCHS=20
	LR_ADJUST_EPO_THRED=2
	LR_SCHEDULER="exponential"
	# for debugging : 'ASN-DFN-PSM-NoDFN'
	#IS_DFN='false'

elif [ $MODEL_NAME == 'ASN-PAC-PSM' ]; then
	let CROP_HEIGHT=256
	let CROP_WIDTH=512+64
	let MAX_DISP=192
	LOG_SUMMARY_STEP=20
  BATCHSIZE=2
  if [ $TASK_TYPE == 'PAC-EMBED' ]; then
		EXP_NAME="${MODEL_NAME_STR}-D${MAX_DISP}-psm-sfepo10-${KT_STR}epo${NUM_EPOCHS_STR}-embedlossW-${EMBED_LOSS_WEIGHT_STR}${LR_STR}"
	elif [ $TASK_TYPE == 'PAC' ]; then
		EXP_NAME="${MODEL_NAME_STR}-D${MAX_DISP}-psm-sfepo10-${KT_STR}epo${NUM_EPOCHS_STR}-woEmbed${LR_STR}"
	fi 
  #RESUME='./checkpoints/saved/psmnet-pretrained/pretrained_sceneflow.tar'
	if [ $START_EPOCH -eq 0 ]; then
    RESUME='./checkpoints/saved/psmnet-pretrained/pretrained_sceneflow.tar'
	else
		RESUME="./checkpoints/saved/${EXP_NAME}/ASN-PAC-PSM/model_epoch_$(printf "%05d" "$START_EPOCH").tar"
	fi

elif [ $MODEL_NAME == 'ASN-Embed-DispNetC' ]; then
	let CROP_HEIGHT=256+64
	#let CROP_WIDTH=768
	let CROP_WIDTH=512+64
	#let CROP_WIDTH=768-128
	#let MAX_DISP=192
	LOG_SUMMARY_STEP=10
	#LOG_SUMMARY_STEP=100
  #BATCHSIZE=16
  BATCHSIZE=8
  #BATCHSIZE=1
  EXP_NAME="${MODEL_NAME_STR}-D${MAX_DISP}-dispnetc-sfepo20-${KT_STR}epo${NUM_EPOCHS_STR}-embedlossW-${EMBED_LOSS_WEIGHT_STR}${LR_STR}"
	if [ $START_EPOCH -eq 0 ]; then
    RESUME='./checkpoints/saved/dispnetV4-D192-BN-corrV1-sfepo20/DispNetC/model_epoch_00020.tar'
	else
		RESUME="./checkpoints/saved/${EXP_NAME}/ASN-Embed-DispNetC/model_epoch_$(printf "%05d" "$START_EPOCH").tar"
	fi

elif [ $MODEL_NAME == 'ASN-PAC-DispNetC' ]; then
	let CROP_HEIGHT=320
	let CROP_WIDTH=768+64
	let MAX_DISP=192
  #LOG_SUMMARY_STEP=10
  LOG_SUMMARY_STEP=50
  BATCHSIZE=4
  if [ $TASK_TYPE == 'PAC-EMBED' ]; then
		EXP_NAME="${MODEL_NAME_STR}-D${MAX_DISP}-dispnetc-sfepo20-${KT_STR}epo${NUM_EPOCHS_STR}-embedlossW-${EMBED_LOSS_WEIGHT_STR}${LR_STR}"
	elif [ $TASK_TYPE == 'PAC' ]; then
	  EXP_NAME="${MODEL_NAME_STR}-D${MAX_DISP}-dispnetc-sfepo20-${KT_STR}epo${NUM_EPOCHS_STR}-woEmbed${LR_STR}"
	fi 
	if [ $START_EPOCH -eq 0 ]; then
	  RESUME='./checkpoints/saved/dispnetV4-D192-BN-corrV1-sfepo20/DispNetC/model_epoch_00020.tar'
	else
		RESUME="./checkpoints/saved/${EXP_NAME}/ASN-PAC-DispNetC/model_epoch_$(printf "%05d" "$START_EPOCH").tar"
	fi

elif [ $MODEL_NAME == 'ASN-DFN-DispNetC' ]; then
	let CROP_HEIGHT=320
	let CROP_WIDTH=768+64
	#let CROP_HEIGHT=256
	#let CROP_WIDTH=512
	let MAX_DISP=192
	LOG_SUMMARY_STEP=25
	BATCHSIZE=16
	EXP_NAME="${MODEL_NAME_STR}-D${MAX_DISP}-dispnetc-sfepo20-${KT_STR}epo${NUM_EPOCHS_STR}-woEmbed${LR_STR}"
	if [ $START_EPOCH -eq 0 ]; then
	  #RESUME='./checkpoints/saved/dispnet-D192-BN-corrV1-sfepo20/DispNetC/model_epoch_00020.tar'
	  RESUME='./checkpoints/saved/dispnetV4-D192-BN-corrV1-sfepo20/DispNetC/model_epoch_00020.tar'
	else
		RESUME="./checkpoints/saved/${EXP_NAME}/ASN-DFN-DispNetC/model_epoch_$(printf "%05d" "$START_EPOCH").tar"
	fi
fi



echo "Kernek size = $K_WIDTH x $K_WIDTH"
#EXP_NAME="asn-embed-sga-sf-epo10"
#EXP_NAME="asn-sga-sf-small-tmp"
echo "EXP_NAME=$EXP_NAME"

CHECKPOINT_DIR="./checkpoints/${EXP_NAME}"
echo "CHECKPOINT_DIR=$CHECKPOINT_DIR"
TRAIN_LOGDIR="./logs/${EXP_NAME}"
echo "TRAIN_LOGDIR=$TRAIN_LOGDIR"
#exit



################################
# Netwrok Training & profiling
################################
flag=false
#flag=true
if [ "$flag" = true ]; then
	cd ${MY_PROJECT_ROOT}
	MODE='train'
	#MODE='debug'
  RESULTDIR="${MY_PROJECT_ROOT}/results/${EXP_NAME}"
	#CUDA_VISIBLE_DEVICES=0 python3.7 -m main_attenStereoNet \
	CUDA_VISIBLE_DEVICES=$1 python3.7 -m main_attenStereoNet \
		--batchSize=${BATCHSIZE} \
		--crop_height=$CROP_HEIGHT \
		--crop_width=$CROP_WIDTH \
		--max_disp=$MAX_DISP \
		--train_logdir=$TRAIN_LOGDIR \
		--thread=${NUM_WORKERS} \
		--data_path=$DATA_PATH \
		--training_list=$TRAINING_LIST \
		--test_list=$TEST_LIST \
		--checkpoint_dir=$CHECKPOINT_DIR \
		--log_summary_step=${LOG_SUMMARY_STEP} \
		--resume=$RESUME \
		--model_name=$MODEL_NAME \
		--nEpochs=$NUM_EPOCHS \
		--startEpoch=$START_EPOCH \
		--sigma_s=$SIGMA_S \
		--sigma_v=$SIGMA_V \
		--is_embed=$IS_EMBED \
		--kitti2012=$KT2012 \
		--kitti2015=$KT2015 \
		--virtual_kitti2=$VIRTUAL_KITTI2 \
		--mode=$MODE \
		--saved_embednet_checkpoint=$RESUME_EMBEDNET \
		--isFreezeEmbed=$IS_FREEZE_EMBED \
		--resultDir=$RESULTDIR \
		--embed_loss_weight=$EMBED_LOSS_WEIGHT \
		--dilation=$DILATION \
		--cost_filter_grad=$COST_FILTER_GRAD \
		--is_dfn=$IS_DFN \
		--dfn_kernel_size=$DFN_K_WIDTH \
		--is_pac=$IS_PAC \
		--pac_kernel_size=$PAC_K_WIDTH \
		--pac_native_imple=$PAC_NATIVE_IMPLE \
		--is_sga_guide_from_img=$IS_SGA_GUIDE_FROM_IMG \
		--is_quarter_size_cost_volume_gcnet=$IS_QUARTER_SIZE_COST_VOLUME_GCNET \
		--is_kendall_version_gcnet=$IS_KENDALL_VERSION_GCNET \
		--lr_adjust_epo_thred=$LR_ADJUST_EPO_THRED \
    --lr_scheduler=$LR_SCHEDULER \
		--lr=$LEARNING_RATE \
	  --kt12_image_mode=$KT12_IMAGE_MODE \
		--lr_epoch_steps=$LR_EPOCH_STEPS
	  exit
fi

######################
# Netwrok Testing 
######################
#flag=false
flag=true
if [ "$flag" = true ]; then
	MODE='test'
	#kt15/12: crop_height=384, crop_width=1248
	#sceneflow: crop_height=576, crop_width=960
	#echo $MODEL_NAME
	let CROP_HEIGHT=384
	#Use double brackets and wildcards *, for 
	if [[ $MODEL_NAME == *"DispNetC"* ]]; then
		let CROP_WIDTH=1280 # multiple of 64, due to DispNetC
		ENCODER_DS=64
	elif [[ $MODEL_NAME == *"GANet"* ]]; then
		let CROP_WIDTH=1248 # multiple of 48, due to GANet
		ENCODER_DS=48
	elif [[ $MODEL_NAME == *"GCNet"* ]]; then
		let CROP_WIDTH=1280 # multiple of 64, due to GCNet
		ENCODER_DS=64
	elif [[ $MODEL_NAME == *"SGA"* ]]; then
		let CROP_WIDTH=1280 # multiple of 64, due to SGA module
		ENCODER_DS=64
	else
		let CROP_WIDTH=1248
		ENCODER_DS=32
  fi
	
	declare -a ALL_EPOS_TEST=(50 100 150 200 250 300 350 400 450 500 550 600 650 700 750 800 850 900 950 1000)
	declare -a ALL_EPOS_TEST=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60)
	declare -a ALL_EPOS_TEST=(15 14 13 12 7 6 5 4 11 10 9 8 7 6 5 4 3 2 1) 
	declare -a ALL_EPOS_TEST=(15)
	declare -a ALL_EPOS_TEST=(25 50 75 100 125 150 175 200 225 250 275 300 325 350 375 400 425 450 475 500 525 550 575 600 625 650 675 700 725 750 775 800)
	#for idx in $(seq 1 19)
	#for idx in $(seq 16 31)
	#for idx in $(seq 1 15)
	#for idx in $(seq 0 23)
	for idx in $(seq 0 0)
	do # epoch model loop
		EPO_TEST=${ALL_EPOS_TEST[idx]}
		EPO_TEST=250
		echo $EPO_TEST
		#-------------------------------
		# baseline1: pre-trained GANet:
		#-------------------------------
		if [ "$1" == 0 ]; then
			#echo "test GANet baseline: SF --> KITTI !!!"
			#RESUME="./checkpoints/saved/ganet-sf-epo10-${KT_STR}-epoch100/model_epoch_00100.tar"
			#EXP_NAME="ganet-sfepo10-${KT_STR}epo200/"
			RESUME='./checkpoints/saved/ganet-pretrained/sceneflow_epoch_10.pth'
			IS_EMBED='false'
			KT2015=0
			if [ "$KT2015" == 1 ]; then
				KT2012=0
				TEST_LIST="lists/kitti2015.list"
				EXP_NAME="ganet-sfepo10-testKT15"
				echo "test GANet baseline: SF --> KT15 !!!"
			else
				KT2012=1
				KT2015=0
				TEST_LIST="lists/kitti2012.list"
				EXP_NAME="ganet-sfepo10-testKT12"
				echo "test GANet baseline: SF --> KT12 !!!"
			fi
		#----------------------
		#  Method 1) : GANet + Embednet (which is frozen during training)
		#----------------------
		elif [ "$1" == 1 ]; then
			echo "test Method 1: GANet + Froezen Embednet !!!"
			RESUME="./checkpoints/saved/asn-froznEmbed-k13-sga-sfepo10-${KT_STR}epo200/model_epoch_00200.tar"
			EXP_NAME="asn-embed-k13-sga-sfepo10-${KT_STR}epo200/"
			IS_EMBED='true'
			RESUME_EMBEDNET='./checkpoints/pascalvoc-embednet-epo30/vgg-like-embed/best_model_epoch_00030_valloss_1.2961.tar'
			IS_FREEZE_EMBED='true'
		#----------------------
		#  Method 2) : GANet + Embednet (training together)
		#----------------------
		elif [ "$1" == 2 ]; then
			echo "test Method 2: GANet + Embednet !!!"
			TMP_MODEL_NAME="asn-embed-k5-d2-D192-ganetdeep-sfepo10-vkt2epo20-embedlossW-0.06-lr-0.001-p-eposteps-5-18"
			RESUME="./checkpoints/saved/${TMP_MODEL_NAME}/ASN-Embed-GANet-Deep/model_epoch_$(printf "%05d" "$EPO_TEST").tar"
			EXP_NAME="${TMP_MODEL_NAME}/disp-epo-$(printf "%03d" "$EPO_TEST")"
			##already contained in the RESUME:
			RESUME_EMBEDNET=''
		
		#-------------------------------
		# baseline2: PSMNet:
		#-------------------------------
		elif [ "$1" == 3 ]; then
			echo "test PSM baseline: SF + KITTI !!!"
			#EPO_TEST=400
			RESUME="./checkpoints/saved/psm-sfepo10-${KT_STR}epo400/model_epoch_$(printf "%05d" "$EPO_TEST").tar"
			EXP_NAME="psm-sfepo10-${KT_STR}epo400/disp-epo-$(printf "%03d" "$EPO_TEST")"
			IS_EMBED='false'
		
		#----------------------
		#  Method 3) : PSMNet + Embednet (training together)
		#----------------------
		elif [ "$1" == 4 ]; then
			echo "test Method 3: PSMNet + Embednet !!!"
			# experiment 4:
			if [ 1 -eq 1 ]; then
				#TMP_MODEL_NAME="asn-embed-k5-d2-D192-psm-sfepo10-${KT_STR}epo400-embedlossW-0.06"
				#TMP_MODEL_NAME="asn-embed-k5-d2-D192-psm-sfepo10-kt12epo400-embedlossW-no"
				#TMP_MODEL_NAME="asn-embed-k5-d2-D192-psm-sfepo10-kt15epo400-kt12epo400-embedlossW-no"
				TMP_MODEL_NAME="asn-embed-k5-d2-D192-psm-sfepo10-vkt2epo10-embedlossW-0.06-lr-0.001-e-epothrd-22"
				#RESUME="./checkpoints/${TMP_MODEL_NAME}/ASN-Embed-PSM/model_epoch_$(printf "%05d" "$EPO_TEST").tar"
				RESUME="./checkpoints/saved/${TMP_MODEL_NAME}/ASN-Embed-PSM/model_epoch_$(printf "%05d" "$EPO_TEST").tar"
				EXP_NAME="${TMP_MODEL_NAME}/disp-epo-$(printf "%03d" "$EPO_TEST")"
				##already contained in the RESUME:
				RESUME_EMBEDNET=''
			fi

		#----------------------
		#  Method 4) : PSMNet + DFN(Dynamic Filter Network)
		#----------------------
		elif [ "$1" == 5 ]; then
			echo "test Method 4: PSMNet + DFN !!!"
			# experiment 1:
			if [ 1 -eq 1 ]; then
				#TMP_MODEL_NAME="asn-dfn-k5-d2-D192-psm-sfepo10-${KT_STR}epo400-woEmbed"
				#TMP_MODEL_NAME="asn-dfn-k5-d2-D192-psm-sfepo10-${KT_STR}epo400-woEmbed-lr-0.001"
				#TMP_MODEL_NAME="asn-psm-sfepo10-${KT_STR}epo400-lr-0.001"
				TMP_MODEL_NAME="asn-dfn-k5-d2-D192-psm-sfepo10-${KT_STR}epo20-woEmbed"
				RESUME="./checkpoints/${TMP_MODEL_NAME}/ASN-DFN-PSM/model_epoch_$(printf "%05d" "$EPO_TEST").tar"
				EXP_NAME="${TMP_MODEL_NAME}/disp-epo-$(printf "%03d" "$EPO_TEST")"
				#already contained in the RESUME:
				RESUME_EMBEDNET=''
			fi
		
		#----------------------
		#  Method 5) : PSMNet + PAC(Pixel-adaptive Convolution Network)
		#----------------------
		elif [ "$1" == 6 ]; then
			echo "test Method 5: PSMNet + PAC !!!"
			# experiment 1: PAC + Featue-of-left-image
			if [ 1 -eq 1 ]; then
				#TMP_MODEL_NAME="asn-npac-k5-d2-D192-psm-sfepo10-${KT_STR}epo400-woEmbed"
				TMP_MODEL_NAME="asn-npac-k5-d2-D192-psm-sfepo10-vkt2epo20-woEmbed-lr-0.001-e-epothrd-2"
        BATCH_IN_IMAGE='true'
				RESUME="./checkpoints/saved/${TMP_MODEL_NAME}/ASN-PAC-PSM/model_epoch_$(printf "%05d" "$EPO_TEST").tar"
				EXP_NAME="${TMP_MODEL_NAME}/disp-epo-$(printf "%03d" "$EPO_TEST")"
				#already contained in the RESUME:
				RESUME_EMBEDNET=''
			fi
		#----------------------
		#  Method 6) : PSMNet + Embed-Bilateral
		#----------------------
		elif [ "$1" == 7 ]; then
			echo "test Method 6: Embed+ DispNetC !!!"
			# experiment 1:
			if [ 1 -eq 1 ]; then
				#TMP_MODEL_NAME="asn-embed-k3-d2-D192-dispnetc-sfepo10-${KT_STR}epo400-embedlossW-0.06"
				#TMP_MODEL_NAME="asn-embed-k5-d2-D192-dispnetc-sfepo20-${KT_STR}epo800-embedlossW-no"
				TMP_MODEL_NAME="asn-embed-k5-d2-D192-dispnetc-sfepo20-${KT_STR}epo20-embedlossW-0.06"
				TMP_MODEL_NAME="asn-embed-k${K_WIDTH}-d${DILATION}-D${MAX_DISP}-dispnetc-sfepo20-${KT_STR}epo${NUM_EPOCHS_STR}-embedlossW-${EMBED_LOSS_WEIGHT_STR}${LR_STR}"
				RESUME="./checkpoints/saved/${TMP_MODEL_NAME}/ASN-Embed-DispNetC/model_epoch_$(printf "%05d" "$EPO_TEST").tar"
				EXP_NAME="${TMP_MODEL_NAME}/disp-epo-$(printf "%03d" "$EPO_TEST")"
				#already contained in the RESUME:
				RESUME_EMBEDNET=''
			fi
		#----------------------
		#  Method 7) : DispNetC + DFN
		#----------------------
	  elif [ "$1" == 8 ]; then
			echo "test Method 7: DFN + DispNetC !!!"
			# experiment 1:
			if [ 1 -eq 1 ]; then
				#TMP_MODEL_NAME="asn-dfn-embed-k5-d2-D192-dispnetc-sfepo9-${KT_STR}epo400-woEmbed"
				TMP_MODEL_NAME="asn-dfn-k5-d2-D192-dispnetc-sfepo20-${KT_STR}epo20-woEmbed"
				#RESUME="./checkpoints/saved/${TMP_MODEL_NAME}/ASN-DFN-DispNetC/model_epoch_$(printf "%05d" "$EPO_TEST").tar"
				RESUME="./checkpoints/${TMP_MODEL_NAME}/ASN-DFN-DispNetC/model_epoch_$(printf "%05d" "$EPO_TEST").tar"
				EXP_NAME="${TMP_MODEL_NAME}/disp-epo-$(printf "%03d" "$EPO_TEST")"
				#already contained in the RESUME:
				RESUME_EMBEDNET=''
			fi
		#----------------------
		#  Method 8) : DispNetC + PAC
		#----------------------
	  elif [ "$1" == 9 ]; then
			echo "test Method 8: PAC + DispNetC !!!"
			# experiment 1:
			if [ 1 -eq 1 ]; then
				TMP_MODEL_NAME="asn-npac-k5-d2-D192-dispnetc-sfepo20-${KT_STR}epo1000-woEmbed"
				TMP_MODEL_NAME="asn-npac-embed-k5-d2-D192-dispnetc-sfepo20-${KT_STR}epo400-embedlossW-0.06"
				TMP_MODEL_NAME="asn-npac-embed-k5-d2-D192-dispnetcV4-sfepo9-${KT_STR}epo400-embedlossW-0.06"
				TMP_MODEL_NAME="asn-npac-embed-k5-d2-D192-dispnetcV4-sfepo20-${KT_STR}epo1000-embedlossW-0.06"
				TMP_MODEL_NAME="asn-pac-k5-d2-D192-dispnetc-sfepo20-${KT_STR}epo20-woEmbed"
				TMP_MODEL_NAME="asn-npac-k5-d2-D192-dispnetc-sfepo20-${KT_STR}epo20-woEmbed"
				RESUME="./checkpoints/${TMP_MODEL_NAME}/ASN-PAC-DispNetC/model_epoch_$(printf "%05d" "$EPO_TEST").tar"
				EXP_NAME="${TMP_MODEL_NAME}/disp-epo-$(printf "%03d" "$EPO_TEST")"
				#already contained in the RESUME:
				RESUME_EMBEDNET=''
			fi

		#----------------------
		#  Method 9) : SGA + PSM
		#----------------------
	  elif [ "$1" == 10 ]; then
			echo "test Method 9: SGA + PSMNet !!!"
			# experiment 1:
			if [ 1 -eq 1 ]; then
				#TMP_MODEL_NAME="asn-sga-k0-d2-D192-psm-sfepo10-${KT_STR}epo400-woEmbed"
				TMP_MODEL_NAME="asn-sga-k0-d2-D192-psm-sfepo10-vkt2epo20-woEmbed-lr-0.001-e-epothrd-2"
				RESUME="./checkpoints/saved/${TMP_MODEL_NAME}/ASN-SGA-PSM/model_epoch_$(printf "%05d" "$EPO_TEST").tar"
				EXP_NAME="${TMP_MODEL_NAME}/disp-epo-$(printf "%03d" "$EPO_TEST")"
				#already contained in the RESUME:
				RESUME_EMBEDNET=''
			fi

		#----------------------
		#  Method 10) : SGA + DispNetC
		#----------------------
	  elif [ "$1" == 11 ]; then
			echo "test Method 10: SGA + DispNetC !!!"
			# experiment 1:
			if [ 1 -eq 1 ]; then
				TMP_MODEL_NAME="asn-sga-k0-d2-D192-dispnetc-sfepo20-${KT_STR}epo400-woEmbed"
				RESUME="./checkpoints/saved/${TMP_MODEL_NAME}/ASN-SGA-DispNetC/model_epoch_$(printf "%05d" "$EPO_TEST").tar"
				EXP_NAME="${TMP_MODEL_NAME}/disp-epo-$(printf "%03d" "$EPO_TEST")"
				#already contained in the RESUME:
				RESUME_EMBEDNET=''
			fi
		
    #----------------------
		#  Method 11) : DFN + GANet-Deep
		#----------------------
	  elif [ "$1" == 12 ]; then
			echo "test Method 11: DFN + GANet-Deep !!!"
			# experiment 1:
			if [ 1 -eq 1 ]; then
				#TMP_MODEL_NAME="asn-dfn-k5-d2-D${MAX_DISP}-ganetdeep-sfepo10-${KT_STR}epo400-woEmbed"
				TMP_MODEL_NAME="asn-dfn-k5-d2-D${MAX_DISP}-ganetdeep-sfepo10-vkt2epo20-woEmbed-lr-0.001-p-eposteps-5-18"
				RESUME="./checkpoints/saved/${TMP_MODEL_NAME}/ASN-DFN-GANet-Deep/model_epoch_$(printf "%05d" "$EPO_TEST").tar"
				EXP_NAME="${TMP_MODEL_NAME}/disp-epo-$(printf "%03d" "$EPO_TEST")"
				#already contained in the RESUME:
				RESUME_EMBEDNET=''
			fi
    
		#----------------------
		#  Method 12) : DFN + GCNet
		#----------------------
	  elif [ "$1" == 13 ]; then
			echo "test Method 12: DFN + GCNet !!!"
			# experiment 1:
			if [ 1 -eq 1 ]; then
				#TMP_MODEL_NAME="asn-dfn-k5-d2-D192-gcnetAKQ-sfepo10-${KT_STR}epo400-woEmbed"
				TMP_MODEL_NAME="asn-dfn-k5-d2-D192-gcnetAKQ-sfepo10-vkt2epo20-woEmbed-lr-0.001-e-epothrd-2"
				RESUME="./checkpoints/saved/${TMP_MODEL_NAME}/ASN-DFN-GCNet/model_epoch_$(printf "%05d" "$EPO_TEST").tar"
				EXP_NAME="${TMP_MODEL_NAME}/disp-epo-$(printf "%03d" "$EPO_TEST")"
				#already contained in the RESUME:
				RESUME_EMBEDNET=''
			fi
		
		#----------------------
		#  Method 13) : PAC + GANet-Deep
		#----------------------
	  elif [ "$1" == 14 ]; then
			echo "test Method 13: PAC + GANet-Deep !!!"
			# experiment 1:
			if [ 1 -eq 1 ]; then
				#TMP_MODEL_NAME="asn-npac-k5-d2-D192-ganetdeep-sfepo10-${KT_STR}epo400-woEmbed"
				TMP_MODEL_NAME="asn-pac-k5-d2-D${MAX_DISP}-ganetdeep-sfepo10-${KT_STR}epo20-woEmbed-lr-0.001-p-eposteps-5-18"
				RESUME="./checkpoints/saved/${TMP_MODEL_NAME}/ASN-PAC-GANet-Deep/model_epoch_$(printf "%05d" "$EPO_TEST").tar"
				EXP_NAME="${TMP_MODEL_NAME}/disp-epo-$(printf "%03d" "$EPO_TEST")"
				#already contained in the RESUME:
				RESUME_EMBEDNET=''
				echo "14: RESUME= $RESUME"
				#BATCH_IN_IMAGE='true'
				#BATCH_H=240
				#BATCH_H=144
        #TEST_LIST="./lists/kitti2015_val_small.list"
			fi
		
		#----------------------
		#  Method 14) : SGA + GCNet
		#----------------------
	  elif [ "$1" == 15 ]; then
			echo "test Method 14: SGA + GCNet !!!"
			# experiment 1:
			if [ 1 -eq 1 ]; then
				#TMP_MODEL_NAME="asn-sga-k0-d2-D192-gcnet-sfepo20-${KT_STR}epo400-woEmbed"
				TMP_MODEL_NAME="asn-sga-k0-d2-D192-gcnetAKQ-sfepo10-${KT_STR}epo400-woEmbed"
				RESUME="./checkpoints/saved/${TMP_MODEL_NAME}/ASN-SGA-GCNet/model_epoch_$(printf "%05d" "$EPO_TEST").tar"
				EXP_NAME="${TMP_MODEL_NAME}/disp-epo-$(printf "%03d" "$EPO_TEST")"
				#already contained in the RESUME:
				RESUME_EMBEDNET=''
				#BATCH_IN_IMAGE='true'
				#BATCH_H=256
			fi
		
		#----------------------
		#  Method 15) : PAC + GCNet
		#----------------------
	  elif [ "$1" == 16 ]; then
			echo "test Method 15: PAC + GCNet !!!"
			# experiment 1:
			if [ 1 -eq 1 ]; then
				#TMP_MODEL_NAME="asn-npac-k5-d2-D192-gcnet-sfepo10-${KT_STR}epo400-woEmbed"
				#TMP_MODEL_NAME="asn-npac-k5-d2-D192-gcnetAKQ-sfepo10-${KT_STR}epo400-woEmbed"
				#TMP_MODEL_NAME="asn-pac-k5-d2-D192-gcnetAKQ-sfepo10-${KT_STR}epo30-woEmbed-lr-0.001-e-epothrd-2"
				TMP_MODEL_NAME="asn-pac-k5-d2-D192-gcnetAKQ-sfepo10-${KT_STR}epo30-woEmbed-lr-0.0001-p-eposteps-5-25"
				RESUME="./checkpoints/saved/${TMP_MODEL_NAME}/ASN-PAC-GCNet/model_epoch_$(printf "%05d" "$EPO_TEST").tar"
				EXP_NAME="${TMP_MODEL_NAME}/disp-epo-$(printf "%03d" "$EPO_TEST")"
				#already contained in the RESUME:
				RESUME_EMBEDNET=''
			fi
		
		#---------------------------
		#  Method 16) : EBF + GCNet
		#---------------------------
	  elif [ "$1" == 17 ]; then
			echo "test Method 16: EBF + GCNet !!!"
			# experiment 1:
			if [ 1 -eq 1 ]; then
				#TMP_MODEL_NAME="asn-embed-k5-d2-D192-gcnet-sfepo20-${KT_STR}epo400-embedlossW-0.06"
				#TMP_MODEL_NAME="asn-embed-k5-d2-D192-gcnetQ-sfepo10-${KT_STR}epo400-embedlossW-0.06"
				#TMP_MODEL_NAME="asn-embed-k5-d2-D192-gcnetAKQ-sfepo10-${KT_STR}epo400-embedlossW-0.06"
				TMP_MODEL_NAME="asn-embed-k5-d2-D192-gcnetAKQ-sfepo10-vkt2epo20-embedlossW-0.06-lr-0.001-e-epothrd-2"
				RESUME="./checkpoints/saved/${TMP_MODEL_NAME}/ASN-Embed-GCNet/model_epoch_$(printf "%05d" "$EPO_TEST").tar"
				EXP_NAME="${TMP_MODEL_NAME}/disp-epo-$(printf "%03d" "$EPO_TEST")"
				#already contained in the RESUME:
				RESUME_EMBEDNET=''
			fi


		else 
			echo "You have to specify a argument to bash!!!"
			exit
    fi
		
		RESULTDIR="${MY_PROJECT_ROOT}/results/${EXP_NAME}"
		#cd /home/${USER}/atten-stereo
		cd ${MY_PROJECT_ROOT}
		CUDA_VISIBLE_DEVICES=0 python3.7 -m main_attenStereoNet \
			--batchSize=${BATCHSIZE} \
			--crop_height=$CROP_HEIGHT \
			--crop_width=$CROP_WIDTH \
			--max_disp=$MAX_DISP \
			--train_logdir=$TRAIN_LOGDIR \
			--thread=${NUM_WORKERS} \
			--data_path=$DATA_PATH \
			--training_list=$TRAINING_LIST \
			--test_list=$TEST_LIST \
			--checkpoint_dir=$CHECKPOINT_DIR \
			--log_summary_step=${LOG_SUMMARY_STEP} \
			--resume=$RESUME \
			--model_name=$MODEL_NAME \
			--nEpochs=$NUM_EPOCHS \
			--startEpoch=$START_EPOCH \
			--sigma_s=$SIGMA_S \
			--sigma_v=$SIGMA_V \
			--is_embed=$IS_EMBED \
			--kitti2012=$KT2012 \
			--kitti2015=$KT2015 \
		  --virtual_kitti2=$VIRTUAL_KITTI2 \
			--mode=$MODE \
			--saved_embednet_checkpoint=$RESUME_EMBEDNET \
			--isFreezeEmbed=$IS_FREEZE_EMBED \
			--resultDir=$RESULTDIR \
			--embed_loss_weight=$EMBED_LOSS_WEIGHT \
			--dilation=$DILATION \
			--cost_filter_grad=$COST_FILTER_GRAD \
			--is_dfn=$IS_DFN \
			--dfn_kernel_size=$DFN_K_WIDTH \
			--is_pac=$IS_PAC \
			--pac_kernel_size=$PAC_K_WIDTH \
		  --pac_native_imple=$PAC_NATIVE_IMPLE \
		  --is_sga_guide_from_img=$IS_SGA_GUIDE_FROM_IMG \
			--batch_in_image=$BATCH_IN_IMAGE \
			--is_quarter_size_cost_volume_gcnet=$IS_QUARTER_SIZE_COST_VOLUME_GCNET \
			--is_kendall_version_gcnet=$IS_KENDALL_VERSION_GCNET \
			--batch_h=$BATCH_H \
			--encoder_ds=$ENCODER_DS \
		  --kt12_image_mode=$KT12_IMAGE_MODE \
			--lr_epoch_steps=$LR_EPOCH_STEPS
	  
		if [ $KT2015 -eq 1 ] || [ $KT2012 -eq 1 ]; then
			# move pfm files to val-30 subdir
			makeDir "$RESULTDIR/val-30"
			cd $RESULTDIR
			for i in *.pfm; do
				mv -i -- "$i" "./val-30"
			done	
			if [ -f "./bad-err.csv" ]; then
				cat "./bad-err.csv" >> "${MY_PROJECT_ROOT}/results/bad-err-evalu.csv"
			fi

		elif [ $VIRTUAL_KITTI2 -eq 1 ]; then
			# move pfm files to disp-pfm subdir
			makeDir "$RESULTDIR/val-2620"
			cd $RESULTDIR
			for i in *.pfm; do
				mv -i -- "$i" "./val-2620"
			done	
			if [ -f "./bad-err.csv" ]; then
				cat "./bad-err.csv" >> "${MY_PROJECT_ROOT}/results/bad-err-evalu.csv"
			fi
		else
			# move pfm files to val-30 subdir
			makeDir "$RESULTDIR/disp-pfm"
			cd $RESULTDIR
			for i in *.pfm; do
				mv -i -- "$i" "./disp-pfm"
				#echo "$i"
			done
			if [ -f "./bad-err.csv" ]; then
				cat "./bad-err.csv" >> "${MY_PROJECT_ROOT}/results/bad-err-evalu.csv"
			fi
		fi
	done
fi # end of Netwrok Testing
