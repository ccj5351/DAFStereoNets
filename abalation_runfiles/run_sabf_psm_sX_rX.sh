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
if [ ! -d $DATA_ROOT ];then
	DATA_ROOT="/data/ccjData"
	echo "Updated : setting data_root = ${DATA_ROOT}"
fi

if [ ! -d $DATA_ROOT ];then
	DATA_ROOT="/diskb/ccjData2"
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

	NUM_EPOCHS=300
	NUM_EPOCHS_STR=300
  LR_ADJUST_EPO_THRED=200
	LR_SCHEDULER="piecewise"
					
	# new try
	LEARNING_RATE=0.001
	LR_SCHEDULER="piecewise"
	LR_EPOCH_STEPS="200"

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
#NUM_EPOCHS=20
#START_EPOCH=400
#NUM_EPOCHS=400
#NUM_EPOCHS=800
#NUM_EPOCHS=20


#NUM_WORKERS=1
NUM_WORKERS=4

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
#DILATION=2
DILATION=$1
if [ "$DILATION" = "-h" ]; then
	echo "#1=DILATION=1,2 #2=SIMPLE_WINOW_SIZE=3,5,7,9,11 #3=BATCH_SIZE=1,2, #4=GPU_ID"
	exit
elif [ "$DILATION" = 1 ]; then
	echo "Set DILATION=1"
elif [ "$DILATION" = 2 ]; then
	echo "Set DILATION=2"
else
	echo "Wrong input! DILATION should be 1 or 2!"
	exit
fi

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



#----------------------------------------------------------#
#-----The only input parameters you have to provide  ------#
#----------------------------------------------------------#
#SIMPLE_TASK_TYPE=$1
SIMPLE_TASK_TYPE='sabf-psm'

echo "Your input SIMPLE_TASK_TYPE : ${SIMPLE_TASK_TYPE} !!!"

if [ $SIMPLE_TASK_TYPE == 'psm' ]; then
  TASK_TYPE='BASELINE'
  MODEL_NAME='PSMNet'
	EPO_TEST=19
elif [ $SIMPLE_TASK_TYPE == 'dispnetc' ]; then
  TASK_TYPE='BASELINE'
  MODEL_NAME='DispNetC'
	EPO_TEST=20
elif [ $SIMPLE_TASK_TYPE == 'ganet' ]; then
  TASK_TYPE='BASELINE'
  MODEL_NAME='GANet-Deep'
	EPO_TEST=10
elif [ $SIMPLE_TASK_TYPE == 'gcnet' ]; then
  TASK_TYPE='BASELINE'
  MODEL_NAME='GCNet'
	EPO_TEST=10

elif [ $SIMPLE_TASK_TYPE == 'sabf-dispnetc' ]; then
  TASK_TYPE='EMBED_BILATERAL'
  MODEL_NAME='ASN-Embed-DispNetC'
	EPO_TEST=17
	TASK_IDX=7
elif [ $SIMPLE_TASK_TYPE == 'sabf-psm' ]; then
  TASK_TYPE='EMBED_BILATERAL'
  MODEL_NAME='ASN-Embed-PSM'
	EPO_TEST=30
elif [ $SIMPLE_TASK_TYPE == 'sabf-ganet' ]; then
  TASK_TYPE='EMBED_BILATERAL'
  MODEL_NAME='ASN-Embed-GANet-Deep'
	EPO_TEST=11
elif [ $SIMPLE_TASK_TYPE == 'sabf-gcnet' ]; then
  TASK_TYPE='EMBED_BILATERAL'
  MODEL_NAME='ASN-Embed-GCNet'
	EPO_TEST=20

elif [ $SIMPLE_TASK_TYPE == 'dfn-dispnetc' ]; then
  TASK_TYPE='DFN'
  MODEL_NAME='ASN-DFN-DispNetC'
	EPO_TEST=18
elif [ $SIMPLE_TASK_TYPE == 'dfn-psm' ]; then
  TASK_TYPE='DFN'
  MODEL_NAME='ASN-DFN-PSM'
	EPO_TEST=18
elif [ $SIMPLE_TASK_TYPE == 'dfn-ganet' ]; then
  TASK_TYPE='DFN'
  MODEL_NAME='ASN-DFN-GANet-Deep'
	# only this one case: D = 180
	MAX_DISP=180 
	EPO_TEST=13
elif [ $SIMPLE_TASK_TYPE == 'dfn-gcnet' ]; then
  TASK_TYPE='DFN'
  MODEL_NAME='ASN-DFN-GCNet'
	EPO_TEST=20
	

elif [ $SIMPLE_TASK_TYPE == 'pac-dispnetc' ]; then
	TASK_TYPE='PAC'
  MODEL_NAME='ASN-PAC-DispNetC'
  PAC_NATIVE_IMPLE='true'
	EPO_TEST=19
elif [ $SIMPLE_TASK_TYPE == 'pac-psm' ]; then
	TASK_TYPE='PAC'
  MODEL_NAME='ASN-PAC-PSM'
  PAC_NATIVE_IMPLE='true'
	EPO_TEST=31
elif [ $SIMPLE_TASK_TYPE == 'pac-ganet' ]; then
	TASK_TYPE='PAC'
  MODEL_NAME='ASN-PAC-GANet-Deep'
  PAC_NATIVE_IMPLE='false'
	EPO_TEST=20
elif [ $SIMPLE_TASK_TYPE == 'pac-gcnet' ]; then
	TASK_TYPE='PAC'
  MODEL_NAME='ASN-PAC-GCNet'
  PAC_NATIVE_IMPLE='false'
	EPO_TEST=60

elif [ $SIMPLE_TASK_TYPE == 'sga-dispnetc' ]; then
  TASK_TYPE='SGA'
  MODEL_NAME='ASN-SGA-DispNetC'
	EPO_TEST=18
elif [ $SIMPLE_TASK_TYPE == 'sga-psm' ]; then
  TASK_TYPE='SGA'
	MODEL_NAME='ASN-SGA-PSM'
	EPO_TEST=24
elif [ $SIMPLE_TASK_TYPE == 'sga-gcnet' ]; then
  TASK_TYPE='SGA'
  MODEL_NAME='ASN-SGA-GCNet'
	EPO_TEST=15
else
	echo "Wrong SIMPLE_TASK_TYPE=${SIMPLE_TASK_TYPE}!!! Try it again!!"
	exit
fi




#--------------------------#
#-----Task Type Here ------#
#--------------------------#
#TASK_TYPE='EMBED_BILATERAL'
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

SIMPLE_WINOW_SIZE=$2

if [ $TASK_TYPE == 'BASELINE' ]; then
	echo "No filters"
	IS_DFN='false'
	IS_EMBED='false'
	IS_PAC='false'

elif [ $TASK_TYPE == 'EMBED_BILATERAL' ]; then
	echo 'TASK_TYPE : EMBED_BILATERAL !!!'
	IS_EMBED='true'
	#IS_FREEZE_EMBED='true'
	IS_FREEZE_EMBED='false'
	#SIGMA_S=3.0 # window 21 x 21
	#SIGMA_S=2.0 # window 15 x 15
	#SIGMA_S=1.4 # window 11 x 11
	#SIGMA_S=1.0 # window 9 x 9
	#SIGMA_S=0.7 # window 7 x 7
	#SIGMA_S=0.3 # window 3 x 3
	#SIGMA_S=0.5 # window 5 x 5
	if [ "$SIMPLE_WINOW_SIZE" = 3 ]; then
		SIGMA_S=0.3
		echo "Set SIGMA_S=${SIGMA_S}, due to SIMPLE_WINOW_SIZE=3"
	elif [ "$SIMPLE_WINOW_SIZE" = 5 ]; then
		SIGMA_S=0.5
		echo "Set SIGMA_S=${SIGMA_S}, due to SIMPLE_WINOW_SIZE=5"
	elif [ "$SIMPLE_WINOW_SIZE" = 7 ]; then
		SIGMA_S=0.7
		echo "Set SIGMA_S=${SIGMA_S}, due to SIMPLE_WINOW_SIZE=7"
	elif [ "$SIMPLE_WINOW_SIZE" = 9 ]; then
		SIGMA_S=1.0
		echo "Set SIGMA_S=${SIGMA_S}, due to SIMPLE_WINOW_SIZE=9"
	elif [ "$SIMPLE_WINOW_SIZE" = 11 ]; then
		SIGMA_S=1.4
		echo "Set SIGMA_S=${SIGMA_S}, due to SIMPLE_WINOW_SIZE=11"
	else
		echo "Wrong input! SIMPLE_WINOW_SIZE should be 3,5,7,9,11"
		exit
	fi
	
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
#MODEL_NAME='ASN-Embed-DispNetC'
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
echo "MODEL_NAME=${MODEL_NAME}"
echo "MODEL_NAME_STR=${MODEL_NAME_STR}"

if [ $MODEL_NAME == 'ASN-Embed-PSM' ]; then
	let CROP_HEIGHT=256
	let CROP_WIDTH=512
	let MAX_DISP=192
	BATCHSIZE=$3
	#LOG_SUMMARY_STEP=150
	LOG_SUMMARY_STEP=50
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
fi


#EXP_NAME='tmp'
VKT2_ABLATION='kt15-ablation-runtime'
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
	pwd
	MODE='train'
	#MODE='debug'
  RESULTDIR="${MY_PROJECT_ROOT}/results/${EXP_NAME}"
	#CUDA_VISIBLE_DEVICES=0 python3.7 -m main_attenStereoNet \
	CUDA_VISIBLE_DEVICES=$4 python3.7 -m main_attenStereoNet \
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
flag=false
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
  
	declare -a ALL_EPOS_TEST=(20 40 60 80 100 120 140 160 180 200 220 240 260 280 300)	
	declare -a ALL_EPOS_TEST=(10 20 30 40 50 60 70 80 90 100)	
	for idx in $(seq 0 9)
	do # epoch model loop
		EPO_TEST=${ALL_EPOS_TEST[idx]}
		echo $EPO_TEST
    
	  # for baselines, we always use (dummy) Embed + Baseline, but disable Embed !!!
		if [ $MODEL_NAME == 'PSMNet' ]; then
			MODEL_NAME='ASN-Embed-PSM'
			echo "test baseline: PSMNet, via Dummy Embed + PSMNet !!!"
			TMP_MODEL_NAME="psmnet-D192-sfepo10-vkt2epo20"
			RESUME="./checkpoints/${VKT2_ABLATION}/${TMP_MODEL_NAME}/model_epoch_$(printf "%05d" "$EPO_TEST").tar"
			EXP_NAME="${TMP_MODEL_NAME}/disp-epo-$(printf "%03d" "$EPO_TEST")"
			#already contained in the RESUME:
			RESUME_EMBEDNET=''

	  # for baselines, we always use (dummy) Embed + Baseline, but disable Embed !!!
	  elif [ $MODEL_NAME == 'DispNetC' ]; then
			MODEL_NAME='ASN-Embed-DispNetC'
			echo "test baseline: DispNetC, via Dummy Embed + DispNetC !!!"
			TMP_MODEL_NAME="dispnetV4-D192-BN-corrV1-sfepo20-vkt2epo20"
			RESUME="./checkpoints/${VKT2_ABLATION}/${TMP_MODEL_NAME}/DispNetC/model_epoch_$(printf "%05d" "$EPO_TEST").tar"
			EXP_NAME="${TMP_MODEL_NAME}/disp-epo-$(printf "%03d" "$EPO_TEST")"
			#already contained in the RESUME:
			RESUME_EMBEDNET=''
		
	  # for baselines, we always use (dummy) Embed + Baseline, but disable Embed !!!
	  elif [ $MODEL_NAME == 'GANet-Deep' ]; then
			MODEL_NAME='ASN-Embed-GANet-Deep'
			echo "test baseline: GANet-Deep, via Dummy Embed + GANet-Deep !!!"
			TMP_MODEL_NAME="ganet-deep-D192-sfepo10-vkt2epo20"
			RESUME="./checkpoints/${VKT2_ABLATION}/${TMP_MODEL_NAME}/model_epoch_$(printf "%05d" "$EPO_TEST").tar"
			EXP_NAME="${TMP_MODEL_NAME}/disp-epo-$(printf "%03d" "$EPO_TEST")"
			#already contained in the RESUME:
			RESUME_EMBEDNET=''
	  
		# for baselines, we always use (dummy) Embed + Baseline, but disable Embed !!!
	  elif [ $MODEL_NAME == 'GCNet' ]; then
			MODEL_NAME='ASN-Embed-GCNet'
			echo "test baseline: GCNet, via Dummy Embed + GCNet !!!"
			TMP_MODEL_NAME="gcnetAKQ-D192-sfepo10-vkt2epo20"
			RESUME="./checkpoints/${VKT2_ABLATION}/${TMP_MODEL_NAME}/GCNet/model_epoch_$(printf "%05d" "$EPO_TEST").tar"
			EXP_NAME="${TMP_MODEL_NAME}/disp-epo-$(printf "%03d" "$EPO_TEST")"
			#already contained in the RESUME:
			RESUME_EMBEDNET=''

		elif [ $MODEL_NAME == 'ASN-Embed-GANet-Deep' ]; then
		#----------------------
		#  Method 2) : GANet + Embednet (training together)
		#----------------------
			echo "test Method 2: GANet + Embednet !!!"
			TMP_MODEL_NAME="asn-embed-k5-d2-D192-ganetdeep-sfepo10-vkt2epo20-embedlossW-0.06-lr-0.001-p-eposteps-5-18"
			RESUME="./checkpoints/${VKT2_ABLATION}/${TMP_MODEL_NAME}/ASN-Embed-GANet-Deep/model_epoch_$(printf "%05d" "$EPO_TEST").tar"
			EXP_NAME="${TMP_MODEL_NAME}/disp-epo-$(printf "%03d" "$EPO_TEST")"
			##already contained in the RESUME:
			RESUME_EMBEDNET=''
		
		#----------------------
		#  Method 6) : DispNetC + Embed-Bilateral
		#----------------------
		elif [ $MODEL_NAME == 'ASN-Embed-DispNetC' ]; then
			echo "test Method 6: Embed+ DispNetC !!!"
			TMP_MODEL_NAME="asn-embed-k5-d2-D192-dispnetc-sfepo20-vkt2epo20-embedlossW-0.06"
			RESUME="./checkpoints/${VKT2_ABLATION}/${TMP_MODEL_NAME}/ASN-Embed-DispNetC/model_epoch_$(printf "%05d" "$EPO_TEST").tar"
			EXP_NAME="${TMP_MODEL_NAME}/disp-epo-$(printf "%03d" "$EPO_TEST")"
			#already contained in the RESUME:
			RESUME_EMBEDNET=''
		
		#----------------------
		#  Method 3) : PSMNet + Embednet (training together)
		#----------------------
		elif [ $MODEL_NAME == 'ASN-Embed-PSM' ]; then
			echo "test Method 3: PSMNet + Embednet !!!"
			#TMP_MODEL_NAME="asn-embed-k5-d2-D192-psm-sfepo10-vkt2epo30-embedlossW-0.06-lr-0.001-e-epothrd-22"
			#TMP_MODEL_NAME="asn-embed-k${K_WIDTH}-d${DILATION}-D${MAX_DISP}-psm-sfepo10-${KT_STR}epo300-embedlossW-0.06-lr-0.001-p-eposteps-200"
			TMP_MODEL_NAME="asn-embed-k${K_WIDTH}-d${DILATION}-D${MAX_DISP}-psm-sfepo10-${KT_STR}epo100-embedlossW-0.06-lr-0.001-p-eposteps-80"
			RESUME="./checkpoints/${VKT2_ABLATION}/${TMP_MODEL_NAME}/ASN-Embed-PSM/model_epoch_$(printf "%05d" "$EPO_TEST").tar"
			EXP_NAME="${TMP_MODEL_NAME}/disp-epo-$(printf "%03d" "$EPO_TEST")"
			##already contained in the RESUME:
			RESUME_EMBEDNET=''
		
		#---------------------------
		#  Method 16) : EBF + GCNet
		#---------------------------
		elif [ $MODEL_NAME == 'ASN-Embed-GCNet' ]; then
			echo "test Method 16: EBF + GCNet !!!"
			TMP_MODEL_NAME="asn-embed-k5-d2-D192-gcnetAKQ-sfepo10-vkt2epo20-embedlossW-0.06-lr-0.001-e-epothrd-2"
			RESUME="./checkpoints/${VKT2_ABLATION}/${TMP_MODEL_NAME}/ASN-Embed-GCNet/model_epoch_$(printf "%05d" "$EPO_TEST").tar"
			EXP_NAME="${TMP_MODEL_NAME}/disp-epo-$(printf "%03d" "$EPO_TEST")"
			#already contained in the RESUME:
			RESUME_EMBEDNET=''

		#----------------------
		#  Method 4) : PSMNet + DFN(Dynamic Filter Network)
		#----------------------
		elif [ $MODEL_NAME == 'ASN-DFN-PSM' ]; then
			echo "test Method 4: PSMNet + DFN !!!"
			TMP_MODEL_NAME="asn-dfn-k5-d2-D192-psm-sfepo10-vkt2epo20-woEmbed-lr-0.001-e-epothrd-2"
			RESUME="./checkpoints/${VKT2_ABLATION}/${TMP_MODEL_NAME}/ASN-DFN-PSM/model_epoch_$(printf "%05d" "$EPO_TEST").tar"
			EXP_NAME="${TMP_MODEL_NAME}/disp-epo-$(printf "%03d" "$EPO_TEST")"
			#already contained in the RESUME:
			RESUME_EMBEDNET=''
		
		#----------------------
		#  Method 5) : PSMNet + PAC(Pixel-adaptive Convolution Network)
		#----------------------
		elif [ $MODEL_NAME == 'ASN-PAC-PSM' ]; then
			echo "test Method 5: PSMNet + PAC !!!"
			TMP_MODEL_NAME="asn-npac-k5-d2-D192-psm-sfepo10-vkt2epo20-woEmbed-lr-0.001-e-epothrd-2"
			BATCH_IN_IMAGE='true'
			RESUME="./checkpoints/${VKT2_ABLATION}/${TMP_MODEL_NAME}/ASN-PAC-PSM/model_epoch_$(printf "%05d" "$EPO_TEST").tar"
			EXP_NAME="${TMP_MODEL_NAME}/disp-epo-$(printf "%03d" "$EPO_TEST")"
			#already contained in the RESUME:
			RESUME_EMBEDNET=''
		#----------------------
		#  Method 7) : DispNetC + DFN
		#----------------------
		elif [ $MODEL_NAME == 'ASN-DFN-DispNetC' ]; then
			echo "test Method 7: DFN + DispNetC !!!"
			TMP_MODEL_NAME="asn-dfn-k5-d2-D192-dispnetc-sfepo20-vkt2epo20-woEmbed"
			RESUME="./checkpoints/${VKT2_ABLATION}/${TMP_MODEL_NAME}/ASN-DFN-DispNetC/model_epoch_$(printf "%05d" "$EPO_TEST").tar"
			EXP_NAME="${TMP_MODEL_NAME}/disp-epo-$(printf "%03d" "$EPO_TEST")"
			#already contained in the RESUME:
			RESUME_EMBEDNET=''
		#----------------------
		#  Method 8) : DispNetC + PAC
		#----------------------
		elif [ $MODEL_NAME == 'ASN-PAC-DispNetC' ]; then
			echo "test Method 8: PAC + DispNetC !!!"
			TMP_MODEL_NAME="asn-npac-k5-d2-D192-dispnetc-sfepo20-vkt2epo20-woEmbed"
			RESUME="./checkpoints/${VKT2_ABLATION}/${TMP_MODEL_NAME}/ASN-PAC-DispNetC/model_epoch_$(printf "%05d" "$EPO_TEST").tar"
			EXP_NAME="${TMP_MODEL_NAME}/disp-epo-$(printf "%03d" "$EPO_TEST")"
			#already contained in the RESUME:
			RESUME_EMBEDNET=''

		#----------------------
		#  Method 9) : SGA + PSM
		#----------------------
		elif [ $MODEL_NAME == 'ASN-SGA-PSM' ]; then
			echo "test Method 9: SGA + PSMNet !!!"
			TMP_MODEL_NAME="asn-sga-k0-d2-D192-psm-sfepo10-vkt2epo20-woEmbed-lr-0.001-e-epothrd-2"
			RESUME="./checkpoints/${VKT2_ABLATION}/${TMP_MODEL_NAME}/ASN-SGA-PSM/model_epoch_$(printf "%05d" "$EPO_TEST").tar"
			EXP_NAME="${TMP_MODEL_NAME}/disp-epo-$(printf "%03d" "$EPO_TEST")"
			#already contained in the RESUME:
			RESUME_EMBEDNET=''

		#----------------------
		#  Method 10) : SGA + DispNetC
		#----------------------
		elif [ $MODEL_NAME == 'ASN-SGA-DispNetC' ]; then
			echo "test Method 10: SGA + DispNetC !!!"
			TMP_MODEL_NAME="asn-sga-k0-d2-D192-dispnetc-sfepo20-vkt2epo20-woEmbed"
			RESUME="./checkpoints/${VKT2_ABLATION}/${TMP_MODEL_NAME}/ASN-SGA-DispNetC/model_epoch_$(printf "%05d" "$EPO_TEST").tar"
			EXP_NAME="${TMP_MODEL_NAME}/disp-epo-$(printf "%03d" "$EPO_TEST")"
			#already contained in the RESUME:
			RESUME_EMBEDNET=''
		
    #----------------------
		#  Method 11) : DFN + GANet-Deep
		#----------------------
		elif [ $MODEL_NAME == 'ASN-DFN-GANet-Deep' ]; then
			echo "test Method 11: DFN + GANet-Deep !!!"
			TMP_MODEL_NAME="asn-dfn-k5-d2-D${MAX_DISP}-ganetdeep-sfepo10-vkt2epo20-woEmbed-lr-0.001-p-eposteps-5-18"
			RESUME="./checkpoints/${VKT2_ABLATION}/${TMP_MODEL_NAME}/ASN-DFN-GANet-Deep/model_epoch_$(printf "%05d" "$EPO_TEST").tar"
			EXP_NAME="${TMP_MODEL_NAME}/disp-epo-$(printf "%03d" "$EPO_TEST")"
			#already contained in the RESUME:
			RESUME_EMBEDNET=''
    
		#----------------------
		#  Method 12) : DFN + GCNet
		#----------------------
		elif [ $MODEL_NAME == 'ASN-DFN-GCNet' ]; then
			echo "test Method 12: DFN + GCNet !!!"
			TMP_MODEL_NAME="asn-dfn-k5-d2-D192-gcnetAKQ-sfepo10-vkt2epo20-woEmbed-lr-0.001-e-epothrd-2"
			RESUME="./checkpoints/${VKT2_ABLATION}/${TMP_MODEL_NAME}/ASN-DFN-GCNet/model_epoch_$(printf "%05d" "$EPO_TEST").tar"
			EXP_NAME="${TMP_MODEL_NAME}/disp-epo-$(printf "%03d" "$EPO_TEST")"
			#already contained in the RESUME:
			RESUME_EMBEDNET=''
		
		#----------------------
		#  Method 13) : PAC + GANet-Deep
		#----------------------
		elif [ $MODEL_NAME == 'ASN-PAC-GANet-Deep' ]; then
			echo "test Method 13: PAC + GANet-Deep !!!"
			TMP_MODEL_NAME="asn-pac-k5-d2-D192-ganetdeep-sfepo10-vkt2epo20-woEmbed-lr-0.001-p-eposteps-5-18"
			RESUME="./checkpoints/${VKT2_ABLATION}/${TMP_MODEL_NAME}/ASN-PAC-GANet-Deep/model_epoch_$(printf "%05d" "$EPO_TEST").tar"
			EXP_NAME="${TMP_MODEL_NAME}/disp-epo-$(printf "%03d" "$EPO_TEST")"
			#already contained in the RESUME:
			RESUME_EMBEDNET=''
		
		#----------------------
		#  Method 14) : SGA + GCNet
		#----------------------
		elif [ $MODEL_NAME == 'ASN-SGA-GCNet' ]; then
			echo "test Method 14: SGA + GCNet !!!"
			TMP_MODEL_NAME="asn-sga-k0-d2-D192-gcnetAKQ-sfepo10-vkt2epo20-woEmbed-lr-0.001-p-eposteps-5-18"
			RESUME="./checkpoints/${VKT2_ABLATION}/${TMP_MODEL_NAME}/ASN-SGA-GCNet/model_epoch_$(printf "%05d" "$EPO_TEST").tar"
			EXP_NAME="${TMP_MODEL_NAME}/disp-epo-$(printf "%03d" "$EPO_TEST")"
			#already contained in the RESUME:
			RESUME_EMBEDNET=''
		
		#----------------------
		#  Method 15) : PAC + GCNet
		#----------------------
		elif [ $MODEL_NAME == 'ASN-PAC-GCNet' ]; then
			echo "test Method 15: PAC + GCNet !!!"
			TMP_MODEL_NAME="asn-pac-k5-d2-D192-gcnetAKQ-sfepo10-vkt2epo30-woEmbed-lr-0.001-e-epothrd-2"
			RESUME="./checkpoints/${VKT2_ABLATION}/${TMP_MODEL_NAME}/ASN-PAC-GCNet/model_epoch_$(printf "%05d" "$EPO_TEST").tar"
			EXP_NAME="${TMP_MODEL_NAME}/disp-epo-$(printf "%03d" "$EPO_TEST")"
			#already contained in the RESUME:
			RESUME_EMBEDNET=''

		else 
			echo "You have to specify a argument to bash!!!"
			exit
    fi
		
		RESULTDIR="./results/${VKT2_ABLATION}/${EXP_NAME}"
		#cd /home/${USER}/atten-stereo
	  cd ${MY_PROJECT_ROOT}
		CUDA_VISIBLE_DEVICES=$4 python3.7 -m main_attenStereoNet \
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

#############################
# Netwrok Inference Runtime 
#############################
flag=false
#flag=true
if [ "$flag" = true ]; then
	MODE='inferencetime'
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
	
	#----------------------
	#  Method 6) : Embed + PSMNet
	#----------------------
	echo "runtime Method 6: Embed+ PSMNet !!!"
	EPO_TEST=100
	#TMP_MODEL_NAME="asn-embed-k${K_WIDTH}-d${DILATION}-D${MAX_DISP}-psm-sfepo10-${KT_STR}epo300-embedlossW-0.06-lr-0.001-p-eposteps-200"
	TMP_MODEL_NAME="asn-embed-k${K_WIDTH}-d${DILATION}-D${MAX_DISP}-psm-sfepo10-${KT_STR}epo100-embedlossW-0.06-lr-0.001-p-eposteps-80"
	#TMP_MODEL_NAME="asn-embed-k${K_WIDTH}-d${DILATION}-D${MAX_DISP}-psm-sfepo10-${KT_STR}epo200-embedlossW-0.06-lr-0.001-p-eposteps-150"
	RESUME="./checkpoints/kt15-ablation/${TMP_MODEL_NAME}/ASN-Embed-PSM/model_epoch_$(printf "%05d" "$EPO_TEST").tar"
	#already contained in the RESUME:
	RESUME_EMBEDNET=''	
	RESULTDIR="${MY_PROJECT_ROOT}/results/tmp-runtime"
  CHECKPOINT_DIR="./checkpoints/tmp-runtime"
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

fi # end of Netwrok Inference Runtime
