#ccj's experiments

#t=14400
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


DATA_ROOT="/media/ccjData2"
if [ ! -d $DATA_ROOT ];then
	DATA_ROOT="/data/ccjData"
	echo "Updated : setting data_root = ${DATA_ROOT}"
fi

SCENE_FLOW_DATA="$DATA_ROOT/datasets/SceneFlowDataset/"
MODEL_RESUME='/home/ccj/atten-stereo/checkpoints/saved/psmnet-pretrained/pretrained_sceneflow.tar'
#MODEL_RESUME=''
EXP_NAME="psm-tmp"
SAVE_MODEL_PATH="/home/ccj/atten-stereo/checkpoints/$EXP_NAME"
BATCH_SIZE=3
FRACTION=0.01
#FRACTION=1.0
MODEL_NAME_STR='psmnet'
MAX_DISP=192

#-------------------------
#--- Train ---------------
#-------------------------
flag=false
#flag=true
if [ "$flag" = true ]; then
	NUM_EPOCHS=10
	DATA_TYPE_STR='sf'
	#EXP_NAME="psm-tmp"
  EXP_NAME="${MODEL_NAME_STR}-D${MAX_DISP}-${DATA_TYPE_STR}epo${NUM_EPOCHS}" 
	SAVE_MODEL_PATH="/home/ccj/atten-stereo/checkpoints/$EXP_NAME"
  makeDir "$SAVE_MODEL_PATH"
	CUDA_VISIBLE_DEVICES=0 python3.7 main.py \
		--maxdisp=${MAX_DISP} \
		--model=stackhourglass \
		--datapath=$SCENE_FLOW_DATA \
		--epochs=${NUM_EPOCHS} \
		--loadmodel=$MODEL_RESUME \
		--batch_size=$BATCH_SIZE \
		--savemodel=$SAVE_MODEL_PATH \
		--fraction=$FRACTION
	exit
fi

#-------------------------
#--- Finetune ---------------
#-------------------------
NUM_EPOCHS=400
NUM_EPOCHS=20
#flag=true
flag=false
#DATA_TYPE='2015'
#DATA_TYPE='2012'
DATA_TYPE='virtual_kt_2'

KT15_TRAIN_LIST="/home/ccj/atten-stereo/lists/kitti2015_train170.list"
KT15_VAL_LIST="/home/ccj/atten-stereo/lists/kitti2015_val30.list"

KT12_TRAIN_LIST="/home/ccj/atten-stereo/lists/kitti2012_train164.list"
KT12_VAL_LIST="/home/ccj/atten-stereo/lists/kitti2012_val30.list"

VKT2_TRAIN_LIST="/home/ccj/atten-stereo/lists/virtual_kitti2_wo_scene06_fixed_train.list"
VKT2_VAL_LIST="/home/ccj/atten-stereo/lists/virtual_kitti2_wo_scene06_fixed_test.list"
#VKT2_TRAIN_LIST="/home/ccj/atten-stereo/lists/virtual_kitti2_wo_scene06_fixed_train_small.list"
#VKT2_VAL_LIST="/home/ccj/atten-stereo/lists/virtual_kitti2_wo_scene06_fixed_test_small.list"

if [ "$DATA_TYPE" = '2015' ]; then
	DATA_PATH="${DATA_ROOT}/datasets/KITTI-2015/training/"
	DATA_TYPE_STR='kt15'
elif [ "$DATA_TYPE" = '2012' ]; then
	DATA_PATH="${DATA_ROOT}/datasets/KITTI-2012/training/"
	DATA_TYPE_STR='kt12'
elif [ "$DATA_TYPE" = 'virtual_kt_2' ]; then
	DATA_PATH="${DATA_ROOT}/datasets/Virtual-KITTI-V2/"
	DATA_TYPE_STR='vkt2'
fi
TMP='/home/ccj/atten-stereo/checkpoints/saved/psmnet-pretrained'
MODEL_RESUME="$TMP/pretrained_sceneflow.tar"
#MODEL_RESUME="$TMP/pretrained_model_KITTI2012.tar"
#MODEL_RESUME="$TMP/pretrained_model_KITTI2015.tar"
if [ "$flag" = true ]; then
	#EXP_NAME="psm-sfepo10-kt15epo400_try2"
  EXP_NAME="${MODEL_NAME_STR}-D${MAX_DISP}-sfepo10-${DATA_TYPE_STR}epo${NUM_EPOCHS}"
  TRAIN_LOGDIR="/home/ccj/atten-stereo/logs/${EXP_NAME}"
  BATCH_SIZE=6
  LOG_SUMMARY_STEP=40
  #BATCH_SIZE=1
  #LOG_SUMMARY_STEP=200
	SAVE_MODEL_PATH="/home/ccj/atten-stereo/checkpoints/$EXP_NAME"
  makeDir "$SAVE_MODEL_PATH"
	echo "SAVE_MODEL_PATH=$SAVE_MODEL_PATH"
	#exit
	CUDA_VISIBLE_DEVICES=0 python3.7 finetune.py \
		--maxdisp=${MAX_DISP} \
		--model="stackhourglass" \
		--datatype=$DATA_TYPE \
		--datapath=$DATA_PATH \
		--epochs=${NUM_EPOCHS} \
		--loadmodel=$MODEL_RESUME \
		--batch_size=$BATCH_SIZE \
		--savemodel=$SAVE_MODEL_PATH \
		--kt15_train_list=${KT15_TRAIN_LIST} \
		--kt15_val_list=${KT15_VAL_LIST} \
		--kt12_train_list=${KT12_TRAIN_LIST} \
		--kt12_val_list=${KT12_VAL_LIST} \
		--vkt2_train_list=${VKT2_TRAIN_LIST} \
		--vkt2_val_list=${VKT2_VAL_LIST} \
		--log_summary_step=${LOG_SUMMARY_STEP} \
		--train_logdir=$TRAIN_LOGDIR
	exit
fi



#-------------------------------------
# -- Added by CCJ --------------------
#--- KT15 Val-30 validation ----------
#-------------------------------------
flag=false
#flag=true
#DATA_TYPE='2015'
DATA_TYPE='2012'
if [ "$DATA_TYPE" = '2015' ]; then
	DATA_PATH="${DATA_ROOT}/datasets/KITTI-2015/training/"
	DATA_TYPE_STR='kt15'
	VAL_LIST="/home/ccj/atten-stereo/lists/kitti2015_val30.list"
elif [ "$DATA_TYPE" = '2012' ]; then
	DATA_PATH="${DATA_ROOT}/datasets/KITTI-2012/training/"
	DATA_TYPE_STR='kt12'
	VAL_LIST="/home/ccj/atten-stereo/lists/kitti2012_val30.list"
elif [ "$DATA_TYPE" = 'virtual_kt_2' ]; then
	DATA_PATH="${DATA_ROOT}/datasets/Virtual-KITTI-V2/"
	DATA_TYPE_STR='vkt2'
  VAL_LIST="/home/ccj/atten-stereo/lists/virtual_kitti2_wo_scene06_fixed_test.list"
fi

if [ "$flag" = true ]; then
	TMP='/home/ccj/atten-stereo/checkpoints/saved/'
  EXP_NAME="${MODEL_NAME_STR}-D${MAX_DISP}-sfepo10-${DATA_TYPE_STR}epo${NUM_EPOCHS}"
	declare -a ALL_EPOS_TEST=(25 50 75 100 125 150 175 200 225 250 275 300 325 350 375 400)
	#for idx in $(seq 1 15)
	#for idx in $(seq 15 15)
	for idx in $(seq 1 15)
  do # epoch model loop
		EPO_TEST=${ALL_EPOS_TEST[idx]}
		echo "Loading model at epoch $EPO_TEST"
	  MODEL_RESUME="$TMP/${EXP_NAME}/model_epoch_$(printf "%05d" "$EPO_TEST").tar"
		RESULTDIR="/home/ccj/atten-stereo/results/${EXP_NAME}/disp-epo-$(printf "%03d" "$EPO_TEST")"
    makeDir "$RESULTDIR"
	  cd /home/ccj/atten-stereo/src/baselines/PSMNet
		CUDA_VISIBLE_DEVICES=0 python3.7 submission.py \
			--result_dir $RESULTDIR \
			--maxdisp ${MAX_DISP} \
			--model stackhourglass \
			--KITTI ${DATA_TYPE} \
			--datapath $DATA_PATH \
			--loadmodel $MODEL_RESUME \
			--file_txt_path $VAL_LIST
		# move pfm files to val-30 subdir
		makeDir "$RESULTDIR/val-30"
		cd $RESULTDIR
		for i in *.pfm; do
			mv -i -- "$i" "./val-30"
			#echo "$i"
		done
	done
fi
#exit

#-----------------------------------------
#--- Evaluation of Virtual KT 2 ----------
#-----------------------------------------
#flag=false
flag=true

if [ "$flag" = true ]; then
  DATA_TYPE='virtual_kt_2'
	if [ "$DATA_TYPE" = 'virtual_kt_2' ]; then
		DATA_PATH="${DATA_ROOT}/datasets/Virtual-KITTI-V2/"
		DATA_TYPE_STR='vkt2'
		VAL_LIST="/home/ccj/atten-stereo/lists/virtual_kitti2_wo_scene06_fixed_test.list"
		VAL_LIST="/home/ccj/atten-stereo/lists/virtual_kitti2_wo_scene06_fixed_test_small.list"
	  EXP_NAME="psmnet-D192-sfepo10-${DATA_TYPE_STR}epo20"
	  MODEL_RESUME="/home/ccj/atten-stereo/checkpoints/saved/psmnet-D192-sfepo10-vkt2epo20/best_model_epoch_00019_valerr_1.9549.tar"
	fi

	declare -a ALL_EPOS_TEST=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20)
	for idx in $(seq 0 0)
  #for idx in $(seq 0 19)
	do # epoch model loop
		EPO_TEST=${ALL_EPOS_TEST[idx]}
	  TMP='/home/ccj/atten-stereo/checkpoints'
		echo "Loading model at epoch $EPO_TEST"
	  #MODEL_RESUME="$TMP/saved/${EXP_NAME}/model_epoch_$(printf "%05d" "$EPO_TEST").tar"
	  MODEL_RESUME="$TMP/${EXP_NAME}/model_epoch_$(printf "%05d" "$EPO_TEST").tar"
		RESULTDIR="/home/ccj/atten-stereo/results/${EXP_NAME}/disp-epo-$(printf "%03d" "$EPO_TEST")"
    makeDir "$RESULTDIR"
	  cd /home/ccj/atten-stereo/src/baselines/PSMNet
		CUDA_VISIBLE_DEVICES=0 python3.7 submission.py \
			--maxdisp 192 \
			--model stackhourglass \
			--KITTI ${DATA_TYPE} \
			--datapath $DATA_PATH \
			--loadmodel $MODEL_RESUME \
			--result_dir $RESULTDIR \
			--file_txt_path $VAL_LIST

		
		# move pfm files to val-30 subdir
		makeDir "$RESULTDIR/disp-pfm"
		cd $RESULTDIR
		for i in *.pfm; do
			mv -i -- "$i" "./disp-pfm"
			#echo "$i"
		done
		if [ -f "./bad-err.csv" ]; then
			cat "./bad-err.csv" >> "/home/ccj/atten-stereo/results/bad-err-evalu.csv"
		fi
  done
	exit
fi

#-------------------------
#--- Submission ----------
#-------------------------
flag=false
#flag=true
if [ "$flag" = true ]; then
	DATA_TYPE='2015'
	#DATA_TYPE='2012'
  #DATA_TYPE='virtual_kt_2'
	TMP='/home/ccj/atten-stereo/checkpoints/saved/psmnet-pretrained'
	if [ "$DATA_TYPE" = '2015' ]; then
		DATA_PATH="${DATA_ROOT}/datasets/KITTI-2015/testing/"
		DATA_TYPE_STR='kt15'
	  MODEL_RESUME="$TMP/pretrained_model_KITTI2015.tar"
	  EXP_NAME="psm-offical-test-${DATA_TYPE_STR}"
	
	elif [ "$DATA_TYPE" = '2012' ]; then
		DATA_PATH="${DATA_ROOT}/datasets/KITTI-2012/testing/"
		DATA_TYPE_STR='kt12'
	  MODEL_RESUME="$TMP/pretrained_model_KITTI2012.tar"
	  EXP_NAME="psm-offical-test-${DATA_TYPE_STR}"
	fi
	
	#TMP='/home/ccj/atten-stereo/checkpoints/saved/psmnet-pretrained'
	#MODEL_RESUME="$TMP/pretrained_sceneflow.tar"
	#MODEL_RESUME="$TMP/pretrained_model_KITTI2012.tar"
	RESULTDIR="/home/ccj/atten-stereo/results/${EXP_NAME}"
	CUDA_VISIBLE_DEVICES=0 python3.7 submission.py \
		--maxdisp 192 \
		--model stackhourglass \
		--KITTI ${DATA_TYPE} \
		--datapath $DATA_PATH \
		--loadmodel $MODEL_RESUME \
		--result_dir $RESULTDIR
fi



#---------------------------------------
#--- Test You Own Stereo Pair ----------
#---------------------------------------
flag=false
#flag=true
if [ "$flag" = true ]; then
	TMP='/home/ccj/atten-stereo/checkpoints/saved/psmnet-pretrained'
	#MODEL_RESUME="$TMP/pretrained_sceneflow.tar"
	#MODEL_RESUME="$TMP/pretrained_model_KITTI2012.tar"
	MODEL_RESUME="$TMP/pretrained_model_KITTI2015.tar"
	LIMG='./dataset/2011_10_03_drive_0034_sync/left/0000000001.png'
	RIMG='./dataset/2011_10_03_drive_0034_sync/right/0000000001.png'
	IS_GRAY_IMG=false
	CUDA_VISIBLE_DEVICES=0 python3.7 test_img.py \
		--loadmodel=$MODEL_RESUME \
		--leftimg=$LIMG \
		--rightimg=$RIMG \
		--isgray=${IS_GRAY_IMG}
fi
