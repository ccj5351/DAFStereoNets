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

KT2012=0
KT2015=0
VIRTUAL_KITTI2=0
#IS_KT12_GRAY='false'
#IS_KT12_GRAY='true'
#KT12_IMAGE_MODE='gray'
#KT12_IMAGE_MODE='gray2rgb'
KT12_IMAGE_MODE='rgb'

IS_DATA_AUGMENT="false"
#IS_DATA_AUGMENT="true"

START_EPOCH=0
#NUM_EPOCHS=20
#NUM_EPOCHS=10
NUM_WORKERS=12
BATCHSIZE=16
#BATCHSIZE=40
#BATCHSIZE=1
LOG_SUMMARY_STEP=9
#LOG_SUMMARY_STEP=50

IS_BN='true'
#IS_BN='false'
CORR_FUNC_STR='corrV1'
#CORR_FUNC_STR='corrV2'
if [ "$CORR_FUNC_STR" = corrV1 ]; then
	CORR_FUNC='correlation1D_map_V1'
elif [ "$CORR_FUNC_STR" = corrV2 ]; then
	CORR_FUNC='corr1D_v2'
fi

if [ "$IS_BN" = true ]; then
	BN_STR='BN'
else
	BN_STR='noBN'
fi

if [ $KT2012 -eq 1 ]; then
	DATA_PATH="${DATA_ROOT}/datasets/KITTI-2012/training/"
	KT_STR='kt12'
	if [ "$IS_KT12_GRAY" = true ]; then
		KT_STR="${KT_STR}gray"
	fi
	if [ "$IS_DATA_AUGMENT" = true ]; then
		KT_STR="${KT_STR}Aug"
	fi
  TRAINING_LIST="lists/kitti2012_train164.list"
  TEST_LIST="lists/kitti2012_val30.list"
	#revise parameter settings and run "train.sh" and "predict.sh" for 
	#training, finetuning and prediction/testing. 
	#Note that the “crop_width” and “crop_height” must be multiple of 64, 
	#"max_disp" must be multiple of 4 (default: 192).
	let CROP_HEIGHT=320
	let CROP_WIDTH=768+64
	let MAX_DISP=192
  NUM_EPOCHS=800
  BATCHSIZE=16
  LOG_SUMMARY_STEP=9
  EXP_NAME="dispnetV4-D${MAX_DISP}-${BN_STR}-${CORR_FUNC_STR}-sfepo20-${KT_STR}epo${NUM_EPOCHS}"
	RESUME='./checkpoints/saved/dispnetV4-D192-BN-corrV1-sfepo20/DispNetC/model_epoch_00020.tar'

elif [ $KT2015 -eq 1 ]; then
	DATA_PATH="${DATA_ROOT}/datasets/KITTI-2015/training/"
	KT_STR='kt15'
	if [ "$IS_DATA_AUGMENT" = true ]; then
		KT_STR="${KT_STR}Aug"
	fi
	TRAINING_LIST="lists/kitti2015_train170.list"
  TEST_LIST="lists/kitti2015_val30.list"
	let CROP_HEIGHT=320
	let CROP_WIDTH=768+64
	let MAX_DISP=192
	#let MAX_DISP=180
  BATCHSIZE=16
  LOG_SUMMARY_STEP=4
  NUM_EPOCHS=400
  EXP_NAME="dispnetV4-D${MAX_DISP}-${BN_STR}-${CORR_FUNC_STR}-sfepo20-${KT_STR}epo${NUM_EPOCHS}"
	RESUME='./checkpoints/saved/dispnetV4-D192-BN-corrV1-sfepo20/DispNetC/model_epoch_00020.tar'

elif [ $VIRTUAL_KITTI2 -eq 1 ]; then
	DATA_PATH="${DATA_ROOT}/datasets/Virtual-KITTI-V2/"
	KT_STR='vkt2'
	if [ "$IS_DATA_AUGMENT" = true ]; then
		KT_STR="${KT_STR}Aug"
	fi
	# no shuffle
	TRAINING_LIST="lists/virtual_kitti2_wo_scene06_fixed_train.list"
	TEST_LIST="lists/virtual_kitti2_wo_scene06_fixed_test.list"
	#TEST_LIST="lists/virtual_kitti2_wo_scene06_fixed_test_small.list"
	# with shuffle
	#TRAINING_LIST="lists/virtual_kitti2_wo_scene06_random_train.list"
  #TEST_LIST="lists/virtual_kitti2_wo_scene06_random_test.list"
	#let CROP_HEIGHT=240-96-48
	#let CROP_WIDTH=576-96*2
	let CROP_HEIGHT=256
	let CROP_WIDTH=512
	let MAX_DISP=192
  BATCHSIZE=16
  LOG_SUMMARY_STEP=25
  NUM_EPOCHS=20
  EXP_NAME="dispnetV4-D${MAX_DISP}-${BN_STR}-${CORR_FUNC_STR}-sfepo20-${KT_STR}epo${NUM_EPOCHS}"
	RESUME='./checkpoints/saved/dispnetV4-D192-BN-corrV1-sfepo20/DispNetC/model_epoch_00020.tar'


else
	DATA_PATH="${DATA_ROOT}/datasets/SceneFlowDataset/"
  TRAINING_LIST="lists/sceneflow_train.list"
  #TRAINING_LIST="lists/sceneflow_train_small.list"
  TEST_LIST="lists/sceneflow_test_select.list"
	let CROP_HEIGHT=384-64
	let CROP_WIDTH=768-64
	let MAX_DISP=192
	SF_STR='sf'
	if [ "$IS_DATA_AUGMENT" = true ]; then
		SF_STR="${SF_STR}Aug"
	fi
  BATCHSIZE=16
  LOG_SUMMARY_STEP=50
  NUM_EPOCHS=20
  EXP_NAME="dispnetV4-D${MAX_DISP}-${BN_STR}-${CORR_FUNC_STR}-${SF_STR}epo${NUM_EPOCHS}"
	RESUME=''
fi
echo "DATA_PATH=$DATA_PATH"


#############################
#### TASK TYPES #############
#TASK_TYPE='train'
#TASK_TYPE='val-30'
TASK_TYPE='cross-val'
#############################

#EXP_NAME="dispnet-sf-small-tmp"
#EXP_NAME="dispnetV2-D${MAX_DISP}-${BN_STR}-${CORR_FUNC_STR}-sfepo20-kt15epo${NUM_EPOCHS}/"

TRAIN_LOGDIR="./logs/${EXP_NAME}"
CHECKPOINT_DIR="./checkpoints/${EXP_NAME}"
echo "EXP_NAME=$EXP_NAME"
echo "TRAIN_LOGDIR=$TRAIN_LOGDIR"
echo "CHECKPOINT_DIR=$CHECKPOINT_DIR"

#exit

cd /home/ccj/atten-stereo
################################
# Netwrok Training & profiling
################################
if [ "$TASK_TYPE" = 'train' ]; then
	flag=true
else
	flag=false
fi
echo "TASK_TYPE=$TASK_TYPE, flag=$flag"
#flag=false
#flag=true
if [ "$flag" = true ]; then
	MODE='train'
	#MODE='debug'
  RESULTDIR="./results/${EXP_NAME}"
	#RESUME=''
	CUDA_VISIBLE_DEVICES=0 python3.7 -m src.baselines.DispNet.main_DispNet \
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
		--nEpochs=$NUM_EPOCHS \
		--startEpoch=$START_EPOCH \
		--kitti2012=$KT2012 \
		--kitti2015=$KT2015 \
		--virtual_kitti2=$VIRTUAL_KITTI2 \
		--mode=$MODE \
		--resultDir=$RESULTDIR \
		--is_bn=$IS_BN \
	  --corr_func=$CORR_FUNC \
		--kt12_image_mode=$KT12_IMAGE_MODE \
		--is_data_augment=$IS_DATA_AUGMENT
	  exit
fi

######################
# Netwrok Testing 
######################
if [ "$TASK_TYPE" = 'val-30' ]; then
	flag=true
else
	flag=false
fi
#flag=false
#flag=true
if [ "$flag" = true ]; then
	MODE='test'
	#KT15/12: crop_height=384, crop_width=1248
	#sceneflow: crop_height=576, crop_width=960
	#let CROP_HEIGHT=384
	#let CROP_WIDTH=1248
	let MAX_DISP=192
	#let MAX_DISP=192-48

	if [ $KT2012 -eq 1 ] || [ $KT2015 -eq 1 ] || [ $VIRTUAL_KITTI2 -eq 1 ]; then
		let CROP_HEIGHT=384
		#let CROP_WIDTH=1248
		let CROP_WIDTH=1280 # multiple of 64
	else
		let CROP_HEIGHT=576
		let CROP_WIDTH=960
	  let MAX_DISP=192
  fi
		
	declare -a ALL_EPOS_TEST=( 25 50 75 100 125 150 175 200 225 250 275 300 325 350 375 400 425 450 475 500 525 550 575 600 625 650 675 700 725 750 775 800 )
	declare -a ALL_EPOS_TEST=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20)
	#declare -a ALL_EPOS_TEST=(25)
	for idx in $(seq 0 19)
	#for idx in $(seq 1 15)
	#for idx in $(seq 1 31)
	#for idx in $(seq 6 6)
	do # epoch model loop
		EPO_TEST=${ALL_EPOS_TEST[idx]}
		echo "Testing using model at epoch = $EPO_TEST"
		#-------------------------------
		# baseline1: pre-trained DispNet:
		#-------------------------------
	  if [ "$1" == 0 ]; then
			#echo "test GCNet baseline: SF --> KITTI !!!"
			if [ "$KT2015" == 1 ]; then
				echo "test DispNet baseline: SF + KT15 !!!"
				#TMP_MODEL_NAME='dispnet-D192-noBN-corrV1-sfepo10-kt15epo400'
				TMP_MODEL_NAME='dispnet-D192-BN-corrV1-sfepo20-kt15epo400'
				TMP_MODEL_NAME='dispnetV2-D192-BN-corrV1-sfepo20-kt15epo400'
				TMP_MODEL_NAME='dispnetV4-D192-BN-corrV1-sfepo20-kt15epo400'
			  RESUME="./checkpoints/saved/${TMP_MODEL_NAME}/DispNetC/model_epoch_$(printf "%05d" "$EPO_TEST").tar"
				KT2012=0
				VIRTUAL_KITTI2=0
				TEST_LIST="lists/kitti2015_val30.list"
				EXP_NAME="${TMP_MODEL_NAME}/disp-epo-$(printf "%03d" "$EPO_TEST")"
			elif [ "$KT2012" == 1 ]; then
				echo "test DispNet baseline: SF + KT12 !!!"
			  #RESUME="./checkpoints/saved/dispnet-D192-noBN-sfepo20-kt12epo400/DispNetC/model_epoch_$(printf "%05d" "$EPO_TEST").tar"
				VIRTUAL_KITTI2=0
				KT2015=0
				TEST_LIST="lists/kitti2012_val30.list"
				TMP_MODEL_NAME='dispnetV4-D192-BN-corrV1-sfepo20-kt12epo800'
			  RESUME="./checkpoints/saved/${TMP_MODEL_NAME}/DispNetC/model_epoch_$(printf "%05d" "$EPO_TEST").tar"
				EXP_NAME="${TMP_MODEL_NAME}/disp-epo-$(printf "%03d" "$EPO_TEST")"
			
			elif [ "$VIRTUAL_KITTI2" == 1 ]; then
				echo "test DispNet baseline: SF + VIRTUAL_KITTI2 !!!"
			  #RESUME="./checkpoints/saved/dispnet-D192-noBN-sfepo20-kt12epo400/DispNetC/model_epoch_$(printf "%05d" "$EPO_TEST").tar"
				KT2015=0
				KT2012=0
	      TEST_LIST="lists/virtual_kitti2_wo_scene06_fixed_test.list"
	      #TEST_LIST="lists/virtual_kitti2_wo_scene06_fixed_test_small.list"
				TMP_MODEL_NAME='dispnetV4-D192-BN-corrV1-sfepo20-vkt2epo20'
			  RESUME="./checkpoints/${TMP_MODEL_NAME}/DispNetC/model_epoch_$(printf "%05d" "$EPO_TEST").tar"
				EXP_NAME="${TMP_MODEL_NAME}/disp-epo-$(printf "%03d" "$EPO_TEST")"
			
			else
				echo "test DispNet baseline: SF !!!"
				#TMP_MODEL_NAME='dispnet-D192-noBN-corrV1-sfepo10'
				#TMP_MODEL_NAME='dispnet-D192-noBN-corr2-sfepo10'
				TMP_MODEL_NAME='dispnet-D192-BN-corrV1-sfepo20'
				TMP_MODEL_NAME='dispnetV4-D192-BN-corrV1-sfepo20'
			  RESUME="./checkpoints/saved/${TMP_MODEL_NAME}/DispNetC/model_epoch_$(printf "%05d" "$EPO_TEST").tar"
				KT2012=0
				KT2015=0
				TEST_LIST="lists/sceneflow_val.list"
				EXP_NAME="${TMP_MODEL_NAME}-sfVal2k/disp-epo-$(printf "%03d" "$EPO_TEST")"
			fi
		
		else 
			echo "You have to specify a argument to bash!!!"
			exit
		fi
		
		RESULTDIR="./results/${EXP_NAME}"
		cd /home/ccj/atten-stereo
	  CUDA_VISIBLE_DEVICES=0 python3.7 -m src.baselines.DispNet.main_DispNet \
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
			--nEpochs=$NUM_EPOCHS \
			--startEpoch=$START_EPOCH \
			--kitti2012=$KT2012 \
			--kitti2015=$KT2015 \
			--virtual_kitti2=$VIRTUAL_KITTI2 \
			--mode=$MODE \
			--resultDir=$RESULTDIR \
			--is_bn=$IS_BN \
			--corr_func=$CORR_FUNC \
		  --kt12_image_mode=$KT12_IMAGE_MODE \
		  --is_data_augment=$IS_DATA_AUGMENT
		
		if [ $KT2015 -eq 1 ] || [ $KT2012 -eq 1 ]; then
			# move pfm files to val-30 subdir
			makeDir "$RESULTDIR/val-30"
			cd $RESULTDIR
			for i in *.pfm; do
				mv -i -- "$i" "./val-30"
			done	
		  cat "./bad-err.csv" >> "/home/ccj/atten-stereo/results/bad-err-evalu.csv"
	  elif [ $VIRTUAL_KITTI2 -eq 1 ]; then
			# move pfm files to disp-pfm subdir
			makeDir "$RESULTDIR/val-2620"
			cd $RESULTDIR
			for i in *.pfm; do
				mv -i -- "$i" "./val-2620"
			done
		  cat "./bad-err.csv" >> "/home/ccj/atten-stereo/results/bad-err-evalu.csv"
	  else
			# move pfm files to disp-pfm subdir
			makeDir "$RESULTDIR/disp-pfm"
			cd $RESULTDIR
			for i in *.pfm; do
				mv -i -- "$i" "./disp-pfm"
			done
		  cat "./bad-err.csv" >> "/home/ccj/atten-stereo/results/bad-err-evalu.csv"
		fi
		pwd

	done
fi # end of Netwrok Testing

################################################
# For Generalization experiments: CBMVNet paper 
################################################
if [ "$TASK_TYPE" = 'cross-val' ]; then
	flag=true
else
	flag=false
fi
KT2012=0
KT2015=1

if [ "$flag" = true ]; then
	MODE='test'
	#KT15/12: crop_height=384, crop_width=1248
	#sceneflow: crop_height=576, crop_width=960
	#let CROP_HEIGHT=384
	#let CROP_WIDTH=1248
	let MAX_DISP=192
	#let MAX_DISP=192-48

	if [ $KT2012 -eq 1 ] || [ $KT2015 -eq 1 ]; then
		let CROP_HEIGHT=384
		#let CROP_WIDTH=1248
		let CROP_WIDTH=1280 # multiple of 64
	else
		let CROP_HEIGHT=576
		let CROP_WIDTH=960
  fi
		
	#declare -a ALL_EPOS_TEST=(20)
	declare -a ALL_EPOS_TEST=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20)
	#for idx in $(seq 0 0)
	for idx in $(seq 0 19)
	do # epoch model loop
		EPO_TEST=${ALL_EPOS_TEST[idx]}
		echo "Testing using model at epoch = $EPO_TEST"
	  if [ "$1" == 0 ]; then
			#echo "test GCNet baseline: SF --> KITTI !!!"
			#TMP_MODEL_NAME='dispnetV4-D192-BN-corrV1-sfepo20'
			TMP_MODEL_NAME='dispnetV4-D192-BN-corrV1-sfAugepo20'
			#RESUME="./checkpoints/saved/${TMP_MODEL_NAME}/DispNetC/model_epoch_$(printf "%05d" "$EPO_TEST").tar"
			RESUME="./checkpoints/${TMP_MODEL_NAME}/DispNetC/model_epoch_$(printf "%05d" "$EPO_TEST").tar"
			if [ "$KT2015" == 1 ]; then
				echo "sf->tk15: test KT15 train-200 !!!"
				DATA_PATH="${DATA_ROOT}/datasets/KITTI-2015/training/"
				KT2012=0
				TEST_LIST="lists/kitti2015.list"
				#TEST_LIST="lists/kitti2015_val_small.list"
				EXP_NAME="${TMP_MODEL_NAME}-testKT15/disp-epo-$(printf "%03d" "$EPO_TEST")"

			elif [ "$KT2012" == 1 ]; then
				echo "sf->tk12: test KT12 train-194 !!!"
				KT2012=1
				KT2015=0
				TEST_LIST="lists/kitti2012.list"
			fi
		
		else 
			echo "You have to specify a argument to bash!!!"
			exit
		fi
		
		RESULTDIR="./results/${EXP_NAME}"
		cd /home/ccj/atten-stereo
	  CUDA_VISIBLE_DEVICES=0 python3.7 -m src.baselines.DispNet.main_DispNet \
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
			--nEpochs=$NUM_EPOCHS \
			--startEpoch=$START_EPOCH \
			--kitti2012=$KT2012 \
			--kitti2015=$KT2015 \
			--virtual_kitti2=$VIRTUAL_KITTI2 \
			--mode=$MODE \
			--resultDir=$RESULTDIR \
			--is_bn=$IS_BN \
			--corr_func=$CORR_FUNC \
		  --kt12_image_mode=$KT12_IMAGE_MODE \
		  --is_data_augment=$IS_DATA_AUGMENT
		
		if [ $KT2015 -eq 1 ] || [ $KT2012 -eq 1 ]; then
			# move pfm files to disp-pfm subdir
			makeDir "$RESULTDIR/disp-pfm"
			cd $RESULTDIR
			for i in *.pfm; do
				mv -i -- "$i" "./disp-pfm"
			done	
    fi
		
		cat "${RESULTDIR}/bad-err.csv" >> "/home/ccj/atten-stereo/results/bad-err-evalu.csv"
	done
fi # end of Netwrok Testing
