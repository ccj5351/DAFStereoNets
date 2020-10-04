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

if [ ! -d $DATA_ROOT ];then
	DATA_ROOT="/home/$USER"
	echo "Updated : setting data_root = ${DATA_ROOT}"
fi

#---------------------------------------#
#-----Common Hyperparameters Here ------#
#---------------------------------------#
KT2012=0
KT2015=0
VIRTUAL_KITTI2=1
#NUM_EPOCHS=400
#NUM_EPOCHS=20
NUM_EPOCHS=20
NUM_WORKERS=8
#BATCHSIZE=2
BATCHSIZE=4
LOG_SUMMARY_STEP=5
#LOG_SUMMARY_STEP=200

MODEL_NAME='GANet_deep'
#MODEL_NAME='GANet11'


if [ "$MODEL_NAME" = 'GANet_deep' ]; then
	MODEL_NAME_STR='ganet-deep' 
elif [ "$MODEL_NAME" = 'GANet11' ]; then
	MODEL_NAME_STR='ganet11'
else
	echo "Wrong MODEL_NAME!!!"
	exit
fi


if [ $KT2012 -eq 1 ]; then
	DATA_PATH="${DATA_ROOT}/datasets/KITTI-2012/training/"
	KT_STR='kt12'
  TRAINING_LIST="lists/kitti2012_train170.list"
  TEST_LIST="lists/kitti2012_val24.list"
	#revise parameter settings and run "train.sh" and "predict.sh" for 
	#training, finetuning and prediction/testing. 
	#Note that the “crop_width” and “crop_height” must be multiple of 48, 
	#"max_disp" must be multiple of 12 (default: 192).
	let CROP_HEIGHT=240
	let CROP_WIDTH=528
	let MAX_DISP=192
  EXP_NAME="${MODEL_NAME_STR}-D${MAX_DISP}-sfepo10-kt12epo${NUM_EPOCHS}"

elif [ $KT2015 -eq 1 ]; then
	DATA_PATH="${DATA_ROOT}/datasets/KITTI-2015/training/"
	KT_STR='kt15'
	TRAINING_LIST="lists/kitti2015_train170.list"
  TEST_LIST="lists/kitti2015_val30.list"
	let CROP_HEIGHT=240
	let CROP_WIDTH=528
	let MAX_DISP=192
  EXP_NAME="${MODEL_NAME_STR}-D${MAX_DISP}-sfepo10-kt15epo${NUM_EPOCHS}"

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
	let CROP_HEIGHT=240
	let CROP_WIDTH=528-96
	let MAX_DISP=192
  EXP_NAME="${MODEL_NAME_STR}-D${MAX_DISP}-sfepo10-vkt2epo${NUM_EPOCHS}"

else
	DATA_PATH="${DATA_ROOT}/datasets/SceneFlowDataset/"
  TRAINING_LIST="lists/sceneflow_train.list"
  #TRAINING_LIST="lists/sceneflow_train_small.list"
  TEST_LIST="lists/sceneflow_test_select.list"
  #TEST_LIST="lists/sceneflow_test_small.list"
	#let CROP_HEIGHT=256
	#let CROP_WIDTH=512
	let CROP_HEIGHT=240
	let CROP_WIDTH=528
	let MAX_DISP=192
  #EXP_NAME="${MODEL_NAME_STR}-D${MAX_DISP}-sf-small-tmp"
  EXP_NAME="${MODEL_NAME_STR}-D${MAX_DISP}-sfepo${NUM_EPOCHS}"
fi


echo "DATA_PATH=$DATA_PATH"
echo "EXP_NAME=$EXP_NAME"
TRAIN_LOGDIR="./logs/${EXP_NAME}"
echo "TRAIN_LOGDIR=$TRAIN_LOGDIR"
CHECKPOINT_DIR="./checkpoints/${EXP_NAME}"
echo "CHECKPOINT_DIR=$CHECKPOINT_DIR" 
#exit



################################
# Netwrok Training & profiling
################################
flag=false
#flag=true
if [ "$flag" = true ]; then
	MODE='train'
	#MODE='debug'
  RESULTDIR="./results/${EXP_NAME}"
  RESUME="/home/${USER}/atten-stereo/checkpoints/saved/ganet-pretrained/${MODEL_NAME_STR}/sceneflow_epoch_10.pth"
  #RESUME=''
  cd /home/${USER}/atten-stereo
	CUDA_VISIBLE_DEVICES=0 python3.7 -m src.baselines.GANet.train \
		--batchSize=${BATCHSIZE} \
		--crop_height=$CROP_HEIGHT \
		--crop_width=$CROP_WIDTH \
		--max_disp=$MAX_DISP \
		--train_logdir=$TRAIN_LOGDIR \
		--thread=${NUM_WORKERS} \
		--data_path=$DATA_PATH \
		--training_list=$TRAINING_LIST \
		--checkpoint_dir=$CHECKPOINT_DIR \
		--log_summary_step=${LOG_SUMMARY_STEP} \
		--resume=$RESUME \
		--nEpochs=$NUM_EPOCHS \
		--kitti2012=$KT2012 \
		--kitti2015=$KT2015 \
		--virtual_kitti2=$VIRTUAL_KITTI2 \
		--model=$MODEL_NAME
	exit
fi

######################
# Netwrok Testing 
######################
#flag=false
flag=true
if [ "$flag" = true ]; then
	MODE='test'
	let MAX_DISP=192	
	#Note that the “crop_width” and “crop_height” must be multiple of 48, 
	#"max_disp" must be multiple of 12 (default: 192).

	if [ $KT2015 -eq 1 ] || [ $KT2012 -eq 1 ] || [ $VIRTUAL_KITTI2 -eq 1 ] ; then
		let CROP_HEIGHT=384
		let CROP_WIDTH=1248

	else
		let CROP_HEIGHT=576
		let CROP_WIDTH=960
  fi

	task_type=$1
	
	#declare -a ALL_EPOS_TEST=(25 50 75 100 125 150 175 200 225 250 275 300 325 350 375 400)
	#declare -a ALL_EPOS_TEST=(10)
	declare -a ALL_EPOS_TEST=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40)
	for idx in $(seq 11 11)
	#for idx in $(seq 1 15)
	#for idx in $(seq 0 0)
	do # epoch model loop
		EPO_TEST=${ALL_EPOS_TEST[idx]}
		echo "Testing using model at epoch = $EPO_TEST"
		#-------------------------------
		# baseline1: pre-trained GANet: SF
		#-------------------------------
		if [ "$1" == 0 ]; then
			#RESUME='./checkpoints/saved/ganet-sf-epo10-kt15-epoch100/model_epoch_00100.tar'
			EXP_NAME="${MODEL_NAME_STR}-D${MAX_DISP}-sfepo10/disp-epo-$(printf "%03d" "$EPO_TEST")"
      RESUME="./checkpoints/saved/ganet-pretrained/${MODEL_NAME_STR}/sceneflow_epoch_10.pth"
			KT2015=0
			KT2012=0
			echo "test GANet baseline: on SF!!!"
		#-------------------------------
		# baseline2: fine-tuning GANet:
		#-------------------------------
	  elif [ "$1" == 1 ]; then
			#RESUME="./checkpoints/saved/${MODEL_NAME_STR}-D${MAX_DISP}-sfepo10-kt15epo400/??/model_epoch_$(printf "%05d" "$EPO_TEST").tar"
			if [ "$KT2015" == 1 ]; then
				echo "test GANet baseline: SF --> KT15 !!!"
				KT2012=0
				VIRTUAL_KITTI2=0
				TEST_LIST="lists/kitti2015_val30.list"
			  EXP_NAME="${MODEL_NAME_STR}-D${MAX_DISP}-sfepo10-kt15epo400/disp-epo-$(printf "%03d" "$EPO_TEST")"
        RESUME="./checkpoints/saved/ganet-pretrained/${MODEL_NAME_STR}/kitti2015_final.pth"
			
			elif [ "$KT2012" == 1 ]; then
				echo "test GANet baseline: SF --> KT12 !!!"
				KT2015=0
				VIRTUAL_KITTI2=0
				TEST_LIST="lists/kitti2012_val30.list"
				#EXP_NAME="ganet-D192-sfepo20-kt12epo400/disp-epo-$(printf "%03d" "$EPO_TEST")"
			  EXP_NAME="${MODEL_NAME_STR}-D${MAX_DISP}-sfepo10-kt12epo400/disp-epo-$(printf "%03d" "$EPO_TEST")"
        RESUME="./checkpoints/saved/ganet-pretrained/${MODEL_NAME_STR}/kitti2012_final.pth"

			elif [ "$VIRTUAL_KITTI2" == 1 ]; then	
				echo "test GANet baseline: SF --> Virtual KT 2 !!!"
				KT2015=0
				KT2012=0
				#EXP_NAME="ganet-D192-sfepo20-kt12epo400/disp-epo-$(printf "%03d" "$EPO_TEST")"
			  #EXP_NAME="${MODEL_NAME_STR}-D${MAX_DISP}-sfepo10-kt12epo400/disp-epo-$(printf "%03d" "$EPO_TEST")"
        #RESUME="./checkpoints/saved/ganet-pretrained/${MODEL_NAME_STR}/kitti2012_final.pth"
        #RESUME="./checkpoints/saved/${EXP_NAME}"

				TMP_MODEL_NAME="${MODEL_NAME_STR}-D${MAX_DISP}-sfepo10-${KT_STR}epo20"
				RESUME="./checkpoints/saved/${TMP_MODEL_NAME}/model_epoch_$(printf "%05d" "$EPO_TEST").tar"
				EXP_NAME="${TMP_MODEL_NAME}/disp-epo-$(printf "%03d" "$EPO_TEST")"
			fi
		
		else 
			echo "You have to specify a argument to bash!!!"
			exit
		fi
		
		RESULTDIR="./results/${EXP_NAME}"
		cd /home/${USER}/atten-stereo
	  CUDA_VISIBLE_DEVICES=0 python3.7 -m src.baselines.GANet.predict \
			--crop_height=$CROP_HEIGHT \
			--crop_width=$CROP_WIDTH \
			--max_disp=$MAX_DISP \
			--data_path=$DATA_PATH \
			--test_list=$TEST_LIST \
			--resume=$RESUME \
			--kitti2012=$KT2012 \
			--kitti2015=$KT2015 \
		  --virtual_kitti2=$VIRTUAL_KITTI2 \
		  --model=$MODEL_NAME \
			--resultDir=$RESULTDIR
		
		if [ $KT2015 -eq 1 ] || [ $KT2012 -eq 1 ]; then
			# move pfm files to val-30 subdir
			makeDir "$RESULTDIR/val-30"
			cd $RESULTDIR
			for i in *.pfm; do
				mv -i -- "$i" "./val-30"
			done
	  
		elif [ $VIRTUAL_KITTI2 -eq 1 ]; then
			makeDir "$RESULTDIR/val-2620"
			cd $RESULTDIR
			for i in *.pfm; do
				mv -i -- "$i" "./val-2620"
			done
			
			if [ -f "./bad-err.csv" ]; then
				cat "./bad-err.csv" >> "/home/${USER}/atten-stereo/results/bad-err-evalu.csv"
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
				cat "./bad-err.csv" >> "/home/${USER}/atten-stereo/results/bad-err-evalu.csv"
			fi
		fi

	done
fi # end of Netwrok Testing


#######################
# Benchmark Submission 
#######################
KT2012=0
KT2015=1
flag=false
#flag=true
if [ "$flag" = true ]; then
	MODE='test'
	#KT15/12: crop_height=384, crop_width=1248
	#sceneflow: crop_height=576, crop_width=960
	#let CROP_HEIGHT=384
	#let CROP_WIDTH=1248
	let MAX_DISP=192
	#let MAX_DISP=192-48

	if [ $KT2012 -eq 1 ]; then
		let CROP_HEIGHT=384
		let CROP_WIDTH=1248

	elif [ $KT2015 -eq 1 ]; then
		let CROP_HEIGHT=384
		let CROP_WIDTH=1248

	else
		let CROP_HEIGHT=576
		let CROP_WIDTH=960
  fi

	task_type=$1
		
	declare -a ALL_EPOS_TEST=(375)
	for idx in $(seq 0 0)
	do # epoch model loop
		EPO_TEST=${ALL_EPOS_TEST[idx]}
		echo "Testing using model at epoch = $EPO_TEST"
		#-------------------------------
		# baseline2: pre-trained GANet:
		#-------------------------------
	  if [ "$1" == 8 ]; then
			#echo "test GANet baseline: SF --> KITTI !!!"
			RESUME="./???checkpoints/saved/ganet-D192-sfepo20-kt15epo400/GCNet/model_epoch_$(printf "%05d" "$EPO_TEST").tar"
			if [ "$KT2015" == 1 ]; then
				echo "test GANet baseline: benchmark KT15 !!!"
				DATA_PATH="${DATA_ROOT}/datasets/KITTI-2015/testing/"
				KT2012=0
				TEST_LIST="lists/kitti2015_test.list"
				EXP_NAME="ganet-D192-sfepo20-kt15epo400-KT15Submit/disp-epo-$(printf "%03d" "$EPO_TEST")"
			else
				echo "test GANet baseline: benchmark KT12 !!!"
				KT2012=1
				KT2015=0
				DATA_PATH="${DATA_ROOT}/datasets/KITTI-2012/testing/"
				TEST_LIST="lists/kitti2012_test.list"
				EXP_NAME="ganet-D192-sfepo20-kt12epo400-KT12Submit/disp-epo-$(printf "%03d" "$EPO_TEST")"
			fi
		
		else 
			echo "You have to specify a argument to bash!!!"
			exit
		fi
		
		RESULTDIR="./results/${EXP_NAME}"
		cd /home/${USER}/atten-stereo
	  CUDA_VISIBLE_DEVICES=0 python3.7 -m src.baselines.GANet.predict \
			--crop_height=$CROP_HEIGHT \
			--crop_width=$CROP_WIDTH \
			--max_disp=$MAX_DISP \
			--data_path=$DATA_PATH \
			--test_list=$TEST_LIST \
			--resume=$RESUME \
			--kitti2012=$KT2012 \
			--kitti2015=$KT2015 \
		  --model=$MODEL_NAME \
			--resultDir=$RESULTDIR
		
		if [ $KT2015 -eq 1 ]; then
			# move pfm files to val-30 subdir
			makeDir "$RESULTDIR/disp-pfm"
			cd $RESULTDIR
			for i in *.pfm; do
				mv -i -- "$i" "./disp-pfm"
			done	
	  elif [ $KT2012 -eq 1 ]; then
			# move pfm files to val-30 subdir
			makeDir "$RESULTDIR/disp-pfm"
			cd $RESULTDIR
			for i in *.pfm; do
				mv -i -- "$i" "./disp-pfm"
			done
		fi

	done
fi # end of Netwrok Benchmark Submission
