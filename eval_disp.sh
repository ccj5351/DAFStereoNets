#!/bin/bash
#echo "Hi, I'm sleeping for 1 seconds..."
#sleep 1s 
#echo "all Done."

#***********************
#****** Parameters *****
#***********************
dataroot="/media/ccjData2"
if [ ! -d $dataroot ]; then
	dataroot="/home/ccj"
	echo "Updated : setting dataroot = ${dataroot}"
fi

ktimgdir="$dataroot/datasets/KITTI-2015/training/"
#file_txt="/home/ccj/GCNet/results/cbmvnet-gc-regu-sfepo9-F8-MBV3-ft-fold3_all-testKT15/kt15-1_fold_cv_small.txt"
file_txt=""
mbv3badthresh=1.0
#mbv3gt='/home/ccj/datasets/MiddleBury/MiddEval3/trainingQ/'
flag=false
mbv3gt=''
#data='mbv3q'
#data='mbv3h'
#data='mbv3f'
#data='kt12'
#data='kt15'

#--------------------------------------
# not used here, put here as backup
declare -a disps_to_eval_old=( 
	 'asn-npac-embed-k5-d2-D192-dispnetc-sfepo20-kt15epo400-embedlossW-0.06'
	 'asn-npac-k5-d2-D192-dispnetc-sfepo20-kt15epo1000-woEmbed'
   'asn-pac-embed-k3-d2-D144-psm-sfepo10-kt15epo400-embedlossW-0.06/disp-epo-325/val-30' 
   'asn-pac-embed-k3-d2-D144-psm-sfepo10-kt15epo400-embedlossW-0.06/disp-epo-350/val-30' 
   'asn-pac-embed-k3-d2-D144-psm-sfepo10-kt15epo400-embedlossW-0.06/disp-epo-375/val-30' 
   'asn-pac-embed-k3-d2-D144-psm-sfepo10-kt15epo400-embedlossW-0.06/disp-epo-400/val-30' 
	 
	 'asn-dfn-k5-d2-D192-psm-sfepo10-kt15epo400-woEmbed/disp-epo-325/val-30'
	 'asn-dfn-k5-d2-D192-psm-sfepo10-kt15epo400-woEmbed/disp-epo-350/val-30'
	 'asn-dfn-k5-d2-D192-psm-sfepo10-kt15epo400-woEmbed/disp-epo-375/val-30'
	 'asn-dfn-k5-d2-D192-psm-sfepo10-kt15epo400-woEmbed/disp-epo-400/val-30'

   'atten-embedCitFine-kt15_mccnn_acrt-sgm-D192/ebf-k5-d2-D192/val-200'
   'atten-embedCitFine-kt15_mccnn_acrt-sgm-D192/ebf-k3-d2-D192/val-200'

	 'atten-embedCityCoar-kt15_mccnn_fast-sgm-D192/kt15_mccnn_fast-D192/val-200' 
   'atten-embedCityCoar-kt15_mccnn_fast-sgm-D192/ebf-k3-d2-D192/val-200' 
   'atten-embedCityCoar-kt15_mccnn_fast-sgm-D192/ebf-k5-d2-D192/val-200' 
   
	 'atten-embedCityCoar-kt15_mccnn_fast-sgm-D192/ebf-k19-d2-D192/val-30' 
   'atten-embedCityCoar-kt15_mccnn_fast-sgm-D192/ebf-k21-d2-D192/val-30' 
   'atten-embedCityCoar-kt15_mccnn_fast-sgm-D192/ebf-k15-d2-D192/val-30' 
   'atten-embedCityCoar-kt15_mccnn_fast-sgm-D192/ebf-k17-d2-D192/val-30' 
	 
	 'atten-embedPasc-kt15_mccnn_fast-sgm-D192/ebf-k21-d2-D192/val-30'
	 'atten-embedPasc-kt15_mccnn_fast-sgm-D192/ebf-k15-d2-D192/val-30'
	 'atten-embedPasc-kt15_mccnn_fast-sgm-D192/ebf-k17-d2-D192/val-30'
	 'atten-embedPasc-kt15_mccnn_fast-sgm-D192/ebf-k19-d2-D192/val-30'

	 'atten-embedCityCoar-kt15_mccnn_fast-sgm-D192/kt15_mccnn_fast-D192/val-30' 
   'atten-embedCityCoar-kt15_mccnn_fast-sgm-D192/ebf-k3-d2-D192/val-30' 

   'atten-embedCityCoar-kt15_mccnn_acrt-sgm-D192/ebf-k5-d2-D192/val-200'
   'atten-embedCityCoar-kt15_mccnn_acrt-sgm-D192/ebf-k3-d2-D192/val-200'

	 'atten-embedCityCoar-kt15_mccnn_acrt-sgm-D192/kt15_mccnn_acrt-D192/val-100'

	 'atten-embedCityCoar-kt15_mccnn-sgm-D192/ebf-k13-d2-D192/val-30' 
	 'atten-embedCityCoar-kt15_mccnn-sgm-D192/ebf-k15-d2-D192/val-30'
	 'atten-embedCityCoar-kt15_mccnn-sgm-D192/ebf-k17-d2-D192/val-30' 
	 'atten-embedCityCoar-kt15_mccnn-sgm-D192/ebf-k19-d2-D192/val-30'

	 'atten-kt15_mccnn-sgm-D192/ebf-k11-d2-D192/val-30'
	 'atten-kt15_mccnn-sgm-D192/ebf-k13-d2-D192/val-30'
	 'atten-kt15_mccnn-sgm-D192/ebf-k15-d2-D192/val-30'
	 'atten-kt15_mccnn-sgm-D192/ebf-k17-d2-D192/val-30'
	 'atten-kt15_mccnn-sgm-D192/ebf-k19-d2-D192/val-30'
	 'atten-kt15_mccnn-sgm-D192/ebf-k21-d2-D192/val-30'

	 'atten-sgm-D192-kt15/ebf-k13-d1-D192/val-30'
	 'atten-sgm-D192-kt15/ebf-k15-d1-D192/val-30'
	 'atten-sgm-D192-kt15/ebf-k17-d1-D192/val-30'
	 'atten-sgm-D192-kt15/ebf-k19-d1-D192/val-30'
	 'atten-sgm-D192-kt15/ebf-k21-d1-D192/val-30'

	 'atten-sgm-D192-kt15/sgm-D192/val-30'

   'asn-embed-k7-d2-D192-dispnetc-sfepo20-kt15epo400-embedlossW-0.06/disp-epo-325/val-30'
   'asn-embed-k7-d2-D192-dispnetc-sfepo20-kt15epo400-embedlossW-0.06/disp-epo-350/val-30'
   'asn-embed-k7-d2-D192-dispnetc-sfepo20-kt15epo400-embedlossW-0.06/disp-epo-375/val-30'
   'asn-embed-k7-d2-D192-dispnetc-sfepo20-kt15epo400-embedlossW-0.06/disp-epo-400/val-30'
   
	 'dispnet-D192-BN-corrV1-sfepo10-kt15epo400/disp-epo-325/val-30'
   'dispnet-D192-BN-corrV1-sfepo10-kt15epo400/disp-epo-350/val-30'
   'dispnet-D192-BN-corrV1-sfepo10-kt15epo400/disp-epo-375/val-30'
   'dispnet-D192-BN-corrV1-sfepo10-kt15epo400/disp-epo-400/val-30'

   'dispnet-D192-noBN-corrV2-sfepo10-kt15epo400/disp-epo-325/val-30'
   'dispnet-D192-noBN-corrV2-sfepo10-kt15epo400/disp-epo-350/val-30'
   'dispnet-D192-noBN-corrV2-sfepo10-kt15epo400/disp-epo-375/val-30'
   'dispnet-D192-noBN-corrV2-sfepo10-kt15epo400/disp-epo-400/val-30'

	 'dispnet-D192-noBN-sfepo20-kt15epo400/disp-epo-175/val-30'
	 'gcnet-D192-sfepo20-kt15epo400/disp-epo-325/val-30'
	 'gcnet-D192-sfepo20-kt15epo400/disp-epo-350/val-30'
	 'gcnet-D192-sfepo20-kt15epo400/disp-epo-375/val-30'
	 'gcnet-D192-sfepo20-kt15epo400/disp-epo-400/val-30'
	 'asn-npac-embed-k3-d2-D144-psm-sfepo10-kt15epo400-embedlossW-0.06/disp-epo-325/val-30'
	 'asn-npac-embed-k3-d2-D144-psm-sfepo10-kt15epo400-embedlossW-0.06/disp-epo-350/val-30'
	 'asn-npac-embed-k3-d2-D144-psm-sfepo10-kt15epo400-embedlossW-0.06/disp-epo-375/val-30'
	 'asn-npac-embed-k3-d2-D144-psm-sfepo10-kt15epo400-embedlossW-0.06/disp-epo-400/val-30'
	 'asn-npac-k3-d2-D144-psm-sfepo10-kt15epo400-woEmbed/disp-epo-325/val-30'
	 'asn-npac-k3-d2-D144-psm-sfepo10-kt15epo400-woEmbed/disp-epo-350/val-30'
	 'asn-npac-k3-d2-D144-psm-sfepo10-kt15epo400-woEmbed/disp-epo-375/val-30'
	 'asn-npac-k3-d2-D144-psm-sfepo10-kt15epo400-woEmbed/disp-epo-400/val-30'
	 'asn-npac-embed-k5-d2-D144-psm-sfepo10-kt15epo400-embedlossW-0.06/disp-epo-325/val-30'
	 'asn-npac-embed-k5-d2-D144-psm-sfepo10-kt15epo400-embedlossW-0.06/disp-epo-350/val-30'
	 'asn-npac-embed-k5-d2-D144-psm-sfepo10-kt15epo400-embedlossW-0.06/disp-epo-375/val-30'
	 'asn-npac-embed-k5-d2-D144-psm-sfepo10-kt15epo400-embedlossW-0.06/disp-epo-400/val-30'
	 'asn-pac-k7-d2-D144-psm-sfepo10-kt15epo400-woEmbed/disp-epo-325/val-30'
	 'asn-pac-k7-d2-D144-psm-sfepo10-kt15epo400-woEmbed/disp-epo-350/val-30'
	 'asn-pac-k7-d2-D144-psm-sfepo10-kt15epo400-woEmbed/disp-epo-375/val-30'
	 'asn-pac-k7-d2-D144-psm-sfepo10-kt15epo400-woEmbed/disp-epo-400/val-30'
	 'asn-pac-embed-k15-d2-D144-psm-sfepo10-kt15epo400-embedlossW-0.06/disp-epo-725/val-30'
	 'asn-pac-embed-k15-d2-D144-psm-sfepo10-kt15epo400-embedlossW-0.06/disp-epo-750/val-30'
	 'asn-pac-embed-k15-d2-D144-psm-sfepo10-kt15epo400-embedlossW-0.06/disp-epo-775/val-30'
	 'asn-pac-embed-k15-d2-D144-psm-sfepo10-kt15epo400-embedlossW-0.06/disp-epo-800/val-30'
	 'asn-pac-embed-k15-d2-D144-psm-sfepo10-kt15epo400-embedlossW-0.006/disp-epo-325/val-30'
	 'asn-pac-embed-k15-d2-D144-psm-sfepo10-kt15epo400-embedlossW-0.006/disp-epo-350/val-30'
	 'asn-pac-embed-k15-d2-D144-psm-sfepo10-kt15epo400-embedlossW-0.006/disp-epo-375/val-30'
	 'asn-pac-embed-k15-d2-D144-psm-sfepo10-kt15epo400-embedlossW-0.006/disp-epo-400/val-30'
	 'asn-pac-k15-d2-D144-psm-sfepo10-kt15epo400-woEmbed/disp-epo-300/val-30'
	 'asn-pac-k15-d2-D144-psm-sfepo10-kt15epo400-woEmbed/disp-epo-325/val-30'
	 'asn-pac-k15-d2-D144-psm-sfepo10-kt15epo400-woEmbed/disp-epo-350/val-30'
	 'asn-pac-k15-d2-D144-psm-sfepo10-kt15epo400-woEmbed/disp-epo-375/val-30'
	 'asn-pac-k15-d2-D144-psm-sfepo10-kt15epo400-woEmbed/disp-epo-400/val-30'
	 'asn-embed-k21-d1-sga-sfepo10-kt15epo400-embedlossW-0.006/disp-epo-400/val-30' 
	 'asn-embed-k21-d1-sga-sfepo10-kt15epo400-embedlossW-0.006/disp-epo-150/val-30' 
	 'asn-embed-k21-d1-sga-sfepo10-kt15epo400-embedlossW-0.006/disp-epo-100/val-30' 
	 'asn-embed-k21-d1-sga-sfepo10-kt15epo400-embedlossW-0.006/disp-epo-050/val-30' 
	 'asn-embed-k5-d2-sga-sfepo10-kt15epo200-woEmbedLoss/disp-epo-050/val-30' 
	 'asn-embed-k5-d2-sga-sfepo10-kt15epo200-woEmbedLoss/disp-epo-100/val-30' 
	 'asn-embed-k5-d2-sga-sfepo10-kt15epo200-woEmbedLoss/disp-epo-150/val-30' 
	 'asn-embed-k5-d2-sga-sfepo10-kt15epo200-woEmbedLoss/disp-epo-200/val-30' 
	 'ganet-sfepo10-kt15epo100/val-30'
	 'ganet-sfepo10-kt15epo100-w-embed/val-30'
	 'asn-embed-k13-sga-sfepo10-kt15epo200/val-30'
	 )
#--------------------------------------

#--------------------------------------------------
#### ECCV 2020 CBMV Experiments: cross evaluation;
#--------------------------------------------------
#KT2015=1 KT2012=0
#flag=true
flag=false
if [ "$flag" = true ]; then
	declare -a disps_to_eval=( 
	 'resneti2-D192-BN-corrV1-sfAugepo20-testKT15'
	 'resneti2-D192-BN-corrV1-sfAugepo10-testKT15'
	 'dispnetV4-D192-BN-corrV1-sfAugepo20-testKT15'
	 'cbmvdispnetcSoft-D192-sfepo20-testMB14'
	 'cbmvdispnetcSoft-D192-sfepo20-testETH3D'
	 'cbmvdispnetcSoft-D192-sfepo20-testKT12'
	 'cbmvdispnetcSoft-D192-sfepo20-testKT15'
	 'cbmvdispnetcV2SumFus-D192-sfepo20-testKT15'
	 'cbmvdispnetcV2Cosine-D192-sfepo20-testMB14'
	 'cbmvdispnetcV2Cosine-D192-sfepo20-testETH3D'
	 'cbmvdispnetcV2Cosine-D192-sfepo20-testKT12'
	 'cbmvdispnetcV2Cosine-D192-sfepo20-testKT15'
	 'cbmvdispnetcV3Cosine-D192-sfepo20-testMB14'
	 'cbmvdispnetcV3Cosine-D192-sfepo20-testETH3D'
	 'cbmvdispnetcV3Cosine-D192-sfepo20-testKT12'
	 'cbmvdispnetcV3Cosine-D192-sfepo20-testKT15'
	 'cbmvdispnetcV2Cosine-D256-sfepo20-testMB14'
	 'cbmvdispnetcV2Cosine-D256-sfepo20-testETH3D'
	 'cbmvdispnetcV2Cosine-D256-sfepo20-testKT12'
	 'cbmvdispnetcV2Cosine-D256-sfepo20-testKT15'
	 'cbmvresneti2-D192-sfepo30-testMB14'
	 'dispnetV4-D192-BN-corrV1-sfepo20-testETH3D'
	 'dispnetV4-D192-BN-corrV1-sfepo20-testMB14'
	 'cbmvdispnetc-D192-sfepo20-testMB14'
	 'cbmvdispnetc-D192-sfepo20-testETH3D'
	 'cbmvresneti2-D192-sfepo30-testETH3D'
	 'resneti2-D192-BN-corrV1-sfepo20-testKT15'
	 'resneti2-D192-BN-corrV1-sfepo20-testKT12'
	 'resneti2-D192-BN-corrV1-sfepo20-testMB14'
	 'resneti2-D192-BN-corrV1-sfepo20-testETH3D'
	 'cbmvresneti2-D192-sfepo30-testKT12'
	 'cbmvresneti2-D192-sfepo30-testKT15'
	 'cbmvdispnetc-D192-sfepo30-testKT12'
	 'cbmvdispnetc-D192-sfepo30-testKT15'
	 'dispnetV4-D192-BN-corrV1-sfepo20-testKT12'
	 'dispnetV4-D192-BN-corrV1-sfepo20-testKT15'
	 'cbmvgcnet-D192-sfepo10-testKT15'
	 'asn-sga-k0-d2-D192-psm-sfepo10-kt15epo400-woEmbed'
	 'asn-embed-k5-d2-D192-dispnetcV4-sfepo20-kt15epo500-embedlossW-0.06'
	 'asn-pac-embed-k5-d2-D192-dispnetc-sfepo20-kt15epo400-embedlossW-0.06'
	 )
	
	#declare -a ALL_EPOS_TEST=(50 100 150 200 250 300 350 400 450 500 550 600 650 700 750 800 850 900 950 1000)
	#declare -a ALL_EPOS_TEST=(25 50 75 100 125 150 175 200 225 250 275 300 325 350 375 400)
	declare -a ALL_EPOS_TEST=(25)
	declare -a ALL_EPOS_TEST=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20)
	#for idx in 0 
	for idx in 0
	do
		if [[ ${disps_to_eval[idx]} == *"KT15"* ]]; then
			data='kt15'
			ktimgdir="$dataroot/datasets/KITTI-2015/training/"
			csv_file_str='kt-err'
			echo "KT15: It's there!"
		elif [[ ${disps_to_eval[idx]} == *"KT12"* ]]; then
			data='kt12'
			ktimgdir="$dataroot/datasets/KITTI-2012/training/"
			csv_file_str='kt-err'
			echo "KT12: It's there!"
		elif [[ ${disps_to_eval[idx]} == *"ETH3D"* ]]; then
	    data='eth3d'
			csv_file_str='eth3d-err'
			echo "ETH3D: It's there!"
		elif [[ ${disps_to_eval[idx]} == *"MB14Q"* ]]; then
		  data='mbv3q'
      #mbv3gt="$dataroot/datasets/MiddleBury/MiddEval3/trainingQ/"
			#mbv3badthresh=0.5
      mbv3gt=""
			mbv3badthresh=2.0
			csv_file_str='mbv3-err'
			echo "MB14Q: It's there!"
		elif [[ ${disps_to_eval[idx]} == *"MB14"* ]]; then
		  data='mbv3h'
      mbv3gt=''
			mbv3badthresh=2.0
			csv_file_str='mbv3-err'
			echo "MB14H: It's there!"
		fi
		#tmp_name=${disps_to_eval[idx]}
		#for epo_idx in $(seq 0 19)
		for epo_idx in $(seq 10 19)
		#for epo_idx in $(seq 0 0)
		do
			EPO_TEST=${ALL_EPOS_TEST[epo_idx]}
			#tmp_name="${disps_to_eval[idx]}/disp-epo-$(printf "%03d" "$EPO_TEST")/val-30"
			tmp_name="${disps_to_eval[idx]}/disp-epo-$(printf "%03d" "$EPO_TEST")/disp-pfm"
			resultdir="/home/ccj/atten-stereo/results/${tmp_name}/"
		  echo $resultdir
			if [ -f "${resultdir}${csv_file_str}.csv" ]; then
				rm "${resultdir}${csv_file_str}.csv"
			fi
			python3.7 eval_disp.py --dataroot=$dataroot --mbv3gt=$mbv3gt \
				--mbv3badthresh=$mbv3badthresh --ktimgdir=$ktimgdir \
				--resultdir=$resultdir --file=$file_txt --dataset=$data
			cat "${resultdir}${csv_file_str}.csv" >> "/home/ccj/atten-stereo/results/bad-err-evalu.csv"
		done
  done
fi


#-----------------------------------
#### Attention stero matching: evaluating KT15;
#-----------------------------------
KT2015=1 KT2012=0
#KT2015=0 KT2012=1
flag=true
#flag=false
if [ "$flag" = true ]; then
	if [ "$KT2015" == 1 ]; then
		data='kt15'
		ktimgdir="$dataroot/datasets/KITTI-2015/training/"
	elif [ "$KT2012" == 1 ]; then
		data='kt12'
		ktimgdir="$dataroot/datasets/KITTI-2012/training/"
	fi
	declare -a disps_to_eval=(
	 'kt15-ablation-runtime/asn-embed-k9-d2-D192-psm-sfepo10-kt15epo100-embedlossW-0.06-lr-0.001-p-eposteps-80'
	 'kt15-ablation/asn-dfn-k5-d2-D192-psm-sfepo10-kt15epo400-woEmbed'
	 'kt15-ablation/asn-embed-k7-d2-D192-psm-sfepo10-kt15epo200-embedlossW-0.06-lr-0.001-p-eposteps-150'
	 'kt15-ablation/asn-embed-k7-d1-D192-psm-sfepo10-kt15epo200-embedlossW-0.06-lr-0.001-p-eposteps-150'
	 'kt15-ablation/asn-embed-k5-d2-D192-psm-sfepo10-kt15epo300-embedlossW-0.06-lr-0.001-p-eposteps-200'
	 'kt15-ablation/asn-embed-k5-d1-D192-psm-sfepo10-kt15epo300-embedlossW-0.06-lr-0.001-p-eposteps-200'
	 'kt15-ablation/asn-embed-k3-d2-D192-psm-sfepo10-kt15epo300-embedlossW-0.06-lr-0.001-p-eposteps-200'
	 'kt15-ablation/asn-embed-k9-d1-D192-psm-sfepo10-kt15epo100-embedlossW-0.06-lr-0.001-p-eposteps-80'
	 'kt15-ablation/asn-embed-k3-d1-D192-psm-sfepo10-kt15epo300-embedlossW-0.06-lr-0.001-p-eposteps-200'
	 'asn-embed-k3-d1-D192-dispnetc-sfepo20-kt15epo600-embedlossW-0.06-lr-0.001-p-eposteps-300'
	 'asn-embed-k5-d2-D192-dispnetc-sfepo20-kt15epo600-embedlossW-0.06-lr-0.001-p-eposteps-300'
	 'asn-embed-k5-d1-D192-dispnetc-sfepo20-kt15epo600-embedlossW-0.06-lr-0.001-p-eposteps-300'
	 'asn-embed-k7-d2-D192-dispnetc-sfepo20-kt15epo600-embedlossW-0.06-lr-0.001-p-eposteps-300'
	 'asn-embed-k7-d1-D192-dispnetc-sfepo20-kt15epo600-embedlossW-0.06-lr-0.001-p-eposteps-300'
	 'asn-embed-k9-d2-D192-dispnetc-sfepo20-kt15epo600-embedlossW-0.06-lr-0.001-p-eposteps-300'
	 'asn-embed-k9-d1-D192-dispnetc-sfepo20-kt15epo600-embedlossW-0.06-lr-0.001-p-eposteps-300'
	 'asn-embed-k11-d2-D192-dispnetc-sfepo20-kt15epo600-embedlossW-0.06-lr-0.001-p-eposteps-300'
	 'asn-embed-k3-d2-D192-dispnetc-sfepo20-kt15epo600-embedlossW-0.06-lr-0.001-p-eposteps-300'
	 'asn-embed-k11-d1-D192-dispnetc-sfepo20-kt15epo600-embedlossW-0.06-lr-0.001-p-eposteps-300'
	 )
	
	#declare -a ALL_EPOS_TEST=(50 100 150 200 250 300 350 400 450 500 550 600 650 700 750 800 850 900 950 1000)
	declare -a ALL_EPOS_TEST=(25 50 75 100 125 150 175 200 225 250 275 300 325 350 375 400 425 450 475 500 525 550 575 600 625 650 675 700 725 750 775 800 )
	declare -a ALL_EPOS_TEST=(20 40 60 80 100 120 140 160 180 200 220 240 260 280 300)
	declare -a ALL_EPOS_TEST=(10 20 30 40 50 60 70 80 90 100 110 120 130 140 150 160 170 180 190 200)
	declare -a ALL_EPOS_TEST=(350)
	declare -a ALL_EPOS_TEST=(10 20 30 40 50 60 70 80 90 100)
	#for idx in 0 
	for idx in 0
	do
		#tmp_name=${disps_to_eval[idx]}
		for epo_idx in $(seq 0 9)
		#for epo_idx in $(seq 0 19)
		#for epo_idx in $(seq 0 23)
		#for epo_idx in $(seq 0 0)
		do
			EPO_TEST=${ALL_EPOS_TEST[epo_idx]}
			tmp_name="${disps_to_eval[idx]}/disp-epo-$(printf "%03d" "$EPO_TEST")/val-30"
			resultdir="/home/ccj/atten-stereo/results/${tmp_name}/"
		  echo $resultdir
			if [ -f "${resultdir}kt-err.csv" ]; then
				rm "${resultdir}kt-err.csv"
			fi
			python3.7 eval_disp.py --dataroot=$dataroot --mbv3gt=$mbv3gt \
				--mbv3badthresh=$mbv3badthresh --ktimgdir=$ktimgdir \
				--resultdir=$resultdir --file=$file_txt --dataset=$data
			cat "${resultdir}kt-err.csv" >> "/home/ccj/atten-stereo/results/bad-err-evalu.csv"
		done
  done
fi


#-----------------------------------
#### evaluating KT15;
#-----------------------------------

#flag=true
flag=false
if [ "$flag" = true ]; then
	data='kt15'
  ktimgdir="$dataroot/datasets/KITTI-2015/training/"
	#for epo in 25 335 476
	#for epo in 114
	for epo in 22 23 25
	do
		#-----------------
		#----sf-all ------
		#-----------------
	  tmp_name="cbmvnet-gc-F8-RMSp-sf-epo26Based-epo30-4dsConv-k5"
	  #tmp_name="gcnet-F8-RMSp-sf-epo30-4dsConv-k5"
		#-----------------
		#----sf-3k ------
		#-----------------
	  #tmp_name="cbmvnet-gc-F8-RMSp-sf3k-epo20-4dsConv-k5"
	  #tmp_name="gcnet-F8-RMSp-sf3k-epo20-4dsConv-k5"
		#-----------------
		#----sfF3k ------
		#-----------------
	  #tmp_name="cbmvnet-gc-F8-RMSp-sfF3kSimR-epo30-4dsConv-k5"
	  #tmp_name="gcnet-F8-RMSp-sfF3kSimR-epo30-4dsConv-k5"
		#-----------------
		#----sfM3k ------
		#-----------------
	  #tmp_name="cbmvnet-gc-F8-RMSp-sfM3kSimR-epo30-4dsConv-k5"
	  #tmp_name="gcnet-F8-RMSp-sfM3kSimR-epo30-4dsConv-k5"
		#-----------------
		#----sfD3k ------
		#-----------------
	  #tmp_name="cbmvnet-gc-F8-RMSp-sfD3kSimR-epo30-4dsConv-k5"
	  #tmp_name="gcnet-F8-RMSp-sfD3kSimR-epo30-4dsConv-k5"
		#-----------------
		#----sf+kt12 ------
		#-----------------
		#tmp_name="cbmvnet-gc-F8-RMSp-sfepo26-kt12epo300-4dsConv-k5"
		#tmp_name="gcnet-F8-RMSp-sfepo30-kt12epo300-4dsConv-k5"
		#-------------------------------
		#----sf+eth-ad-200 images ------
		#-------------------------------
		#tmp_name="cbmvnet-gc-F8-RMSp-sfepo26-ethad200epo300-4dsConv-k5"
		#tmp_name="gcnet-F8-RMSp-sfepo30-ethad200epo300-4dsConv-k5"
		#-------------------------------
		#----sf+kt12+eth-ad-200 images ------
		#-------------------------------
		#tmp_name="cbmvnet-gc-F8-RMSp-sfepo26-ethad200kt12epo300-4dsConv-k5"
		#tmp_name="gcnet-F8-RMSp-sfepo30-ethad200kt12epo300-4dsConv-k5"

		resultdir="/home/ccj/GCNet/results/${tmp_name}-testKT15/disp-epo-$(printf "%03d" "$epo")/"
		echo $resultdir
		if [ -f "${resultdir}kt-err.csv" ]; then
			rm "${resultdir}kt-err.csv"
		fi
		python3.7 eval_disp.py --dataroot=$dataroot --mbv3gt=$mbv3gt --mbv3badthresh=$mbv3badthresh --ktimgdir=$ktimgdir --resultdir=$resultdir --file=$file_txt --dataset=$data
		cat "${resultdir}kt-err.csv" >> "/home/ccj/GCNet/results/bad-err-evalu.csv"
  done
fi


#-----------------------------------
#### evaluating KT12;
#-----------------------------------
#flag=true
flag=false
if [ "$flag" = true ]; then
	data='kt12'
  ktimgdir="$dataroot/datasets/KITTI-2012/training/"
	for epo in 21 29 25 17
	do
		#-----------------
		#----sf-all ------
		#-----------------
	  tmp_name="cbmvnet-gc-F8-RMSp-sf-epo26Based-epo30-4dsConv-k5"
	  tmp_name="gcnet-F8-RMSp-sf-epo30-4dsConv-k5"
		#-----------------
		#----sf-3k ------
		#-----------------
	  #tmp_name="cbmvnet-gc-F8-RMSp-sf3k-epo20-4dsConv-k5"
	  tmp_name="gcnet-F8-RMSp-sf3k-epo20-4dsConv-k5"
		#-----------------
		#----sfF3k ------
		#-----------------
	  #tmp_name="cbmvnet-gc-F8-RMSp-sfF3kSimR-epo30-4dsConv-k5"
	  tmp_name="gcnet-F8-RMSp-sfF3kSimR-epo30-4dsConv-k5"
		#-----------------
		#----sfM3k ------
		#-----------------
	  tmp_name="cbmvnet-gc-F8-RMSp-sfM3kSimR-epo30-4dsConv-k5"
	  #tmp_name="gcnet-F8-RMSp-sfM3kSimR-epo30-4dsConv-k5"
		#-----------------
		#----sfD3k ------
		#-----------------
	  #tmp_name="cbmvnet-gc-F8-RMSp-sfD3kSimR-epo30-4dsConv-k5"
	  #tmp_name="gcnet-F8-RMSp-sfD3kSimR-epo30-4dsConv-k5"

		resultdir="/home/ccj/GCNet/results/${tmp_name}-testKT12/disp-epo-$(printf "%03d" "$epo")/"
		echo $resultdir
		if [ -f "${resultdir}kt-err.csv" ]; then
			rm "${resultdir}kt-err.csv"
		fi
		python3.7 eval_disp.py --dataroot=$dataroot --mbv3gt=$mbv3gt --mbv3badthresh=$mbv3badthresh --ktimgdir=$ktimgdir --resultdir=$resultdir --file=$file_txt --dataset=$data
		cat "${resultdir}kt-err.csv" >> "/home/ccj/GCNet/results/bad-err-evalu.csv"
  done
fi



#-----------------------------------
#### evaluating ETH3D;
#-----------------------------------
flag=false
#flag=true
if [ "$flag" = true ]; then
	data='eth3d'
	#for epo in 539 231 139 95 31
	for epo in 21 29 25 17
	do
		#-----------------
		#----sf-all ------
		#-----------------
	  #tmp_name="cbmvnet-gc-F8-RMSp-sf-epo26Based-epo30-4dsConv-k5"
	  tmp_name="gcnet-F8-RMSp-sf-epo30-4dsConv-k5"
		#-----------------
		#----sf-3k ------
		#-----------------
	  #tmp_name="cbmvnet-gc-F8-RMSp-sf3k-epo20-4dsConv-k5"
	  tmp_name="gcnet-F8-RMSp-sf3k-epo20-4dsConv-k5"
		#-----------------
		#----sfF3k ------
		#-----------------
	  #tmp_name="cbmvnet-gc-F8-RMSp-sfF3kSimR-epo30-4dsConv-k5"
	  tmp_name="gcnet-F8-RMSp-sfF3kSimR-epo30-4dsConv-k5"
		#-----------------
		#----sfM3k ------
		#-----------------
	  tmp_name="cbmvnet-gc-F8-RMSp-sfM3kSimR-epo30-4dsConv-k5"
	  #tmp_name="gcnet-F8-RMSp-sfM3kSimR-epo30-4dsConv-k5"
		#-----------------
		#----sfD3k ------
		#-----------------
	  #tmp_name="cbmvnet-gc-F8-RMSp-sfD3kSimR-epo30-4dsConv-k5"
	  #tmp_name="gcnet-F8-RMSp-sfD3kSimR-epo30-4dsConv-k5"
		
		resultdir="/home/ccj/GCNet/results/${tmp_name}-testETH3D/disp-epo-$(printf "%03d" "$epo")/"
		if [ ! -d $resultdir ]; then
			resultdir="/home/ccj/GCNet/results/${tmp_name}-testETH3D/disp64-epo-$(printf "%03d" "$epo")/"
		fi
		echo $resultdir
		if [ -f "${resultdir}eth3d-err.csv" ]; then
			rm "${resultdir}eth3d-err.csv"
		fi
		python3.7 eval_disp.py --dataroot=$dataroot --mbv3gt=$mbv3gt --mbv3badthresh=$mbv3badthresh --ktimgdir=$ktimgdir --resultdir=$resultdir --file=$file_txt --dataset=$data
		cat "${resultdir}eth3d-err.csv" >> "/home/ccj/GCNet/results/bad-err-evalu.csv"
	done
fi


#-----------------------------------
#### evaluating MBV3 H-size;
#-----------------------------------
flag=false
#flag=true
if [ "$flag" = true ]; then
	for epo in 224
	#for epo in $(seq 1 10)
	do
		data='mbv3h'
		mbv3badthresh=2.0
		#-----------------
		#----sf-all ------
		#-----------------
	  #tmp_name="cbmvnet-gc-F8-RMSp-sf-epo26Based-epo30-4dsConv-k5"
	  #tmp_name="gcnet-F8-RMSp-sf-epo30-4dsConv-k5"
		#-----------------
		#----sf-3k ------
		#-----------------
	  #tmp_name="cbmvnet-gc-F8-RMSp-sf3k-epo20-4dsConv-k5"
	  #tmp_name="gcnet-F8-RMSp-sf3k-epo20-4dsConv-k5"
		#-----------------
		#----sfF3k ------
		#-----------------
	  #tmp_name="cbmvnet-gc-F8-RMSp-sfF3kSimR-epo30-4dsConv-k5"
	  #tmp_name="gcnet-F8-RMSp-sfF3kSimR-epo30-4dsConv-k5"
		#-----------------
		#----sfM3k ------
		#-----------------
	  #tmp_name="cbmvnet-gc-F8-RMSp-sfM3kSimR-epo30-4dsConv-k5"
	  #tmp_name="gcnet-F8-RMSp-sfM3kSimR-epo30-4dsConv-k5"
		#-----------------
		#----sfD3k ------
		#-----------------
	  #tmp_name="cbmvnet-gc-F8-RMSp-sfD3kSimR-epo30-4dsConv-k5"
	  #tmp_name="gcnet-F8-RMSp-sfD3kSimR-epo30-4dsConv-k5"
		#-----------------
		#----sf+kt12 ------
		#-----------------
		#tmp_name="cbmvnet-gc-F8-RMSp-sfepo26-kt12epo300-4dsConv-k5"
		tmp_name="gcnet-F8-RMSp-sfepo30-kt12epo300-4dsConv-k5"
		#-------------------------------
		#----sf+eth-ad-200 images ------
		#-------------------------------
		#tmp_name="cbmvnet-gc-F8-RMSp-sfepo26-ethad200epo300-4dsConv-k5"
		#tmp_name="gcnet-F8-RMSp-sfepo30-ethad200epo300-4dsConv-k5"
		#-------------------------------
		#----sf+kt12+eth-ad-200 images ------
		#-------------------------------
		#tmp_name="cbmvnet-gc-F8-RMSp-sfepo26-ethad200kt12epo300-4dsConv-k5"
		#tmp_name="gcnet-F8-RMSp-sfepo30-ethad200kt12epo300-4dsConv-k5"

		
		resultdir="/home/ccj/GCNet/results/${tmp_name}-testMBV3/disp-epo-$(printf "%03d" "$epo")/trainingH/"
		if [ ! -d $resultdir ]; then
			resultdir="/home/ccj/GCNet/results/${tmp_name}-testMBV3/disp-epo-$(printf "%03d" "$epo")/"
		fi
		echo $resultdir
		if [ -f "${resultdir}mbv3-err.csv" ]; then
			rm "${resultdir}mbv3-err.csv"
		fi
		python3.7 eval_disp.py --dataroot=$dataroot --mbv3badthresh=$mbv3badthresh --ktimgdir=$ktimgdir --resultdir=$resultdir --file=$file_txt --dataset=$data
		cat "${resultdir}mbv3-err.csv" >> "/home/ccj/GCNet/results/bad-err-evalu.csv"
	done
fi


#-----------------------------------
#### evaluating MBV3 Q-size;
#### Either using F-size as ground truth and bad-2.0 as threshold;
#### or using Q-size as ground truth and bad-0.5 ad threshold;
#-----------------------------------
#flag=true
flag=false
if [ "$flag" = true ]; then
	for t in trainingQ
	do
		for epo in $(seq 1 10)
		do
			data='mbv3q'
			mbv3gt=''
			mbv3badthresh=2.0
			#mbv3gt="$dataroot/datasets/MiddleBury/MiddEval3/trainingQ/"
			#mbv3badthresh=0.5
			resultdir="/home/ccj/GCNet/results/cbmvgc-sfepo10-F8-RMSp-4dsConv-k5-crop608x320_d192-testMBV3/disp-epo-$(printf "%03d" "$epo")/$t/"
			echo $resultdir
			if [ -f "${resultdir}mbv3-err.csv" ]; then
				rm "${resultdir}mbv3-err.csv"
			fi
			python3.7 eval_disp.py --dataroot=$dataroot --mbv3gt=$mbv3gt --mbv3badthresh=$mbv3badthresh --ktimgdir=$ktimgdir --resultdir=$resultdir --file=$file_txt --dataset=$data
			cat "${resultdir}mbv3-err.csv" >> "/home/ccj/GCNet/results/bad-err-evalu.csv"
		done
	done
fi

#for i in $(seq 145 149)
#for i in 20 60 80 100 120 140 160 180 200 213 220 240 250 260 263 280 300
#for i in 10 20 50 80 100 150 160 200
