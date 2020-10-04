#!/bin/bash
#echo "Hi, I'm sleeping for 1 seconds..."
#sleep 1s 
#echo "all Done."
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

#-----------------------------------
#### Attention stero matching: evaluating KT15;
#-----------------------------------
flag=true
#flag=false
if [ "$flag" = true ]; then
	declare -a disps_to_eval=(
	 'asn-embed-k9-d2-D192-dispnetc-sfepo20-kt15epo600-embedlossW-0.06-lr-0.001-p-eposteps-300'
	 #'asn-embed-k7-d1-D192-dispnetc-sfepo20-kt15epo600-embedlossW-0.06-lr-0.001-p-eposteps-300'
	 #'asn-embed-k7-d2-D192-dispnetc-sfepo20-kt15epo600-embedlossW-0.06-lr-0.001-p-eposteps-300'
	 #'asn-embed-k5-d1-D192-dispnetc-sfepo20-kt15epo600-embedlossW-0.06-lr-0.001-p-eposteps-300'
	 #'asn-embed-k5-d2-D192-dispnetc-sfepo20-kt15epo600-embedlossW-0.06-lr-0.001-p-eposteps-300'
	 #'asn-embed-k3-d1-D192-dispnetc-sfepo20-kt15epo600-embedlossW-0.06-lr-0.001-p-eposteps-300'
	 )
	
	declare -a ALL_EPOS_TEST=(25 50 75 100 125 150 175 200 225 250 275 300 325 350 375 400 425 450 475 500 525 550 575 600 625 650 675 700 725 750 775 800 )
	#declare -a ALL_EPOS_TEST=(250)
	#for idx in 0 
	for idx in 0
	do
		#tmp_name=${disps_to_eval[idx]}
		for epo_idx in $(seq 0 23)
		do
			EPO_TEST=${ALL_EPOS_TEST[epo_idx]}
			tmp_name="${disps_to_eval[idx]}/disp-epo-$(printf "%03d" "$EPO_TEST")/val-30"
			dst_name="${disps_to_eval[idx]}/disp-epo-$(printf "%03d" "$EPO_TEST")"
			#resultdir="/home/${USER}/Downloads/${dst_name}"
			resultdir="~/Downloads/${dst_name}"
		  echo $tmp_name "," $resultdir
			scp -r changjiang@10.145.83.35:/diskb/ccjData2/atten-stereo/results/${tmp_name} ${resultdir}
		done
  done
fi
