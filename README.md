# Deep Adaptive Filtering (DAF) Stereo Networks: DAF-StereoNets (Code Work in Progress!!!)
[Do End-to-end Stereo Algorithms Under-utilize Information?](https://arxiv.org/abs/2010.07350) 

<img align="center" src="files/network-architecture.png">


## Introduction

### Cost Volume in SOTA Stereo Matching
End-to-end stereo matching methods can be generally grouped into two categories: 
2D CNNs for correlation-based (i.e., generating a 3D cost volume with dimension DxHxW) 
disparity estimation and 3DCNNs for cost volume (i.e., generating a 4D cost volume with 
dimension FxDxHxW) based disparity regression.
The following figure demonstrates the cost volume in 2D and 3D CNNs for stereo matching.
<img align="center" src="files/2D-3D-deep-stereo-nets.png">

## Building Requirements

## How to Use?

## Pretrained Models

## Results

## Reference:
If you find the code useful, please cite our paper:

		@misc{cai2020deep_adaptive_stereo,
			title={Do End-to-end Stereo Algorithms Under-utilize Information?}, 
			author={Changjiang Cai and Philippos Mordohai},
			year={2020},
			eprint={2010.07350},
			archivePrefix={arXiv},
			primaryClass={cs.CV}
		}

