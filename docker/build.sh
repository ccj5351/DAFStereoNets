#!/bin/bash

#> See: Verify the version of ubuntu running in a Docker container,
#> at https://stackoverflow.com/questions/38003194/verify-the-version-of-ubuntu-running-in-a-docker-container;

#It's simple variables that are shell script friendly, so you can run

if [ ! -f /etc/lsb-release ]; then
	echo "lsb-release missing, unlikely to be a Ubuntu system"
	exit 1
fi

. /etc/lsb-release
#For the single-bracket syntax, you can use “-a” for and and “-o” for or.
if [ "$DISTRIB_ID" = "Ubuntu" -a "$DISTRIB_RELEASE" = "16.04" ]; then
	echo "Linux install appears to be Ubuntu 16.04"
	UBUNTU_VERSION="16.04"
	#CUDA_VERSION="9.2"
	#CUDA_VERSION="10.2"
	CUDA_VERSION="10.0"
	PYTHON_VERSION="3.7"
	PYTORCH_VERSION="1.2"
elif [ "$DISTRIB_ID" = "Ubuntu" -a "$DISTRIB_RELEASE" = "18.04" ]; then
	echo "Linux install appears to be Ubuntu 18.04"
	UBUNTU_VERSION="18.04"
	PYTHON_VERSION="3.7"
	CUDA_VERSION="10.0"
	PYTORCH_VERSION="1.2"
	#PYTORCH_VERSION="1.5"
	#CUDA_VERSION="10.2"
else
	echo "Linux install doesn't appear to be Ubuntu 16.04 or 18.04"
	exit 1
fi

PYTORCH_VERSION_MAJOR=${PYTORCH_VERSION}
CUDA_VERSION_MAJOR=${CUDA_VERSION}
echo "UBUNTU_VERSION=${UBUNTU_VERSION}, CUDA_VERSION=${CUDA_VERSION}, PYTHON_VERSION=${PYTHON_VERSION}, PyTorch=${PYTORCH_VERSION}"
echo "PYTORCH_VERSION_MAJOR=${PYTORCH_VERSION_MAJOR}"
echo "CUDA_VERSION_MAJOR=${CUDA_VERSION_MAJOR}"
echo "current user is: ${USER}"
#exit

docker build --tag changjiang/atten_stereo:1.0 \
	--build-arg CUDA_VERSION=${CUDA_VERSION} \
	--build-arg username=${USER} \
	--build-arg PYTHON_VERSION=${PYTHON_VERSION} \
	--build-arg PYTORCH=${PYTORCH_VERSION} \
	--build-arg CUDA_VERSION_MAJOR=${CUDA_VERSION_MAJOR} \
	--build-arg PYTORCH_VERSION_MAJOR=${PYTORCH_VERSION_MAJOR} \
	--build-arg UBUNTU_VERSION=${UBUNTU_VERSION} .
# could add :
# --no-cache .
