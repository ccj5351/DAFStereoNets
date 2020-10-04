# this Dockerfile is adapted from:
# 1) open-mmlab/mmdetection, please see: https://github.com/open-mmlab/mmdetection/blob/master/docker/Dockerfile;
# 2) pytorch/docker/pytorch/Dockerfile, please see: https://github.com/pytorch/pytorch/blob/master/docker/pytorch/Dockerfile;
# 3) anibali/docker-pytorch/dockerfiles, please see: https://github.com/anibali/docker-pytorch/tree/master/dockerfiles;


# > see: https://stackoverflow.com/questions/47459614/docker-argument-to-a-run-echo-command;
# > see: 1) First: Make sure, your ARG comes after your FROM, see details at https://docs.docker.com/engine/reference/builder/#understand-how-arg-and-from-interact;
#           Details: An ARG declared before a FROM is outside of a build stage, 
#                    so it can’t be used in any instruction after a FROM. i
#                    To use the default value of an ARG declared before i
#                    the first FROM use an ARG instruction without a value inside of a build stage;
#        2) Second: variables won't be interpretad inside '', so use "" instead.
ARG PYTORCH="1.5"
# > see: https://hub.docker.com/r/pytorch/pytorch/tags 
# > for available pytorch images;
# > see: How to pass arguments to a Dockerfile? 
# at 1) https://stackoverflow.com/questions/34254200/how-to-pass-arguments-to-a-dockerfile;
#    2) https://docs.docker.com/engine/reference/builder/#arg;
#syn: ARG <name>[=<default value>]
ARG CUDA_VERSION="10.2"
ARG CUDNN="7"
ARG PYTHON_VERSION="3.7"
ARG UBUNTU_VERSION="18.04"
#ARG UBUNTU_VERSION="16.04"
#ARG WITH_TORCHVISION=1

# > see: What is the difference between devel and runtime tag for a Docker container?
# > at https://stackoverflow.com/questions/56405159/what-is-the-difference-between-devel-and-runtime-tag-for-a-docker-container;
# > 1) runtime: extends the base image by adding all the shared libraries from the CUDA toolkit. Use this image if you have a pre-built application using multiple CUDA libraries.
# > 2) devel: extends the runtime image by adding the compiler toolchain, the debugging tools, the headers and the static libraries. Use this image to compile a CUDA application from sources.
#FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

#FROM nvidia/cuda:9.2-base-ubuntu16.04
#FROM nvidia/cuda:10.2-base-ubuntu18.04

#FROM nvidia/cuda:${CUDA_VERSION}-base-ubuntu${UBUNTU_VERSION}
#> See: To get access to the CUDA development tools, you should use the `devel` images instead. 
#       These are the relevant tags:
#        1) nvidia/cuda:10.2-devel-ubuntu18.04
#        2) nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04
#       These would then give you access to the tools such as nvcc and the 
#       cuDNN header files that are required for development.
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION}

#NOTE: To use the default value of an ARG declared before the first FROM 
# use an ARG instruction without a value inside of a build stage:
ARG CUDA_VERSION
ARG UBUNTU_VERSION
ARG PYTHON_VERSION
ARG PYTORCH
RUN echo "We did FROM nvidia/cuda:${CUDA_VERSION}-base-ubuntu${UBUNTU_VERSION}, and PYTHON_VERSION=${PYTHON_VERSION}, PYTORCH=${PYTORCH}"



#ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
#ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
#ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

RUN apt-get update && apt-get install -y git \
        build-essential cmake curl \
        libjpeg-dev libjpeg-dev \
        ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
        vim tmux \
        #required by opencv3
        libgtk2.0-0 libcanberra-gtk-module \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create a working directory
RUN mkdir /app
WORKDIR /app

# Create a non-root user and switch to it
ARG username=ccj
ENV USER $username
ENV HOME /home/$USER
RUN adduser --disabled-password --gecos 'non-root user' --shell /bin/bash --home ${HOME} ${USER} \
 && chown -R ${USER}:${USER} /app
#RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER ${USER}

# All users can use /home/user as their home directory
#RUN chmod 777 ${HOME}
RUN chmod 755 ${HOME}


# Install Miniconda and Python 3.7
ARG CONDA_VERSION=4.8.2
#ARG CONDA_VERSION=latest
ENV CONDA_AUTO_UPDATE_CONDA=false
ENV PATH=${HOME}/miniconda/bin:$PATH
RUN echo "To do: curl -o ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-py37_${CONDA_VERSION}-Linux-x86_64.sh"
RUN curl -o ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-py37_${CONDA_VERSION}-Linux-x86_64.sh && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p ~/miniconda && \
     rm ~/miniconda.sh
RUN echo "To do: conda install -y python=${PYTHON_VERSION} numpy pyyaml scipy ipython mkl mkl-include ninja cython typing"
RUN  conda install -y python=${PYTHON_VERSION} numpy pyyaml scipy ipython mkl mkl-include ninja cython typing && \
     conda clean -ya

# Create a Python 3.7 environment
#RUN conda install conda-build \
# && conda create -y --name py37 python=3.7 \
# && conda clean -ya
#ENV CONDA_DEFAULT_ENV=py37
#ENV CONDA_PREFIX=${HOME}/miniconda/envs/${CONDA_DEFAULT_ENV}
#ENV PATH=${CONDA_PREFIX}/bin:$PATH

# CUDA 9.2-specific steps
#RUN conda install -y -c pytorch \
#    cuda92=1.0 \
#    magma-cuda92=2.3.0 \
#    && conda clean -ya

#RUN echo "To do: conda install pytorch torchvision cudatoolkit=${CUDA_VERSION} -c pytorch"
#RUN conda install pytorch torchvision cudatoolkit=${CUDA_VERSION} -c pytorch
#NOTE: here for cuda=10.2, we can use conda to install pytorch;
# but for cua=9.2, I tried to use conda but failed. So use pip to install pytorch with cuda=9.2;

# The -a and -o known from the single-bracket syntax is supported, too.
# or Note that the and operator has precedence over the or operator, meaning that “&&” or “-a” will be evaluated before “||” or “-o”.

ARG CUDA_VERSION_MAJOR
ARG PYTORCH_VERSION_MAJOR
RUN if [ "$CUDA_VERSION_MAJOR" = "9.2" -a "$PYTORCH_VERSION_MAJOR" = "1.5" ] ; then echo "9.2+1.5 ==> Major CUDA ${CUDA_VERSION_MAJOR} and Major Pytorch ${PYTORCH_VERSION_MAJOR}" \
    && pip install torch==1.5.0+cu92 torchvision==0.6.0+cu92 -f https://download.pytorch.org/whl/torch_stable.html ; else echo "9.2+1.5 != Major CUDA ${CUDA_VERSION_MAJOR} and Major Pytorch ${PYTORCH_VERSION_MAJOR}! Do nothing here!"; fi

RUN if [ "$CUDA_VERSION_MAJOR" = "9.2" -a "$PYTORCH_VERSION_MAJOR" = "1.2" ]; then echo "9.2+1.2 ==> Major CUDA ${CUDA_VERSION_MAJOR} and Major Pytorch ${PYTORCH_VERSION_MAJOR}" \
    && conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=9.2 -c pytorch && conda clean -ya; else echo "9.2+1.2 != Major CUDA ${CUDA_VERSION_MAJOR} and Major Pytorch ${PYTORCH_VERSION_MAJOR}! Do nothing here!"; fi

RUN if [ "$CUDA_VERSION_MAJOR" = "10.0" -a "$PYTORCH_VERSION_MAJOR" = "1.2" ]; then echo "10.2+1.2 ==> Major CUDA ${CUDA_VERSION_MAJOR} and Major Pytorch ${PYTORCH_VERSION_MAJOR}" \
    && conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=10.0 -c pytorch && conda clean -ya; else echo "10.0+1.2 != Major CUDA ${CUDA_VERSION_MAJOR} and Major Pytorch ${PYTORCH_VERSION_MAJOR}! Do nothing here!"; fi


RUN if [ "$CUDA_VERSION_MAJOR" = "10.2" -a "$PYTORCH_VERSION_MAJOR" = "1.5" ] ; then echo "10.2+1.5 ==> Major CUDA ${CUDA_VERSION_MAJOR} and Major Pytorch ${PYTORCH_VERSION_MAJOR}" \
    && conda install pytorch torchvision cudatoolkit=10.2 -c pytorch && conda clean -ya; else echo "10.2+1.5 != Major CUDA ${CUDA_VERSION_MAJOR} and Major Pytorch ${PYTORCH_VERSION_MAJOR}! Do nothing here!"; fi
#RUN pip install torch==1.5.0+cu92 torchvision==0.6.0+cu92 -f https://download.pytorch.org/whl/torch_stable.html

# This must be done before pip so that requirements.txt is available
# Create a working directory
ENV PROJECT_DIR $HOME/atten-stereo
RUN mkdir ${PROJECT_DIR}
WORKDIR ${PROJECT_DIR}

# Copy the requirement file to the continer;
COPY /requirements.txt .

RUN TORCH_CUDA_ARCH_LIST="3.5 5.2 6.0 6.1 7.0+PTX" TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" \
    pip install -vr requirements.txt

RUN rm requirements.txt

RUN pip install Pillow==6.2.1  matplotlib tensorboard future

# if pytorch = 1.5.0, then install kornia==0.3.1(the currently latest version)
# if pytorch = 1.2.0, then install kornia<=0.2.2, here we choose kornia==0.2.2
#RUN pip install kornia==0.2.2
RUN if [ "${PYTORCH_VERSION_MAJOR}" = "1.2" ]; then echo "install pip install kornia==0.2.2" \
    && pip install kornia==0.2.2; else echo "PYTORCH version ${PYTORCH} > 1.2" ; fi
RUN if [ "${PYTORCH_VERSION_MAJOR}" = "1.5" ]; then echo "install pip install kornia==0.3.1" \
    && pip install kornia==0.3.1; else echo "PYTORCH version ${PYTORCH} != 1.5" ; fi

# Set the default command to python3
CMD ["python3"]

#Install the COCO API
#RUN git clone https://github.com/cocodataset/cocoapi.git /cocoapi
#WORKDIR /cocoapi/PythonAPI
#RUN make install
#WORKDIR ${PROJECT_DIR}
#RUN chmod -R a+w .

#ARG WITH_TORCHVISION=0
#RUN if [ "$WITH_TORCHVISION" = "1" ] ; then git clone https://github.com/pytorch/vision.git && cd vision && pip install -v . ; else echo "building without torchvision" ; fi
