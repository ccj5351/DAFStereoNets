#!/bin/bash
DATA_ROOT="/diskb/ccjData2/"

if [ ! -d $DATA_ROOT ];then
  DATA_ROOT="/media/ccjData2"
	echo "Updated : setting data_root = ${DATA_ROOT}"
fi

if [ ! -d $DATA_ROOT ];then
	DATA_ROOT="/data/ccjData"
	echo "Updated : setting data_root = ${DATA_ROOT}"
fi

if [ ! -d $DATA_ROOT ];then
	DATA_ROOT="/home/ccj"
	echo "Updated : setting data_root = ${DATA_ROOT}"
fi

echo "Current user is : ${USER}"
#exit
#docker run --gpus '"device=0,1,2,3"' --rm -it changjiang/image_alignment_ss:1.0 bash
#exit

#NOTE: the error: "groups: cannot find name for group ID 998"
# Solutions:
# > see: https://stackoverflow.com/questions/46018102/how-can-i-use-matplotlib-pyplot-in-a-docker-container
# > see: https://stackoverflow.com/questions/50658883/how-to-correctly-use-system-user-in-docker-container
# > Note: This sort of error will happen when the uid/gid does not exist in the /etc/passwd or /etc/group 
#         file inside the container. There are various ways to work around that. 
#         One is to directly map these files from your host into the container with something like:
#         "-v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro"

#NOTE: ERROR: Unexpected bus error encountered in worker. This might be caused by insufficient shared memory (shm).
#> see: https://github.com/ultralytics/yolov3/issues/283
# "--num-workers 0" will slow down training.
# I fixed it by adding "--ipc=host" in my docker container configuration.

# NOTE: docker run :
# --rm: Automatically remove the container when it exits;
# --tty , -t		Allocate a pseudo-TTY;
# --volume , -v		Bind mount a volume;
# --interactive , -i		Keep STDIN open even if not attached;
# --init : Run an init inside the container that forwards signals and reaps processes;
# > see how to use --init at https://stackoverflow.com/questions/41097652/how-to-fix-ctrlc-inside-a-docker-container/62529065#62529065
echo "running docker"
docker run --gpus '"device=0,1,2,3"' --ipc=host --rm \
	--user=$(id -u $USER):$(id -g $USER) \
	--env="DISPLAY" \
	--volume="/home/${USER}/.Xauthority:/home/${USER}/.Xauthority:rw" --net=host \
	--volume="/home/${USER}/atten-stereo:/home/${USER}/atten-stereo" \
	--volume="/home/${USER}/.bashrc:/home/${USER}/.bashrc" \
	--volume="${DATA_ROOT}/datasets/:/home/${USER}/datasets" \
	--volume="/etc/group:/etc/group:ro" \
	--volume="/etc/passwd:/etc/passwd:ro" \
	--volume="/etc/shadow:/etc/shadow:ro" \
	--volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
	-it --init changjiang/atten_stereo:1.0 bash
