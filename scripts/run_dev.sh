#!/bin/bash -e
#
# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
source $ROOT/utils/print_color.sh

function usage() {
    print_info "Usage: run_dev.sh" {isaac_ros_dev directory path OPTIONAL}
    print_into "Copyright (c) 2021, NVIDIA CORPORATION."
}

ISAAC_ROS_DEV_DIR="$1"
if [[ -z "$ISAAC_ROS_DEV_DIR" ]]; then
    ISAAC_ROS_DEV_DIR="$HOME/workspaces/isaac_ros-dev"
    print_warning "isaac_ros_dev not specified, assuming $ISAAC_ROS_DEV_DIR"
else
    shift 1
fi

ON_EXIT=()
function cleanup {
    for command in "${ON_EXIT[@]}"
    do
        $command
    done
}
trap cleanup EXIT

pushd .
cd $ROOT
ON_EXIT+=("popd")

PLATFORM="$(uname -m)"

BASE_NAME="isaac_ros_dev-$PLATFORM"
CONTAINER_NAME="$BASE_NAME-container"

# Arguments for docker build
BUILD_ARGS+=("--build-arg USERNAME="admin"")
BUILD_ARGS+=("--build-arg USER_UID=`id -u`")
BUILD_ARGS+=("--build-arg USED_GID=`id -g`")

# Check if GPU is installed
if [[ $PLATFORM == "x86_64" ]]; then
    if type nvidia-smi &>/dev/null; then
        GPU_ATTACHED=(`nvidia-smi -a | grep "Attached GPUs"`)
        if [ ! -z $GPU_ATTACHED ]; then
            BUILD_ARGS+=("--build-arg HAS_GPU="true"")
        fi
    fi
fi

# Build image
print_info "Building $PLATFORM base as image: $BASE_NAME"
docker build -f $ROOT/../docker/Dockerfile.$PLATFORM.base \
    -t $BASE_NAME \
    ${BUILD_ARGS[@]} \
    $ROOT/../docker

# Map host's display socket to docker
DOCKER_ARGS+=("-v /tmp/.X11-unix:/tmp/.X11-unix")
DOCKER_ARGS+=("-e DISPLAY")
DOCKER_ARGS+=("-e NVIDIA_VISIBLE_DEVICES=all")
DOCKER_ARGS+=("-e NVIDIA_DRIVER_CAPABILITIES=all")

if [[ $PLATFORM == "aarch64" ]]; then
    DOCKER_ARGS+=("-v /opt/nvidia:/opt/nvidia")
    DOCKER_ARGS+=("-v /usr/bin/tegrastats:/usr/bin/tegrastats")
    DOCKER_ARGS+=("-v /usr/share/vpi1:/usr/share/vpi1")
    DOCKER_ARGS+=("-v /usr/lib/aarch64-linux-gnu/tegra:/usr/lib/aarch64-linux-gnu/tegra")
    DOCKER_ARGS+=("-v /usr/local/cuda-10.2/targets/aarch64-linux/lib:/usr/local/cuda-10.2/targets/aarch64-linux/lib")
    DOCKER_ARGS+=("-v /usr/lib/aarch64-linux-gnu/tegra-egl:/usr/lib/aarch64-linux-gnu/tegra-egl")
    DOCKER_ARGS+=("-v /usr/lib/aarch64-linux-gnu/libcudnn.so.8.2.1:/usr/lib/aarch64-linux-gnu/libcudnn.so.8.2.1")
    DOCKER_ARGS+=("-v /dev/video*:/dev/video*")
fi

# Run container from image
print_info "Running $CONTAINER_NAME"
docker run -it --rm \
    --privileged --network host \
    ${DOCKER_ARGS[@]} \
    -v $ISAAC_ROS_DEV_DIR:/workspaces/isaac_ros-dev \
    --name "$CONTAINER_NAME" \
    --gpus all \
    --user="admin" \
    --entrypoint /home/admin/workspace-entrypoint.sh \
    $@ \
    $BASE_NAME \
    /bin/bash


