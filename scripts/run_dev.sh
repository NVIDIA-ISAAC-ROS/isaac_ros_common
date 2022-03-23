#!/bin/bash
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

# Get the entry in the dpkg status file corresponding to the provied package name
# Prepend two newlines so it can be safely added to the end of any existing
# dpkg/status file.
function get_dpkg_status() {
    echo -e "\n"
    awk '/Package: '"$1"'/,/^$/' /var/lib/dpkg/status
}

ISAAC_ROS_DEV_DIR="$1"
if [[ -z "$ISAAC_ROS_DEV_DIR" ]]; then
    ISAAC_ROS_DEV_DIR="$HOME/workspaces/isaac_ros-dev"
    if [[ ! -d "$ISAAC_ROS_DEV_DIR" ]]; then
        ISAAC_ROS_DEV_DIR=$(pwd)
    fi
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

# Prevent running as root.
if [[ $(id -u) -eq 0 ]]; then
    print_error "This script cannot be executed with root privileges."
    print_error "Please re-run without sudo and follow instructions to configure docker for non-root user if needed."
    exit 1
fi

# Check if user can run docker without root.
RE="\<docker\>"
if [[ ! $(groups $USER) =~ $RE ]]; then
    print_error "User |$USER| is not a member of the 'docker' group and cannot run docker commands without sudo."
    print_error "Run 'sudo usermod -aG docker \$USER && newgrp docker' to add user to 'docker' group, then re-run this script."
    print_error "See: https://docs.docker.com/engine/install/linux-postinstall/"
    exit 1
fi

# Check if able to run docker commands.
if [[ -z "$(docker ps)" ]] ;  then
    print_error "Unable to run docker commands. If you have recently added |$USER| to 'docker' group, you may need to log out and log back in for it to take effect."
    print_error "Otherwise, please check your Docker installation."
    exit 1
fi    
    
PLATFORM="$(uname -m)"

BASE_NAME="isaac_ros_dev-$PLATFORM"
CONTAINER_NAME="$BASE_NAME-container"

# Remove any exited containers.
if [ "$(docker ps -a --quiet --filter status=exited --filter name=$CONTAINER_NAME)" ]; then
    docker rm $CONTAINER_NAME > /dev/null
fi

# Re-use existing container.
if [ "$(docker ps -a --quiet --filter status=running --filter name=$CONTAINER_NAME)" ]; then
    print_info "Attaching to running container: $CONTAINER_NAME"
    docker exec -i -t -u admin --workdir /workspaces/isaac_ros-dev $CONTAINER_NAME /bin/bash $@
    exit 0
fi

# Arguments for docker build
BUILD_ARGS+=("--build-arg" "USERNAME="admin"")
BUILD_ARGS+=("--build-arg" "USER_UID=`id -u`")
BUILD_ARGS+=("--build-arg" "USER_GID=`id -g`")

# Check if GPU is installed
if [[ $PLATFORM == "x86_64" ]]; then
    if type nvidia-smi &>/dev/null; then
        GPU_ATTACHED=(`nvidia-smi -a | grep "Attached GPUs"`)
        if [ ! -z $GPU_ATTACHED ]; then
            BUILD_ARGS+=("--build-arg" "HAS_GPU="true"")
        fi
    fi
fi

if [[ "$PLATFORM" == "aarch64" ]]; then
    # Get information about the user's installed cuda packages
    BUILD_ARGS+=("--build-arg" "DPKG_STATUS=$(get_dpkg_status cuda-cudart-10-2)$(get_dpkg_status libcufft-10-2)")

    # Make sure the nvidia docker runtime will be used for builds
    DEFAULT_RUNTIME=$(docker info | grep "Default Runtime: nvidia" ; true)
    if [[ -z "$DEFAULT_RUNTIME" ]]; then
        print_error "Default docker runtime is not nvidia!, please make sure the following line is"
        print_error "present in /etc/docker/daemon.json"
        print_error '"default-runtime": "nvidia",'
        print_error ""
        print_error "And then restart the docker daemon"
        exit 1
    fi
fi

# Build image
print_info "Building $PLATFORM base as image: $BASE_NAME"
docker build -f $ROOT/../docker/Dockerfile.$PLATFORM.base \
    -t $BASE_NAME \
    "${BUILD_ARGS[@]}" \
    $ROOT/../docker

# Map host's display socket to docker
DOCKER_ARGS+=("-v /tmp/.X11-unix:/tmp/.X11-unix")
DOCKER_ARGS+=("-e DISPLAY")
DOCKER_ARGS+=("-e NVIDIA_VISIBLE_DEVICES=all")
DOCKER_ARGS+=("-e NVIDIA_DRIVER_CAPABILITIES=all")

if [[ $PLATFORM == "aarch64" ]]; then
    DOCKER_ARGS+=("-v /usr/bin/tegrastats:/usr/bin/tegrastats")
    DOCKER_ARGS+=("-v /tmp/argus_socket:/tmp/argus_socket")
    DOCKER_ARGS+=("-v /usr/lib/aarch64-linux-gnu/tegra:/usr/lib/aarch64-linux-gnu/tegra")
    DOCKER_ARGS+=("-v /usr/src/jetson_multimedia_api:/usr/src/jetson_multimedia_api")
    DOCKER_ARGS+=("-v /opt/nvidia/nsight-systems-cli:/opt/nvidia/nsight-systems-cli")
    DOCKER_ARGS+=("--pid=host")
fi

# Optionally load custom docker arguments from file
DOCKER_ARGS_FILE="$ROOT/.isaac_ros_dev-dockerargs"
if [[ -f "$DOCKER_ARGS_FILE" ]]; then
    print_info "Using additional Docker run arguments from $DOCKER_ARGS_FILE"
    readarray -t DOCKER_ARGS_FILE_LINES < $DOCKER_ARGS_FILE
    for arg in "${DOCKER_ARGS_FILE_LINES[@]}"; do
        DOCKER_ARGS+=($(eval "echo $arg | envsubst"))
    done
fi

# Run container from image
print_info "Running $CONTAINER_NAME"
docker run -it --rm \
    --privileged --network host \
    ${DOCKER_ARGS[@]} \
    -v $ISAAC_ROS_DEV_DIR:/workspaces/isaac_ros-dev \
    -v /dev/shm:/dev/shm \
    --name "$CONTAINER_NAME" \
    --runtime nvidia \
    --user="admin" \
    --entrypoint /usr/local/bin/scripts/workspace-entrypoint.sh \
    --workdir /workspaces/isaac_ros-dev \
    $@ \
    $BASE_NAME \
    /bin/bash
