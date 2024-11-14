#!/bin/bash

xhost +

# Determine platform architecture
PLATFORM="$(uname -m)"
USER_ID=$(id -u)
GROUP_ID=$(id -g)
DOCKER_USER="admin"
if [[ $PLATFORM == "aarch64" ]]; then
    PLATFORM_NAME="jetson"
elif [[ $PLATFORM == "x86_64" ]]; then
    if [[ $USER_ID == 1001 ]]; then
        PLATFORM_NAME="desktop"
    elif [[ $USER_ID == 1003 ]]; then
        PLATFORM_NAME="dgx"
        DOCKER_USER="orx_user"
    else
        echo "Unsupported user id: $USER_ID"
        exit 1
    fi
else
    echo "Unsupported platform: $PLATFORM"
    exit 1
fi

DOCKER_IMAGE_NAME=vschorp98/orx-middleware-isaac-ros-"$PLATFORM_NAME"-ros2_humble
echo "Running: $DOCKER_IMAGE_NAME with user $DOCKER_USER"

docker run --rm -it --gpus all --runtime=nvidia \
    --privileged \
    --network host \
    -e DISPLAY \
    -e ROS_ROOT=/opt/ros/humble \
    -e ROS_DOMAIN_ID=1 \
    -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
    -e CYCLONEDDS_URI=/home/"$DOCKER_USER"/cyclone_profile.xml \
    -v /home/"$USER"/dev/orx/cyclone_profile.xml:/home/"$DOCKER_USER"/cyclone_profile.xml \
    -v /home/"$USER"/dev/orx/data:/home/"$DOCKER_USER"/data \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    --user $DOCKER_USER \
    --workdir /home/"$DOCKER_USER" \
    $DOCKER_IMAGE_NAME \
    bash
