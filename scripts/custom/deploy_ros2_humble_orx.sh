#!/bin/bash

xhost +

# Determine platform architecture
PLATFORM="$(uname -m)"
if [[ $PLATFORM == "aarch64" ]]; then
    PLATFORM_NAME="jetson"
elif [[ $PLATFORM == "x86_64" ]]; then
    PLATFORM_NAME="desktop"
else
    echo "Unsupported platform: $PLATFORM"
    exit 1
fi

docker run --rm -it --gpus all --runtime=nvidia \
    --privileged \
    --network host \
    -e ROS_DOMAIN_ID=1 \
    -e ROS_ROOT=/opt/ros/humble \
    --user admin \
    --workdir /home/admin \
    vschorp98/orx-middleware-isaac-ros-"$PLATFORM_NAME"-ros2_humble \
    bash