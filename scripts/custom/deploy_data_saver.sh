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
    -e DISPLAY \
    --network host \
    -e ROS_DOMAIN_ID=1 \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /home/vschorp/dev/orx/orx_middleware/ros2_ws:/home/admin/ros2_ws \
    -v /home/vschorp/dev/orx/data/data_saver_dump:/home/admin/data_saver_dump \
    -e ROS_ROOT=/opt/ros/humble \
    --user admin \
    --workdir /home/admin \
    vschorp98/orx-middleware-isaac-ros-"$PLATFORM_NAME"-data_saver \