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
    -e ROS_ROOT=/opt/ros/humble \
    -e ROS_DOMAIN_ID=1 \
    -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
    -e CYCLONEDDS_URI=/home/admin/cyclone_profile.xml \
    -v /home/vschorp/dev/orx/cyclone_profile.xml:/home/admin/cyclone_profile.xml \
    -v /home/vschorp/dev/orx/orx_middleware/ros2_ws:/home/admin/ros2_ws \
    -v /home/vschorp/dev/orx/data/data_saver_dump:/home/admin/data_saver_dump \
    vschorp98/orx-middleware-isaac-ros-"$PLATFORM_NAME"-data_saver \