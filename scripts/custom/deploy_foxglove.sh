#!/bin/bash

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

docker_name=foxglove

docker run --rm -it --gpus all --runtime=nvidia \
    --name $docker_name \
    --privileged \
    --network host \
    -e ROS_ROOT=/opt/ros/humble \
    -e ROS_DOMAIN_ID=1 \
    -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
    -e CYCLONEDDS_URI=/home/admin/cyclone_profile.xml \
    -v /home/$USER/dev/orx/cyclone_profile.xml:/home/admin/cyclone_profile.xml \
    -p 8765:8765 \
    vschorp98/orx-middleware-isaac-ros-"$PLATFORM_NAME"-foxglove

    # https://app.foxglove.dev/balgrist-orx/dashboard