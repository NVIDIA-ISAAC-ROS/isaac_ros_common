#!/bin/bash

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

# Set default absolute path for the config file
default_config_path="/home/$USER/dev/orx/rosbag_recorder_config"

# Use the first argument as the config path, or the specified default path
config_path="${1:-$default_config_path}"

# Check if the config dir exists
if [ ! -d "$config_path" ]; then
    echo "Configuration dir not found at: $config_path"
    exit 1
fi

DOCKER_IMAGE_NAME=girf/orx-middleware-isaac-ros-"$PLATFORM_NAME"-rosbag_mp4_exporter
echo "Running: $DOCKER_IMAGE_NAME with user $DOCKER_USER"

docker_name="rosbag_mp4_exporter"

docker run --rm -it --gpus all --runtime=nvidia \
    --name $docker_name \
    --privileged \
    --network host \
    --group-add 1009 \
    -e USERNAME=$DOCKER_USER \
    -e ROS_ROOT=/opt/ros/humble \
    -e ROS_DOMAIN_ID=1 \
    -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
    -e ROSBAG_RECORDER_CONFIG_FPATH=/home/"$DOCKER_USER"/config/config.yaml \
    -e CYCLONEDDS_URI=/home/"$DOCKER_USER"/cyclone_profile.xml \
    -v /home/"$USER"/dev/orx/cyclone_profile.xml:/home/"$DOCKER_USER"/cyclone_profile.xml \
    -v /home/"$USER"/dev/orx/data:/home/"$DOCKER_USER"/data \
    -v $config_path:/home/"$DOCKER_USER"/config \
    --user $DOCKER_USER \
    --workdir /home/"$DOCKER_USER" \
    $DOCKER_IMAGE_NAME \
    bash -lc "source /opt/ros/humble/setup.bash && ros2 launch rosbag_mp4_exporter export_mp4.launch.py config:=/home/$DOCKER_USER/config/config.yaml"

