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

# Set default absolute path for the config file
default_config_path="/home/$USER/dev/orx/data/experiment_config/datahub_01/intel_realsense_d435_0"

# Use the first argument as the config path, or the specified default path
config_path="${1:-$default_config_path}"

# Check if the config file exists
if [ ! -f "$config_path" ]; then
    echo "Configuration file not found at: $config_path"
    exit 1
fi

docker_name=$(basename ${config_path})_encoder

docker run --rm -it --gpus all --runtime=nvidia \
    --name $docker_name \
    --privileged \
    --network host \
    --cpus 4 \
    -e ROS_DOMAIN_ID=1 \
    -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
    -e CYCLONEDDS_URI=/home/admin/cyclone_profile.xml \
    -v /home/"$USER"/dev/orx/cyclone_profile.xml:/home/admin/cyclone_profile.xml \
    -v /dev/input:/dev/input \
    -v "$config_path":/intel_realsense_d435_ros_config.yaml \
    girf/orx-middleware-isaac-ros-"$PLATFORM_NAME"-realsense_d435_encoder