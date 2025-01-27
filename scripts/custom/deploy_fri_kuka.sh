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
default_config_path="/home/$USER/dev/orx/data/experiment_config/datahub_01/"

# Use the first argument as the config path, or the specified default path
config_path="${1:-$default_config_path}"

# Check if the config dir exists
if [ ! -d "$config_path" ]; then
    echo "Configuration dir not found at: $config_path"
    exit 1
fi

docker_name=$(basename ${config_path})

docker run --rm -it --gpus all --runtime=nvidia \
    --name $docker_name \
    --privileged \
    --network host \
    -e ROS_DOMAIN_ID=1 \
    -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
    -e CYCLONEDDS_URI=/home/admin/cyclone_profile.xml \
    -v /home/"$USER"/dev/orx/cyclone_profile.xml:/home/admin/cyclone_profile.xml \
    -v "$config_path":/home/admin/fri_kuka_config \
    vschorp98/orx-middleware-isaac-ros-"$PLATFORM_NAME"-fri_kuka