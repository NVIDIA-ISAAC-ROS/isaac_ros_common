#!/bin/bash

# Set default absolute path for the config file
default_config_path="/home/vschorp/dev/orx/data/experiment_config/datahub_01/intel_realsense_d405_0"

# Use the first argument as the config path, or the specified default path
config_path="${1:-$default_config_path}"

# Check if the config file exists
if [ ! -f "$config_path" ]; then
    echo "Configuration file not found at: $config_path"
    exit 1
fi

docker run --rm -it --gpus all --runtime=nvidia \
    --network host \
    -e ROS_DOMAIN_ID=1 \
    -v /dev/input:/dev/input \
    -v "$config_path":/intel_realsense_d405_ros_config.yaml \
    --privileged \
    vschorp98/orx-middleware-isaac-ros-jetson-realsense