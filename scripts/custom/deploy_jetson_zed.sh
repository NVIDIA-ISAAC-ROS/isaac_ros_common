#!/bin/bash

# Set default absolute path for the config file
default_config_path="/home/vschorp/dev/orx/data/experiment_config/datahub_01/zed_mini_0"

# Use the first argument as the config path, or the specified default path
config_path="${1:-$default_config_path}"

# Check if the config file exists
if [ ! -f "$config_path" ]; then
    echo "Configuration file not found at: $config_path"
    exit 1
fi

docker run --rm -it --gpus all --runtime=nvidia \
    -e ROS_DOMAIN_ID=1 \
    -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
    -e CYCLONEDDS_URI=/cyclone_profile.xml \
    --network host \
    -v /home/vschorp/dev/cyclone_profile.xml:/cyclone_profile.xml \
    -v /dev/input:/dev/input \
    -v "/usr/local/zed/settings:/usr/local/zed/settings" \
    -v "/usr/local/zed/resources:/usr/local/zed/resources" \
    -v "$config_path":/zed_mini_ros_config.yaml \
    --privileged \
    vschorp98/orx-middleware-isaac-ros-jetson-zed