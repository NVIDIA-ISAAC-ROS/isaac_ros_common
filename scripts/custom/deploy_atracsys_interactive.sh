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

# Set default absolute path for the config file
default_config_path="/home/$USER/dev/orx/data/experiment_config/datahub_01/atracsys_fusion_500_0"

# Use the first argument as the config path, or the specified default path
config_path="${1:-$default_config_path}"

# Check if the config dir exists
# if [ ! -d "$config_path" ]; then
#     echo "Configuration dir not found at: $config_path"
#     exit 1
# fi

docker run --rm -it --gpus all --runtime=nvidia \
    --privileged \
    --network host \
    -it \
    -e DISPLAY \
    -e ROS_DOMAIN_ID=1 \
    -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
    -e CYCLONEDDS_URI=/home/admin/cyclone_profile.xml \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /home/"$USER"/dev/orx/cyclone_profile.xml:/home/admin/cyclone_profile.xml \
    -v /home/"$USER"/dev/orx/data:/home/admin/data \
    -v "$config_path":/home/admin/atracsys_fusion_500_config/ \
    girf/orx-middleware-isaac-ros-"$PLATFORM_NAME"-atracsys bash