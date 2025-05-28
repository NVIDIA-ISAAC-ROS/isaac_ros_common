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
default_config_path="/home/$USER/dev/orx/data/experiment_config/datahub_03/depthai_oakdprow_0"

# Use the first argument as the config path, or the specified default path
config_path="${1:-$default_config_path}"

# Check if the config file exists
if [ ! -f "$config_path" ]; then
    echo "Configuration file not found at: $config_path"
    exit 1
fi

# # Run this the first time when deploying the DepthAI camera
# default_udev_rules_path="/home/$USER/dev/orx/orx_middleware/isaac_ros_common/docker/udev_rules/80-movidius.rules"
# # Check if the udev rules file exists
# if [ ! -f "$default_udev_rules_path" ]; then
#     echo "Udev rules file not found at: $default_udev_rules_path"
#     exit 1
# fi
# # Copy udev rules to the system
# sudo cp "$default_udev_rules_path" /etc/udev/rules.d/80-movidius.rules
# # Reload udev rules
# sudo udevadm control --reload-rules
# # Trigger udev to apply the new rules
# sudo udevadm trigger

docker_name=$(basename ${config_path})

docker run --rm -it\
    --name $docker_name \
    --privileged \
    --network host \
    -e ROS_DOMAIN_ID=1 \
    -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
    -e CYCLONEDDS_URI=/home/admin/cyclone_profile.xml \
    -v /home/$USER/git/orx_middleware/cyclone_profile.xml:/home/admin/cyclone_profile.xml \
    -v "$config_path":/home/admin/depthai_cam.yaml \
    -v /dev/:/dev/ \
    girf/orx-middleware-isaac-ros-"$PLATFORM_NAME"-depthai \

