#!/bin/bash
#
# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# Build ROS dependency
echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> ~/.bashrc
source /opt/ros/${ROS_DISTRO}/setup.bash

# sudo apt-get update
# rosdep update

# Restart udev daemon
sudo service udev restart

echo "source /workspaces/isaac_ros-dev/install/setup.bash" >> ~/.bashrc
source /workspaces/isaac_ros-dev/install/setup.bash

# Setup before starting BE server
sudo chown 1000:1000 /usr/config/
pip3 install typing-extensions --upgrade

# Start the applications
#ros2 launch yolox_ros_cpp yolox_tensorrt_jetson.launch.py &
python3 /workspaces/isaac_ros-dev/src/backend_components/backend_ui_server/backend_ui_server/main.py \
    --conn_string_path /usr/config/connection.txt \
    --default_config_path /workspaces/isaac_ros-dev/src/backend_components/backend_ui_server/backend_ui_server/default_machine_config.json \
    --config_path /usr/config/machine_config.json &
ros2 launch micro_ros_agent micro_ros_agent_launch.py
$@
