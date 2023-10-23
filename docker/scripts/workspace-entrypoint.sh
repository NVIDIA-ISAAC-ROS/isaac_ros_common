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

sudo apt-get update
rosdep update
rosdep install -i --from-path src --rosdistro $ROS_DISTRO -y --skip-keys "nvblox"

# Restart udev daemon
sudo service udev restart

# Build and source the workspace
# cd /workspaces/isaac_ros-dev && \
# colcon build --symlink-install && \
# source install/setup.bash

# Automatically launch the apriltags package
#ros2 launch isaac_ros_apriltag isaac_ros_apriltag_realsense.launch.py

# Automatically launch the nvblox package
# ros2 launch nvblox_nav2 nvblox_launch.py

$@
