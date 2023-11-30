#!/bin/bash

#

# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.

#

# NVIDIA CORPORATION and its licensors retain all intellectual property

# and proprietary rights in and to this software, related documentation

# and any modifications thereto. Any use, reproduction, disclosure or

# distribution of this software and related documentation without an express

# license agreement from NVIDIA CORPORATION is strictly prohibited.

#Get platform
PLATFORM="$(uname -m)"

# Build ROS dependency

echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> ~/.bashrc

source /opt/ros/${ROS_DISTRO}/setup.bash

 

#sudo apt-get update

#rosdep update

# If VS Code is installed
if [[ "$VSCODE" == true ]]; then
    code --install-extension ms-python.python --force --user-data-dir $HOME/.vscode/ 
    code --install-extension codium.codium --force --user-data-dir $HOME/.vscode/
    code --install-extension github.copilot --force --user-data-dir $HOME/.vscode/
    code --install-extension ms-azuretools.vscode-docker --force --user-data-dir $HOME/.vscode/
    code --install-extension github.vscode-pull-request-github --force --user-data-dir $HOME/.vscode/
    code --install-extension eamodio.gitlens --force --user-data-dir $HOME/.vscode/
fi
 

# Restart udev daemon

sudo service udev restart

colcon build

echo "source /workspaces/isaac_ros-dev/install/setup.bash" >> ~/.bashrc
source /workspaces/isaac_ros-dev/install/setup.bash

# Setup before starting BE server
sudo chown 1000:1000 /usr/config/
sudo chown 1000:1000 /usr/data/

if [[ "$PLATFORM" == "aarch64" ]]; then
    pip3 install typing-extensions --upgrade
fi

export RUN_DEV=true


# Start the applications
python3 /workspaces/isaac_ros-dev/src/backend_components/backend_ui_server/backend_ui_server/main.py &
   
ros2 launch micro_ros_agent micro_ros_agent_launch.py &

#Install can if not already installed
if [ -d "/sys/class/net/can0" ]; then
    echo "CAN Installed"
    ros2 run py_ui_messaging run_msgs
else
    echo "CAN Controller is not configured on this device!"
fi

$@

# add ! bin bash back here
#
# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# Build ROS dependency
# echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> ~/.bashrc
# source /opt/ros/${ROS_DISTRO}/setup.bash

# # sudo apt-get update
# # rosdep update

# # Restart udev daemon
# sudo service udev restart

# echo "source /workspaces/isaac_ros-dev/install/setup.bash" >> ~/.bashrc
# source /workspaces/isaac_ros-dev/install/setup.bash

# # Setup before starting BE server
# sudo chown 1000:1000 /usr/config/
# pip3 install typing-extensions --upgrade

# # Start the applications
# #ros2 launch yolox_ros_cpp yolox_tensorrt_jetson.launch.py &
# python3 /workspaces/isaac_ros-dev/src/backend_components/backend_ui_server/backend_ui_server/main.py \
#     --conn_string_path /usr/config/connection.txt \
#     --default_config_path /workspaces/isaac_ros-dev/src/backend_components/backend_ui_server/backend_ui_server/default_machine_config.json \
#     --config_path /usr/config/machine_config.json &
# ros2 launch micro_ros_agent micro_ros_agent_launch.py
# $@
