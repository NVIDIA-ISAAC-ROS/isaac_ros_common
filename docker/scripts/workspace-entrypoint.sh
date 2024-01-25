#!/bin/bash


#Get platform
PLATFORM="$(uname -m)"

# Build ROS dependency

echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> ~/.bashrc

source /opt/ros/${ROS_DISTRO}/setup.bash

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

#Install can if not already installed
if [ -d "/sys/class/net/can0" ]; then
    echo "CAN Installed"
    ros2 run py_ui_messaging run_msgs &
else
    echo "CAN Controller is not configured on this device!"
fi

# If VS Code is installed
if [[ "$VSCODE" == true ]]; then
    code --install-extension ms-python.python --force --user-data-dir $HOME/.vscode/ 
    code --install-extension codium.codium --force --user-data-dir $HOME/.vscode/
    code --install-extension github.copilot --force --user-data-dir $HOME/.vscode/
    code --install-extension ms-azuretools.vscode-docker --force --user-data-dir $HOME/.vscode/
    code --install-extension github.vscode-pull-request-github --force --user-data-dir $HOME/.vscode/
    code --install-extension eamodio.gitlens --force --user-data-dir $HOME/.vscode/
    code --disable-gpu
fi

ros2 launch micro_ros_agent micro_ros_agent_launch.py &

# Start the applications
ros2 run backend_ui_server server

$@