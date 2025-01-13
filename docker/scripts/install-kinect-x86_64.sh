## Install x86 dependencies for Kinect (Ubuntu 22.04)

# Get libsoundio1 package and add Microsoft repository
curl -sSL https://packages.microsoft.com/keys/microsoft.asc | sudo tee /etc/apt/trusted.gpg.d/microsoft.asc
sudo apt-add-repository https://packages.microsoft.com/ubuntu/22.04/prod
sudo apt-get update
wget mirrors.kernel.org/ubuntu/pool/universe/libs/libsoundio/libsoundio1_1.1.0-1_amd64.deb
sudo dpkg -i libsoundio1_1.1.0-1_amd64.deb

# Install the SDK
sudo apt-get update
sudo apt-get install libk4a1.4
sudo apt install libk4a1.4-dev
sudo apt install k4a-tools

# Cleanup
sudo rm /etc/apt/trusted.gpg.d/microsoft.asc
sudo rm /etc/apt/sources.list.d/microsoft-prod.list
sudo apt-get update

# Everything also needs to be installed on the host system for the docker to work properly
# Uncomment the lines below and run the script on the host system
# Also, run the below to install the driver on the docker container
# Assumes ROS2 Humble is installed on the host system
# Refer to the following links for debugging:
# https://robotics.stackexchange.com/questions/24529/azure-kinect-with-ros2-humble
# https://github.com/microsoft/Azure_Kinect_ROS_Driver/tree/humble
# https://github.com/juancarlosmiranda/azure_kinect_notes

# source /opt/ros/humble/setup.bash
# git clone https://github.com/microsoft/Azure_Kinect_ROS_Driver.git -b humble
# pip3 install xacro
# sudo apt install ros-humble-joint-state-publisher
# cd Azure_Kinect_ROS_Driver
# colcon build 
# source install/setup.bash

# Test your installation by trying to run
# ros2 launch azure_kinect_ros_driver driver.launch.py
# Or to test just the SDK
# k4arecorder -l 5 out.mkv