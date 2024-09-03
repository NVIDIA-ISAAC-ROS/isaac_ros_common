#!/bin/bash
#
# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# tmux
git clone https://github.com/tmux-plugins/tpm ~/.tmux/plugins/tpm
wget https://raw.githubusercontent.com/SasaKuruppuarachchi/SasaKuruppuarachchi/main/.tmux.conf -P ~/
echo "alias runagipix='cd /workspaces/isaac_ros-dev/src/isaac_ros_common && ./run.sh'" >> ~/.bashrc
echo "alias runas2='cd /workspaces/aerostack2_ws/src/aerostack2/as2_aerial_platforms/project_agipix/ && ./launch_as2.bash -s -t -g'" >> ~/.bashrc
echo "alias stopas2='cd /workspaces/aerostack2_ws/src/project_agipix/ && ./stop.bash'" >> ~/.bashrc
echo "alias sorpx4='source /workspaces/aerostack2_ws/install/setup.bash'" >> ~/.bashrc
echo "export AEROSTACK2_WORKSPACE=/workspaces/aerostack2_ws" >> ~/.bashrc
echo "export PX4_FOLDER=/workspaces/aerostack2_ws/src/thirdparty/PX4-Autopilot" >> ~/.bashrc

echo "alias runagi='cd /workspaces/agipix_control/src/px4_ros2_offboard/tmux/ && ./start.sh -s'" >> ~/.bashrc
echo "alias sorcon='source /workspaces/agipix_control/install/setup.bash'" >> ~/.bashrc
echo "alias bilcon='cd /workspaces/agipix_control && colcon build'" >> ~/.bashrc

# Build ROS dependency
echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> ~/.bashrc
source /opt/ros/${ROS_DISTRO}/setup.bash

#tentative
sudo apt-get update
rosdep update
# curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
# python get-pip.py
cd /workspaces/dds/Micro-XRCE-DDS-Agent/build && sudo make install && sudo ldconfig /usr/local/lib/
cd /workspaces/isaac_ros-dev
#rosdep fix-permissions
#rosdep install -y -r -q --from-paths src --ignore-src

# Restart udev daemon
sudo service udev restart

$@
