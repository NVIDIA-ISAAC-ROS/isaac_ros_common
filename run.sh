#!/bin/bash

tmux has-session -t agipix_test
if [ $? == 0 ]; then
    tmux kill-session -t agipix_test
fi
tmux new-session -s agipix_test -n main -d # new session
tmux new-window -n debug -t agipix_test # new window
tmux new-window -n dds -t agipix_test # new window
tmux new-window -n v-slam -t agipix_test # new window
tmux new-window -n sensor_test -t agipix_test # new window

tmux select-window -t main

# divide
tmux split-window -h -t 1
tmux split-window -v -t 1

# run
sleep 1
tmux send-keys -t 1 "" C-m
#tmux send-keys -t 3 "ISAAC_PY sim_launch.py" C-m

#OV

tmux send-keys -t 2 "" C-m




#simulation
tmux select-window -t dds
tmux split-window -v -t 1
tmux send-keys -t 1 "cd /workspaces/dds/Micro-XRCE-DDS-Agent" C-m
tmux send-keys -t 1 "cd build" C-m
tmux send-keys -t 1 "cmake .." C-m
tmux send-keys -t 1 "make" C-m
tmux send-keys -t 1 "sudo make install" C-m
tmux send-keys -t 1 "sudo ldconfig /usr/local/lib/" C-m
tmux send-keys -t 1 "sudo MicroXRCEAgent serial --dev /dev/ttyUSB0 -b 921600" C-m
sleep 1

tmux send-keys -t 2 "cd /workspaces/px4_ros2" C-m
tmux send-keys -t 2 "source install/setup.bash" C-m
tmux send-keys -t 2 "ros2 launch px4_ros_com sensor_combined_listener.launch.py" C-m

#eval
tmux select-window -t sensor_test
tmux split-window -v -t 1
tmux send-keys -t 1 "cd /workspaces/isaac_ros-dev" C-m
tmux send-keys -t 1 "colcon build --symlink-install" C-m
tmux send-keys -t 1 "source install/setup.bash" C-m
tmux send-keys -t 1 "ros2 launch realsense2_camera rs_launch.py pointcloud.enable:=True enable_infra1:=True enable_infra2:=True enable_gyro:=True enable_accel:=True" C-m

tmux send-keys -t 2 "rviz2 -d /workspaces/isaac_ros-dev/src/realsense-ros/realsense2_camera/launch/default.rviz"



#yolact_test
tmux select-window -t v-slam
tmux split-window -v -t 1
tmux send-keys -t 1 "cd /workspaces/isaac_ros-dev" C-m
#tmux send-keys -t 1 "colcon build --symlink-install" C-m
tmux send-keys -t 1 "source install/setup.bash" C-m
tmux send-keys -t 1 "ros2 launch isaac_ros_visual_slam isaac_ros_visual_slam_realsense.launch.py"

tmux send-keys -t 2 "rviz2 -d /workspaces/isaac_ros-dev/src/isaac_ros_visual_slam/isaac_ros_visual_slam/rviz/realsense1.cfg.rviz"

#debug
tmux select-window -t debug
tmux send-keys -t 1 "cd /workspaces/px4_ros2" C-m
tmux send-keys -t 1 "source install/setup.bash" C-m
tmux send-keys -t 1 "ros2 topic list" C-m

tmux select-window -t main
#----------------------------------------------------------------------------
tmux attach -t agipix_test # needed to run
