#!/bin/bash


#Get platform
PLATFORM="$(uname -m)"

# Build ROS dependency

echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> ~/.bashrc

source /opt/ros/${ROS_DISTRO}/setup.bash

# Restart udev daemon

sudo service udev restart

colcon build  --continue-on-error --packages-select \
    backend_msgs \
    backend_ui_server \
    can_ros_nodes \
    custom_nitros_image \
    custom_nitros_string \
    depthai_bridge \
    depthai_descriptions \
    depthai_examples \
    depthai_filters \
    depthai-ros \
    depthai_ros_driver \
    depthai_ros_msgs \
    isaac_ros_bi3d_interfaces \
    isaac_ros_common \
    isaac_ros_depth_image_proc \
    isaac_ros_detectnet \
    isaac_ros_dnn_image_encoder \
    isaac_ros_gxf \
    isaac_ros_image_pipeline \
    isaac_ros_image_proc \
    isaac_ros_managed_nitros \
    isaac_ros_nitros \
    isaac_ros_nitros_bridge_interfaces \
    isaac_ros_nitros_camera_info_type \
    isaac_ros_nitros_compressed_image_type \
    isaac_ros_nitros_detection2_d_array_type \
    isaac_ros_nitros_detection3_d_array_type \
    isaac_ros_nitros_disparity_image_type \
    isaac_ros_nitros_imu_type \
    isaac_ros_nitros_image_type \
    isaac_ros_nitros_interfaces \
    isaac_ros_nitros_occupancy_grid_type \
    isaac_ros_nitros_odometry_type \
    isaac_ros_nitros_point_cloud_type \
    isaac_ros_nitros_pose_array_type \
    isaac_ros_nitros_pose_cov_stamped_type \
    isaac_ros_nitros_std_msg_type \
    isaac_ros_nitros_tensor_list_type \
    isaac_ros_pointcloud_interfaces \
    isaac_ros_stereo_image_proc \
    isaac_ros_tensor_list_interfaces \
    isaac_ros_tensor_rt \
    isaac_ros_test \
    isaac_ros_triton \
    isaac_ros_visual_slam \
    isaac_ros_visual_slam_interfaces \
    isaac_ros_yolov8 \
    isaac_slam_saver \
    microcdr \
    micro_ros_agent \
    micro_ros_msgs \
    microxrcedds_client \
    mmc_ui_msgs \
    xacro \

    # Skip these packages for now

    # isaac_ros_apriltag_interfaces \
    # isaac_ros_nitros_april_tag_detection_array_type \
    # isaac_ros_nitros_battery_state_type \
    # isaac_ros_nitros_correlated_timestamp_type \
    # isaac_ros_nitros_encoder_ticks_type \
    # isaac_ros_nitros_flat_scan_type \
    # isaac_ros_nitros_twist_type \
    # isaac_ros_nova_interfaces \
    # isaac_ros_nvblox \
    # network_performance_measurement \
    # nvblox \
    # nvblox_cpu_gpu_tools \
    # nvblox_examples_bringup \
    # nvblox_image_padding \
    # nvblox_isaac_sim \
    # nvblox_msgs \
    # nvblox_nav2
    # nvblox_performance_measurement \
    # nvblox_performance_measurement_msgs \
    # nvblox_ros \
    # nvblox_ros_common \
    # nvblox_rviz_plugin \
    # semantic_label_conversion \
    # odometry_flattener \
    # realsense2_camera \
    # realsense2_camera_msgs \
    # realsense2_description \
    # realsense_splitter \


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