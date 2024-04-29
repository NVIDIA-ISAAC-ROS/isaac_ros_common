#!/bin/bash

echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> ~/.bashrc

source /opt/ros/${ROS_DISTRO}/setup.bash

sudo chown -R admin /workspaces/isaac_ros-dev/

# Patch for launch_testing
sudo cp /workspaces/isaac_ros-dev/src/isaac_ros_common/docker/patches/hooks.py /opt/ros/humble/lib/python3.8/site-packages/launch_testing/pytest/hooks.py

colcon build  --packages-select \
    backend_msgs \
    backend_ui_server \
    can_ros_nodes \
    custom_nitros_image \
    custom_nitros_string \
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
    isaac_ros_nvblox \
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
    map_saver_2d \
    microcdr \
    micro_ros_agent \
    micro_ros_msgs \
    microxrcedds_client \
    mmc_ui_msgs \
    nvblox \
    nvblox_cpu_gpu_tools \
    nvblox_examples_bringup \
    nvblox_image_padding \
    nvblox_isaac_sim \
    nvblox_msgs \
    nvblox_nav2 \
    nvblox_performance_measurement \
    nvblox_performance_measurement_msgs \
    nvblox_ros \
    nvblox_ros_common \
    nvblox_rviz_plugin \
    odometry_flattener \
    realsense2_camera \
    realsense2_camera_msgs \
    realsense2_description \
    realsense_splitter \
    semantic_label_conversion \
    serial_ros_nodes \
    mli_ros_simulator \
    ros_simulator_web_app \

    # Skip these packages for now
    # isaac_ros_apriltag_interfaces \
    # isaac_ros_nitros_april_tag_detection_array_type \
    # isaac_ros_nitros_battery_state_type \
    # isaac_ros_nitros_correlated_timestamp_type \
    # isaac_ros_nitros_encoder_ticks_type \
    # isaac_ros_nitros_flat_scan_type \
    # isaac_ros_nitros_twist_type \
    # isaac_ros_nova_interfaces \
    # network_performance_measurement \
    # depthai_bridge \
    # depthai_descriptions \
    # depthai_examples \
    # depthai_filters \
    # depthai-ros \
    # depthai_ros_driver \
    # depthai_ros_msgs \

echo "source /workspaces/isaac_ros-dev/install/setup.bash" >> ~/.bashrc

source /workspaces/isaac_ros-dev/install/setup.bash

rm -rf /workspaces/isaac_ros-dev/test_results

mkdir -p /workspaces/isaac_ros-dev/test_results

echo Starting Tests...

# -vs to show logs
pytest -v --disable-warnings \
    /workspaces/isaac_ros-dev/src/backend_components/backend_ui_server/backend_ui_server/tests/ \
    /workspaces/isaac_ros-dev/src/configurator/tests/ \
    --junitxml=/workspaces/isaac_ros-dev/test_results/test-result.xml \
    --cov=. \
    --cov-report xml:/workspaces/isaac_ros-dev/test_results/coverage-result.xml
