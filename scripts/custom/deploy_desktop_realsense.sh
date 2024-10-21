docker run --rm -it --gpus all --runtime=nvidia \
    -v /dev/input:/dev/input \
    -v /home/vschorp/dev/orx/orx_middleware/orx_interface/config_gui/experiment_config/datahub_01/intel_realsense_d405_0:/intel_realsense_d405_ros_config.yaml \
    --privileged \
    vschorp98/orx-middleware-isaac-ros-desktop-realsense