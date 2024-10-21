docker run --rm -it --gpus all --runtime=nvidia \
    -v /dev/input:/dev/input \
    -v /home/vschorp/dev/orx/orx_middleware/orx_interface/config_gui/experiment_config/datahub_01/zed_mini_0:/zed_mini_ros_config.yaml \
    --privileged \
    vschorp98/orx-middleware-isaac-ros-jetson-zed