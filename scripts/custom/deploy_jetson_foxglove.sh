docker run --rm -it --gpus all --runtime=nvidia \
    -e ROS_ROOT=/opt/ros/humble \
    -e ROS_DOMAIN_ID=1 \
    -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
    --network host \
    --privileged \
    -p 8765:8765 \
    vschorp98/orx-middleware-isaac-ros-jetson-foxglove

    # https://app.foxglove.dev/balgrist-orx/dashboard