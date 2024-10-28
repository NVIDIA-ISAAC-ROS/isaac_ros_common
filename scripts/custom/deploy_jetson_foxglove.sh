docker run --rm -it --gpus all --runtime=nvidia \
    -e ROS_ROOT=/opt/ros/humble \
    -e ROS_DOMAIN_ID=1 \
    --privileged \
    -p 8765:8765 \
    --network host \
    vschorp98/orx-middleware-isaac-ros-desktop-foxglove

    # https://app.foxglove.dev/balgrist-orx/dashboard