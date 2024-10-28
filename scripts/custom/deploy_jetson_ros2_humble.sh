xhost +
docker run --rm -it --gpus all --runtime=nvidia \
    --privileged \
    --network host \
    -e ROS_DOMAIN_ID=1 \
    -e DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -e ROS_ROOT=/opt/ros/humble \
    --user admin \
    --workdir /home/admin \
    vschorp98/orx-middleware-isaac-ros-jetson-ros2_humble \
    bash