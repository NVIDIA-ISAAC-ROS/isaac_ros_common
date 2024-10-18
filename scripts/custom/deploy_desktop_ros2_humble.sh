xhost +
docker run --rm -it --gpus all --runtime=nvidia \
    --privileged \
    -e DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -e ROS_ROOT=/opt/ros/humble \
    vschorp98/orx-middleware-isaac-ros-desktop-ros2_humble \
    bash