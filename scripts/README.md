# Isaac ROS Dev Build Scripts


For Jetson or x86_64:  
  `run_dev.sh` creates a dev environment with ROS2 installed. By default, the directory `/workspace` in the container is mapped from `~/workspaces/isaac_ros-dev` on the host machine, but the directory the container is mapping from can be replaced by running the script and passing a path as the first argument:
  `run_dev.sh <path to workspace>`
