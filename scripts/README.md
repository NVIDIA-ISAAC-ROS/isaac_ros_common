# Isaac ROS Dev Build Scripts

For Jetson or x86_64:  
  `run_dev.sh` creates a dev environment with ROS 2 installed. By default, the directory `/workspaces/isaac_ros-dev` in the container is mapped from `~/workspaces/isaac_ros-dev` on the host machine if it exists OR the current working directory from where the script was invoked otherwise. The host directory the container maps to can be explicitly set by running the script with the desired path as the first argument:
  `run_dev.sh <path to workspace>`
