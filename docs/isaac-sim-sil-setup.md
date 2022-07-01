# Isaac Sim Setup for Software in the Loop (SIL)
> **Note: Isaac Sim 2022.1.0 published on 6/3/2022 does not support ROS2 Humble. Please follow one of the [workarounds](#isaac-sim-202210-workarounds) mentioned below before continuing with the tutorial.**

> **Note:** This tutorial runs a software in the loop simulation. In order to run a hardware in the loop simulation, follow the steps in the [corresponding tutorial](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common/blob/main/docs/isaac-sim-hil-setup.md).

Software in the Loop (SIL) refers to a configuration where the software being tested is not running on the target hardware platform. For example, Isaac ROS packages being tested on x86 before deployment on Jetson is SIL.

1. Install Isaac Sim by following the steps on the [basic installation page](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/install_basic.html). 

    Choose the appropriate working environment:
    - [Native](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/install_basic.html)
    - [Docker or Cloud](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/install_advanced.html)
     
    > **Note:** This tutorial assumes the Native environment for Isaac Sim.

2. Connect to the Nucleus server as shown on the [Nucleus installation page](https://docs.omniverse.nvidia.com/prod_nucleus/prod_nucleus/workstation/installation.html).
<div align="center"><img src="../resources/isaac_sim_nucleus_setup.png" width="800px"/></div>

3. See the [basic installation](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/install_basic.html#isaac-sim-setup-native-workstation-launcher) section to launch Isaac Sim from the app launcher and click on the **Launch** button. Once the launch completes, you should see the below screen:
<div align="center"><img src="../resources/isaac_sim_initial_screen.png" width="800px"/></div>

4. Disable ROS1 bridge and enable the ROS2 bridge as described [here](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/ext_omni_isaac_ros_bridge.html#ros2-bridge).

> **Warning:** Make sure to complete this step before opening a USD file.
<div align="center"><img src="../resources/isaac_sim_ros_bridge.png" width="800px"/></div>

5. Continue with the next steps in your specfic Isaac ROS package tutorial.

## Isaac Sim 2022.1.0 Workarounds

- Run Isaac Sim and the Isaac ROS Packages on different systems. For example, you can run Isaac Sim 2022.1.0 on your laptop and run the Isaac ROS Package on an AGX Xavier. 
    > **Note:** Both systems must be connected to the same network.

- If you are running Isaac Sim 2022.1.0 and the Docker container launched by `isaac_ros_common/scripts/run_dev.sh` on the same machine, please make the following patch to prevent port clashes:
    
    `isaac_ros_common/scripts/run_dev.sh`
    ```diff
    docker run -it --rm \
        --privileged \
    --    --network host \
        ${DOCKER_ARGS[@]} \
        -v $ISAAC_ROS_DEV_DIR:/workspaces/isaac_ros-dev \
    --    -v /dev/shm:/dev/shm \
    --    -v /dev/*:/dev/* \
        --name "$CONTAINER_NAME" \
        --runtime nvidia \
        --user="admin" \
        --entrypoint /usr/local/bin/scripts/workspace-entrypoint.sh \
        --workdir /workspaces/isaac_ros-dev \
        $@ \
        $BASE_NAME \
        /bin/bash
    ```
