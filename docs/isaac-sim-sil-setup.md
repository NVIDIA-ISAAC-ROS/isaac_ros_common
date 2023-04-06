# Isaac Sim Setup for Software in the Loop (SIL)

> **Note**: This tutorial runs a software in the loop simulation. In order to run a hardware in the loop simulation, follow the steps in the [corresponding tutorial](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common/blob/main/docs/isaac-sim-hil-setup.md).

Software in the Loop (SIL) refers to a configuration where the software being tested is not running on the target hardware platform. For example, Isaac ROS packages being tested on x86 before deployment on Jetson is SIL.

1. Install Isaac Sim by following the steps on the [basic installation page](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/install_basic.html).

    Choose the appropriate working environment:
    - [Native](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/install_basic.html)
    - [Docker or Cloud](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/install_advanced.html)

    > **Note**: This tutorial assumes the Native environment for Isaac Sim.

2. Connect to the Nucleus server as shown on the [Nucleus installation page](https://docs.omniverse.nvidia.com/prod_nucleus/prod_nucleus/workstation/installation.html).

    <div align="center"><img src="../resources/isaac_sim_nucleus_setup.png" width="800px"/></div>

3. See the [basic installation](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/install_basic.html#isaac-sim-setup-native-workstation-launcher) section to launch Isaac Sim from the app launcher and click on the **Launch** button. Once the launch completes, you should see the below screen:

    <div align="center"><img src="../resources/isaac_sim_initial_screen.png" width="800px"/></div>

4. Disable ROS bridge and enable the ROS 2 Humble bridge as described [here](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/ext_omni_isaac_ros_bridge.html#ros2-bridge).

    > **Warning**: Make sure to complete this step before opening a USD file.
    <div align="center"><img src="../resources/isaac_sim_ros_bridge.png" width="800px"/></div>

5. Continue with the next steps in your specfic Isaac ROS package tutorial.
