# Isaac Sim Setup for Hardware in the Loop (HIL)

The following instructions are for a setup where we can run a sample on a Jetson device and Isaac Sim on a x86 machine. We will use the `ROS_DOMAIN_ID` environment variable to have separate logical networks for Isaac Sim and the sample application.

Hardware in the Loop (HIL) refers to a configuration where the software is being tested on the target hardware platform. For example, Isaac ROS packages being tested on Jetson before deployment on Jetson is HIL.

> **Note**: Make sure to set the `ROS_DOMAIN_ID` variable before executing any of the ROS commands.

1. Complete the Quickstart section in the main README of the package of interest.
2. Install Isaac Sim and Nucleus following steps 1-2 of the [Isaac Sim Software in the Loop (SIL) guide](./isaac-sim-sil-setup.md).
3. Open the location of the Isaac Sim package in the terminal by clicking the [**Open in Terminal**](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/user_interface_launcher.html) button.

   <div align="center"><img src="../resources/Isaac_sim_app_launcher.png" width="400px"/></div>
4. In the terminal opened by the previous step, set the `ROS_DOMAIN_ID` as shown:

   ```bash
   export ROS_DOMAIN_ID=<some_number>
   ```

5. Launch Isaac Sim from the script as shown:

   ```bash
   ./isaac-sim.sh
   ```

   <div align="center"><img src="../resources/Isaac_sim_app_terminal.png" width="600px"/></div>

   > Make sure to set the `ROS_DOMAIN_ID` variable before running the sample application.

6. Continue from step 4 of the [Isaac Sim Software in the Loop (SIL) guide](./isaac-sim-sil-setup.md).
