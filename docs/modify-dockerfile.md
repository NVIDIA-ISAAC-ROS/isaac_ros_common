# Customizing Dockerfiles and the Isaac ROS Docker Container

## Change the Config Image Key

Providing a value to the `$CONFIG_IMAGE_KEY` variable in `isaac_ros_common/scripts/.isaac_ros_common-config` before running the `isaac_ros_common/scripts/run_dev.sh` script allows you to select a different base image upon which to build the Isaac ROS layers.

For example, you may wish to bundle a dependency for the RealSense camera as part of the base Docker image. The `isaac_ros_common/docker/realsense-dockerfile-example` folder contains an example setup:

- Associated shell scripts from [JetsonHacks](https://jetsonhacks.com/) that prepare build on `aarch64` and ROS 2 Humble
- `Dockerfile.realsense` that invokes the shell scripts
- `.isaac_ros_common-config` that specifies the new value of `$CONFIG_IMAGE_KEY` to point to the new Dockerfile
    > **NOTE**: This configuration file must be moved to `isaac_ros_common/scripts/` for it to be used while running `run_dev.sh`

Please see [this section](../README.md#configuring-run_devsh) for additional details.
