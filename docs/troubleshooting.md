# Troubleshooting

## Table of Contents
- [Troubleshooting](#troubleshooting)
  - [Table of Contents](#table-of-contents)
  - [Realsense driver doesn't work with ROS2 Humble](#realsense-driver-doesnt-work-with-ros2-humble)
    - [Symptom](#symptom)
    - [Solution](#solution)
  - [Input images must have even height and width](#input-images-must-have-even-height-and-width)
    - [Symptom](#symptom-1)
    - [Solution](#solution-1)
  - [RealSense `ros2 launch realsense2_camera rs_launch.py` error](#realsense-ros2-launch-realsense2_camera-rs_launchpy-error)
    - [Symptoms](#symptoms)
    - [Solution](#solution-2)
  - [Failed to get valid output from Isaac ROS Rectify or Apriltag Node](#failed-to-get-valid-output-from-isaac-ros-rectify-or-apriltag-node)
    - [Symptom](#symptom-2)
    - [Solution](#solution-3)

## Realsense driver doesn't work with ROS2 Humble
### Symptom
As of June 30 2022, the [Realsense ROS2 wrapper packager](https://github.com/IntelRealSense/realsense-ros/tree/ros2) does not support ROS2 Humble. Building the package on Humble produces the below error:

`CMake Error at CMakeLists.txt:142 (message):
  Unsupported ROS Distribution: humble`
### Solution
Please make the following changes in the `realsense2_camera` source files:

`realsense-ros/realsense2_camera/CMakeLists.txt`
```diff
elseif("$ENV{ROS_DISTRO}" STREQUAL "rolling")
  message(STATUS "Build for ROS2 Rolling")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DROLLING")
  set(SOURCES "${SOURCES}" src/ros_param_backend_foxy.cpp)
+elseif("$ENV{ROS_DISTRO}" STREQUAL "humble")
+  message(STATUS "Build for ROS2 Humble")
+  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHUMBLE")
+  set(SOURCES "${SOURCES}" src/ros_param_backend_foxy.cpp)
else()
  message(FATAL_ERROR "Unsupported ROS Distribution: " "$ENV{ROS_DISTRO}")
endif()
```
`realsense-ros/realsense2_camera/src/base_realsense_node.cpp`

```diff
-#ifdef GALACTIC
+#if defined(GALACTIC) || defined(HUMBLE)
        rclcpp::Duration elapsed_camera(rclcpp::Duration::from_nanoseconds(elapsed_camera_ns));
#else
        rclcpp::Duration elapsed_camera(elapsed_camera_ns);
#endif
```

## Input images must have even height and width
### Symptom
Isaac ROS nodes terminate when given images that have odd width or height:
```
[component_container_mt-1] [INFO] [1655745951.986574909] [NitrosContext]: [NitrosContext] Loading application: '/workspaces/isaac_ros-dev/ros_ws/install/isaac_ros_nitros/share/isaac_ros_nitros/config/type_adapter_nitros_context_graph.yaml'
[component_container_mt-1] [ERROR] [1655747175.384136779] [NitrosImage]: [convert_to_custom] Image width/height must be even for creation of gxf::VideoBuffer
[component_container_mt-1] terminate called after throwing an instance of 'std::runtime_error'
[component_container_mt-1]   what():  [convert_to_custom] Odd Image width or height.
[ERROR] [component_container_mt-1]: process has died [pid 28538, exit code -6, cmd '/opt/ros/humble/install/lib/rclcpp_components/component_container_mt --ros-args -r __node:=apriltag_container -r __ns:=/'].
```
### Solution
Replace the input image source with one that produces images that have even width and height.
## RealSense `ros2 launch realsense2_camera rs_launch.py` error
### Symptoms
Launching the realsense node using `ros2 launch realsense2_camera rs_launch.py`, produces the below error message:

```
[realsense2_camera_node-1]  29/06 01:10:27,431 WARNING [140061797918464] (rs.cpp:310) null pointer passed for argument "device"
[realsense2_camera_node-1] [WARN] [1656465033.464454660] [camera.camera]: Device 1/1 failed with exception: failed to set power state
[realsense2_camera_node-1] [ERROR] [1656465033.464505994] [camera.camera]: The requested device with  is NOT found. Will Try again.
[realsense2_camera_node-1]  29/06 01:10:33,463 ERROR [140061781133056] (handle-libusb.h:51) failed to open usb interface: 0, error: RS2_USB_STATUS_NO_DEVICE
[realsense2_camera_node-1]  29/06 01:10:33,463 ERROR [140061797918464] (sensor.cpp:572) acquire_power failed: failed to set power state
```

### Solution
Please make the following changes in the file `isaac_ros_common/scripts/run_dev.sh` and start a new container:

`isaac_ros_common/scripts/run_dev.sh`
```diff
if [ "$(docker ps -a --quiet --filter status=running --filter name=$CONTAINER_NAME)" ]; then
    print_info "Attaching to running container: $CONTAINER_NAME"
--    docker exec -i -t -u admin --workdir /workspaces/isaac_ros-dev $CONTAINER_NAME /bin/bash $@
++    docker exec -i -t --workdir /workspaces/isaac_ros-dev $CONTAINER_NAME /bin/bash $@
    exit 0
fi
```

`isaac_ros_common/scripts/run_dev.sh`
```diff
# Run container from image
print_info "Running $CONTAINER_NAME"
docker run -it --rm \
    --privileged \
    --network host \
    ${DOCKER_ARGS[@]} \
    -v $ISAAC_ROS_DEV_DIR:/workspaces/isaac_ros-dev \
    -v /dev/shm:/dev/shm \
    -v /dev/*:/dev/* \
    --name "$CONTAINER_NAME" \
    --runtime nvidia \
--    --user="admin" \
    --entrypoint /usr/local/bin/scripts/workspace-entrypoint.sh \
    --workdir /workspaces/isaac_ros-dev \
    $@ \
    $BASE_NAME \
    /bin/bash
```
## Failed to get valid output from Isaac ROS Rectify or Apriltag Node
### Symptom
If there is no available calibration data for an Argus camera, you will see warning messages similar to:

```
WARN  extensions/hawk/argus_camera.cpp@677: Failed to get calibration data from Argus!
```

### Solution
Most camera modules require calibration for lens distortion. Without camera calibration, rectification will output invalid results and impact the accuracy of downstream Nodes in the pipeline, such as AprilTag detection. 

Please refer to [this guide](./camera-calibration.md) to calibrate the camera. If you use Argus cameras, the Isaac ROS Argus Node provides options to get the calibration parameters from either the device driver or a `.ini` file. Please refer to [this section](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_argus_camera/blob/main/README.md#camerainfo-message) for additional details.
