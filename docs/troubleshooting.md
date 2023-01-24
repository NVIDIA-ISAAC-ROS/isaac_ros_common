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
  - [RealSense does not streaming IR stereo images](#realsense-does-not-streaming-ir-stereo-images)
    - [Symptom](#symptom-3)
    - [Solution](#solution-4)
  - [RealSense error `Failed to resolve the request`](#realsense-error-failed-to-resolve-the-request)
    - [Symptom](#symptom-4)
    - [Solution](#solution-5)
  - [RealSense `incompatible QoS Policy` error](#realsense-incompatible-qos-policy-error)
    - [Symptom](#symptom-5)
    - [Solution](#solution-6)
  - [Linker error `foo.so: file format not recognized; treating as linker script`](#linker-error-fooso-file-format-not-recognized-treating-as-linker-script)
    - [Symptom](#symptom-6)
    - [Solution](#solution-7)


## Realsense driver doesn't work with ROS2 Humble

### Symptom

As of Sep 13 2022, the [Realsense ROS2 wrapper package's `ros2` branch](https://github.com/IntelRealSense/realsense-ros/tree/ros2) does not support ROS2 Humble. Building the package on Humble produces the below error:

`CMake Error at CMakeLists.txt:142 (message):
  Unsupported ROS Distribution: humble`

### Solution

Use the [RealSense ROS2 driver `ros2-development` branch](https://github.com/IntelRealSense/realsense-ros/tree/ros2-development)

## Input images must have even height and width

### Symptom

Isaac ROS nodes terminate when given images that have odd width or height:

```log
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

```log
[realsense2_camera_node-1]  29/06 01:10:27,431 WARNING [140061797918464] (rs.cpp:310) null pointer passed for argument "device"
[realsense2_camera_node-1] [WARN] [1656465033.464454660] [camera.camera]: Device 1/1 failed with exception: failed to set power state
[realsense2_camera_node-1] [ERROR] [1656465033.464505994] [camera.camera]: The requested device with  is NOT found. Will Try again.
[realsense2_camera_node-1]  29/06 01:10:33,463 ERROR [140061781133056] (handle-libusb.h:51) failed to open usb interface: 0, error: RS2_USB_STATUS_NO_DEVICE
[realsense2_camera_node-1]  29/06 01:10:33,463 ERROR [140061797918464] (sensor.cpp:572) acquire_power failed: failed to set power state
```

### Solution

**Before** starting the isaac_ros-dev docker container using run_dev.sh, run [setup_udev_rules.sh](https://github.com/IntelRealSense/librealsense/blob/master/scripts/setup_udev_rules.sh) from [librealsense](https://github.com/IntelRealSense/librealsense) **in a terminal outside docker**

## Failed to get valid output from Isaac ROS Rectify or Apriltag Node

### Symptom

If there is no available calibration data for an Argus camera, you will see warning messages similar to:

```log
WARN  extensions/hawk/argus_camera.cpp@677: Failed to get calibration data from Argus!
```

### Solution

Most camera modules require calibration for lens distortion. Without camera calibration, rectification will output invalid results and impact the accuracy of downstream Nodes in the pipeline, such as AprilTag detection.

Please refer to [this guide](./camera-calibration.md) to calibrate the camera. If you use Argus cameras, the Isaac ROS Argus Node provides options to get the calibration parameters from either the device driver or a `.ini` file. Please refer to [this section](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_argus_camera/blob/main/README.md#camerainfo-message) for additional details.

## RealSense does not streaming IR stereo images

### Symptom

**Within the Docker container**: Cannot see any IR images in `realsense-viewer` but depth images can be streamed.

**Outside the Docker container**: No metadata containing projector status attached to the IR frames.

### Solution

1. Download and install the dkms for kernel 5.15 from [here](https://github.com/mengyui/librealsense2-dkms/releases/tag/initial-support-for-kernel-5.15)
   1. Download the .deb file named `librealsense2-dkms-dkms_1.3.14_amd64.deb`
   2. In the directory where you downloaded the file run: `sudo apt install ./librealsense2-dkms-dkms_1.3.14_amd64.deb`
2. Manually build librealsense within the container without CUDA. Run the following commands within the docker container:

- `git clone https://github.com/JetsonHacksNano/installLibrealsense`
- `cd installLibrealsense`
- `./installLibrealsense.sh`
- `./buildLibrealsense.sh --no_cuda`

> **Note**: For more details on this issue, refer to the [nvblox troubleshooting section](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_nvblox/blob/main/docs/troubleshooting-nvblox-vslam-realsense.md)

## RealSense error `Failed to resolve the request`

### Symptom

When any RealSense tutorial is launched, no images are streamed and the following error is found in the terminal logs:

```log
[component_container_mt-1] [ERROR] [1665684669.343408502] [camera]: /workspaces/isaac_ros-dev/ros_ws/src/third_party/realsense-ros/realsense2_camera/src/rs_node_setup.cpp:344:An exception has been thrown: 
[component_container_mt-1] Failed to resolve the request: 
[component_container_mt-1]  Format: Y8, width: 640, height: 480
[component_container_mt-1]  Format: Y8, width: 640, height: 480
[component_container_mt-1] 
[component_container_mt-1] Into:
[component_container_mt-1]  Formats: 
[component_container_mt-1]   Y8I
[component_container_mt-1] 
[component_container_mt-1] 
[component_container_mt-1] [ERROR] [1665684669.343506391] [camera]: Error starting device: 
[component_container_mt-1] Failed to resolve the request: 
[component_container_mt-1]  Format: Y8, width: 640, height: 480
[component_container_mt-1]  Format: Y8, width: 640, height: 480
[component_container_mt-1] 
[component_container_mt-1] Into:
[component_container_mt-1]  Formats: 
[component_container_mt-1]   Y8I
[component_container_mt-1]
```

### Solution

Update the RealSense driver to the latest version using the following steps:

1. Install librealsense on your local system using instructions from [this link](https://github.com/IntelRealSense/librealsense/blob/master/doc/distribution_linux.md)
2. Download the latest fimrware binary(eg:Signed_Image_UVC_5_13_0_55.bin) from [this link](https://dev.intelrealsense.com/docs/firmware-releases)
3. Extract the binary from the .zip file
4. Run the following command(Replace the <binary_filename> with the name of the binary you downloaded):

```bash
rs-fw-update -f <binary_filename>
```

> **Note**: For more information refer to the [RealSense update tool](https://dev.intelrealsense.com/docs/firmware-update-tool)

## RealSense `incompatible QoS Policy` error
### Symptom

When any RealSense tutorial is launched, the output topic from RealSense is not subscribed and the following warning is found in the terminal logs:
```log
[component_container_mt-2] [WARN] [1666117836.386565040] [left_encoder_node]: New publisher discovered on topic '/color/image_raw', offering incompatible QoS. No messages will be sent to it. Last incompatible policy: RELIABILITY_QOS_POLICY
[component_container_mt-2] [WARN] [1666117836.392021093] [camera]: New subscription discovered on topic '/color/image_raw', requesting incompatible QoS. No messages will be sent to it. Last incompatible policy: RELIABILITY_QOS_POLICY
```

### Solution

Change the QoS policy in `realsense.yaml` file from the launched package. Take `isaac_ros_h264_encoder` as an example, change [policy](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_compression/blob/main/isaac_ros_h264_encoder/config/realsense.yaml#L7) from `SENSOR_DATA` to `SYSTEM_DEFAULT` could make the RealSense rgb_camera publisher compatible with the subscriber of `isaac_ros_h264_encoder`.
> **Note**: For more information on Quality of Service compatibilities refer to [this link](https://docs.ros.org/en/humble/Concepts/About-Quality-of-Service-Settings.html)

## Linker error `foo.so: file format not recognized; treating as linker script`
### Symptom
When attempting to build an Isaac ROS package by running `colcon build`, the linker complains with an error that resembles the following:
```
--- stderr: isaac_ros_nitros                                                                                      
/usr/bin/ld:/workspaces/isaac_ros-dev/src/isaac_ros_nitros/isaac_ros_nitros/gxf/lib/gxf_x86_64_cuda_11_7/core/libgxf_core.so: file format not recognized; treating as linker script
/usr/bin/ld:/workspaces/isaac_ros-dev/src/isaac_ros_nitros/isaac_ros_nitros/gxf/lib/gxf_x86_64_cuda_11_7/core/libgxf_core.so:1: syntax error
collect2: error: ld returned 1 exit status
make[2]: *** [CMakeFiles/isaac_ros_nitros.dir/build.make:311: libisaac_ros_nitros.so] Error 1
make[1]: *** [CMakeFiles/Makefile2:139: CMakeFiles/isaac_ros_nitros.dir/all] Error 2
make: *** [Makefile:146: all] Error 2
---
Failed   <<< isaac_ros_nitros [0.52s, exited with code 2]
```

### Solution
This error typically arises when the local copy of a `.so` binary on your machine contains only the Git LFS pointer, instead of the actual binary contents.

> **Note:** If you're encountering this error, you may have forgotten to install and perform one-time setup for Git LFS on your machine. Please refer to the instructions for [development environment setup](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common/blob/main/docs/dev-env-setup.md#quickstart).

To correct this, navigate to the Isaac ROS package that is failing to build and re-pull the binary files using Git LFS.

For example, if `isaac_ros_nitros` is failing to build as shown above:

1. Navigate to the package:
   ```bash
   cd ~/workspaces/isaac_ros-dev/src/isaac_ros_nitros
   ```
2. Pull from Git LFS:
   ```bash
   git lfs pull
   ```
3. Return to the workspace and run the build command again:
   ```bash
   cd ~/workspaces/isaac_ros-dev && colcon build --symlink-install
   ```
