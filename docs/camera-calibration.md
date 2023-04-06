# Monocular Camera Calibration

> **Note**: These instructions are specifically for a monocular camera. To calibrate a stereoscopic camera, please follow [these instructions](http://wiki.ros.org/camera_calibration).

1. Set up your development environment by following the instructions [here](./dev-env-setup.md).
2. Print a large checkerboard with known dimensions.
  
    This tutorial uses a 6x8 checkerboard with 200mm squares. Calibration uses the interior vertex points of the checkerboard, so a “7x9” board uses the interior vertex parameter “6x8” as in the example below. Checkerboards with specific dimensions can be downloaded [here](https://calib.io/pages/camera-calibration-pattern-generator).
3. Clone the ROS 2 `usb_cam` package:

   ```bash
    cd ~/workspaces/isaac_ros-dev/src && \
      git clone -b ros2 https://github.com/ros-drivers/usb_cam
    ```

    > **Note**: Your camera vendor may offer a specific ROS 2-compatible camera driver package that can be used in place of the `usb_cam` package.

4. Launch the Docker container using the `run_dev.sh` script:

    ```bash
    cd ~/workspaces/isaac_ros-dev/src/isaac_ros_common && \
      ./scripts/run_dev.sh
    ```

5. Inside the container, build and source the workspace:  

    ```bash
    cd /workspaces/isaac_ros-dev && \
      colcon build --symlink-install && \
      source install/setup.bash
    ```

6. Run the `usb_cam` image publisher:  

    ```bash
    ros2 run usb_cam usb_cam_node_exe --remap __ns:=/my_camera --ros-args -p framerate:=30.0 -p image_height:=720 -p image_width:=1280
    ```

7. Attach another terminal to the Docker container using the `run_dev.sh` script:

    ```bash
    cd ~/workspaces/isaac_ros-dev/src/isaac_ros_common && \
      ./scripts/run_dev.sh
    ```

8. Run the camera calibrator:

    ```bash
    ros2 run camera_calibration cameracalibrator --size 6x8 --square 0.20 image:=/my_camera/image_raw camera:=/my_camera
    ```

9. Complete steps 6-8 from [this tutorial](https://navigation.ros.org/tutorials/docs/camera_calibration.html).
10. Make sure you press **Calibrate** and then the **Save** button.

    You should see the following line in the second terminal:

    ```bash
    ('Wrote calibration data to', '/tmp/calibrationdata.tar.gz')
    ```

    The calibration file will be stored at `/tmp/calibrationdata.tar.gz`.
11. Once the calibration file has been saved, enter `Ctrl+C` in each terminal to stop the nodes.
12. Move the calibration file to the required location

    ```bash
      cd /workspaces/isaac_ros-dev/src/<isaac_ros_metapackage>/<isaac_ros_package>/config/ && \
      tar -xvf /tmp/calibrationdata.tar.gz -C ./ ost.yaml && \
      mv ost.yaml camera_info.yaml
    ```
