# Isaac ROS Common

Isaac ROS common utilities and scripts for use in conjunction with the Isaac ROS suite of packages.

## Docker Scripts
`run_dev.sh` creates a dev environment with ROS2 installed and key versions of NVIDIA frameworks prepared for both x86_64 and Jetson. By default, the directory `/workspaces/isaac_ros-dev` in the container is mapped from `~/workspaces/isaac_ros-dev` on the host machine if it exists OR the current working directory from where the script was invoked otherwise. The host directory the container maps to can be explicitly set by running the script with the desired path as the first argument:
```
scripts/run_dev.sh <path to workspace>
```

For solutions to known issues, please visit the [Troubleshooting](#troubleshooting) section below.

## System Requirements
This script is designed and tested to be compatible with ROS2 Foxy on Jetson hardware in addition to on x86 systems with an Nvidia GPU. 

### Jetson
- [Jetson AGX Xavier or Xavier NX](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/)
- [JetPack 4.6.1](https://developer.nvidia.com/embedded/jetpack)

### x86_64
- Ubuntu 20.04+
- CUDA 11.4 supported discrete GPU
- VPI 1.1.11

You must first install the [Nvidia Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) to make use of the Docker container development/runtime environment.

Configure `nvidia-container-runtime` as the default runtime for Docker by editing `/etc/docker/daemon.json` to include the following:
```
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    },
    "default-runtime": "nvidia"
```
and then restarting Docker: `sudo systemctl daemon-reload && sudo systemctl restart docker`

**Note:** For best performance on Jetson, ensure that power settings are configured appropriately ([Power Management for Jetson](https://docs.nvidia.com/jetson/l4t/index.html#page/Tegra%20Linux%20Driver%20Package%20Development%20Guide/power_management_jetson_xavier.html#wwpID0EUHA)).

## Troubleshooting
### `run_dev.sh` on x86 fails with `vpi-lib-1.1.11-cuda11-x86_64-linux.deb` is not a Debian format archive
When building a Docker image, `run_dev.sh` may fail because some files seem to be invalid. Debian packages for VPI on x86 are packaged in Isaac ROS using `git-lfs`. These files need to be fetched in order to install VPI in the Docker image.

#### Symptoms
```
dpkg-deb: error: 'vpi-lib-1.1.11-cuda11-x86_64-linux.deb' is not a Debian format archive
dpkg: error processing archive vpi-lib-1.1.11-cuda11-x86_64-linux.deb (--install):
 dpkg-deb --control subprocess returned error exit status 2
Errors were encountered while processing:
 vpi-lib-1.1.11-cuda11-x86_64-linux.deb
```
#### Solution
Run `git lfs pull` in each Isaac ROS repository you have checked out, especially `isaac_ros_common`, to ensure all of the large binary files have been downloaded.

### Nodes crashed on initial launch or failed to build, reporting shared libraries have a file format not recognized
Many dependent shared library binary files are stored in `git-lfs`. These files need to be fetched in order for Isaac ROS nodes to function correctly.

#### Symptoms
```
/usr/bin/ld:/workspaces/isaac_ros-dev/ros_ws/src/isaac_ros_common/isaac_ros_nvengine/gxf/lib/gxf_jetpack46/core/libgxf_core.so: file format not recognized; treating as linker script
/usr/bin/ld:/workspaces/isaac_ros-dev/ros_ws/src/isaac_ros_common/isaac_ros_nvengine/gxf/lib/gxf_jetpack46/core/libgxf_core.so:1: syntax error
collect2: error: ld returned 1 exit status
make[2]: *** [libgxe_node.so] Error 1
make[1]: *** [CMakeFiles/gxe_node.dir/all] Error 2
make: *** [all] Error 2
```
#### Solution
Run `git lfs pull` in each Isaac ROS repository you have checked out, especially `isaac_ros_common`, to ensure all of the large binary files have been downloaded.

# Updates

| Date | Changes |
| -----| ------- |
| 2021-10-20 | Migrated to [NVIDIA-ISAAC-ROS](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common), added `isaac_ros_nvengine` and `isaac_ros_nvengine_interfaces` packages  |
| 2021-08-11 | Initial release to [NVIDIA-AI-IOT](https://github.com/NVIDIA-AI-IOT/isaac_ros_common) |
