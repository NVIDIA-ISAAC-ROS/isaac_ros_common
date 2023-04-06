# Isaac ROS Common

## Latest Update

Update 2023-04-05: Merged 'Dockerfile.humble' and 'Dockerfile.nav2' into 'Dockerfile.ros2_humble' as image key 'ros2_humble'.

## Overview

The [Isaac ROS Common](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common) repository contains a number of scripts and Dockerfiles to help streamline development and testing with the Isaac ROS suite.

<div align="center"><img src="resources/isaac_ros_common_tools.png" width="800" title="Isaac ROS DevOps tools"/></div>

The Docker images included in this package provide pre-compiled binaries for ROS 2 Humble on Ubuntu 20.04 Focal.  

Additionally, on x86_64 platforms, Docker containers allow you to quickly set up a sensitive set of frameworks and dependencies to ensure a smooth experience with Isaac ROS packages. The Dockerfiles for this platform are based on the version 22.03 image from [Deep Learning Frameworks Containers](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html). On Jetson platforms, JetPack manages all of these dependencies for you.

Use of Docker images enables CI|CD systems to scale with DevOps work and run automated testing in cloud native platforms on Kubernetes.

For solutions to known issues, please visit the [Troubleshooting](./docs/troubleshooting.md) section.

### Docker Development Scripts

`run_dev.sh` sets up a development environment containing ROS 2 and key versions of NVIDIA frameworks prepared for both x86_64 and Jetson. This script prepares a Docker image with a supported configuration for the
host machine and delivers you into a bash prompt running inside the container. From here, you are ready to execute ROS 2 build/run commands with your host workspace files, which are mounted into the container and available to edit both on the host and in the container. If you run this script again while it is running, it will attach a new shell to the same container.

By default, the `/workspaces/isaac_ros-dev` directory in the container is mapped from `~/workspaces/isaac_ros-dev` on the host machine if it exists; otherwise, it is mapped using the current working directory where the script was invoked. You can set the host directory that the container maps to explicitly by running the script with the desired path as the first argument:

```bash
scripts/run_dev.sh <path to workspace>
```

### Configuring run_dev.sh

`run_dev.sh` prepares a base Docker image and mounts your target workspace into the running container. The Docker image is assembled by parsing a period-delimited image key into matching Dockerfiles and building each one in sequence. For example, an image key of `first.second.third` could match {`Dockerfile.first`, `Dockerfile.second`, `Dockerfile.third`}, where `run_dev.sh` will build the image for `Dockerfile.first`, then use that image as the base image while building `Dockerfile.second`, and so on. The file matching looks for the largest subsequence, so `first.second.third` could also match {`Dockerfile.first.second`, `Dockerfile.third`} depending on which files exist in the search paths. Using this pattern, you can add your own layers on top of the base images provided in Isaac ROS to set up your environment your way, such as installing additional packages to use.

If you write a file with the name `.isaac_ros_common-config` in the same directory as the `run_dev.sh`, you can configure the development environment. The following keys can be configured in `.isaac_ros_common-config`:

| Key                         | Type                 | Description                                                 | Examples                              |
| --------------------------- | -------------------- | ----------------------------------------------------------- | ------------------------------------- |
| `CONFIG_IMAGE_KEY`          | String               | Image key with period-delimited components                  | `ros2_humble.realsense` <br> `first.second.third` |
| `CONFIG_DOCKER_SEARCH_DIRS` | Bash array of string | List of directories to search for Dockerfiles when matching | `($HOME/ros_ws/docker $HOME/docker)`  |

For example, if you had the following directory structure:

```log
workspaces/isaac_ros-dev
    workspaces/isaac_ros-dev/ros_ws
        workspaces/isaac_ros-dev/ros_ws/src
        workspaces/isaac_ros-dev/ros_ws/src/isaac_ros_common
        workspaces/isaac_ros-dev/ros_ws/src/isaac_ros_common/scripts
        workspaces/isaac_ros-dev/ros_ws/src/isaac_ros_common/scripts/.isaac_ros_common-config

        workspaces/isaac_ros-dev/ros_ws/docker
        workspaces/isaac_ros-dev/ros_ws/docker/Dockerfile.mine
        workspaces/isaac_ros-dev/ros_ws/docker/myfile.txt
```

where `Dockerfile.mine` can be your own custom image layers of the form using its host directory for the Docker build context:

```Dockerfile
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

... steps ...

COPY myfile.txt /myfile.txt

... more steps ...
```

You could extend the base image launched as a container by `run_dev.sh` as follows.

`workspaces/isaac_ros-dev/ros_ws/src/isaac_ros_common/scripts/.isaac_ros_common-config`

```vim
CONFIG_IMAGE_KEY="ros2_humble.mine"
CONFIG_DOCKER_SEARCH_DIRS=(workspaces/isaac_ros-dev/ros_ws/docker)
```

This configures the image key to match with `mine` included and specifies where to look first for the Dockerfiles before the default.

## Table of Contents

- [Isaac ROS Common](#isaac-ros-common)
  - [Overview](#overview)
    - [Docker Development Scripts](#docker-development-scripts)
    - [Configuring run\_dev.sh](#configuring-run_devsh)
  - [Table of Contents](#table-of-contents)
  - [Latest Update](#latest-update)
  - [Supported Platforms](#supported-platforms)
  - [Updates](#updates)

## Latest Update

Update 2023-04-05: Experimental support for WSL2

## Supported Platforms

This package is designed and tested to be compatible with ROS 2 Humble running on [Jetson](https://developer.nvidia.com/embedded-computing) or an x86_64 system with an NVIDIA GPU.

> **Note**: Versions of ROS 2 earlier than Humble are **not** supported. This package depends on specific ROS 2 implementation features that were only introduced beginning with the Humble release.

| Platform | Hardware                                                                                                                                                                                                 | Software                                                                                                           | Notes                                                                                                                                                                                   |
| -------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Jetson   | [Jetson Orin](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/) <br> [Jetson Xavier](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-agx-xavier/) | [JetPack 5.1.1](https://developer.nvidia.com/embedded/jetpack)                                                     | For best performance, ensure that [power settings](https://docs.nvidia.com/jetson/archives/r34.1/DeveloperGuide/text/SD/PlatformPowerAndPerformance.html) are configured appropriately. |
| x86_64   | NVIDIA GPU                                                                                                                                                                                               | [Ubuntu 20.04+](https://releases.ubuntu.com/20.04/) <br> [CUDA 11.8+](https://developer.nvidia.com/cuda-downloads) |

## Updates

| Date       | Changes                                                                                                                                                       |
| ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 2023-04-05 | Experimental support for WSL2, merged image keys 'humble' and 'nav2' into 'ros2_humble'                                                                                                                     |
| 2022-10-19 | Minor updates and bugfixes                                                                                                                                    |
| 2022-08-31 | Update to be compatible with JetPack 5.0.2                                                                                                                    |
| 2022-06-30 | Support ROS 2 Humble and miscellaneous bug fixes.                                                                                                             |
| 2022-06-16 | Update `run_dev.sh` and removed `isaac_ros_nvengine`                                                                                                          |
| 2021-10-20 | Migrated to [NVIDIA-ISAAC-ROS](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common), added `isaac_ros_nvengine` and `isaac_ros_nvengine_interfaces` packages |
| 2021-08-11 | Initial release to [NVIDIA-AI-IOT](https://github.com/NVIDIA-AI-IOT/isaac_ros_common)                                                                         |
