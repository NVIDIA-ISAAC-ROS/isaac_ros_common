# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## ⚠️ IMPORTANT: You Are Inside the Docker Container

**When this CLAUDE.md file is loaded, you (Claude Code) are already running inside the Isaac ROS development container.** You do NOT need to:
- Run `./scripts/run_dev.sh` to enter the container
- Use `docker exec` commands
- Worry about being on the host system

**Your current environment:**
- You are user `admin` with sudo access
- Default workspace: `/workspaces/isaac_ros-dev`
- ROS2 Humble is installed and ready to use
- All CUDA, TensorRT, VPI, and ROS dependencies are available
- The host workspace is already mounted at `/workspaces/isaac_ros-dev`

**All commands you run execute directly in the container environment.** Simply use standard ROS2 and colcon commands as documented below.

---

## Repository Overview

This is a **fork** of NVIDIA's Isaac ROS Common repository. The fork maintains minimal modifications focused on:
- Docker configuration in `docker/Dockerfile.ros2_humble`
- Development script `scripts/run_dev.sh`

The repository provides foundational infrastructure for the Isaac ROS ecosystem: Docker-based development environments, testing frameworks, build utilities, and common interfaces for ROS2 Humble on NVIDIA hardware (Jetson/x86_64 with CUDA).

## Development Environment

### Docker Container Environment (You Are Here)

Development occurs **inside Docker containers**, not on the host system. You (Claude) are already inside this container environment.

**How users typically enter this container** (for context only):
```bash
# From isaac_ros_common directory on host
./scripts/run_dev.sh

# Inside container - default workspace
cd /workspaces/isaac_ros-dev
```

**Key container characteristics:**
- Reuses existing running containers (attaches instead of creating new)
- Removes exited containers automatically on exit
- Maps host workspace directory to `/workspaces/isaac_ros-dev` in container
- Runs as non-root user 'admin' with sudo access
- Uses `--rm` flag: containers are ephemeral, data must be on mounted volumes

**Configuration files:**
- `~/.isaac_ros_common-config`: Container and image configuration
- `~/.isaac_ros_dev-dockerargs`: Additional Docker arguments (volumes, devices, env vars)

### Building the Codebase

```bash
# Build entire workspace
cd /workspaces/isaac_ros-dev
colcon build --symlink-install

# Build specific package
colcon build --packages-select <package_name>

# Build with specific upstream package
colcon build --packages-up-to <package_name>

# Clean build
rm -rf build install log
colcon build --symlink-install
```

Build configuration:
- Build system: `colcon` + `ament_cmake_auto`
- All packages use C++17, Release mode by default
- CUDA architectures auto-configured per platform
- Jetson: 87 (Orin), x86_64: 89;86;80;75;70 (Ada, Ampere, Turing, Volta)


## Architecture Overview

### Package Categories

**Development Infrastructure:**
- `isaac_ros_common`: Core C++/Python utilities (QoS, VPI), CMake modules
- `isaac_ros_test`: Python testing framework with `IsaacROSBaseTest` base class
- `isaac_ros_test_cmake`: CMake integration for launch-based tests
- `isaac_ros_launch_utils`: Launch file helpers with `ArgumentContainer` pattern

**Common Libraries:**
- `isaac_common`: ROS-agnostic C++ utilities
- `isaac_common_py`: Python utilities

**Interface Packages:** (ROS2 message/service definitions)
- `isaac_ros_apriltag_interfaces`, `isaac_ros_bi3d_interfaces`, `isaac_ros_nova_interfaces`, `isaac_ros_pointcloud_interfaces`, `isaac_ros_tensor_list_interfaces`, `isaac_ros_nitros_bridge_interfaces`

**Utilities:**
- `isaac_ros_rosbag_utils`: Rosbag manipulation
- `isaac_ros_r2b_galileo`: Internal tooling

### Docker Layer Architecture

Multi-stage build system:

1. **Base layer** (`docker/Dockerfile.base`): Platform + CUDA 12.6 + core dependencies
2. **ROS2 layer** (`docker/Dockerfile.ros2_humble`): ROS2 Humble + patched packages
3. **Dev layer**: Mounted workspaces (not baked into image)
4. **Deploy layer**: Specific workspace + launch command

Patched packages (fixes applied during Docker build):
- `rclcpp`: Multithreaded executor deadlock fix
- `image_proc`: Resize node fix
- `negotiated`: Built from source

## Common Patterns

### CMake Package Structure

All packages follow this pattern:

```cmake
cmake_minimum_required(VERSION 3.22.1)
project(my_package)

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

# Automatically includes isaac_ros_common-extras.cmake which sets:
# - C++17 standard
# - Release build type
# - CUDA paths and architectures
# - Version embedding via generate_version_info()

ament_auto_add_library(${PROJECT_NAME} SHARED src/my_node.cpp)
rclcpp_components_register_nodes(${PROJECT_NAME} "my_namespace::MyNode")

ament_auto_package()
```

### Python Testing Pattern

Tests inherit from `IsaacROSBaseTest`:

```python
from isaac_ros_test import IsaacROSBaseTest

class MyNodeTest(IsaacROSBaseTest):
    @classmethod
    def generate_test_description(cls):
        # Return launch description with nodes to test
        return cls.generate_test_description(
            [container, load_nodes],
            node_startup_delay=2.0
        )

    def test_functionality(self):
        # Use namespace isolation
        self.generate_namespace_lookup(['input', 'output'], 'my_namespace')

        # Create logging subscribers
        received = {}
        subs = self.create_logging_subscribers(
            [('output', Image)], received
        )

        # Publish test data, spin, assert results
```

Key features:
- `DEFAULT_NAMESPACE = 'isaac_ros_test'` for isolation
- `@IsaacROSBaseTest.for_each_test_case()` decorator for data-driven tests
- Test case folders in `test_cases/` (add `SKIP` file to skip a case)
- Image comparison helpers with configurable tolerance


### QoS Configuration

Both C++ and Python support QoS parameter parsing:

**C++:**
```cpp
#include <isaac_ros_common/qos.hpp>
rclcpp::QoS qos = isaac_ros::common::AddQosParameter(node, "SENSOR_DATA", "qos_param");
```

**Python:**
```python
from isaac_ros_common.qos import add_qos_parameter
qos = add_qos_parameter(node, default_qos='SENSOR_DATA', parameter_name='qos')
```

Profiles: `SYSTEM_DEFAULT`, `SENSOR_DATA`, `PARAMETERS`, `SERVICES_DEFAULT`

## Key Build Files

### CMake Modules (in `isaac_ros_common/cmake/`)

- `isaac_ros_common-extras.cmake`: Auto-loaded, sets C++17/Release/CUDA config
- `isaac_ros_common-version-info.cmake`: Embeds git version info into builds
- `isaac_ros_common-extras-assets.cmake`: Asset installation system


### Nova Robot Detection

For NVIDIA Nova platforms:
```python
from isaac_ros_launch_utils import get_nova_robot, NovaRobot
robot = get_nova_robot()  # Reads /etc/nova/manager_selection
# Returns: NOVA_CARTER, NOVA_DEVELOPER_KIT, NOVA_BENCHTOP, or UNKNOWN
```

## Common Commands

**Remember: You are already inside the container, so use these commands directly:**

```bash
# Build workspace
colcon build --symlink-install

# Run tests for a package
colcon test --packages-select <package_name>
colcon test-result --verbose

# Source workspace
source install/setup.bash

# Check package dependencies
rosdep install -i --from-path src --rosdistro humble -y

# List ROS packages
ros2 pkg list

# Check node info
ros2 node info <node_name>

# Monitor topics
ros2 topic list
ros2 topic echo <topic_name>
```

### Host-Side Commands (For Reference Only - Not Used by Claude)

These commands run on the host system outside the container. Since Claude operates inside the container, these are documented for reference but should not be used:

```bash
# Enter development container (NOT NEEDED - you're already inside)
cd /path/to/isaac_ros_common
./scripts/run_dev.sh

# Build custom Docker layers (requires host system)
./scripts/build_image_layers.sh <image_key>
# Example: ./scripts/build_image_layers.sh x86_64.ros2_humble

# Create deployment image (requires host system)
./scripts/docker_deploy.sh \
  -b "x86_64.ros2_humble" \
  -w /workspaces/isaac_ros-dev/ros_ws \
  -p my_package \
  -f my_launch.launch.py \
  -n my_deployed_image
```

## Important Caveats

1. **Containers are ephemeral**: Use `--rm` flag, so data must be on mounted volumes
2. **Git LFS required**: Repository uses Git LFS for large files (checked on startup)
3. **Non-root user**: Container creates user matching host UID/GID to avoid permission issues
4. **FastRTPS configuration**: Uses custom UDP profile in `docker/middleware_profiles/rtps_udp_profile.xml`
5. **Asset downloads**: Large models installed via scripts (not bundled), support EULA enforcement
6. **Setuptools pinned**: Version 65.7.0 due to upstream issue

## File Location Reference

```
/workspaces/isaac_ros-dev/src/isaac_ros_common/
├── docker/                          # Dockerfiles and configs
│   ├── Dockerfile.base             # Platform base image
│   ├── Dockerfile.ros2_humble      # ROS2 layer
│   └── middleware_profiles/        # FastRTPS configs
├── scripts/                         # Development scripts
│   ├── run_dev.sh                  # Main dev container launcher
│   ├── build_image_layers.sh       # Image builder
│   └── docker_deploy.sh            # Deployment builder
├── isaac_ros_common/               # Core utilities
│   ├── cmake/                      # CMake modules
│   ├── include/isaac_ros_common/   # C++ headers
│   └── isaac_ros_common/           # Python module
├── isaac_ros_test/                 # Testing framework
│   └── isaac_ros_test/
│       └── isaac_ros_base_test.py  # Base test class
├── isaac_ros_launch_utils/         # Launch helpers
│   └── isaac_ros_launch_utils/
│       └── core.py                 # ArgumentContainer, etc.
└── isaac_ros_*_interfaces/         # Message definitions
```

## Dependencies

- **ROS2**: Humble (Ubuntu 22.04)
- **CUDA**: 12.6.1+
- **Python**: 3.10
- **CMake**: 3.22.1+
- **TensorRT**: 10.x
- **VPI**: 3.2.4
- **PyTorch**: 2.5.0 (Jetson), latest (x86)
