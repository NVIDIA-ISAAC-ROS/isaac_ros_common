# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.22.1)
project(isaac_ros_test_cmake)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

# Install additional cmake functions
ament_index_register_resource(isaac_ros_test_cmake_cmake_path CONTENT
    "${CMAKE_INSTALL_PREFIX}/share/${PROJECT_NAME}/cmake")
list(APPEND ${PROJECT_NAME}_CONFIG_EXTRAS
  cmake/isaac_ros_test_cmake-extras-graph-test.cmake
)

# Register the dummy bag as a resource
ament_index_register_resource(dummy_bag CONTENT
    "${CMAKE_INSTALL_PREFIX}/share/${PROJECT_NAME}/resources/dummy_bag")

# Register the test as a resource
set(ISAAC_ROS_GRAPH_STARTUP_TEST_PATH
    "${CMAKE_INSTALL_PREFIX}/share/${PROJECT_NAME}/isaac_ros_test_cmake/isaac_ros_graph_startup_test.py")
ament_index_register_resource(graph_startup_test CONTENT "${ISAAC_ROS_GRAPH_STARTUP_TEST_PATH}")

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()

  include(cmake/isaac_ros_test_cmake-extras-graph-test.cmake)
  add_graph_startup_test(test_graph_startup_test
    test/test_launchfile.launch.py
    TIMEOUT 5
    # NOTE: We have to manually pass this path in this file because the resource
    # is not ready at build time.
    ISAAC_ROS_GRAPH_STARTUP_TEST_PATH "${ISAAC_ROS_GRAPH_STARTUP_TEST_PATH}"
  )
endif()


# Embed versioning information into installed files
ament_index_get_resource(ISAAC_ROS_COMMON_CMAKE_PATH isaac_ros_common_cmake_path isaac_ros_common)
include("${ISAAC_ROS_COMMON_CMAKE_PATH}/isaac_ros_common-version-info.cmake")
generate_version_info(${PROJECT_NAME})

ament_auto_package(INSTALL_TO_SHARE test resources isaac_ros_test_cmake)
