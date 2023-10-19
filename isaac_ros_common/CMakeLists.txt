# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
project(isaac_ros_common LANGUAGES C CXX)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/isaac_ros_common-extras.cmake)

# Dependencies
find_package(vpi REQUIRED)

# vpi utilities
ament_auto_add_library(${PROJECT_NAME} SHARED
  src/vpi_utilities.cpp
)
target_link_libraries(${PROJECT_NAME} vpi)

# Register cmake in install.
ament_index_register_resource(isaac_ros_common_cmake_path CONTENT
    "${CMAKE_INSTALL_PREFIX}/share/${PROJECT_NAME}/cmake")
list(APPEND ${PROJECT_NAME}_CONFIG_EXTRAS
  cmake/isaac_ros_common-extras.cmake
  cmake/isaac_ros_common-extras-modules.cmake)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)

  # Ignore copyright notices since we use custom NVIDIA Isaac ROS Software License
  set(ament_cmake_copyright_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_auto_package(INSTALL_TO_SHARE cmake)