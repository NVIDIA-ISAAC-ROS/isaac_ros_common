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
project(isaac_ros_r2b_galileo)

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

include(FetchContent)

set(DATASET_NAME r2b_galileo)
set(BASE_URL https://api.ngc.nvidia.com/v2/resources/org/nvidia/team/isaac/r2bdataset2024/1/files?redirect=true&path=r2b_galileo)

FetchContent_Declare(
  metadata
  URL ${BASE_URL}/metadata.yaml
  DOWNLOAD_DIR ${CMAKE_CURRENT_BINARY_DIR}/data/${DATASET_NAME}/
  DOWNLOAD_NO_EXTRACT TRUE
)

FetchContent_Declare(
  mcap
  URL ${BASE_URL}/${DATASET_NAME}_0.mcap
  DOWNLOAD_DIR ${CMAKE_CURRENT_BINARY_DIR}/data/${DATASET_NAME}/
  DOWNLOAD_NO_EXTRACT TRUE
)

FetchContent_MakeAvailable(metadata mcap)

# Embed versioning information into installed files
ament_index_get_resource(ISAAC_ROS_COMMON_CMAKE_PATH isaac_ros_common_cmake_path isaac_ros_common)
include("${ISAAC_ROS_COMMON_CMAKE_PATH}/isaac_ros_common-version-info.cmake")
generate_version_info(${PROJECT_NAME})

ament_auto_package(INSTALL_TO_SHARE ${CMAKE_CURRENT_BINARY_DIR}/data)
