# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Common flags and cmake commands for all Isaac ROS packages.
message(STATUS "Loading isaac_ros_common extras")

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

# Default to Release build
if(NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif()
message( STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}" )

# for #include <cuda_runtime.h>
set(CUDA_MIN_VERSION "11.4")
find_package(CUDA REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")

# Target Ada is CUDA 11.8 or greater
if( ${CUDA_VERSION} GREATER_EQUAL 11.8)
  set(CMAKE_CUDA_ARCHITECTURES "89;87;86;80;75;72;70;61;60")
else()
  set(CMAKE_CUDA_ARCHITECTURES "87;86;80;75;72;70;61;60")
endif()
