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

# Create TensorRT imported targets
#
# This module defines TensorRT_FOUND if all GXF libraries are found or
# if the required libraries (COMPONENTS property in find_package)
# are found.
#
# A new imported target is created for each component (library)
# under the TensorRT namespace (TensorRT::${component_name})
#
# Note: this leverages the find-module paradigm [1]. The config-file paradigm [2]
# is recommended instead in CMake.
# [1] https://cmake.org/cmake/help/latest/manual/cmake-packages.7.html#config-file-packages
# [2] https://cmake.org/cmake/help/latest/manual/cmake-packages.7.html#find-module-packages

# Find headers
find_path(TENSORRT_INCLUDE_DIR NAMES NvInferVersion.h REQUIRED)
mark_as_advanced(TENSORRT_INCLUDE_DIR)

# Find version
function(read_version name str)
    string(REGEX MATCH "${name} ([0-9]\\d*)" _ ${str})
    set(${name} ${CMAKE_MATCH_1} PARENT_SCOPE)
endfunction()

file(READ "${TENSORRT_INCLUDE_DIR}/NvInferVersion.h" _TRT_VERSION_FILE)
read_version(NV_TENSORRT_MAJOR ${_TRT_VERSION_FILE})
read_version(NV_TENSORRT_MINOR ${_TRT_VERSION_FILE})
read_version(NV_TENSORRT_PATCH ${_TRT_VERSION_FILE})
set(TENSORRT_VERSION "${NV_TENSORRT_MAJOR}.${NV_TENSORRT_MINOR}.${NV_TENSORRT_PATCH}")
unset(_TRT_VERSION_FILE)

# Find libs, and create the imported target
macro(find_trt_library libname)
    find_library(TENSORRT_${libname}_LIBRARY NAMES ${libname} REQUIRED)
    mark_as_advanced(TENSORRT_${libname}_LIBRARY)
    add_library(TENSORRT::${libname} SHARED IMPORTED GLOBAL)
    set_target_properties(TENSORRT::${libname} PROPERTIES
        IMPORTED_LOCATION "${TENSORRT_${libname}_LIBRARY}"
        INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${TENSORRT_INCLUDE_DIR}"
    )
endmacro()

find_trt_library(nvinfer)
find_trt_library(nvinfer_plugin)
find_trt_library(nvcaffe_parser)
find_trt_library(nvonnxparser)
find_trt_library(nvparsers)

# Generate TENSORRT_FOUND
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(TENSORRT
    FOUND_VAR TENSORRT_FOUND
    VERSION_VAR TENSORRT_VERSION
    REQUIRED_VARS TENSORRT_INCLUDE_DIR # no need for libs/targets, since find_library is REQUIRED
)
