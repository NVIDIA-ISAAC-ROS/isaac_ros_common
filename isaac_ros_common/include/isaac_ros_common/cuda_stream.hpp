/*
Copyright 2025 NVIDIA CORPORATION

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#pragma once

#include <cstdio>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

#include "cuda_runtime.h"  // NOLINT - include .h without directory

#define CHECK_CUDA_ERROR(cu_result, fmt, ...) \
  do { \
    cudaError_t err_ = (cu_result); \
    if (err_ != cudaSuccess) { \
      /* Build the user message with optional printf-style args */ \
      char user_msg_[1024]; \
      std::snprintf(user_msg_, sizeof(user_msg_), fmt, ## __VA_ARGS__); \
      /* Build a rich error message with CUDA details */ \
      std::ostringstream oss_; \
      oss_ << user_msg_ << ", cuda_error: " << cudaGetErrorName(err_) \
           << ", error_str: " << cudaGetErrorString(err_); \
      /* Throw a standard C++ exception with the message */ \
      throw std::runtime_error(oss_.str()); \
    } \
  } while (0)

namespace nvidia
{
namespace isaac_ros
{
namespace common
{

cudaError_t initNamedCudaStream(cudaStream_t & stream, const std::string & name);

void nameExistingCudaStream(cudaStream_t & stream, const std::string & name);

}  // namespace common
}  // namespace isaac_ros
}  // namespace nvidia
