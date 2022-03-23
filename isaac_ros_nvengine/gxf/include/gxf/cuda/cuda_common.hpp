/*
Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#ifndef NVIDIA_GXF_CUDA_CUDA_COMMON_HPP_
#define NVIDIA_GXF_CUDA_CUDA_COMMON_HPP_

#include <cuda_runtime.h>

#include "common/assert.hpp"
#include "common/logger.hpp"

#define CHECK_CUDA_ERROR(cu_result, fmt, ...)                                  \
    do {                                                                       \
        cudaError_t err = (cu_result);                                          \
        if (err != cudaSuccess) {                                               \
            GXF_LOG_ERROR(fmt ", cuda_error: %s, error_str: %s", ##__VA_ARGS__, \
                cudaGetErrorName(err), cudaGetErrorString(err));                \
            return Unexpected{GXF_FAILURE};                                     \
        }                                                                       \
    } while (0)

#define CONTINUE_CUDA_ERROR(cu_result, fmt, ...)                               \
    do {                                                                       \
        cudaError_t err = (cu_result);                                          \
        if (err != cudaSuccess) {                                               \
            GXF_LOG_ERROR(fmt ", cuda_error: %s, error_str: %s", ##__VA_ARGS__, \
                cudaGetErrorName(err), cudaGetErrorString(err));                \
        }                                                                       \
    } while (0)

#endif  // NVIDIA_GXF_CUDA_CUDA_COMMON_HPP_
