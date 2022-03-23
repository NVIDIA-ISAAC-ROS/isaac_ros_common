/*
Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#ifndef NVIDIA_GXF_CUDA_CUDA_STREAM_POOL_HPP_
#define NVIDIA_GXF_CUDA_CUDA_STREAM_POOL_HPP_

#include <cuda_runtime.h>

#include <memory>
#include <mutex>
#include <queue>
#include <unordered_map>
#include <vector>

#include "gxf/core/component.hpp"
#include "gxf/core/expected.hpp"
#include "gxf/cuda/cuda_stream.hpp"
#include "gxf/std/allocator.hpp"

namespace nvidia {
namespace gxf {

// A memory pools which provides a maximum number of equally sized blocks of
// memory.
class CudaStreamPool : public Allocator {
 public:
  CudaStreamPool() = default;
  ~CudaStreamPool();

  gxf_result_t registerInterface(Registrar* registrar) override;
  gxf_result_t initialize() override;
  gxf_result_t is_available_abi(uint64_t size) override;
  gxf_result_t allocate_abi(uint64_t size, int32_t type, void** pointer) override;
  gxf_result_t free_abi(void* pointer) override;
  gxf_result_t deinitialize() override;

  // Allocate a cudastream for other components
  Expected<Handle<CudaStream>> allocateStream();
  // Free a cudastream
  Expected<void> releaseStream(Handle<CudaStream> stream);

 private:
  Expected<Entity> createNewStreamEntity();
  Expected<void> reserveStreams();

  Parameter<int32_t> dev_id_;
  Parameter<uint32_t> stream_flags_;
  Parameter<int32_t> stream_priority_;
  Parameter<uint32_t> reserved_size_;
  Parameter<uint32_t> max_size_;

  std::mutex mutex_;
  // map of <entity_id, Entity>
  std::unordered_map<gxf_uid_t, std::unique_ptr<Entity>> streams_;
  std::queue<Entity> reserved_streams_;
};

}  // namespace gxf
}  // namespace nvidia

#endif  // NVIDIA_GXF_CUDA_CUDA_STREAM_POOL_HPP_
