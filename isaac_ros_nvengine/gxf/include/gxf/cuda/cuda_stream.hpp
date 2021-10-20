/*
Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#ifndef NVIDIA_GXF_CUDA_CUDA_STREAM_HPP_
#define NVIDIA_GXF_CUDA_CUDA_STREAM_HPP_

#include <cuda_runtime.h>

#include <functional>
#include <mutex>
#include <queue>
#include <vector>

#include "gxf/core/component.hpp"
#include "gxf/core/expected.hpp"
#include "gxf/core/handle.hpp"
#include "gxf/cuda/cuda_event.hpp"


namespace nvidia {
namespace gxf {

class CudaStreamPool;

// Holds and provides access to cudaStream_t. CudaStream is allocated and
// recycled by CudaStreamPool
class CudaStream {
 public:
  CudaStream() = default;
  ~CudaStream();

  CudaStream(const CudaStream&) = delete;
  CudaStream(CudaStream&&) = delete;
  void operator=(const CudaStream&) = delete;

  using EventDestroy = std::function<void(cudaEvent_t)>;
  using SyncedCallback = std::function<void()>;

  // Retrieves cudaSteam_t
  Expected<cudaStream_t> stream() const;
  // Get device id which owns this stream
  int dev_id() const { return dev_id_; }

  // Record event to extend Entity life until event synchronized.
  Expected<void> record(Handle<CudaEvent> event, Entity input_entity,
                        SyncedCallback synced_cb = nullptr);
  // Record event on the stream for an async callback.
  // The callback would be delayed untill CudaStreamSync ticks.
  // The Callback usually is used to recyle dependent resources.
  // If record failed, callback would not be called. User need to check return results.
  Expected<void> record(cudaEvent_t event, EventDestroy cb);

  // Reset all events and callback all the hook functions to release resource.
  Expected<void> resetEvents();

  // Sync all streams, meanwhile clean all recorded events and callback recycle functions
  Expected<void> syncStream();

 private:
  friend class CudaStreamPool;
  // Initialize new cuda stream if was not set by external
  Expected<void> initialize(uint32_t flags = 0, int dev_id = -1, int32_t priority = 0);
  Expected<void> deinitialize();

  Expected<void> prepareResourceInternal(int dev_id);

  Expected<void> recordEventInternal(cudaEvent_t e);
  Expected<void> syncEventInternal(cudaEvent_t e);

  Expected<void> resetEventsInternal(std::queue<CudaEvent::EventPtr>& q);

  mutable std::shared_timed_mutex mutex_;
  int dev_id_ = 0;
  cudaStream_t stream_ = 0;

  // store all recorded event with destory functions.
  std::queue<CudaEvent::EventPtr> recorded_event_queue_;
  // event is defined for for synchronization of stream
  CudaEvent::EventPtr sync_event_;
};

}  // namespace gxf
}  // namespace nvidia

#endif  // NVIDIA_GXF_CUDA_CUDA_STREAM_HPP_
