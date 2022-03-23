/*
Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#ifndef NVIDIA_GXF_CUDA_CUDA_EVENT_HPP_
#define NVIDIA_GXF_CUDA_CUDA_EVENT_HPP_

#include <cuda_runtime.h>

#include <memory>
#include <utility>

#include "gxf/core/component.hpp"
#include "gxf/core/expected.hpp"
#include "gxf/core/handle.hpp"

namespace nvidia {
namespace gxf {

class CudaStream;

// Holds and provides access to cudaEvent_t. The event could be set via initWithEvent() or
// via created by init(flags, dev_id).
class CudaEvent {
 public:
  CudaEvent() = default;
  ~CudaEvent();

  CudaEvent(const CudaEvent&) = delete;
  void operator=(const CudaEvent&) = delete;
  CudaEvent(CudaEvent&& other) {
    resetInternal();
    dev_id_ = other.dev_id_;
    other.dev_id_ = -1;
    event_ = std::move(other.event_);
  }
  void operator=(CudaEvent&& other) {
    resetInternal();
    dev_id_ = other.dev_id_;
    other.dev_id_ = -1;
    event_ = std::move(other.event_);
  }

  friend class CudaStream;

  using EventDestroy = std::function<void(cudaEvent_t)>;

  // Initialize an external event that shall be used
  Expected<void> initWithEvent(cudaEvent_t event, int dev_id = -1, EventDestroy free_fnc = nullptr);
  // Intialize a new event internally
  Expected<void> init(uint32_t flags = 0, int dev_id = -1);
  // Deinitialize cudaevent. In case that user does not called it explicitly,
  // cuda event could be freed in destructor.
  Expected<void> deinit();

  // Retrieves cuda event
  Expected<cudaEvent_t> event() const;
  // Get device id which owns this stream
  int dev_id() const { return dev_id_; }

 private:
  template<class T>
  using PtrT = std::unique_ptr<T, std::function<void(T*)>>;
  using EventPtr = PtrT<cudaEvent_t>;

  void resetInternal();

  static Expected<CudaEvent::EventPtr> createEventInternal(uint32_t flags, int gpuid);
  static Expected<CudaEvent::EventPtr> createEventInternal(
    cudaEvent_t event, EventDestroy free_event);

  int32_t dev_id_ = -1;
  EventPtr event_;
};

}  // namespace gxf
}  // namespace nvidia

#endif  // NVIDIA_GXF_CUDA_CUDA_EVENT_HPP_
