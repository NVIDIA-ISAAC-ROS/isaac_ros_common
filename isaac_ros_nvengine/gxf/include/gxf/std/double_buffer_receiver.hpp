/*
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#ifndef DOUBLE_BUFFER_RECEIVER_HPP
#define DOUBLE_BUFFER_RECEIVER_HPP

#include <memory>

#include "gxf/core/component.hpp"
#include "gxf/core/entity.hpp"
#include "gxf/core/handle.hpp"
#include "gxf/std/gems/staging_queue/staging_queue.hpp"
#include "gxf/std/receiver.hpp"

namespace nvidia {
namespace gxf {

// A receiver which uses a double-buffered queue where new messages are first pushed to a
// backstage. Incoming messages are not immediately available and need to be moved to the mainstage
// first.
class DoubleBufferReceiver : public Receiver {
 public:
  using queue_t = ::gxf::staging_queue::StagingQueue<Entity>;

  gxf_result_t registerInterface(Registrar* registrar) override;

  gxf_result_t initialize() override;

  gxf_result_t deinitialize() override;

  gxf_result_t pop_abi(gxf_uid_t* uid) override;

  gxf_result_t push_abi(gxf_uid_t other) override;

  gxf_result_t peek_abi(gxf_uid_t* uid, int32_t index) override;

  gxf_result_t peek_back_abi(gxf_uid_t* uid, int32_t index) override;

  size_t capacity_abi() override;

  size_t size_abi() override;

  gxf_result_t receive_abi(gxf_uid_t* uid) override;

  size_t back_size_abi() override;

  gxf_result_t sync_abi() override;

  Parameter<uint64_t> capacity_;
  Parameter<uint64_t> policy_;

 private:
  std::unique_ptr<queue_t> queue_;
};

}  // namespace gxf
}  // namespace nvidia

#endif
