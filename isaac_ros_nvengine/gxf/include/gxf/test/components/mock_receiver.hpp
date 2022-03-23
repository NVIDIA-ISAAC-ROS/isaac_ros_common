/*
Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#ifndef NVIDIA_GXF_TEST_COMPONENTS_MOCK_RECEIVER_HPP_
#define NVIDIA_GXF_TEST_COMPONENTS_MOCK_RECEIVER_HPP_

#include <deque>
#include <shared_mutex>  // NOLINT

#include "gxf/std/receiver.hpp"

namespace nvidia {
namespace gxf {
namespace test {

// Entity receiver used to facilitate testing
class MockReceiver : public Receiver {
 public:
  gxf_result_t registerInterface(Registrar* registrar) override;
  gxf_result_t initialize() override;
  gxf_result_t deinitialize() override;

  gxf_result_t pop_abi(gxf_uid_t* uid) override;
  gxf_result_t push_abi(gxf_uid_t other) override;
  gxf_result_t peek_abi(gxf_uid_t* uid, int32_t index) override;
  size_t capacity_abi() override { return max_capacity_; }
  size_t size_abi() override;

  gxf_result_t receive_abi(gxf_uid_t* uid) override;
  size_t back_size_abi() override { return 0; }
  gxf_result_t peek_back_abi(gxf_uid_t* uid, int32_t index) override { return GXF_NOT_IMPLEMENTED; }
  gxf_result_t sync_abi() override;

 private:
  // Structure to organize performance metrics
  struct Metrics {
    size_t received;
    size_t syncs;
    size_t peak;
  };

  // Checks if all entities were received
  Expected<void> checkForUnreceivedEntities();
  // Prints performance metrics
  void printMetrics();

  Parameter<bool> ignore_unreceived_entities_;
  Parameter<bool> fail_on_receive_;
  Parameter<bool> fail_on_sync_;
  Parameter<size_t> max_capacity_;

  // Entity buffer
  std::deque<Entity> entities_;
  // Performance metrics
  Metrics metrics_;
  // Mutex for guarding concurrent access to buffer and metrics
  std::shared_timed_mutex mutex_;
};

}  // namespace test
}  // namespace gxf
}  // namespace nvidia

#endif  // NVIDIA_GXF_TEST_COMPONENTS_MOCK_RECEIVER_HPP_
