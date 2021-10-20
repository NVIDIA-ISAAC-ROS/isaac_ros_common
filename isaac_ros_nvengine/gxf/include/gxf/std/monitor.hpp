/*
Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#ifndef NVIDIA_GXF_STD_MONITOR_HPP_
#define NVIDIA_GXF_STD_MONITOR_HPP_

#include "gxf/core/component.hpp"

namespace nvidia {
namespace gxf {

// Interface for monitoring entities during runtime
class Monitor : public Component {
 public:
  virtual ~Monitor() = default;

  // Callback for after an entity executes
  //         eid - ID of entity that finished execution
  //   timestamp - execution timestamp
  //        code - execution result
  virtual gxf_result_t on_execute_abi(gxf_uid_t eid, uint64_t timestamp, gxf_result_t code) = 0;

  // C++ API wrapper
  Expected<void> onExecute(gxf_uid_t eid, uint64_t timestamp, gxf_result_t code);
};

}  // namespace gxf
}  // namespace nvidia

#endif  // NVIDIA_GXF_STD_MONITOR_HPP_
