/*
Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#ifndef NVIDIA_GXF_TEST_EXTENSIONS_ENTITY_MONITOR_HPP_
#define NVIDIA_GXF_TEST_EXTENSIONS_ENTITY_MONITOR_HPP_

#include "gxf/std/monitor.hpp"

namespace nvidia {
namespace gxf {
namespace test {

// Monitors entity execution during runtime and logs status to console
class EntityMonitor : public Monitor {
 public:
  gxf_result_t on_execute_abi(gxf_uid_t eid, uint64_t timestamp, gxf_result_t code) override;
};

}  // namespace test
}  // namespace gxf
}  // namespace nvidia

#endif  // NVIDIA_GXF_TEST_EXTENSIONS_ENTITY_MONITOR_HPP_
