/*
Copyright (c) 2021 NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#ifndef NVIDIA_GXF_BT_CONSTANT_BEHAVIOR_HPP_
#define NVIDIA_GXF_BT_CONSTANT_BEHAVIOR_HPP_

#include "gxf/std/codelet.hpp"
#include "gxf/std/controller.hpp"
#include "gxf/std/scheduling_terms.hpp"

namespace nvidia {
namespace gxf {

// Constant Behavior Codelet switches its own status to the configured desired
// ||constant_status| after each tick.
class ConstantBehavior : public Codelet {
 public:
  virtual ~ConstantBehavior() = default;

  gxf_result_t registerInterface(Registrar* registrar) override;
  gxf_result_t initialize() override;
  gxf_result_t start() override { return GXF_SUCCESS; }
  gxf_result_t tick() override;
  gxf_result_t stop() override { return GXF_SUCCESS; }

 private:
  // the desired status to switch to during each tick
  Parameter<size_t> constant_status_;
  enum ConstantBehaviorType {
    CONSTANT_SUCCESS = 0,
    CONSTANT_FAILURE = 1,
  };

  // its own scheduling term to start/stop itself
  Parameter<Handle<nvidia::gxf::BTSchedulingTerm>> s_term_;

  SchedulingConditionType ready_conditions;
  SchedulingConditionType never_conditions;
};

}  // namespace gxf
}  // namespace nvidia

#endif  // NVIDIA_GXF_BT_CONSTANT_BEHAVIOR_HPP_
