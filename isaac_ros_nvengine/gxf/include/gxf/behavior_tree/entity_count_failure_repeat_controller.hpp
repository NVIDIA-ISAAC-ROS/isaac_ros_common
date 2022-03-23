/*
Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#ifndef NVIDIA_GXF_BT_ENTITY_COUNT_FAILURE_REPEAT_CONTROLLER_HPP_
#define NVIDIA_GXF_BT_ENTITY_COUNT_FAILURE_REPEAT_CONTROLLER_HPP_

#include "gxf/std/controller.hpp"
#include "gxf/std/parameter_parser_std.hpp"

namespace nvidia {
namespace gxf {

// Repeat the entity on failure up to |max_repeat_count_| times, then deactivate
// the entity
class EntityCountFailureRepeatController : public Controller {
 public:
  gxf_result_t registerInterface(Registrar* registrar) override;
  gxf_result_t initialize() override;
  gxf_controller_status_t control(gxf_uid_t eid, Expected<void> code) override;

 private:
  size_t repeat_count_;
  Parameter<size_t> max_repeat_count_;
  gxf_controller_status_t controller_status;
  // Take gxf_result_t codelet::tick() to entity_state_t
  // controller_status.behavior_status
  entity_state_t setBehaviorStatus(gxf_result_t tick_result);
  // Set execution status based on behavior status and repeat counter
  gxf_execution_status_t setExecStatus(entity_state_t& behavior_status);
  // if code = EntityItem::execute() returns FAILURE & exec_status !=
  // GXF_EXECUTE_FAILURE set behavior status to GXF_BEHAVIOR_RUNNING so that
  // parent codelet knows it's running
  Parameter<bool> return_behavior_running_if_failure_repeat_;
};

}  // namespace gxf
}  // namespace nvidia

#endif  // NVIDIA_GXF_BT_ENTITY_COUNT_FAILURE_REPEAT_CONTROLLER_HPP_
