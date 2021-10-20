/*
Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#ifndef NVIDIA_GXF_STD_SCHEDULING_CONDITION_HPP
#define NVIDIA_GXF_STD_SCHEDULING_CONDITION_HPP
#include <stdint.h>

namespace nvidia {
namespace gxf {

// The type of a scheduling condition
enum class SchedulingConditionType {
  // Will never execute again
  NEVER = 0,
  // Ready to execute now
  READY = 1,
  // May execute again at some point in the future
  WAIT = 2,
  // Will execute after a certain known time interval. Negative or zero interval will result in
  // immediate execution.
  WAIT_TIME = 3,
  // Waiting for an event with unknown interval time. Entity will be put in a waiting queue until
  // event done notification is signalled
  WAIT_EVENT = 4,
};

// A condition for which the scheduling term is waiting.
struct SchedulingCondition {
  // Describes the type of the condition
  SchedulingConditionType type;
  // A timestamp needed for certain conditions. Might not be set always.
  int64_t target_timestamp;
};

// Merges two scheduling conditions with an And-like logic. For example NEVER and READY will result
// in NEVER.
SchedulingCondition AndCombine(SchedulingCondition a, SchedulingCondition b);

}  // namespace gxf
}  // namespace nvidia

#endif  // NVIDIA_GXF_STD_SCHEDULING_CONDITION_HPP
