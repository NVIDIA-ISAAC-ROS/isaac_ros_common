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
#ifndef NVIDIA_GXF_STD_SCHEDULING_TERM_HPP
#define NVIDIA_GXF_STD_SCHEDULING_TERM_HPP

#include "gxf/core/component.hpp"
#include "gxf/std/scheduling_condition.hpp"

namespace nvidia {
namespace gxf {

/// @brief Base class for scheduling terms
///
/// Scheduling terms are used by a scheduler to determine if codelets in an entity are ready for
/// execution.
class SchedulingTerm : public Component {
 public:
  virtual ~SchedulingTerm() = default;

  // Get the condition on which the scheduling waits before allowing execution. If the term is
  // waiting for a time event 'target_timestamp' will contain the target timestamp.
  virtual gxf_result_t check_abi(int64_t timestamp, SchedulingConditionType* type,
                                 int64_t* target_timestamp) const = 0;

  // Called each time after the entity of this term was executed.
  virtual gxf_result_t onExecute_abi(int64_t dt) = 0;

  Expected<SchedulingCondition> check(int64_t timestamp) {
    SchedulingConditionType status;
    int64_t target_timestamp;
    const gxf_result_t error = check_abi(timestamp, &status, &target_timestamp);
    return ExpectedOrCode(error, SchedulingCondition{status, target_timestamp});
  }

  Expected<void> onExecute(int64_t timestamp) { return ExpectedOrCode(onExecute_abi(timestamp)); }
};

}  // namespace gxf
}  // namespace nvidia

#endif
