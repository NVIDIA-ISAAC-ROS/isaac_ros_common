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
#pragma once

#include <cstdint>

#include "gxf/core/component.hpp"

namespace nvidia {
namespace gxf {

// Codelets are special components which allow the execution of custom code. The user can
// create her own codelets by deriving from this class and overriding the functions initialize,
// start, tick, stop, and deinitialize.
class Codelet : public Component {
 public:
  virtual ~Codelet() = default;

  // This function is called during the start phase of the codelet. It allows derived classes to
  // execute custom code during the start phase. This is a good place to obtain resources which
  // are necessary for ticking the codelet. This function is guaranteed to be called before the
  // first call to tick.
  virtual gxf_result_t start() = 0;

  // This function is called whenever the codelet is expected to do work, e.g. when an event was
  // received or periodically. The tick method can be specified with various other member functions.
  // This function is the main work horse of the codelet.
  virtual gxf_result_t tick() = 0;

  // This function is called during the stop phase of the codelet. It allows derived classes to
  // execute custom code during the stop phase. This is a good place to clean up any resources which
  // where obtained during 'start'. After the codelet is stopped it should be in the same state as
  // it was before 'start' was called. Be careful to not leave any unintended left overs as 'start'
  // might be called again afterwards. It is guaranteed that stop is called after the last
  // call to tick. When start was called stop will be called, too.
  virtual gxf_result_t stop() = 0;

  // Timestamp (in nanoseconds) of the beginning of the start, tick or stop function. The execution
  // timestamp does not change during the start, tick or stop function.
  int64_t getExecutionTimestamp() const { return execution_timestamp_; }

  // Similar to getExecutionTimestamp but returns time as a floating point number and using seconds
  // as unit. Equivalent to 'ToSeconds(getExecutionCount())'.
  double getExecutionTime() const { return execution_time_; }

  // The delta between the current execution time and the execution time of the previous execution.
  // During the start function this will return 0.
  double getDeltaTime() const { return delta_time_; }

  // Returns the number of times a codelet is executed. This will return 0 during start and 1 during
  // the first tick.
  int64_t getExecutionCount() const { return execution_count_; }

  // Returns true if this is the first time tick is called after start.
  bool isFirstTick() const { return getExecutionCount() == 1; }

 private:
  // Class is friend to allow EntityExecutor to call private member functions
  friend class EntityExecutor;

  // Called by EntityExecutor before each 'start'
  void beforeStart(int64_t timestamp);

  // Called by EntityExecutor before each 'tick'
  void beforeTick(int64_t timestamp);

  // Called by EntityExecutor before each 'stop'
  void beforeStop();

  // The number of times the codelet tick function was called.
  int64_t execution_count_;
  // The timestamp of the previous execution. Equal to 'execution_timestamp' during 'start'.
  int64_t previous_execution_timestamp_;
  // The timestamp of the current execution in nanoseconds.
  int64_t execution_timestamp_;
  // Same as execution_timestamp_ but in seconds and as a floating point.
  double execution_time_;
  // The difference between the current and the previous execution time in seconds.
  double delta_time_;
};

}  // namespace gxf
}  // namespace nvidia
