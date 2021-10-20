/*
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#ifndef NVIDIA_GXF_STD_SCHEDULING_TERMS_HPP_
#define NVIDIA_GXF_STD_SCHEDULING_TERMS_HPP_

#include <string>
#include <utility>
#include <vector>

#include "gxf/core/component.hpp"
#include "gxf/core/handle.hpp"
#include "gxf/std/clock.hpp"
#include "gxf/std/parameter_parser_std.hpp"
#include "gxf/std/receiver.hpp"
#include "gxf/std/scheduling_term.hpp"
#include "gxf/std/transmitter.hpp"

namespace nvidia {
namespace gxf {

// A scheduling term which permits execution only after a minium time period has passed since the
// last execution.
class PeriodicSchedulingTerm : public SchedulingTerm {
 public:
  gxf_result_t registerInterface(Registrar* registrar) override;
  gxf_result_t initialize() override;
  gxf_result_t check_abi(int64_t timestamp, SchedulingConditionType* type,
                         int64_t* target_timestamp) const override;
  gxf_result_t onExecute_abi(int64_t timestamp) override;

  // Minimum time which needs to elapse between two executions (in nano seconds).
  int64_t recess_period_ns() const { return recess_period_ns_; }

 private:
  // Parses given text to return the desired period in nanoseconds.
  Expected<int64_t> parseRecessPeriodString(std::string text);

  Parameter<std::string> recess_period_;

  int64_t recess_period_ns_;
  Expected<int64_t> last_run_timestamp_ = Unexpected{GXF_UNINITIALIZED_VALUE};
};

// A scheduling term which permits execution only a limited number of times.
class CountSchedulingTerm : public SchedulingTerm {
 public:
  gxf_result_t registerInterface(Registrar* registrar) override;
  gxf_result_t initialize() override;
  gxf_result_t check_abi(int64_t timestamp, SchedulingConditionType* type,
                         int64_t* target_timestamp) const override;
  gxf_result_t onExecute_abi(int64_t timestamp) override;

 private:
  Parameter<int64_t> count_;

  int64_t remaining_;  // The remaining number of permitted executions.
};

// A component which specifies that an entity shall be executed if the receiver for a given
// transmitter can accept new messages.
class DownstreamReceptiveSchedulingTerm : public SchedulingTerm {
 public:
  gxf_result_t registerInterface(Registrar* registrar) override;
  gxf_result_t check_abi(int64_t timestamp, SchedulingConditionType* type,
                         int64_t* target_timestamp) const override;
  gxf_result_t onExecute_abi(int64_t timestamp) override;

  // The transmitter which needs to be able to publish a message.
  Handle<Transmitter> transmitter() const { return transmitter_.get(); }
  // The receiver which is connected to the transmitter and which needs to have room for messages.
  void setReceiver(Handle<Receiver> receiver) { receiver_ = std::move(receiver); }

 private:
  Parameter<Handle<Transmitter>> transmitter_;
  Parameter<uint64_t> min_size_;

  Handle<Receiver> receiver_;  // The receiver connected to the transmitter (if any).
};

// A scheduling term which permits execution at a user-specified timestamp. The timestamp is
// specified on the clock provided.
class TargetTimeSchedulingTerm : public SchedulingTerm {
 public:
  gxf_result_t registerInterface(Registrar* registrar) override;
  gxf_result_t initialize() override;
  gxf_result_t check_abi(int64_t timestamp, SchedulingConditionType* type,
                         int64_t* target_timestamp) const override;
  gxf_result_t onExecute_abi(int64_t timestamp) override;

  // Needs to be called to determine how long to wait for the next execution. If it is not called,
  // the scheduling condition will stay as WAIT.
  gxf_result_t setNextTargetTime(int64_t target_timestamp);

 private:
  Parameter<Handle<Clock>> clock_;

  // The timestamp at which the most recent execution cycle began
  int64_t last_timestamp_;
  // The timestamp at which the next execution cycle is requested to begin
  mutable Expected<int64_t> target_timestamp_ = Unexpected{GXF_UNINITIALIZED_VALUE};
  // The timestamp at which the next execution cycle is locked to begin
  mutable Expected<int64_t> locked_target_timestamp_ = Unexpected{GXF_UNINITIALIZED_VALUE};
};

// A component which specifies that an entity shall be executed when a queue has at least a certain
// number of elements.
class MessageAvailableSchedulingTerm : public SchedulingTerm {
 public:
  gxf_result_t registerInterface(Registrar* registrar) override;
  gxf_result_t check_abi(int64_t timestamp, SchedulingConditionType* type,
                         int64_t* target_timestamp) const override;
  gxf_result_t onExecute_abi(int64_t timestamp) override;

 private:
  // Returns true if the condition imposed by min_size is true.
  bool checkMinSize() const;

  // Returns true if the condition imposed by front_stage_max_size is true.
  bool checkFrontStageMaxSize() const;

  Parameter<Handle<Receiver>> receiver_;
  Parameter<size_t> min_size_;
  Parameter<size_t> front_stage_max_size_;
};

// A scheduling term which specifies that an entity can be executed when a list of provided input
// channels combined have at least a given number of messages.
class MultiMessageAvailableSchedulingTerm : public SchedulingTerm {
 public:
  gxf_result_t registerInterface(Registrar* registrar) override;
  gxf_result_t check_abi(int64_t timestamp, SchedulingConditionType* type,
                         int64_t* target_timestamp) const override;
  gxf_result_t onExecute_abi(int64_t timestamp) override;

 private:
  Parameter<std::vector<Handle<Receiver>>> receivers_;
  Parameter<size_t> min_size_;
};

// A scheduling term which tries to wait for specified number of messages in receiver.
// When the first message in the queue mature after specified delay since arrival it would fire
// regardless.
class ExpiringMessageAvailableSchedulingTerm : public SchedulingTerm {
 public:
  gxf_result_t registerInterface(Registrar* registrar) override;
  gxf_result_t initialize() override;
  gxf_result_t check_abi(int64_t timestamp, SchedulingConditionType* type,
                         int64_t* target_timestamp) const override;
  gxf_result_t onExecute_abi(int64_t timestamp) override;

 private:
  Parameter<int64_t> max_batch_size_;
  Parameter<int64_t> max_delay_ns_;
  Parameter<Handle<Receiver>> receiver_;  // The receiver to check
  Parameter<Handle<Clock>> clock_;
};

// A scheduling term which acts as a boolean AND term to control execution of the
// entity
class BooleanSchedulingTerm : public SchedulingTerm {
 public:
  // Enable entity execution
  void enable_tick();
  // Disable entity execution
  void disable_tick();

  gxf_result_t check_abi(int64_t timestamp, SchedulingConditionType* type,
                         int64_t* target_timestamp) const override;
  gxf_result_t onExecute_abi(int64_t dt) override;

 private:
  bool enable_tick_ = true;
};

enum class AsynchronousEventState {
  READY = 0,             // Init state, first tick is pending
  WAIT,                  // Request to async service yet to be sent, nothing to do but wait
  EVENT_WAITING,         // Request sent to an async service, pending event done notification
  EVENT_DONE,            // Event done notification received, entity ready to be ticked
  EVENT_NEVER,           // Entity does not want to be ticked again, end of execution
};

// A scheduling term which waits on an asynchronous event from the codelet which can happen outside
// of the regular tick function.
class AsynchronousSchedulingTerm : public SchedulingTerm {
 public:
  gxf_result_t initialize() override;
  gxf_result_t check_abi(int64_t timestamp, SchedulingConditionType* type,
                         int64_t* target_timestamp) const override;
  gxf_result_t onExecute_abi(int64_t dt) override;
  void setEventState(AsynchronousEventState state);
  AsynchronousEventState getEventState() const;

 private:
  AsynchronousEventState event_state_{AsynchronousEventState::READY};
  mutable std::mutex event_state_mutex_;
};

}  // namespace gxf
}  // namespace nvidia

#endif  // NVIDIA_GXF_STD_SCHEDULING_TERMS_HPP_
