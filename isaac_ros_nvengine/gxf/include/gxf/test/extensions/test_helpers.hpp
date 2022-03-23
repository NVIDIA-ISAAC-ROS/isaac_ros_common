/*
Copyright (c) 2020-2021, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#ifndef NVIDIA_GXF_TEST_EXTENSIONS_TEST_HELPERS_HPP
#define NVIDIA_GXF_TEST_EXTENSIONS_TEST_HELPERS_HPP

#include <inttypes.h>

#include <atomic>
#include <cmath>
#include <cstring>
#include <string>
#include <thread>
#include <utility>

#include "common/logger.hpp"
#include "gxf/core/component.hpp"
#include "gxf/core/entity.hpp"
#include "gxf/std/allocator.hpp"
#include "gxf/std/clock.hpp"
#include "gxf/std/codelet.hpp"
#include "gxf/std/receiver.hpp"
#include "gxf/std/scheduling_terms.hpp"
#include "gxf/std/tensor.hpp"
#include "gxf/std/timestamp.hpp"
#include "gxf/std/transmitter.hpp"

namespace nvidia {
namespace gxf {
namespace test {

// Tests that the component has certain parameters set to correct values
class ParameterTest : public Component {
 public:
  gxf_result_t initialize() override {
    if (fact_ != true) return GXF_FAILURE;
    if (rumor_ != false) return GXF_FAILURE;
    if (forty_two_ != 42) return GXF_FAILURE;
    if (minus_one_ != -1) return GXF_FAILURE;
    if (some_text_.get() != std::string("hello")) return GXF_FAILURE;
    constexpr const char* expected_more =
        "- a: st\n  b: ee\n- c: an\n  d: en\n- e:\n    - f: figy\n      g: g";
    if (more_.get() != std::string(expected_more)) return GXF_FAILURE;
    return GXF_SUCCESS;
  }

  gxf_result_t registerInterface(Registrar* registrar) override {
    Expected<void> result;
    result &= registrar->parameter(fact_, "fact");
    result &= registrar->parameter(rumor_, "rumor");
    result &= registrar->parameter(forty_two_, "forty_two");
    result &= registrar->parameter(minus_one_, "minus_one");
    result &= registrar->parameter(some_text_, "some_text");
    result &= registrar->parameter(more_, "more");
    return ToResultCode(result);
  }

 private:
  Parameter<bool> fact_;
  Parameter<bool> rumor_;
  Parameter<int> forty_two_;
  Parameter<int> minus_one_;
  Parameter<std::string> some_text_;
  Parameter<std::string> more_;
};

// Tests that the codelet was stepped a certain number of times.
class StepCount : public Codelet {
 public:
  gxf_result_t initialize() override {
    is_initialized_ = true;
    count_start_ = 0;
    count_tick_ = 0;
    count_stop_ = 0;
    return GXF_SUCCESS;
  }

  gxf_result_t start() override {
    if (use_assert_) {
      GXF_ASSERT(is_initialized_, "not initialized");
    } else {
      if (!is_initialized_) return GXF_CONTRACT_INVALID_SEQUENCE;
    }
    count_start_++;
    return GXF_SUCCESS;
  }

  gxf_result_t tick() override {
    if (use_assert_) {
      GXF_ASSERT_GT(count_start_, 0);
    } else {
      if (count_start_ == 0) return GXF_CONTRACT_INVALID_SEQUENCE;
    }
    count_tick_++;
    return GXF_SUCCESS;
  }

  gxf_result_t stop() override {
    if (use_assert_) {
      GXF_ASSERT_EQ(count_start_, count_stop_ + 1);
    } else {
      if (count_start_ != count_stop_ + 1) return GXF_CONTRACT_INVALID_SEQUENCE;
    }
    count_stop_++;
    return GXF_SUCCESS;
  }

  gxf_result_t deinitialize() override {
    if (use_assert_) {
      GXF_ASSERT(is_initialized_, "not initialized");
      GXF_ASSERT_EQ(count_start_, expected_start_count_.get());
      GXF_ASSERT_EQ(count_tick_, expected_count_.get());
      GXF_ASSERT_EQ(count_start_, count_stop_);
    } else {
      if (!is_initialized_) return GXF_CONTRACT_INVALID_SEQUENCE;
      if (count_start_ != expected_start_count_) return GXF_FAILURE;
      if (count_tick_ != expected_count_) return GXF_FAILURE;
      if (count_start_ != count_stop_) return GXF_CONTRACT_INVALID_SEQUENCE;
    }
    return GXF_SUCCESS;
  }

  gxf_result_t registerInterface(Registrar* registrar) override {
    Expected<void> result;
    result &= registrar->parameter(use_assert_, "use_assert", "Use ASSERT",
                                   "If enabled the codelet "
                                   "will assert when test conditions are not true",
                                   false);
    result &= registrar->parameter(expected_start_count_, "expected_start_count", "", "", 1UL);
    result &= registrar->parameter(expected_count_, "expected_count");
    return ToResultCode(result);
  }

 private:
  Parameter<bool> use_assert_;
  Parameter<uint64_t> expected_start_count_;
  Parameter<uint64_t> expected_count_;

  bool is_initialized_ = false;
  uint64_t count_start_;
  uint64_t count_tick_;
  uint64_t count_stop_;
};

// Sends an entity
class PingTx : public Codelet {
 public:
  gxf_result_t start() override { return GXF_SUCCESS; }

  gxf_result_t tick() override {
    auto message = Entity::New(context());
    if (!message) {
      GXF_LOG_ERROR("Failure creating message entity.");
      return message.error();
    }
    auto maybe_clock = clock_.try_get();
    int64_t now;
    if (maybe_clock) {
      now = maybe_clock.value()->timestamp();
    } else {
      now = 0;
    }
    auto result = signal_->publish(message.value(), now);
    return ToResultCode(message);
  }

  gxf_result_t stop() override { return GXF_SUCCESS; }

  gxf_result_t registerInterface(Registrar* registrar) override {
    Expected<void> result;
    result &= registrar->parameter(signal_, "signal");
    result &= registrar->parameter(clock_, "clock", "Clock", "Clock for Timestamp",
                                   Registrar::NoDefaultParameter(), GXF_PARAMETER_FLAGS_OPTIONAL);
    return ToResultCode(result);
  }

  Parameter<Handle<Transmitter>> signal_;
  Parameter<Handle<Clock>> clock_;
};

// Sends a ping after a user-defined delay. If the offset clock parameter is set, an offset is
// added to the acqtime and pubtime before publishing.
class ScheduledPingTx : public Codelet {
 public:
  gxf_result_t start() override {
    // Set the target time so that we tick
    scheduling_term_->setNextTargetTime(execution_clock_->timestamp());

    return GXF_SUCCESS;
  }

  gxf_result_t tick() override {
    auto message = Entity::New(context());
    if (!message) return GXF_FAILURE;

    // Compute the timestamp when the codelet should be executed next
    const int64_t target_timestamp = execution_clock_->timestamp() + delay_;
    scheduling_term_->setNextTargetTime(target_timestamp);

    // Add a timestamp field to the ping message with acquisition and publishing times of the ping
    auto timestamp = message.value().add<Timestamp>("timestamp");
    if (!timestamp) {
      return ToResultCode(timestamp);
    }
    timestamp.value()->acqtime = target_timestamp;
    timestamp.value()->pubtime = target_timestamp;

    // If an offset clock is specified add the offset time
    const auto& maybe_offset_clock = offset_clock_.try_get();
    if (maybe_offset_clock) {
      timestamp.value()->acqtime = maybe_offset_clock.value()->timestamp() + target_timestamp;
      timestamp.value()->pubtime = maybe_offset_clock.value()->timestamp() + target_timestamp;
    }
    const auto result = signal_->publish(std::move(message.value()));
    return ToResultCode(message);
  }

  gxf_result_t stop() override { return GXF_SUCCESS; }

  gxf_result_t registerInterface(Registrar* registrar) override {
    Expected<void> result;
    result &= registrar->parameter(delay_, "delay");
    result &= registrar->parameter(execution_clock_, "execution_clock");
    result &= registrar->parameter(signal_, "signal");
    result &= registrar->parameter(scheduling_term_, "scheduling_term");
    result &= registrar->parameter(offset_clock_, "offset_clock", "", "",
                                   Registrar::NoDefaultParameter(), GXF_PARAMETER_FLAGS_OPTIONAL);
    return ToResultCode(result);
  }
  Parameter<int64_t> delay_;
  Parameter<Handle<Clock>> execution_clock_;
  Parameter<Handle<Clock>> offset_clock_;
  Parameter<Handle<Transmitter>> signal_;
  Parameter<Handle<TargetTimeSchedulingTerm>> scheduling_term_;
};

// Sends an entity asynchronously
// Uses the AsynchronousSchedulingTerm to asynchronously ping the gxf context
// that it is ready to send a new message. Sends at max "count" number of messages.
class AsyncPingTx : public Codelet {
 public:
  void async_ping() {
    GXF_LOG_INFO("Async ping thread entering.");
    while (true) {
      if (should_stop_) {
        GXF_LOG_INFO("Async ping thread exiting.");
        return;
      }

      std::this_thread::sleep_for(std::chrono::milliseconds(delay_));

     if (scheduling_term_->getEventState() == AsynchronousEventState::EVENT_WAITING) {
        scheduling_term_->setEventState(AsynchronousEventState::EVENT_DONE);
      }
    }
  }

  gxf_result_t start() override {
    async_thread_ = std::thread([this] { async_ping(); });
    return GXF_SUCCESS;
  }

  gxf_result_t tick() override {
    auto message = Entity::New(context());
    if (!message) {
      GXF_LOG_ERROR("Failure creating message entity.");
      return message.error();
    }

    ++tick_count_;
    if (tick_count_ == count_) {
      // Reached max count of ticks
      scheduling_term_->setEventState(AsynchronousEventState::EVENT_NEVER);
    } else {
      scheduling_term_->setEventState(AsynchronousEventState::EVENT_WAITING);
    }
    const auto result = signal_->publish(std::move(message.value()));
    return ToResultCode(message);
  }

  gxf_result_t stop() override {
    should_stop_ = true;
    async_thread_.join();

    return GXF_SUCCESS;
  }

  gxf_result_t registerInterface(Registrar* registrar) override {
    Expected<void> result;
    result &= registrar->parameter(delay_, "delay", "Ping delay in ms", "", 10L);
    result &= registrar->parameter(signal_, "signal");
    result &= registrar->parameter(count_, "count", "Ping count", "", 0UL);
    result &= registrar->parameter(scheduling_term_, "scheduling_term");
    return ToResultCode(result);
  }

 private:
  std::atomic<uint64_t> tick_count_{0};
  std::atomic<bool> should_stop_{false};
  std::thread async_thread_;
  Parameter<int64_t> delay_;
  Parameter<Handle<Transmitter>> signal_;
  Parameter<Handle<AsynchronousSchedulingTerm>> scheduling_term_;
  Parameter<uint64_t> count_;
};

// Receives an entity
class PingRx : public Codelet {
 public:
  gxf_result_t start() override { return GXF_SUCCESS; }

  gxf_result_t tick() override {
    auto message = signal_->receive();
    if (!message || message.value().is_null()) {
      return GXF_CONTRACT_MESSAGE_NOT_AVAILABLE;
    }
    return GXF_SUCCESS;
  }

  gxf_result_t stop() override { return GXF_SUCCESS; }

  gxf_result_t registerInterface(Registrar* registrar) override {
    Expected<void> result;
    result &= registrar->parameter(signal_, "signal");
    return ToResultCode(result);
  }

  Parameter<Handle<Receiver>> signal_;
};

// Receives an entity asynchronously
// Uses the AsynchronousSchedulingTerm to asynchronously ping the gxf context
// that it is ready to send a new message.
class AsyncPingRx : public Codelet {
 public:
  void async_ping() {
    GXF_LOG_INFO("Async ping thread entering.");
    while (true) {
      if (should_stop_) {
        GXF_LOG_INFO("Async ping thread exiting.");
        return;
      }

      std::this_thread::sleep_for(std::chrono::milliseconds(delay_));
      if (scheduling_term_->getEventState() == AsynchronousEventState::EVENT_WAITING) {
        scheduling_term_->setEventState(AsynchronousEventState::EVENT_DONE);
      }
    }
  }

  gxf_result_t start() override {
    async_thread_ = std::thread([this] { async_ping(); });
    return GXF_SUCCESS;
  }

  gxf_result_t tick() override {
    size_t size = signal_->size();
    if (size > 0) {
      auto message = signal_->receive();
      if (!message || message.value().is_null()) {
        return GXF_CONTRACT_MESSAGE_NOT_AVAILABLE;
      }
      scheduling_term_->setEventState(AsynchronousEventState::EVENT_WAITING);
    } else {
      GXF_LOG_DEBUG("No messages to read from receiver queue!");
    }
    return GXF_SUCCESS;
  }

  gxf_result_t stop() override {
    should_stop_ = true;
    async_thread_.join();
    return GXF_SUCCESS;
  }

  gxf_result_t registerInterface(Registrar* registrar) override {
    Expected<void> result;
    result &= registrar->parameter(delay_, "delay", "Ping delay in ms", "", 10L);
    result &= registrar->parameter(signal_, "signal");
    result &= registrar->parameter(scheduling_term_, "scheduling_term");
    return ToResultCode(result);
  }

  std::atomic<bool> should_stop_{false};
  std::thread async_thread_;
  Parameter<int64_t> delay_;
  Parameter<Handle<AsynchronousSchedulingTerm>> scheduling_term_;
  Parameter<Handle<Receiver>> signal_;
};

// Receives an entity
class PingPollRx : public Codelet {
 public:
  gxf_result_t start() override {
    counter_ = 0;
    return GXF_SUCCESS;
  }

  gxf_result_t tick() override {
    auto message = signal_->receive();
    if (message && !message.value().is_null()) {
      counter_++;
    }
    return GXF_SUCCESS;
  }

  gxf_result_t stop() override {
    if (counter_ != expected_counter_) {
      GXF_LOG_ERROR("counter does not match: %lld vs %lld", counter_, expected_counter_);
      return GXF_FAILURE;
    }
    return GXF_SUCCESS;
  }

  gxf_result_t registerInterface(Registrar* registrar) override {
    Expected<void> result;
    result &= registrar->parameter(signal_, "signal");
    result &= registrar->parameter(expected_counter_, "expected_counter");
    return ToResultCode(result);
  }

  Parameter<Handle<Receiver>> signal_;
  Parameter<uint64_t> expected_counter_;
  uint64_t counter_ = 0;
};

// Receives an entity in specified batch size
class PingBatchRx : public Codelet {
 public:
  gxf_result_t start() override { return GXF_SUCCESS; }

  gxf_result_t tick() override {
    for (int64_t i = 0; i < batch_size_; i++) {
      auto message = signal_->receive();
      if (assert_full_batch_ && (!message || message.value().is_null())) {
        return GXF_CONTRACT_MESSAGE_NOT_AVAILABLE;
      }
    }
    return GXF_SUCCESS;
  }

  gxf_result_t stop() override { return GXF_SUCCESS; }

  gxf_result_t registerInterface(Registrar* registrar) override {
    Expected<void> result;
    result &= registrar->parameter(signal_, "signal");
    result &= registrar->parameter(batch_size_, "batch_size");
    result &= registrar->parameter(assert_full_batch_, "assert_full_batch", "Assert Full Batch",
                                   "Assert if the batch is not fully populated.", true);
    return ToResultCode(result);
  }

  Parameter<Handle<Receiver>> signal_;
  Parameter<int64_t> batch_size_;
  Parameter<bool> assert_full_batch_;
};

// Generates interger and fibonacci messages
class Generator : public Codelet {
 public:
  gxf_result_t start() override { return GXF_SUCCESS; }

  gxf_result_t tick() override {
    const Expected<Entity> msg1 = createIntegerMessage();
    if (!msg1) return msg1.error();
    integers_->publish(msg1.value());

    const Expected<Entity> msg2 = createFibonacciMessage();
    if (!msg1) return msg1.error();
    fibonacci_->publish(msg2.value());

    return GXF_SUCCESS;
  }

  gxf_result_t stop() override { return GXF_SUCCESS; }

  gxf_result_t registerInterface(Registrar* registrar) override {
    Expected<void> result;
    result &= registrar->parameter(integers_, "integers", "Name of 'integers' channel", "");
    result &= registrar->parameter(fibonacci_, "fibonacci", "Name of 'fibonacci' channel", "");
    result &= registrar->parameter(pool_, "pool", "Pool", "");
    return ToResultCode(result);
  }

 private:
  Expected<Entity> createIntegerMessage() {
    Expected<Entity> message =
        CreateTensorMap(context(), pool_,
                        {{"positive", MemoryStorageType::kHost, {2, 3, 4}, PrimitiveType::kInt32},
                         {"negative", MemoryStorageType::kHost, {2, 3, 4}, PrimitiveType::kInt32}});
    if (!message) return ForwardError(message);

    auto positive = message.value().get<Tensor>("positive");
    if (!positive) return ForwardError(positive);
    Expected<int32_t*> maybe_ptr1 = positive.value()->data<int32_t>();
    if (!maybe_ptr1) return ForwardError(maybe_ptr1);
    int32_t* ptr1 = maybe_ptr1.value();
    for (int i = 0; i < 24; i++) {
      ptr1[i] = i;
    }

    auto negative = message.value().get<Tensor>("negative");
    if (!negative) return ForwardError(negative);
    Expected<int32_t*> maybe_ptr2 = negative.value()->data<int32_t>();
    if (!maybe_ptr2) return ForwardError(maybe_ptr2);
    int32_t* ptr2 = maybe_ptr2.value();
    for (int i = 0; i < 24; i++) {
      ptr2[i] = -i;
    }

    return message;
  }

  Expected<Entity> createFibonacciMessage() {
    Expected<Entity> message = CreateTensorMap(
        context(), pool_, {{"fibonacci", MemoryStorageType::kHost, {8}, PrimitiveType::kFloat32}});
    if (!message) return ForwardError(message);

    auto fibonacci = message.value().get<Tensor>("fibonacci");
    if (!fibonacci) return ForwardError(fibonacci);
    Expected<float*> maybe_ptr = fibonacci.value()->data<float>();
    if (!maybe_ptr) return ForwardError(maybe_ptr);
    float* ptr = maybe_ptr.value();
    ptr[0] = 0;
    ptr[1] = 1;
    for (int i = 2; i < 8; i++) {
      ptr[i] = static_cast<float>(ptr[i - 1] + ptr[i - 2]);
    }

    return message;
  }

  Parameter<Handle<Transmitter>> integers_;
  Parameter<Handle<Transmitter>> fibonacci_;
  Parameter<Handle<Allocator>> pool_;
};

// Forwards incoming messages at the receiver to the transmitter
class Forward : public Codelet {
 public:
  gxf_result_t start() override { return GXF_SUCCESS; }

  gxf_result_t tick() override {
    // in_->receive()
    //     .and_then([] {})
    auto message = in_->receive();
    if (!message) {
      return message.error();
    }
    out_->publish(message.value());
    return GXF_SUCCESS;
  }

  gxf_result_t stop() override { return GXF_SUCCESS; }

  gxf_result_t registerInterface(Registrar* registrar) override {
    Expected<void> result;
    result &= registrar->parameter(in_, "in");
    result &= registrar->parameter(out_, "out");
    return ToResultCode(result);
  }

 private:
  Parameter<Handle<Receiver>> in_;
  Parameter<Handle<Transmitter>> out_;
};

// Pops an incoming message at the reciever
class Pop : public Codelet {
 public:
  gxf_result_t start() override { return GXF_SUCCESS; }

  gxf_result_t tick() override {
    message_->receive();
    return GXF_SUCCESS;
  }

  gxf_result_t stop() override { return GXF_SUCCESS; }

  gxf_result_t registerInterface(Registrar* registrar) override {
    Expected<void> result;
    result &= registrar->parameter(message_, "message");
    return ToResultCode(result);
  }

  Parameter<Handle<Receiver>> message_;
};

// Receives a message with tensors and prints the element count to debug log
class Print : public Codelet {
 public:
  gxf_result_t start() override { return GXF_SUCCESS; }

  gxf_result_t tick() override {
    auto message = message_->receive();
    if (!message) return message.error();

    for (const auto tensor : message.value().findAll<Tensor>()) {
      // FIXME print component name
      GXF_LOG_DEBUG("[%05zu] Received E%05zu with Tensor with %zu elements", cid(),
                    message.value().eid(), tensor->element_count());
    }

    return GXF_SUCCESS;
  }

  gxf_result_t stop() override { return GXF_SUCCESS; }

  gxf_result_t registerInterface(Registrar* registrar) override {
    Expected<void> result;
    result &= registrar->parameter(message_, "message");
    return ToResultCode(result);
  }

  Parameter<Handle<Receiver>> message_;
};

// Prints the elements of a tensor with the given type
// T1 is the type of data
// T2 is the type to use when printing
template <typename T1, typename T2 = T1>
gxf_result_t PrintSingleTensorGivenType(const Tensor& tensor, const char* format) {
  Expected<const T1*> maybe_ptr = tensor.data<T1>();
  if (!maybe_ptr) {
    return maybe_ptr.error();
  }
  const T1* ptr = maybe_ptr.value();
  for (size_t i = 0; i < tensor.element_count(); i++) {
    std::printf(format, static_cast<T2>(ptr[i]));
  }
  std::printf("\n");
  return GXF_SUCCESS;
}

// Prints the elements of a tensor for various types
gxf_result_t PrintSingleTensorAllTypes(const Tensor& tensor) {
  switch (tensor.element_type()) {
    case PrimitiveType::kUnsigned8:
      return PrintSingleTensorGivenType<uint8_t, uint32_t>(tensor, "%u, ");
    case PrimitiveType::kUnsigned16:
      return PrintSingleTensorGivenType<uint16_t, uint32_t>(tensor, "%u, ");
    case PrimitiveType::kUnsigned32:
      return PrintSingleTensorGivenType<uint32_t>(tensor, "%u, ");
    case PrimitiveType::kUnsigned64:
      return PrintSingleTensorGivenType<uint64_t>(tensor, "%" PRIu64 ", ");
    case PrimitiveType::kInt8:
      return PrintSingleTensorGivenType<int8_t, int32_t>(tensor, "%d, ");
    case PrimitiveType::kInt16:
      return PrintSingleTensorGivenType<int16_t, int32_t>(tensor, "%d, ");
    case PrimitiveType::kInt32:
      return PrintSingleTensorGivenType<int32_t>(tensor, "%d, ");
    case PrimitiveType::kInt64:
      return PrintSingleTensorGivenType<int64_t>(tensor, "%" PRId64 ", ");
    case PrimitiveType::kFloat32:
      if (tensor.storage_type() == MemoryStorageType::kHost) {
        return PrintSingleTensorGivenType<float>(tensor, "%f, ");
      } else {
        std::printf("[device-data]\n");
        return GXF_SUCCESS;
      }
    case PrimitiveType::kFloat64:
      if (tensor.storage_type() == MemoryStorageType::kHost) {
        return PrintSingleTensorGivenType<double>(tensor, "%f, ");
      } else {
        std::printf("[device-data]\n");
        return GXF_SUCCESS;
      }
    default:
      std::printf("Unknown type %d\n", static_cast<int>(tensor.element_type()));
      return GXF_SUCCESS;
  }
}

// Receives a message with tensors and prints them to the console.
class PrintTensor : public Codelet {
 public:
  gxf_result_t start() override { return GXF_SUCCESS; }

  gxf_result_t tick() override {
    auto message = tensors_->receive();
    if (!message) return message.error();

    if (silent_) {
      return GXF_SUCCESS;
    }

    for (const auto tensor : message.value().findAll<Tensor>()) {
      // GXF_LOG_DEBUG("[%05zu] Received E%05zu with Tensor '%s' with %zu elements",
      //               cid(), message.value().eid(), tensor->name(), tensor->size());
      std::printf("shape: (");
      for (size_t i = 0; i < tensor->rank(); i++) {
        std::printf("%d", tensor->shape().dimension(i));
        if (i + 1 < tensor->rank()) std::printf(", ");
      }
      std::printf("), %s: ", tensor.name());

      PrintSingleTensorAllTypes(*tensor);
    }

    return GXF_SUCCESS;
  }

  gxf_result_t stop() override { return GXF_SUCCESS; }

  gxf_result_t registerInterface(Registrar* registrar) override {
    Expected<void> result;
    result &= registrar->parameter(tensors_, "tensors", "Input Tensor", "");
    result &= registrar->parameter(silent_, "silent", "Enable silent mode",
                         "If enabled this codelet will not print anything to the command line.",
                         false, GXF_PARAMETER_FLAGS_DYNAMIC);
    return ToResultCode(result);
  }

 private:
  Parameter<Handle<Receiver>> tensors_;
  Parameter<bool> silent_;
};

// Generates an arbitrary precision factorial tensor message
class ArbitraryPrecisionFactorial : public Codelet {
 public:
  gxf_result_t start() override { return GXF_SUCCESS; }

  gxf_result_t tick() override {
    const int32_t digits = digits_;
    const uint32_t factorial = factorial_;

    Expected<Entity> message = CreateTensorMap(
        context(), pool_,
        {{"factorial", MemoryStorageType::kHost, {digits}, PrimitiveType::kUnsigned8}});
    if (!message) return message.error();

    auto tensor = message.value().get<Tensor>("factorial");
    if (!tensor) return tensor.error();

    Expected<uint8_t*> maybe_digit = tensor.value()->data<uint8_t>();
    if (!maybe_digit) return maybe_digit.error();
    uint8_t* digit = maybe_digit.value();
    digit[0] = 1;
    for (int32_t i = 1; i < digits; i++) digit[i] = 0;

    int32_t last = 1;
    for (uint32_t n = 1; n <= factorial; n++) {
      uint32_t carry = 0;
      for (int32_t i = 0; i < last; i++) {
        const uint32_t d = static_cast<uint32_t>(digit[i]) * n + carry;
        digit[i] = static_cast<uint32_t>(d % 10);
        carry = d / 10;
      }
      while (carry > 0) {
        if (last >= digits) {
          // error
          return GXF_FAILURE;
        }
        digit[last] = static_cast<uint32_t>(carry % 10);
        carry = carry / 10;
        last++;
      }
    }

    result_->publish(message.value());

    return GXF_SUCCESS;
  }

  gxf_result_t stop() override { return GXF_SUCCESS; }

  gxf_result_t registerInterface(Registrar* registrar) override {
    Expected<void> result;
    result &= registrar->parameter(digits_, "digits");
    result &= registrar->parameter(factorial_, "factorial");
    result &= registrar->parameter(result_, "result");
    result &= registrar->parameter(pool_, "pool");
    return ToResultCode(result);
  }

  Parameter<int32_t> digits_;
  Parameter<uint64_t> factorial_;
  Parameter<Handle<Transmitter>> result_;
  Parameter<Handle<Allocator>> pool_;
};

// Creates a tensor message with a cumulative sum of sines
class IntegerSinSum : public Codelet {
 public:
  gxf_result_t start() override { return GXF_SUCCESS; }

  gxf_result_t tick() override {
    const size_t count = count_;

    double x = 0.0;
    for (size_t i = 0; i < count; i++) {
      x += std::sin(i);
    }

    Expected<Entity> message = CreateTensorMap(
        context(), pool_, {{"sum", MemoryStorageType::kHost, {1}, PrimitiveType::kFloat32}});
    if (!message) return message.error();

    auto sum = message.value().get<Tensor>("sum");
    if (!sum) return sum.error();
    auto maybe_ptr = sum.value()->data<float>();
    if (!maybe_ptr) return maybe_ptr.error();
    *maybe_ptr.value() = x;

    result_->publish(message.value());

    return GXF_SUCCESS;
  }

  gxf_result_t stop() override { return GXF_SUCCESS; }

  gxf_result_t registerInterface(Registrar* registrar) override {
    Expected<void> result;
    result &= registrar->parameter(count_, "count");
    result &= registrar->parameter(result_, "result");
    result &= registrar->parameter(pool_, "pool");
    return ToResultCode(result);
  }

  Parameter<size_t> count_;
  Parameter<Handle<Transmitter>> result_;
  Parameter<Handle<Allocator>> pool_;
};

class TestTensorStrides : public Codelet {
 public:
  gxf_result_t start() override { return GXF_SUCCESS; }
  gxf_result_t stop() override { return GXF_SUCCESS; }
  gxf_result_t tick() override {
    Shape shape{2, 3, 4};
    auto foo_strides = ComputeRowStrides(shape, 256, PrimitiveTypeTraits<float>::size);
    GXF_ASSERT(foo_strides, "ComputeRowStrides failed");
    Expected<Entity> message = CreateTensorMap(
        context(), pool_,
        {
            TensorDescription{"foo", MemoryStorageType::kHost, shape, PrimitiveType::kFloat32, 0,
                              foo_strides.value()},
            TensorDescription{"bar", MemoryStorageType::kHost, shape, PrimitiveType::kFloat32, 0,
                              ComputeStrides(shape, {256, 64, 8})},
            TensorDescription{"zoo", MemoryStorageType::kHost, shape, PrimitiveType::kFloat64, 0,
                              Unexpected{GXF_UNINITIALIZED_VALUE}},
        });
    if (!message) {
      return message.error();
    }
    auto foo = message.value().get<Tensor>("foo");
    if (!foo) {
      return foo.error();
    }
    Handle<Tensor> foo_tensor = foo.value();
    if (foo_tensor->rank() != 3) {
      GXF_LOG_ERROR("Bad rank");
      return GXF_FAILURE;
    }
    GXF_ASSERT(foo_tensor->bytes_size() == 2 * 256, "Bad foo bytes_size");
    GXF_ASSERT(foo_tensor->stride(0) == 256, "Bad foo stride 0");
    GXF_ASSERT(foo_tensor->stride(1) == 16, "Bad foo stride 1");
    GXF_ASSERT(foo_tensor->stride(2) == 4, "Bad foo stride 1");

    auto bar = message.value().get<Tensor>("bar");
    if (!bar) {
      return bar.error();
    }
    Handle<Tensor> bar_tensor = bar.value();
    if (bar_tensor->rank() != 3) {
      GXF_LOG_ERROR("Bad rank");
      return GXF_FAILURE;
    }
    GXF_ASSERT(bar_tensor->bytes_size() == 2 * 256, "Bad bar bytes_size");
    GXF_ASSERT(bar_tensor->stride(0) == 256, "Bad bar stride 0");
    GXF_ASSERT(bar_tensor->stride(1) == 64, "Bad bar stride 1");
    GXF_ASSERT(bar_tensor->stride(2) == 8, "Bad bar stride 1");

    auto zoo = message.value().get<Tensor>("zoo");
    if (!zoo) {
      return bar.error();
    }
    Handle<Tensor> zoo_tensor = zoo.value();
    GXF_ASSERT(zoo_tensor->bytes_size() == 2 * 3 * 4 * 8 /* sizeof(double) */,
               "Bad zoo bytes_size %lu");
    GXF_ASSERT(zoo_tensor->stride(0) == 3 * 4 * 8, "Bad zoo stride 0");
    GXF_ASSERT(zoo_tensor->stride(1) == 4 * 8, "Bad zoo stride 1");
    GXF_ASSERT(zoo_tensor->stride(2) == 8, "Bad zoo stride 1");

    return GXF_SUCCESS;
  }

  gxf_result_t registerInterface(Registrar* registrar) override {
    Expected<void> result;
    result &= registrar->parameter(pool_, "pool");
    return ToResultCode(result);
  }

  Parameter<Handle<Allocator>> pool_;
};  // namespace test

class TestTimestampTx : public Codelet {
 public:
  gxf_result_t start() override {
    // Set the target time so that we tick
    scheduling_term_->setNextTargetTime(execution_clock_->timestamp());

    return GXF_SUCCESS;
  }

  gxf_result_t tick() override {
    auto message = Entity::New(context());
    if (!message) return GXF_FAILURE;

    // Compute the timestamp when the codelet should be executed next
    const int64_t target_timestamp = execution_clock_->timestamp() + delay_;
    scheduling_term_->setNextTargetTime(target_timestamp);
    const auto result = signal_->publish(message.value(), execution_clock_->timestamp());
    return ToResultCode(message);
  }

  gxf_result_t stop() override { return GXF_SUCCESS; }

  gxf_result_t registerInterface(Registrar* registrar) override {
    Expected<void> result;
    result &= registrar->parameter(delay_, "delay");
    result &= registrar->parameter(execution_clock_, "execution_clock");
    result &= registrar->parameter(signal_, "signal");
    result &= registrar->parameter(scheduling_term_, "scheduling_term");
    result &= registrar->parameter(offset_clock_, "offset_clock", "", "",
                                   Registrar::NoDefaultParameter(), GXF_PARAMETER_FLAGS_OPTIONAL);
    return ToResultCode(result);
  }
  Parameter<int64_t> delay_;
  Parameter<Handle<Clock>> execution_clock_;
  Parameter<Handle<Clock>> offset_clock_;
  Parameter<Handle<Transmitter>> signal_;
  Parameter<Handle<TargetTimeSchedulingTerm>> scheduling_term_;
};

class TestTimestampRx : public Codelet {
 public:
  gxf_result_t start() override { return GXF_SUCCESS; }

  gxf_result_t tick() override {
    auto message = signal_->receive();
    if (!message || message.value().is_null()) { return GXF_CONTRACT_MESSAGE_NOT_AVAILABLE; }
    auto timestamp = message.value().get<Timestamp>();
    if (!timestamp) { return GXF_CONTRACT_MESSAGE_NOT_AVAILABLE; }
    auto pubtime = timestamp.value()->pubtime;
    auto acqtime = timestamp.value()->acqtime;

    if (last_pubtime == 0) {
      last_pubtime = pubtime;
      last_acqtime = acqtime;
      return GXF_SUCCESS;
    }
    if (pubtime != last_pubtime + delay_) { return GXF_CONTRACT_MESSAGE_NOT_AVAILABLE; }
    if (acqtime != last_acqtime + delay_) { return GXF_CONTRACT_MESSAGE_NOT_AVAILABLE; }
    last_pubtime = pubtime;
    last_acqtime = acqtime;
    return GXF_SUCCESS;
  }

  gxf_result_t stop() override { return GXF_SUCCESS; }

  gxf_result_t registerInterface(Registrar* registrar) override {
    Expected<void> result;
    result &= registrar->parameter(signal_, "signal");
    result &= registrar->parameter(delay_, "delay");
    return ToResultCode(result);
  }

  Parameter<Handle<Receiver>> signal_;
  Parameter<int64_t> delay_;

 private:
  int64_t last_pubtime = 0;
  int64_t last_acqtime = 0;
};

// Prints test logs for various severity levels defined in `gxf_severity_t`
class TestLogger : public Codelet {
 public:
  gxf_result_t start() override { return GXF_SUCCESS; }

  gxf_result_t tick() override {
    GXF_LOG_ERROR("This is test error message");
    GXF_LOG_WARNING("This is a test warning message");
    GXF_LOG_INFO("This is a test info message");
    GXF_LOG_DEBUG("This is a test debug message");
    return GXF_SUCCESS;
  }

  gxf_result_t stop() override { return GXF_SUCCESS; }
};

class PeriodicSchedulingTermWithDelay : public PeriodicSchedulingTerm {
 public:
  gxf_result_t check_abi(int64_t timestamp, SchedulingConditionType* type,
                         int64_t* target_timestamp) const final {
    auto seed = static_cast<unsigned int>(timestamp);
    if (!last_run_timestamp()) {
      *type = SchedulingConditionType::READY;
      return GXF_SUCCESS;
    }
    // Adding a random delay to the target timestamp and the scheduling condition type to be ready
    *target_timestamp =
        recess_period_ns() + *last_run_timestamp() + rand_r(&seed) % 100;
    *type = timestamp - rand_r(&seed) % 10> *target_timestamp ? SchedulingConditionType::READY
                                          : SchedulingConditionType::WAIT_TIME;
    return GXF_SUCCESS;
  }
};

}  // namespace test
}  // namespace gxf
}  // namespace nvidia

#endif
