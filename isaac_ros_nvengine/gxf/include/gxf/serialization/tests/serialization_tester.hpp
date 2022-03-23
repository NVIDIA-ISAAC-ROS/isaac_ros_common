/*
Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#ifndef NVIDIA_GXF_SERIALIZATION_TESTS_SERIALIZATION_TESTER_HPP_
#define NVIDIA_GXF_SERIALIZATION_TESTS_SERIALIZATION_TESTER_HPP_

#include <memory>

#include "gxf/serialization/entity_serializer.hpp"
#include "gxf/serialization/serialization_buffer.hpp"
#include "gxf/std/codelet.hpp"
#include "gxf/std/receiver.hpp"
#include "gxf/std/transmitter.hpp"

namespace nvidia {
namespace gxf {
namespace test {

// Codelet that serializes incoming messages and stores them in a buffer
// Messages are immediately deserialized from the buffer and published
class SerializationTester : public Codelet {
 public:
  gxf_result_t registerInterface(Registrar* registrar) override;
  gxf_result_t initialize() override { return GXF_SUCCESS; }
  gxf_result_t deinitialize() override { return GXF_SUCCESS; }

  gxf_result_t start() override { return GXF_SUCCESS; }
  gxf_result_t tick() override;
  gxf_result_t stop() override { return GXF_SUCCESS; }

 private:
  Parameter<Handle<Receiver>> input_;
  Parameter<Handle<Transmitter>> output_;
  Parameter<Handle<EntitySerializer>> entity_serializer_;
  Parameter<Handle<SerializationBuffer>> serialization_buffer_;
};

}  // namespace test
}  // namespace gxf
}  // namespace nvidia

#endif  // NVIDIA_GXF_SERIALIZATION_TESTS_SERIALIZATION_TESTER_HPP_
