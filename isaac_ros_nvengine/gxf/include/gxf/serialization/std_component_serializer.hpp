/*
Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#ifndef NVIDIA_GXF_SERIALIZATION_STD_COMPONENT_SERIALIZER_HPP_
#define NVIDIA_GXF_SERIALIZATION_STD_COMPONENT_SERIALIZER_HPP_

#include "gxf/serialization/component_serializer.hpp"
#include "gxf/std/allocator.hpp"
#include "gxf/std/tensor.hpp"
#include "gxf/std/timestamp.hpp"

namespace nvidia {
namespace gxf {

// Serializer that supports Timestamp and Tensor components
// Valid for sharing data between devices with the same endianness
class StdComponentSerializer : public ComponentSerializer {
 public:
  gxf_result_t registerInterface(Registrar* registrar) override;
  gxf_result_t initialize() override;
  gxf_result_t deinitialize() override { return GXF_SUCCESS; }

 private:
  // Serializes a nvidia::gxf::Timestamp
  Expected<size_t> serializeTimestamp(Timestamp timestamp, Endpoint* endpoint);
  // Deserializes a nvidia::gxf::Timestamp
  Expected<Timestamp> deserializeTimestamp(Endpoint* endpoint);
  // Serializes a nvidia::gxf::Tensor
  Expected<size_t> serializeTensor(const Tensor& tensor, Endpoint* endpoint);
  // Deserializes a nvidia::gxf::Tensor
  Expected<Tensor> deserializeTensor(Endpoint* endpoint);

  Parameter<Handle<Allocator>> allocator_;
};

}  // namespace gxf
}  // namespace nvidia

#endif  // NVIDIA_GXF_SERIALIZATION_STD_COMPONENT_SERIALIZER_HPP_
