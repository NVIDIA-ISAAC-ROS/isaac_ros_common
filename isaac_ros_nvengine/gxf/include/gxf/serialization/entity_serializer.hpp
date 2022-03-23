/*
Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#ifndef NVIDIA_GXF_SERIALIZATION_ENTITY_SERIALIZER_HPP_
#define NVIDIA_GXF_SERIALIZATION_ENTITY_SERIALIZER_HPP_

#include "gxf/core/component.hpp"
#include "gxf/serialization/endpoint.hpp"

namespace nvidia {
namespace gxf {

// Interface for serializing entities
class EntitySerializer : public Component {
 public:
  virtual ~EntitySerializer() = default;

  // Serializes entity and writes to endpoint
  // Returns the size of the serialized entity in bytes
  virtual gxf_result_t serialize_entity_abi(gxf_uid_t eid, Endpoint* endpoint, uint64_t* size) = 0;
  // Reads from endpoint and deserializes entity
  virtual gxf_result_t deserialize_entity_abi(gxf_uid_t eid, Endpoint* endpoint) = 0;

  // C++ API wrappers
  Expected<size_t> serializeEntity(Entity entity, Endpoint* endpoint);
  Expected<void> deserializeEntity(Entity entity, Endpoint* endpoint);

  // Deprecated
  Expected<Entity> deserializeEntity(gxf_context_t context, Endpoint* endpoint);
};

}  // namespace gxf
}  // namespace nvidia

#endif  // NVIDIA_GXF_SERIALIZATION_ENTITY_SERIALIZER_HPP_
