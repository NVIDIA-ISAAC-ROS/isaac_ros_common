/*
Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#ifndef NVIDIA_GXF_SERIALIZATION_ENTITY_REPLAYER_HPP_
#define NVIDIA_GXF_SERIALIZATION_ENTITY_REPLAYER_HPP_

#include <string>

#include "gxf/serialization/entity_serializer.hpp"
#include "gxf/serialization/file_stream.hpp"
#include "gxf/std/codelet.hpp"
#include "gxf/std/scheduling_terms.hpp"
#include "gxf/std/transmitter.hpp"

namespace nvidia {
namespace gxf {

// Replays entities by reading and deserializing from a file.
// The file is processed sequentially and a single entity is published per tick.
class EntityReplayer : public Codelet {
 public:
  gxf_result_t registerInterface(Registrar* registrar) override;
  gxf_result_t initialize() override;
  gxf_result_t deinitialize() override;

  gxf_result_t start() override { return GXF_SUCCESS; }
  gxf_result_t tick() override;
  gxf_result_t stop() override { return GXF_SUCCESS; }

 private:
  Parameter<Handle<Transmitter>> transmitter_;
  Parameter<Handle<EntitySerializer>> entity_serializer_;
  Parameter<Handle<gxf::BooleanSchedulingTerm>> boolean_scheduling_term_;
  Parameter<std::string> directory_;
  Parameter<std::string> basename_;
  Parameter<size_t> batch_size_;
  Parameter<bool> ignore_corrupted_entities_;

  // File stream for entities
  FileStream entity_file_stream_;
  // File stream for index
  FileStream index_file_stream_;
};

}  // namespace gxf
}  // namespace nvidia

#endif  // NVIDIA_GXF_SERIALIZATION_ENTITY_REPLAYER_HPP_
