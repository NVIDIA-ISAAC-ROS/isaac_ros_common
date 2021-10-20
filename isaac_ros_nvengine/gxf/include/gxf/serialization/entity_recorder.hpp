/*
Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#ifndef NVIDIA_GXF_SERIALIZATION_ENTITY_RECORDER_HPP_
#define NVIDIA_GXF_SERIALIZATION_ENTITY_RECORDER_HPP_

#include <string>
#include <vector>

#include "gxf/serialization/component_serializer.hpp"
#include "gxf/serialization/entity_serializer.hpp"
#include "gxf/serialization/file_stream.hpp"
#include "gxf/std/codelet.hpp"
#include "gxf/std/receiver.hpp"

namespace nvidia {
namespace gxf {

// Records incoming entities by serializaing and writing to a file.
// Uses one file to store binary data and a second file as an index to enable random-access.
class EntityRecorder : public Codelet {
 public:
  gxf_result_t registerInterface(Registrar* registrar) override;
  gxf_result_t initialize() override;
  gxf_result_t deinitialize() override;

  gxf_result_t start() override { return GXF_SUCCESS; }
  gxf_result_t tick() override;
  gxf_result_t stop() override { return GXF_SUCCESS; }

 private:
  Parameter<Handle<Receiver>> receiver_;
  Parameter<std::vector<Handle<ComponentSerializer>>> serializers_;
  Parameter<std::string> directory_;
  Parameter<std::string> basename_;
  Parameter<bool> flush_on_tick_;

  // Entity serializer
  EntitySerializer entity_serializer_;
  // File stream for data index
  FileStream index_file_stream_;
  // File stream for binary data
  FileStream binary_file_stream_;
  // Offset into binary file
  size_t binary_file_offset_;
};

}  // namespace gxf
}  // namespace nvidia

#endif
