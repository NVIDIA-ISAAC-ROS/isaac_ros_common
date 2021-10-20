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
#ifndef NVIDIA_GXF_STD_TRANSMITTER_HPP
#define NVIDIA_GXF_STD_TRANSMITTER_HPP

#include "gxf/std/queue.hpp"

namespace nvidia {
namespace gxf {

// Interface for publishing entities.
class Transmitter : public Queue {
 public:
  // Publishes an entity
  virtual gxf_result_t publish_abi(gxf_uid_t uid) = 0;

  // The total number of entities which have previously been published and were moved out of the
  // main stage.
  virtual size_t back_size_abi() = 0;

  // Moves entities which were published recently out of the main stage.
  virtual gxf_result_t sync_abi() = 0;

  Expected<void> publish(const Entity& other);

  Expected<void> publish(Entity& other, const int64_t acq_timestamp);

  size_t back_size();

  Expected<void> sync();
};

}  // namespace gxf
}  // namespace nvidia

#endif
