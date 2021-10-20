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
#ifndef NVIDIA_GXF_STD_COMPONENT_ALLOCATOR_HPP
#define NVIDIA_GXF_STD_COMPONENT_ALLOCATOR_HPP

#include "gxf/core/expected.hpp"

namespace nvidia {
namespace gxf {

// Base class for allocating components.
class ComponentAllocator {
 public:
  virtual ~ComponentAllocator() = default;

  ComponentAllocator(const ComponentAllocator&) = delete;
  ComponentAllocator(ComponentAllocator&&) = delete;
  ComponentAllocator& operator=(const ComponentAllocator&) = delete;
  ComponentAllocator& operator=(ComponentAllocator&&) = delete;

  // Allocates a new component of the specific component type this allocator is handling.
  virtual gxf_result_t allocate_abi(void** out_pointer) = 0;

  // Deallocates a component which was previously allocated by this allocator.
  virtual gxf_result_t deallocate_abi(void* pointer) = 0;

  Expected<void*> allocate();

  Expected<void> deallocate(void* pointer);

 protected:
  ComponentAllocator() = default;
};

}  // namespace gxf
}  // namespace nvidia

#endif
