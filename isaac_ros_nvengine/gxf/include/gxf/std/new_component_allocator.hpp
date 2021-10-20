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
#ifndef NVIDIA_GXF_STD_NEW_COMPONENT_ALLOCATOR_HPP
#define NVIDIA_GXF_STD_NEW_COMPONENT_ALLOCATOR_HPP

#include "gxf/core/expected.hpp"
#include "gxf/std/component_allocator.hpp"

namespace nvidia {
namespace gxf {

// Default implementation for allocating components using new/delete
template <typename T, typename Enabler = void>
class NewComponentAllocator
    : public ComponentAllocator {
 public:
  ~NewComponentAllocator() override = default;

  gxf_result_t allocate_abi(void** out_pointer) override {
    if (out_pointer == nullptr) {
      return GXF_ARGUMENT_NULL;
    }
    *out_pointer = static_cast<void*>(new T());
    if (*out_pointer == nullptr) {
      return GXF_OUT_OF_MEMORY;
    }
    return GXF_SUCCESS;
  }

  gxf_result_t deallocate_abi(void* pointer) override {
    if (pointer == nullptr) {
      return GXF_ARGUMENT_NULL;
    }
    delete static_cast<T*>(pointer);
    return GXF_SUCCESS;
  }
};

// Special case of standard component allocator for abstract components
template <typename T>
class NewComponentAllocator<T, std::enable_if_t<std::is_abstract<T>::value>>
    : public ComponentAllocator {
 public:
  ~NewComponentAllocator() override = default;

  gxf_result_t allocate_abi(void** out_pointer) override {
    return GXF_FACTORY_ABSTRACT_CLASS;
  }

  gxf_result_t deallocate_abi(void* pointer) override {
    return GXF_FACTORY_ABSTRACT_CLASS;
  }
};

}  // namespace gxf
}  // namespace nvidia

#endif
