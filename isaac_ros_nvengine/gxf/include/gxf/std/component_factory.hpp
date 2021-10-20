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
#ifndef NVIDIA_GXF_STD_COMPONENT_FACTORY_HPP
#define NVIDIA_GXF_STD_COMPONENT_FACTORY_HPP

#include "gxf/core/expected.hpp"

namespace nvidia {
namespace gxf {

// Base class for extension factories. An extension factory is used to create instances
// of components.
class ComponentFactory {
 public:
  virtual ~ComponentFactory() = default;

  ComponentFactory(const ComponentFactory&) = delete;
  ComponentFactory(ComponentFactory&&) = delete;
  ComponentFactory& operator=(const ComponentFactory&) = delete;
  ComponentFactory& operator=(ComponentFactory&&) = delete;

  // Allocates a component of the given type
  virtual gxf_result_t allocate_abi(gxf_tid_t tid, void** out_pointer) = 0;

  // Frees a component of the given type
  virtual gxf_result_t deallocate_abi(gxf_tid_t tid, void* pointer) = 0;

  Expected<void*> allocate(gxf_tid_t tid);

  Expected<void> deallocate(gxf_tid_t tid, void* pointer);

 protected:
  ComponentFactory() = default;
};

}  // namespace gxf
}  // namespace nvidia

#endif
