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
#ifndef NVIDIA_GXF_CORE_COMPONENT_HPP
#define NVIDIA_GXF_CORE_COMPONENT_HPP

#include "core/assert.hpp"
#include "gxf/core/entity.hpp"
#include "gxf/core/expected.hpp"
#include "gxf/core/gxf.h"
#include "gxf/core/handle.hpp"
#include "gxf/core/parameter.hpp"
#include "gxf/core/registrar.hpp"
#include "gxf/std/parameter_storage.hpp"

namespace nvidia {
namespace gxf {

// Components are parts of an entity and provide their functionality. The Component class is the
// base class of all GXF components.
class Component {
 public:
  virtual ~Component() = default;

  Component(const Component& component) = delete;
  Component(Component&& component) = delete;
  Component& operator=(const Component& component) = delete;
  Component& operator=(Component&& component) = delete;

  // Used to register all parameters of the components. Do not use this function for other purposes
  // as it might be called at anytime by the runtime.
  //
  // Example:
  //   class Foo : public Component {
  //    public:
  //     gxf_result_t registerInterface(Registrar* registrar) override {
  //       GXF_REGISTER_PARAMETER(count, 1);
  //     }
  //     GXF_PARAMETER(int, count);
  //   };
  virtual gxf_result_t registerInterface(Registrar* registrar) {
    registrar->registerParameterlessComponent();
    return GXF_SUCCESS;
  }

  // Use to start the lifetime of a component and should be used instead of the constructor.
  // Called after all components of an entity are created. The order in which components within
  // the same entity are initialized is undefined.
  virtual gxf_result_t initialize() { return GXF_SUCCESS; }

  // Use to end the lifetime of a component and should be used instead of the deconstructor.
  // The order in which components within the same entity are deinitialized is undefined.
  virtual gxf_result_t deinitialize() { return GXF_SUCCESS; }

  gxf_context_t context() const noexcept { return context_; }
  gxf_uid_t eid() const noexcept { return eid_; }
  gxf_uid_t cid() const noexcept { return cid_; }

  // The entity which owns this component
  Entity entity() const noexcept {
    // FIXME(v1) check that value exists
    return Entity::Shared(context(), eid()).value();
  }

  // Gets the name of the component
  const char* name() const noexcept {
    const char* result;
    const gxf_result_t code = GxfComponentName(context(), cid(), &result);
    return (code == GXF_SUCCESS) ? result : "";
  }

  // This function shall only be called by GXF and is used to setup the component.
  void internalSetup(gxf_context_t context, gxf_uid_t eid, gxf_uid_t cid) {
    context_ = context;
    eid_ = eid;
    cid_ = cid;
  }

 protected:
  Component() = default;

  gxf_context_t context_ = kNullContext;
  gxf_uid_t eid_ = kNullUid;
  gxf_uid_t cid_ = kNullUid;
};

}  // namespace gxf
}  // namespace nvidia

#endif
