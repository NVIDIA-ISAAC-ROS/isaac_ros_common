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
#ifndef NVIDIA_GXF_CORE_ENTITY_HPP_
#define NVIDIA_GXF_CORE_ENTITY_HPP_

#include <utility>
#include <vector>

#include "core/assert.hpp"
#include "core/fixed_vector.hpp"
#include "core/type_name.hpp"
#include "gxf/core/expected.hpp"
#include "gxf/core/gxf.h"
#include "gxf/core/handle.hpp"

namespace nvidia {
namespace gxf {

// All GXF objects are entities. An entity owns multiple components which define the functionality
// of the entity. Entities themselves are nothing more than a unique identifier.
// FIXME This type is a bit strange as it looks like and entity, but is in face just a handle.
class Entity {
 public:
  // Creates a new entity
  static Expected<Entity> New(gxf_context_t context) {
    gxf_uid_t eid;
    const GxfEntityCreateInfo info = {0};
    const gxf_result_t code = GxfCreateEntity(context, &info, &eid);
    if (code != GXF_SUCCESS) {
      return Unexpected{code};
    } else {
      return Shared(context, eid);
    }
  }

  // Creates an entity handle based on an existing ID and takes ownership.
  // Reference count is not increased.
  static Expected<Entity> Own(gxf_context_t context, gxf_uid_t eid) {
    Entity result;
    result.context_ = context;
    result.eid_ = eid;
    return result;
  }

  // Creates an entity handle based on an existing ID and shares ownership.
  // Reference count is increased by one.
  static Expected<Entity> Shared(gxf_context_t context, gxf_uid_t eid) {
    Entity result;
    result.context_ = context;
    result.eid_ = eid;
    const gxf_result_t code = GxfEntityRefCountInc(context, eid);
    if (code != GXF_SUCCESS) {
      return Unexpected{code};
    } else {
      return result;
    }
  }

  Entity() = default;

  Entity(const Entity& other) {
    eid_ = other.eid();
    context_ = other.context();
    if (eid_ != kNullUid) {
      // FIXME(dweikersdorf) How do we deal with failure?
      GxfEntityRefCountInc(context_, eid_);
    }
  }

  Entity(Entity&& other) {
    context_ = other.context_;
    eid_ = other.eid_;
    other.context_ = kNullContext;
    other.eid_ = kNullUid;
  }

  Entity& operator=(const Entity& other) {
    // In case other point to the same entity, nothing needs to be done.
    if (eid_ == other.eid() && context_ == other.context()) {
      return *this;
    }
    if (eid_ != kNullUid) {
      release();
    }
    context_ = other.context();
    eid_ = other.eid();
    if (eid_ != kNullUid) {
      // FIXME(dweikersdorf) How do we deal with failure?
      GxfEntityRefCountInc(context_, eid_);
    }
    return *this;
  }

  Entity& operator=(Entity&& other) {
    // In case other is this, then nothing should be done.
    if (&other == this) {
      return *this;
    }
    if (eid_ != kNullUid) {
      release();
    }
    context_ = other.context_;
    eid_ = other.eid_;
    other.context_ = kNullContext;
    other.eid_ = kNullUid;
    return *this;
  }

  ~Entity() {
    if (eid_ != kNullUid) {
      release();
    }
  }

  // See GxfEntityActivate
  Expected<void> activate() {
    return ExpectedOrCode(GxfEntityActivate(context(), eid()));
  }

  // See GxfEntityDectivate
  Expected<void> deactivate() {
    return ExpectedOrCode(GxfEntityDeactivate(context(), eid()));
  }

  Expected<Entity> clone() const {
    return Shared(context(), eid());
  }

  gxf_context_t context() const { return context_; }
  gxf_uid_t eid() const { return eid_; }
  bool is_null() const { return eid_ == kNullUid; }

  // The name of the entity or empty string if no name has been given to the entity.
  const char* name() const {
    const char* ptr;
    const gxf_result_t result = GxfParameterGetStr(context_, eid_, kInternalNameParameterKey, &ptr);
    return (result == GXF_SUCCESS) ? ptr : "";
  }

  // Adds a component with given type ID
  Expected<UntypedHandle> add(gxf_tid_t tid, const char* name = nullptr) {
    gxf_uid_t cid;
    const auto result = GxfComponentAdd(context(), eid(), tid, name, &cid);
    if (result != GXF_SUCCESS) {
      return Unexpected{result};
    }
    return UntypedHandle::Create(context(), cid);
  }

  // Adds a component with given type
  template <typename T>
  Expected<Handle<T>> add(const char* name = nullptr) {
    gxf_tid_t tid;
    const auto result_1 = GxfComponentTypeId(context(), TypenameAsString<T>(), &tid);
    if (result_1 != GXF_SUCCESS) {
      return Unexpected{result_1};
    }
    gxf_uid_t cid;
    const auto result_2 = GxfComponentAdd(context(), eid(), tid, name, &cid);
    if (result_2 != GXF_SUCCESS) {
      return Unexpected{result_2};
    }
    return Handle<T>::Create(context(), cid);
  }

  // Gets a component by type ID. Asserts if no such component.
  Expected<UntypedHandle> get(gxf_tid_t tid, const char* name = nullptr) const {
    gxf_uid_t cid;
    const auto result = GxfComponentFind(context(), eid(), tid, name, nullptr, &cid);
    if (result != GXF_SUCCESS) {
      return Unexpected{result};
    }
    return UntypedHandle::Create(context(), cid);
  }

  // Gets a component by type. Asserts if no such component.
  template <typename T>
  Expected<Handle<T>> get(const char* name = nullptr) const {
    gxf_tid_t tid;
    const auto result_1 = GxfComponentTypeId(context(), TypenameAsString<T>(), &tid);
    if (result_1 != GXF_SUCCESS) {
      return Unexpected{result_1};
    }
    gxf_uid_t cid;
    const auto result_2 = GxfComponentFind(context(), eid(), tid, name, nullptr, &cid);
    if (result_2 != GXF_SUCCESS) {
      return Unexpected{result_2};
    }
    return Handle<T>::Create(context(), cid);
  }

  // Finds all components.
  Expected<void> findAll(FixedVectorBase<UntypedHandle>& components) const {
    const gxf_context_t c_context = context();
    const gxf_uid_t c_eid = eid();
    for (int offset = 0; ; offset++) {
      gxf_uid_t cid;
      const auto code = GxfComponentFind(c_context, c_eid, GxfTidNull(), nullptr, &offset, &cid);
      if (code != GXF_SUCCESS) { break; }
      const auto handle = UntypedHandle::Create(c_context, cid);
      if (!handle) { return ForwardError(handle); }
      const auto result = components.push_back(handle.value());
      if (!result) { return Unexpected{GXF_EXCEEDING_PREALLOCATED_SIZE}; }
    }
    return Success;
  }

  // Deprecated
  // TODO(ayusmans): Remove for 2.4 release
  std::vector<UntypedHandle> findAll() const {
    std::vector<UntypedHandle> components;
    for (int offset = 0; ; offset++) {
      gxf_uid_t cid;
      const auto result = GxfComponentFind(context(), eid(), GxfTidNull(), nullptr, &offset, &cid);
      if (result != GXF_SUCCESS) {
        return components;
      }
      const auto handle = UntypedHandle::Create(context(), cid);
      if (!handle) {
        return {};  // FIXME
      }
      components.push_back(handle.value());
    }
    return components;
  }

  // Finds all components of given type.
  template <typename T>
  Expected<void> findAll(FixedVectorBase<Handle<T>>& components) const {
    const gxf_context_t c_context = context();
    const gxf_uid_t c_eid = eid();
    gxf_tid_t tid;
    const auto code = GxfComponentTypeId(c_context, TypenameAsString<T>(), &tid);
    if (code != GXF_SUCCESS) { return Unexpected{code}; }
    for (int offset = 0; ; offset++) {
      gxf_uid_t cid;
      const auto code = GxfComponentFind(c_context, c_eid, tid, nullptr, &offset, &cid);
      if (code != GXF_SUCCESS) { break; }
      const auto handle = Handle<T>::Create(c_context, cid);
      if (!handle) { return ForwardError(handle); }
      const auto result = components.push_back(handle.value());
      if (!result) { return Unexpected{GXF_EXCEEDING_PREALLOCATED_SIZE}; }
    }
    return Success;
  }

  // Deprecated
  // TODO(ayusmans): Remove for 2.4 release
  template <typename T>
  std::vector<Handle<T>> findAll() const {
    gxf_tid_t tid;
    const auto result_1 = GxfComponentTypeId(context(), TypenameAsString<T>(), &tid);
    if (result_1 != GXF_SUCCESS) {
      return {};  // FIXME
    }
    std::vector<Handle<T>> components;
    for (int offset = 0; ; offset++) {
      gxf_uid_t cid;
      const auto result_2 = GxfComponentFind(context(), eid(), tid, nullptr, &offset, &cid);
      if (result_2 != GXF_SUCCESS) {
        return components;
      }
      const auto handle = Handle<T>::Create(context(), cid);
      if (!handle) {
        return {};  // FIXME
      }
      components.push_back(handle.value());
    }
    return components;
  }

 private:
  void release() {
    GxfEntityRefCountDec(context_, eid_);  // TODO(v2) We should use the error code, but we can't
                                           //          do anything about it..
    eid_ = kNullUid;
  }

  gxf_context_t context_ = kNullContext;
  gxf_uid_t eid_ = kNullUid;
};

}  // namespace gxf
}  // namespace nvidia

#endif  // NVIDIA_GXF_CORE_ENTITY_HPP_
