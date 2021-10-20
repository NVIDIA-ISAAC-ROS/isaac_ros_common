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
#ifndef NVIDIA_GXF_CORE_HANDLE_HPP
#define NVIDIA_GXF_CORE_HANDLE_HPP

#include "core/assert.hpp"
#include "core/type_name.hpp"
#include "gxf/core/expected.hpp"
#include "gxf/core/gxf.h"

namespace nvidia {
namespace gxf {

// A handle which gives access to a component without specifying its type.
class UntypedHandle {
 public:
  UntypedHandle(const UntypedHandle& component) = default;
  UntypedHandle(UntypedHandle&& component) = default;
  UntypedHandle& operator=(const UntypedHandle& component) = default;
  UntypedHandle& operator=(UntypedHandle&& component) = default;

  // Creates a null untyped handle
  static UntypedHandle Null() {
    return UntypedHandle{kNullContext, kNullUid};
  }

  // Creates a new untyped handle
  static Expected<UntypedHandle> Create(gxf_context_t context, gxf_uid_t cid) {
    UntypedHandle untyped_handle{context, cid};
    gxf_tid_t tid;
    gxf_result_t code = GxfComponentType(context, cid, &tid);
    if (code != GXF_SUCCESS) {
      return Unexpected{code};
    }
    const auto result = untyped_handle.initialize(tid);
    if (!result) {
      return ForwardError(result);
    }
    return untyped_handle;
  }

  // The context to which the component belongs
  gxf_context_t context() const { return context_; }
  // The ID of the component.
  gxf_uid_t cid() const { return cid_; }
  // The type ID describing the component type.
  gxf_tid_t tid() const { return tid_; }
  // Returns null if the handle is equivalent to a nullptr.
  bool is_null() const {
    return context_ == kNullContext || cid_ == kNullUid || pointer_ == nullptr;
  }
  // Same as 'is_null'.
  explicit operator bool() const { return !is_null(); }
  // The component name
  const char* name() const {
    const char* result;
    const gxf_result_t code = GxfComponentName(context(), cid(), &result);
    return (code == GXF_SUCCESS) ? result : "";
  }

 protected:
  UntypedHandle(gxf_context_t context, gxf_uid_t cid)
    : context_{context}, cid_{cid}, tid_{GxfTidNull()}, pointer_{nullptr} { }

  Expected<void> initialize(gxf_tid_t tid) {
    tid_ = tid;
    return ExpectedOrCode(GxfComponentPointer(context_, cid_, tid_, &pointer_));
  }

  Expected<void> initialize(const char* type_name) {
    gxf_tid_t tid;
    gxf_result_t code = GxfComponentTypeId(context_, type_name, &tid);
    if (code != GXF_SUCCESS) {
      return Unexpected{code};
    }
    return initialize(tid);
  }

  Expected<void> verifyPointer() const {
    if (pointer_ == nullptr) { return Unexpected{GXF_FAILURE}; }
    void* raw_pointer;
    if (GXF_SUCCESS != GxfComponentPointer(context(), cid(), tid_, &raw_pointer)) {
      return Unexpected{GXF_FAILURE};
    }
    if (raw_pointer != pointer_) { return Unexpected{GXF_FAILURE}; }
    return Success;
  }

  gxf_context_t context_;
  gxf_uid_t cid_;
  gxf_tid_t tid_;
  void* pointer_;
};

// A handle which gives access to a component with a specific type.
template <typename T>
class Handle : public UntypedHandle {
 public:
  // Creates a null handle
  static Handle Null() {
    return Handle{};
  }

  // An unspecified handle is a unique handle used to denote a component which
  // will be created in the future. A parameter of Handle to a type does not consider
  // "Unspecified" as a valid parameter value and hence this handle must only be used
  // when defining a graph application across different files and the parameters are set
  // in a delayed fashion (sub-graphs and parameter yaml files for example)
  // Entity activation will fail if any of the mandatory parameters are "Unspecified"
  static Handle Unspecified() {
    return Handle(kNullContext, kUnspecifiedUid);
  }

  // Creates a new handle
  static Expected<Handle> Create(gxf_context_t context, gxf_uid_t cid) {
    Handle handle{context, cid};
    const auto result = handle.initialize(TypenameAsString<T>());
    if (!result) {
      return ForwardError(result);
    }
    return handle;
  }

  // Creates a new handle from an untyped handle
  static Expected<Handle> Create(const UntypedHandle& untyped_handle) {
    return Create(untyped_handle.context(), untyped_handle.cid());
  }

  friend bool operator==(const Handle& lhs, const Handle& rhs) {
    return lhs.context() == rhs.context() && lhs.cid() == rhs.cid();
  }

  friend bool operator<(const Handle& lhs, const Handle& rhs) {
    return lhs.cid() < rhs.cid();
  }

  Handle(gxf_context_t context = kNullContext, gxf_uid_t uid = kNullUid)
    : UntypedHandle{context, uid} {}

  ~Handle() = default;

  Handle(const Handle& component) = default;
  Handle(Handle&& component) = default;
  Handle& operator=(const Handle& component) = default;
  Handle& operator=(Handle&& component) = default;

  template <typename Derived>
  Handle(const Handle<Derived>& derived) : UntypedHandle(derived) {
    static_assert(std::is_base_of<T, Derived>::value,
                  "Handle convertion is only allowed from derived class to base class");
  }

  // Allow conversion from handle to pointer
  operator T*() const { return get(); }

  T* operator->() const {
    return get();
  }

  T* get() const {
    GXF_ASSERT(verifyPointer(), "Invalid Component Pointer.");
    return reinterpret_cast<T*>(pointer_);
  }

  Expected<T*> try_get() const {
    if (!verifyPointer()) { return Unexpected{GXF_FAILURE}; }
    return reinterpret_cast<T*>(pointer_);
  }

 private:
  using UntypedHandle::UntypedHandle;
};

}  // namespace gxf
}  // namespace nvidia

#endif
