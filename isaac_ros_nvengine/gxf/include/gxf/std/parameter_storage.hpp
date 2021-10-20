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
#ifndef NVIDIA_GXF_STD_PARAMETER_STORAGE_HPP_
#define NVIDIA_GXF_STD_PARAMETER_STORAGE_HPP_

#include <map>
#include <memory>
#include <shared_mutex>  // NOLINT
#include <string>
#include <utility>

#include "gxf/core/parameter.hpp"

namespace nvidia {
namespace gxf {

// Stores all parameters for an GXF application
//
// Parameters are stored in a dictionary style class. For each parameter the actual parameter value
// and various helper classes to query and set parameters are provided.
class ParameterStorage {
 public:
  ParameterStorage(gxf_context_t context);

  // Registers a parameter with the backend. This is used by components when they register their
  // parameter interface to connect parameters in components to the backend.
  template <typename T>
  Expected<void> registerParameter(nvidia::gxf::Parameter<T>* frontend, gxf_uid_t uid,
                                   const char* key, const char* headline, const char* description,
                                   Expected<T> default_value, gxf_parameter_flags_t flags) {
    if (frontend    == nullptr) { return Unexpected{GXF_ARGUMENT_NULL}; }
    if (key         == nullptr) { return Unexpected{GXF_ARGUMENT_NULL}; }
    if (headline    == nullptr) { return Unexpected{GXF_ARGUMENT_NULL}; }
    if (description == nullptr) { return Unexpected{GXF_ARGUMENT_NULL}; }

    std::unique_lock<std::shared_timed_mutex> lock(mutex_);

    auto it = parameters_.find(uid);
    if (it == parameters_.end()) {
      std::map<std::string, std::unique_ptr<ParameterBackendBase>> tmp;
      it = parameters_.insert({uid, std::move(tmp)}).first;
    }

    const auto jt = it->second.find(key);
    if (jt != it->second.end()) { return Unexpected{GXF_PARAMETER_ALREADY_REGISTERED}; }

    auto backend = std::make_unique<ParameterBackend<T>>();
    backend->context_ = context_;
    backend->uid_ = uid;
    backend->flags_ = flags;
    backend->is_dynamic_ = false;
    backend->key_ = key;
    backend->headline_ = headline;
    backend->description_ = description;
    backend->frontend_ = frontend;
    // FIXME(v1) validator

    frontend->connect(backend.get());

    if (default_value) {
      const auto code = backend->set(std::move(*default_value));
      if (!code) { return ForwardError(code); }
      backend->writeToFrontend();
    }

    it->second.insert({key, std::move(backend)}).first;

    return Success;
  }

  Expected<void> parse(gxf_uid_t uid, const char* key, const YAML::Node& node,
                      const std::string& prefix);

  // Sets a parameter. If the parameter is not yet present a new dynamic parameter will be created.
  // This function fails if a parameter already exists, but with the wrong type.
  template <typename T>
  Expected<void> set(gxf_uid_t uid, const char* key, T value) {
    std::shared_lock<std::shared_timed_mutex> lock(mutex_);

    auto it = parameters_.find(uid);
    if (it == parameters_.end()) {
      std::map<std::string, std::unique_ptr<ParameterBackendBase>> tmp;
      it = parameters_.insert({uid, std::move(tmp)}).first;
    }

    auto jt = it->second.find(key);
    if (jt == it->second.end()) {
      auto ptr = std::make_unique<ParameterBackend<T>>();
      ptr->context_ = context_;
      ptr->uid_ = uid;
      ptr->flags_ = GXF_PARAMETER_FLAGS_OPTIONAL | GXF_PARAMETER_FLAGS_DYNAMIC;
      ptr->is_dynamic_ = true;
      ptr->key_ = key;
      ptr->headline_ = key;
      ptr->description_ = "N/A";
      jt = it->second.insert({key, std::move(ptr)}).first;
    }

    auto* ptr = dynamic_cast<ParameterBackend<T>*>(jt->second.get());
    if (ptr == nullptr) { return Unexpected{GXF_PARAMETER_INVALID_TYPE}; }

    const auto code = ptr->set(std::move(value));
    if (!code) { return ForwardError(code); }

    ptr->writeToFrontend();  // FIXME(v1) Special treatment for codelet parameters
    return Success;
  }

  // Gets a parameter
  template <typename T>
  Expected<T> get(gxf_uid_t uid, const char* key) const {
    std::shared_lock<std::shared_timed_mutex> lock(mutex_);
    return getValuePointer<T>(uid, key).map([](const T* pointer) { return *pointer; });
  }

  // Sets a string parameter. The storage creates it's own internal copy and does not take ownership
  // of the given pointer.
  Expected<void> setStr(gxf_uid_t uid, const char* key, const char* value);

  // Gets a string parameter. A pointer to the internal storage is returned whose contents may
  // change if a call to setStr happens in the meantime.
  Expected<const char*> getStr(gxf_uid_t uid, const char* key) const;

  // Gets a file path parameter. A pointer to the internal storage is returned whose contents may
  // change if a call to setPath happens in the meantime.
  Expected<const char*> getPath(gxf_uid_t uid, const char* key) const;

  // Sets a handle parameter.
  Expected<void> setHandle(gxf_uid_t, const char* key, gxf_uid_t value);

  // Gets a handle parameter.
  Expected<gxf_uid_t> getHandle(gxf_uid_t uid, const char* key) const;

  // Sets a vector of string parameter. The storage creates it's own internal copy and does not take
  // ownership of the given pointer.
  Expected<void> setStrVector(gxf_uid_t uid, const char* key, const char** value, uint64_t length);

  // Adds the given value to a parameter and returns the result. The parameter is initialized to
  // 0 in case it does not exist.
  Expected<int64_t> addGetInt64(gxf_uid_t uid, const char* key, int64_t delta);

  // Returns true if all mandatory parameters are available
  Expected<void> isAvailable() const;

  // Returns true if all mandatory parameters of a component are available
  Expected<void> isAvailable(gxf_uid_t uid) const;

  // Clean up data for specific Entity upon Entity destruction
  Expected<void> clearEntityParameters(gxf_uid_t eid);

 private:
  friend class Runtime;

  // Finds a parameter and returns a pointer to its value.
  template <typename T>
  Expected<const T*> getValuePointer(gxf_uid_t uid, const char* key) const {
    std::shared_lock<std::shared_timed_mutex> lock(mutex_);
    auto backend = getBackendPointerImpl<ParameterBackend<T>>(uid, key);
    if (!backend) { return ForwardError(backend); }
    const auto& maybe = backend.value()->try_get();
    if (!maybe) { return Unexpected{GXF_PARAMETER_NOT_INITIALIZED}; }
    return &(*maybe);
  }

  // Finds a parameter backend and returns a pointer to it.
  template <typename T>
  Expected<const T*> getBackendPointerImpl(gxf_uid_t uid, const char* key) const {
    const auto it = parameters_.find(uid);
    if (it == parameters_.end()) { return Unexpected{GXF_PARAMETER_NOT_FOUND}; }
    const auto jt = it->second.find(key);
    if (jt == it->second.end()) { return Unexpected{GXF_PARAMETER_NOT_FOUND}; }

    const T* ptr = dynamic_cast<const T*>(jt->second.get());
    if (ptr == nullptr) { return Unexpected{GXF_PARAMETER_INVALID_TYPE}; }
    return ptr;
  }

  // Finds a parameter backend and returns a pointer to it.
  template <typename T>
  Expected<T*> getBackendPointerImpl(gxf_uid_t uid, const char* key) {
    const auto it = parameters_.find(uid);
    if (it == parameters_.end()) { return Unexpected{GXF_PARAMETER_NOT_FOUND}; }
    const auto jt = it->second.find(key);
    if (jt == it->second.end()) { return Unexpected{GXF_PARAMETER_NOT_FOUND}; }

    T* ptr = dynamic_cast<T*>(jt->second.get());
    if (ptr == nullptr) { return Unexpected{GXF_PARAMETER_INVALID_TYPE}; }
    return ptr;
  }

  mutable std::shared_timed_mutex mutex_;
  gxf_context_t context_;
  std::map<gxf_uid_t, std::map<std::string, std::unique_ptr<ParameterBackendBase>>> parameters_;
};

}  // namespace gxf
}  // namespace nvidia

#endif  // NVIDIA_GXF_STD_PARAMETER_STORAGE_HPP_
