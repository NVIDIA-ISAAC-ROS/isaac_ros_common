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
#ifndef NVIDIA_GXF_CORE_PARAMETER_HPP_
#define NVIDIA_GXF_CORE_PARAMETER_HPP_

#include <functional>
#include <string>
#include <utility>

#include "core/assert.hpp"
#include "core/optional.hpp"
#include "gxf/core/expected.hpp"
#include "gxf/core/gxf.h"
#include "gxf/core/handle.hpp"
#include "gxf/std/parameter_parser.hpp"

namespace YAML { class Node; }

namespace nvidia {
namespace gxf {

// Base class for parameters stored in ParameterStorage
class ParameterBackendBase {
 public:
  virtual ~ParameterBackendBase() = default;

  // The context to which this parameter backend belongs
  gxf_context_t context() const { return context_; }

  // The object to which this parameter is attached.
  gxf_uid_t uid() const { return uid_; }

  // The name of the parameter
  const char* key() const { return key_; }

  // Returns true if the parameter is guaranteed to always have a value set. Only mandatory
  // parameters can be accessed direclty with 'get' instead of using 'try_get'.
  bool isMandatory() const { return (flags_ & GXF_PARAMETER_FLAGS_OPTIONAL) == 0; }

  // Returns true if the parameter can not be changed after the component has been activated.
  bool isConstant() const { return (flags_ & GXF_PARAMETER_FLAGS_DYNAMIC) == 0; }

  // Sets the latest value from the backend to the frontend.
  virtual void writeToFrontend() = 0;

  // Parses the parameter from the given YAML object.
  virtual Expected<void> parse(const YAML::Node& node, const std::string& prefix) = 0;

  // Returns true if the parameter is set
  virtual bool isAvailable() const = 0;

  // Returns true if it is possible to change this parameter
  bool isImmutable() const {
    if (isConstant()) {
      const bool is_active = false;  // FIXME(v1) Check that component is not active.
      if (is_active) {
        return true;
      }
    }
    return false;
  }

  // FIXME(v1) make private
  gxf_context_t context_;
  gxf_uid_t uid_;
  gxf_parameter_flags_t flags_;
  bool is_dynamic_;
  const char* key_;
  const char* headline_;
  const char* description_;
};

template <typename T>
class Parameter;

// This class stores a parameter of a specific type in the backend.
template <typename T>
class ParameterBackend : public ParameterBackendBase {
 public:
  void writeToFrontend() override;
  Expected<void> parse(const YAML::Node& node, const std::string& prefix) override;
  bool isAvailable() const override { return value_ != std::nullopt; }

  // Sets the parameter to the given value.
  Expected<void> set(T value) {
    // Make sure that the new value passes the validator
    if (validator_&& !validator_(value)) { return Unexpected{GXF_PARAMETER_OUT_OF_RANGE}; }
    // Don't allow modification of a parameter which is currently immutable
    if (isImmutable()) { return Unexpected{GXF_PARAMETER_CAN_NOT_MODIFY_CONSTANT}; }
    // Update the parameter value
    value_ = std::move(value);
    return Success;
  }

  // Gets the current value of the parameter.
  const std::optional<T>& try_get() const { return value_; }

  // FIXME(v1) make private
  nvidia::gxf::Parameter<T>* frontend_ = nullptr;
  std::function<bool(const T&)> validator_;
  std::optional<T> value_;
};

// An intermediate base class for parameters which store a handle.
class HandleParameterBackend : public ParameterBackendBase {
 public:
  virtual ~HandleParameterBackend() = default;

  // Gets the component ID of the handle.
  virtual Expected<gxf_uid_t> get() const = 0;

  // Sets the handle using just a component ID
  virtual Expected<void> set(gxf_uid_t cid) = 0;
};

// A specialization of ParameterBackend<T> for handle types. It derives from the intermediate base
// class HandleParameterBackend so that parameter backends of handle types all have a common base
// class.
template <typename T>
class ParameterBackend<Handle<T>> : public HandleParameterBackend {
 public:
  void writeToFrontend() override;
  Expected<void> parse(const YAML::Node& node, const std::string& prefix) override;
  bool isAvailable() const override {
    if ((value_ == std::nullopt) || (value_ == Handle<T>::Unspecified())) { return false; }
    return true;
  }

  // Sets the handle using just a component ID
  Expected<void> set(gxf_uid_t cid) override {
    auto expected = Handle<T>::Create(context(), cid);
    if (expected) {
      value_ = expected.value();
      return Success;
    } else {
      return ForwardError(expected);
    }
  }

  // Gets the component ID of the handle.
  Expected<gxf_uid_t> get() const override {
    if (!value_) {return gxf::Expected<gxf_uid_t>{gxf::Unexpected{GXF_FAILURE}}; }
    return value_->cid();
  }

  // Sets the parameter to the given value.
  Expected<void> set(Handle<T> value) {
    if (isImmutable()) { return Unexpected{GXF_PARAMETER_CAN_NOT_MODIFY_CONSTANT}; }
    value_ = std::move(value);
    return Success;
  }

  // Gets the current value of the parameter.
  const std::optional<Handle<T>>& try_get() const { return value_; }

  // FIXME(v1) make private
  nvidia::gxf::Parameter<Handle<T>>* frontend_ = nullptr;
  std::optional<Handle<T>> value_;
};


// Common base class for specializations of Parameter<T>.
class ParameterBase {
 public:
  virtual ~ParameterBase() = default;
};

// This type represents a parameter of a component and can be used in custom components. It
// communicates with the backend to set and get parameters as configured.
template <typename T>
class Parameter : public ParameterBase {
 public:
  // Gets the current parameter value. Only valid if the parameter is marked as 'mandatory' in the
  // paramater interface. Otherwise an assert will be raised.
  const T& get() const {
    GXF_ASSERT(backend_ != nullptr, "A parameter with type '%s' was not registered.",
               TypenameAsString<T>());
    GXF_ASSERT(backend_->isMandatory(), "Only mandatory parameters can be accessed with get(). "
               "'%s' is not marked as mandatory", backend_->key());
    GXF_ASSERT(value_, "Mandatory parameter '%s' was not set.", backend_->key());
    return *value_;
  }

  // Convenience function for accessing a mandatory parameter.
  operator const T&() const {
    return get();
  }

  // Tries to get the parameter value. If the parameter is not set std::nullopt is returned.
  const std::optional<T>& try_get() const { return value_; }

  // Sets the parameter to the given value.
  Expected<void> set(T value) {
    GXF_ASSERT(backend_ != nullptr, "Parameter '%s' was not registered.", backend_->key());
    const auto result = backend_->set(value);
    if (!result) {
      return result;
    }
    value_ = std::move(value);
    return Success;
  }

  // Sets the parameter to the given value, but does not notify the backend about the change.
  // This function shall only be used by the ParameterBackend class.
  void setWithoutPropagate(const T& value) {
    value_ = value;
  }

  // Connects this parameter frontend to the corresponding backend.
  void connect(ParameterBackend<T>* backend) {
    backend_ = backend;
  }

  const char* key() {
    return backend_ == nullptr ? nullptr : backend_->key();
  }

 private:
  std::optional<T> value_;
  ParameterBackend<T>* backend_ = nullptr;
};

// A specialization of Parameter<T> for handle types.
template <typename S>
class Parameter<Handle<S>> : public ParameterBase {
 public:
  // Gets the current parameter value. Only valid if the parameter is marked as 'mandatory' in the
  // paramater interface. Otherwise an assert will be raised.
  const Handle<S>& get() const {
    GXF_ASSERT(backend_ != nullptr, "A handle parameter with type '%s' was not registered.",
               TypenameAsString<S>());
    GXF_ASSERT(backend_->isMandatory(), "Only mandatory parameters can be accessed with get(). "
               "'%s' is not marked as mandatory", backend_->key());
    GXF_ASSERT(value_, "Mandatory parameter '%s' was not set.", backend_->key());
    return *value_;
  }

  // Convenience function for accessing a mandatory parameter.
  operator const Handle<S>&() const {
    return get();
  }

  // Tries to get the parameter value. If the parameter is not set std::nullopt is returned.
  const std::optional<Handle<S>>& try_get() const { return value_; }

  // Only if T = Handle<S>
  S* operator->() const {
    return get().get();
  }

  // Sets the parameter to the given value.
  Expected<void> set(Handle<S> value) {
    GXF_ASSERT(backend_ != nullptr, "Parameter '%s' was not registered.", backend_->key());
    const auto result = backend_->set(value);
    if (!result) {
      return result;
    }
    value_ = std::move(value);
    return Success;
  }

  // Sets the parameter to the given value, but does not notify the backend about the change.
  // This function shall only be used by the ParameterBackend class.
  void setWithoutPropagate(const Handle<S>& value) {
    value_ = value;
  }

  // Connects this parameter frontend to the corresponding backend.
  void connect(ParameterBackend<Handle<S>>* backend) {
    backend_ = backend;
  }

  const char* key() {
    return backend_ == nullptr ? nullptr : backend_->key();
  }

 private:
  std::optional<Handle<S>> value_;
  ParameterBackend<Handle<S>>* backend_ = nullptr;
};

// -------------------------------------------------------------------------------------------------

template <typename T>
void ParameterBackend<T>::writeToFrontend() {
  if (frontend_ && value_) {
    frontend_->setWithoutPropagate(*value_);
  }
}

template <typename T>
Expected<void> ParameterBackend<T>::parse(const YAML::Node& node, const std::string& prefix) {
  return ParameterParser<T>::Parse(context(), uid(), key(), node, prefix)
        .map([this] (const T& value) { return set(value); })
        .and_then([this] { writeToFrontend(); });
}

template <typename T>
void ParameterBackend<Handle<T>>::writeToFrontend() {
  if (frontend_ && value_) {
    frontend_->setWithoutPropagate(*value_);
  }
}

template <typename T>
Expected<void> ParameterBackend<Handle<T>>::parse(const YAML::Node& node,
                                                  const std::string& prefix) {
  return ParameterParser<Handle<T>>::Parse(context(), uid(), key(), node, prefix)
        .map([this] (const Handle<T>& value) { return set(value); })
        .and_then([this] { writeToFrontend(); });
}

}  // namespace gxf
}  // namespace nvidia

#endif  // NVIDIA_GXF_CORE_PARAMETER_HPP_
