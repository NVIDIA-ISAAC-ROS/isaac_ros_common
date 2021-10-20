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
#ifndef NVIDIA_GXF_CORE_REGISTRAR_HPP_
#define NVIDIA_GXF_CORE_REGISTRAR_HPP_

#include <map>
#include <memory>
#include <shared_mutex>  // NOLINT
#include <string>
#include <tuple>
#include <utility>

#include "core/optional.hpp"
#include "gxf/core/parameter.hpp"
#include "gxf/std/parameter_registrar.hpp"
#include "gxf/std/parameter_storage.hpp"

namespace nvidia {
namespace gxf {

// Registers parameters and other information of components
class Registrar {
 public:
  // Constant for registering an optional parameter with no default value
  static constexpr Unexpected NoDefaultParameter() {
    return Unexpected{GXF_PARAMETER_NOT_INITIALIZED};
  }

  Registrar() = default;

  // Simplified version of 'parameter'.
  template <typename T>
  Expected<void> parameter(Parameter<T>& parameter, const char* key) {
    return parameterImpl<T>(parameter, {key, key, "N/A"});
  }

  // Simplified version of 'parameter'.
  template <typename T>
  Expected<void> parameter(Parameter<T>& parameter, const char* key, const char* headline) {
    return parameterImpl<T>(parameter, {key, headline, "N/A"});
  }

  // Simplified version of 'parameter'.
  template <typename T>
  Expected<void> parameter(Parameter<T>& parameter, const char* key, const char* headline,
                           const char* description) {
    return parameterImpl<T>(parameter, {key, headline, description});
  }

  // Simplified version of 'parameter'.
  template <typename T>
  Expected<void> parameter(Parameter<T>& parameter, const char* key, const char* headline,
                           const char* description, const T& default_value) {
    ParameterInfo<T> info;
    info.key = key;
    info.headline = headline;
    info.description = description;
    info.value_default = default_value;
    return parameterImpl<T>(parameter, info);
  }

  // Register a component parameter. Every parameter must be registered in the function
  // 'registerInterface'.
  template <typename T>
  Expected<void> parameter(Parameter<T>& parameter, const char* key, const char* headline,
                           const char* description, const T& default_value,
                           gxf_parameter_flags_t flags) {
    ParameterInfo<T> info;
    info.key = key;
    info.headline = headline;
    info.description = description;
    info.value_default = default_value;
    info.flags = flags;
    return parameterImpl<T>(parameter, info);
  }

  // Deprecated
  template <typename T>
  Expected<void> parameter(Parameter<T>& parameter, const char* key, const char* headline,
                           const char* description, std::nullopt_t, gxf_parameter_flags_t flags) {
    GXF_LOG_WARNING("std::nullopt is deprecated, use Registrar::NoDefaultParameter() instead");
    ParameterInfo<T> info;
    info.key = key;
    info.headline = headline;
    info.description = description;
    info.flags = flags;
    return parameterImpl<T>(parameter, info);
  }

  // Register a component parameter. Every parameter must be registered in the function
  // 'registerInterface'.
  template <typename T>
  Expected<void> parameter(Parameter<T>& parameter, const char* key, const char* headline,
                           const char* description, Unexpected, gxf_parameter_flags_t flags) {
    ParameterInfo<T> info;
    info.key = key;
    info.headline = headline;
    info.description = description;
    info.flags = flags;
    return parameterImpl<T>(parameter, info);
  }

  // Registers with a struct of all info.
  template <typename T>
  Expected<void> parameter(nvidia::gxf::Parameter<T>& parameter,
                           const ParameterInfo<T>& parameter_info) {
    return parameterImpl<T>(parameter, parameter_info);
  }

  // Implementation for parameter() above that registers parameter in ParameterRegistrar and
  // ParameterStorage.
  template <typename T>
  Expected<void> parameterImpl(nvidia::gxf::Parameter<T>& parameter,
                               const ParameterInfo<T>& parameter_info) {
    if (parameter_registrar != nullptr) {
      Expected<void> result =
          parameter_registrar->registerComponentParameter(tid, type_name, parameter_info);
      if (!result) { return ForwardError(result); }
    }

    if (parameter_storage == nullptr) { return Unexpected{GXF_CONTEXT_INVALID}; }
    return parameter_storage->registerParameter<T>(
        &parameter, cid, parameter_info.key, parameter_info.headline, parameter_info.description,
        parameter_info.value_default, parameter_info.flags);
  }

  // Registers a component with no parameters
  Expected<void> registerParameterlessComponent() {
    if (parameter_registrar != nullptr) parameter_registrar->addParameterlessType(tid, type_name);
    return Success;
  }

  // Sets the mandatory parameter storage where parameters loaded from YAML are stored.
  void setParameterStorage(ParameterStorage* param_storage) { parameter_storage = param_storage; }

  // Sets parameter registrar
  void setParameterRegistrar(ParameterRegistrar* param_registrar) {
    parameter_registrar = param_registrar;
  }

  ParameterStorage* parameter_storage = nullptr;

  // Stores information about parameter for query
  ParameterRegistrar* parameter_registrar = nullptr;

  // The type id of registering component
  gxf_tid_t tid;
  // The instance uid of registering component
  gxf_uid_t cid = 0;
  // The typename of the registering component
  std::string type_name;
};

}  // namespace gxf
}  // namespace nvidia

#endif  // NVIDIA_GXF_CORE_REGISTRAR_HPP_
