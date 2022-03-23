/*
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#ifndef NVIDIA_GXF_STD_PARAMETER_PARSER_STD_HPP_
#define NVIDIA_GXF_STD_PARAMETER_PARSER_STD_HPP_

#include <array>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "yaml-cpp/yaml.h"

#include "common/assert.hpp"
#include "gxf/core/expected.hpp"
#include "gxf/core/gxf.h"
#include "gxf/core/handle.hpp"
#include "gxf/std/parameter_parser.hpp"
#include "gxf/std/parameter_registrar.hpp"

namespace nvidia {
namespace gxf {

// Parses unknown many of known types.
template <typename T>
struct ParameterParser<std::vector<T>> {
  static Expected<std::vector<T>> Parse(gxf_context_t context, gxf_uid_t component_uid,
                                        const char* key, const YAML::Node& node,
                                        const std::string& prefix) {
    if (!node.IsSequence()) {
      const char* component_name = "UNKNOWN";
      GxfParameterGetStr(context, component_uid, kInternalNameParameterKey, &component_name);
      GXF_LOG_ERROR("Parameter '%s' in component '%s' must be a vector", key, component_name);
      return Unexpected{GXF_PARAMETER_PARSER_ERROR};
    }
    std::vector<T> result(node.size());
    for (size_t i = 0; i < node.size(); i++) {
      const auto maybe = ParameterParser<T>::Parse(context, component_uid, key, node[i], prefix);
      if (!maybe) {
        return ForwardError(maybe);
      }
      result[i] = std::move(maybe.value());
    }
    return result;
  }
};

// Parses known many of known types.
template <typename T, std::size_t N>
struct ParameterParser<std::array<T, N>> {
  static Expected<std::array<T, N>> Parse(gxf_context_t context, gxf_uid_t component_uid,
                                                const char* key, const YAML::Node& node,
                                                const std::string& prefix) {
    if (!node.IsSequence()) {
      const char* component_name = "UNKNOWN";
      GxfParameterGetStr(context, component_uid, kInternalNameParameterKey, &component_name);
      GXF_LOG_ERROR("Parameter '%s' in component '%s' must be an array", key, component_name);
      return Unexpected{GXF_PARAMETER_PARSER_ERROR};
    }
    if (node.size() != N) {
      GXF_LOG_ERROR("Length of parameter array (%zu) does not match required length (%zu)",
                    node.size(), N);
      return Unexpected{GXF_PARAMETER_PARSER_ERROR};
    }
    std::array<T, N> result;
    for (size_t i = 0; i < result.size(); i++) {
      const auto maybe = ParameterParser<T>::Parse(context, component_uid, key, node[i], prefix);
      if (!maybe) { return ForwardError(maybe); }
      result[i] = std::move(maybe.value());
    }
    return result;
  }
};

// Template specializations for ParameterInfo to be used during
// parameter registration in the extension

// Specialized ParameterInfoOverride for parameters of type std::vector<T>
template <typename T>
struct ParameterInfoOverride<std::vector<T>> {
  Expected<void> apply(ParameterRegistrar* registrar,
                       ParameterRegistrar::ComponentParameterInfo& info) {
    // Get the element info
    ParameterInfoOverride<T> override;
    ParameterRegistrar::ComponentParameterInfo element_info;
    const auto result = override.apply(registrar, element_info);
    if (!result) { return ForwardError(result); }
    info.type = element_info.type;
    info.is_arithmetic = element_info.is_arithmetic;
    info.handle_tid = element_info.handle_tid;

    // Fetch the shape of <T> and update it to the current ComponentParameterInfo
    for (int32_t i = 0; i < element_info.rank; ++i) {
      info.shape[i] = element_info.shape[i];
    }
    // A vector increases the rank by 1 and adds shape [-1] to <T>.
    info.shape[element_info.rank] = -1;
    info.rank = element_info.rank + 1;

    return Success;
  }
};

// Specialized ParameterInfoOverride for parameters of type std::array<T,N>
template <typename T, std::size_t N>
struct ParameterInfoOverride<std::array<T, N>> {
  Expected<void> apply(ParameterRegistrar* registrar,
                       ParameterRegistrar::ComponentParameterInfo& info) {
    // Get the element info
    ParameterInfoOverride<T> override;
    ParameterRegistrar::ComponentParameterInfo element_info;
    const auto result = override.apply(registrar, element_info);
    if (!result) { return ForwardError(result); }
    info.type = element_info.type;
    info.is_arithmetic = element_info.is_arithmetic;
    info.handle_tid = element_info.handle_tid;

    // Fetch the shape of <T> and update it to the current ComponentParameterInfo
    for (int32_t i = 0; i < element_info.rank; ++i) {
      info.shape[i] = element_info.shape[i];
    }
    // An increases the rank by 1 and adds shape [N] to <T>.
    info.shape[element_info.rank] = N;
    info.rank = element_info.rank + 1;

    return Success;
  }
};

}  // namespace gxf
}  // namespace nvidia

#endif  // NVIDIA_GXF_STD_PARAMETER_PARSER_STD_HPP_
