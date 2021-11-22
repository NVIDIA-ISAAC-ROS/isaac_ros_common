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
#ifndef NVIDIA_GXF_STD_PARAMETER_PARSER_HPP_
#define NVIDIA_GXF_STD_PARAMETER_PARSER_HPP_

#include <sstream>
#include <string>
#include <type_traits>
#include <utility>

#include "core/assert.hpp"
#include "core/fixed_vector.hpp"
#include "gxf/core/expected.hpp"
#include "gxf/core/gxf.h"
#include "gxf/core/handle.hpp"
#include "yaml-cpp/yaml.h"

namespace nvidia {
namespace gxf {

template <typename T, typename V = void>
struct ParameterParser;

// Parses a parameter from YAML using the default YAML parser. This class can be specialized to
// support custom types.
template <typename T>
struct ParameterParser<T> {
  static Expected<T> Parse(gxf_context_t context, gxf_uid_t component_uid, const char* key,
                           const YAML::Node& node, const std::string& prefix) {
    try {
      return node.as<T>();
    } catch (...) {
      std::stringstream ss;
      ss << node;
      GXF_LOG_ERROR("Could not parse parameter '%s' from '%s'", key, ss.str().c_str());
      return Unexpected{GXF_PARAMETER_PARSER_ERROR};
    }
  }
};

// Specialization of ParameterParser for std::string. The full node is serialized to a string, even
// though it might contain sub children.
template <>
struct ParameterParser<std::string> {
  static Expected<std::string> Parse(gxf_context_t context, gxf_uid_t component_uid,
                                     const char* key, const YAML::Node& node,
                                     const std::string& prefix) {
    try {
      std::stringstream ss;
      ss << node;
      return ss.str();
    } catch (...) {
      std::stringstream ss;
      ss << node;
      GXF_LOG_ERROR("Could not parse parameter '%s' from '%s'", key, ss.str().c_str());
      return Unexpected{GXF_PARAMETER_PARSER_ERROR};
    }
  }
};

struct FilePath : public std::string {
  FilePath() : std::string() {}

  FilePath(const char* s) : std::string(s) {}

  FilePath(const std::string& str) : std::string(str) {}
};

template <>
struct ParameterParser<FilePath> {
  static Expected<FilePath> Parse(gxf_context_t context, gxf_uid_t component_uid,
                                     const char* key, const YAML::Node& node,
                                     const std::string& prefix) {
    try {
      FilePath path;
      std::stringstream ss;
      ss << node;
      path.assign(ss.str());
      return path;
    } catch (...) {
      std::stringstream ss;
      ss << node;
      GXF_LOG_ERROR("Could not parse parameter '%s' from '%s'", key, ss.str().c_str());
      return Unexpected{GXF_PARAMETER_PARSER_ERROR};
    }
  }
};

// Specialization of ParameterParser for uint8_t because it is not supported natively by yaml-cpp
template <>
struct ParameterParser<uint8_t> {
  static Expected<uint8_t> Parse(gxf_context_t context, gxf_uid_t component_uid,
                                     const char* key, const YAML::Node& node,
                                     const std::string& prefix) {
    try {
      return static_cast<uint8_t>(node.as<uint32_t>());
    } catch (...) {
      std::stringstream ss;
      ss << node;
      GXF_LOG_ERROR("Could not parse parameter '%s' from '%s'", key, ss.str().c_str());
      return Unexpected{GXF_PARAMETER_PARSER_ERROR};
    }
  }
};

// Specialization of ParameterParser for FixedVector with stack allocation.
// Substitutes std::array for safety-critical components.
// TODO(ayusmans): parsing support for FixedVector with heap allocation
template <typename T, size_t N>
struct ParameterParser<FixedVector<T, N>> {
  static Expected<FixedVector<T, N>> Parse(gxf_context_t context, gxf_uid_t component_uid,
                                           const char* key, const YAML::Node& node,
                                           const std::string& prefix) {
    if (!node.IsSequence()) {
      return Unexpected{GXF_PARAMETER_PARSER_ERROR};
    }
    if (node.size() > N) {
      GXF_LOG_ERROR("Parameter size (%zu) exceeds vector capacity (%zu)", node.size(), N);
      return Unexpected{GXF_PARAMETER_PARSER_ERROR};
    }
    FixedVector<T, N> vector;
    for (size_t i = 0; i < node.size(); i++) {
      const auto maybe = ParameterParser<T>::Parse(context, component_uid, key, node[i], prefix);
      if (!maybe) {
        return ForwardError(maybe);
      }
      vector.push_back(std::move(maybe.value()));
    }
    return vector;
  }
};

// Specialization of ParameterParser for gxf::Handle. It parses the parameter as a string and
// interprets it as either a component name in the current entity, or as a composed string of the
// form 'entity_name/component_name'.
template <typename S>
struct ParameterParser<Handle<S>> {
  static Expected<Handle<S>> Parse(gxf_context_t context, gxf_uid_t component_uid, const char* key,
                                   const YAML::Node& node, const std::string& prefix) {
    // Parse string from node
    std::string tag;
    try {
      tag = node.as<std::string>();
    } catch (...) {
      std::stringstream ss;
      ss << node;
      GXF_LOG_ERROR("Could not parse parameter '%s' from '%s'", key, ss.str().c_str());
      return Unexpected{GXF_PARAMETER_PARSER_ERROR};
    }

    gxf_uid_t eid;
    std::string component_name;

    const size_t pos = tag.find('/');
    if (pos == std::string::npos) {
      // Get the entity of this component
      const gxf_result_t result_1 = GxfComponentEntity(context, component_uid, &eid);
      if (result_1 != GXF_SUCCESS) {
        return Unexpected{result_1};
      }
      component_name = tag;
    } else {
      component_name = tag.substr(pos + 1);

      // Get the entity
      gxf_result_t result_1_with_prefix = GXF_FAILURE;
      // Try using entity name with prefix
      if (!prefix.empty()) {
        const std::string entity_name = prefix + tag.substr(0, pos);
        result_1_with_prefix = GxfEntityFind(context, entity_name.c_str(), &eid);
        if (result_1_with_prefix != GXF_SUCCESS) {
          GXF_LOG_WARNING("Could not find entity (with prefix) '%s' while parsing parameter '%s' "
                          "of component %zu", entity_name.c_str(), key, component_uid);
        }
      }
      // Try using entity name without prefix, if lookup with prefix failed
      if (result_1_with_prefix != GXF_SUCCESS) {
        const std::string entity_name = tag.substr(0, pos);
        const gxf_result_t result_1_no_prefix  = GxfEntityFind(context, entity_name.c_str(), &eid);
        if (result_1_no_prefix != GXF_SUCCESS) {
          GXF_LOG_ERROR("Could not find entity '%s' while parsing parameter '%s' of component %zu",
                        entity_name.c_str(), key, component_uid);
          return Unexpected{result_1_no_prefix};
        }
      }
    }

    // Get the type id of the component we are are looking for.
    gxf_tid_t tid;
    const gxf_result_t result_2 = GxfComponentTypeId(context, TypenameAsString<S>(), &tid);
    if (result_2 != GXF_SUCCESS) {
      return Unexpected{result_2};
    }

    // Find the component in the indicated entity
    gxf_uid_t cid;
    const gxf_result_t result_3 = GxfComponentFind(context, eid, tid, component_name.c_str(),
                                                   nullptr, &cid);
    if (result_3 != GXF_SUCCESS) {
      if (component_name == "<Unspecified>") {
        GXF_LOG_DEBUG("Using an <Unspecified> handle in entity %zu while parsing parameter '%s'"
        " of component %zu. This handle must be set to a valid component before graph activation",
        eid, key, component_uid);
        return Handle<S>::Unspecified();
      } else {
        GXF_LOG_WARNING("Could not find component '%s' in entity %zu while parsing parameter '%s' "
              "of component %zu", component_name.c_str(), eid, key,
              component_uid);
      }

      return Unexpected{result_3};
    }

    return Handle<S>::Create(context, cid);
  }
};

}  // namespace gxf
}  // namespace nvidia

#endif  // NVIDIA_GXF_STD_PARAMETER_PARSER_HPP_
