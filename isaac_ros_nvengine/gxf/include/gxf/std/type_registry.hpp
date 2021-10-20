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
#ifndef NVIDIA_GXF_STD_TYPE_REGISTRY_HPP
#define NVIDIA_GXF_STD_TYPE_REGISTRY_HPP

#include <map>
#include <set>
#include <shared_mutex>  // NOLINT
#include <string>

#include "gxf/core/expected.hpp"

namespace nvidia {
namespace gxf {

// A type registry which maps a C++ type to it's name and which can be used to
// track base classes.
class TypeRegistry {
 public:
  Expected<void> add(gxf_tid_t tid, const char* component_type_name);

  Expected<void> add_base(const char* component_type_name, const char* base_type_name);

  Expected<gxf_tid_t> id_from_name(const char* component_type_name) const;

  bool is_base(gxf_tid_t derived, gxf_tid_t base) const;

  Expected<const char*> name(gxf_tid_t tid) const;

 private:
  std::map<std::string, gxf_tid_t> tids_;
  std::map<gxf_tid_t, std::set<gxf_tid_t>> bases_;
  mutable std::shared_timed_mutex mutex_;
};

}  // namespace gxf
}  // namespace nvidia

#endif
