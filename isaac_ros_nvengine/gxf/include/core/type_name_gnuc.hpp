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
#ifndef NVIDIA_CORE_TYPE_NAME_GNUC_HPP
#define NVIDIA_CORE_TYPE_NAME_GNUC_HPP

#include <cstdint>

namespace nvidia {

// For the GNU compiler __PRETTY_FUNCTION__ is a null-terminated string with the following form:
//   const char* nvidia::TypenameAsString() [with T = nvidia::gxf::Component]
// We would like to extract the type name as "nvidia::gxf::Component".
// The result is stored in the given string 'output'. If the type name is longer than max_length
// or another error occurs the function returns nullptr; otherwise 'output' is returned.
char* TypenameAsStringGnuC(const char* begin, char* output, int32_t max_length);

// Compiler-specific implementation of the TypenameAsString function.
inline char* TypenameAsStringImpl(const char* begin, char* output, int32_t max_length) {
  return TypenameAsStringGnuC(begin, output, max_length);
}

}  // namespace nvidia

#endif  // NVIDIA_CORE_TYPE_NAME_GNUC_HPP
