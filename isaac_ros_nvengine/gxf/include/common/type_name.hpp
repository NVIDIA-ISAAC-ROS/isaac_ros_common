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
#ifndef NVIDIA_COMMON_TYPE_NAME_HPP_
#define NVIDIA_COMMON_TYPE_NAME_HPP_

#include <cstdint>

#if defined(__clang__)
// Not yet implemented
#elif defined(__GNUC__)
#include "type_name_gnuc.hpp"
#elif defined(_MSC_VER)
// Not yet implemented
#endif

namespace nvidia {

// Gives a string representation of the name of a C++ type.
//
// The function will compute the typename during the first invocation and store it in heap memory.
// Computation of the typename is linear in the length of the typename and does not involve dynamic
// memory allocations.
//
// For example:
//   namespace foo {
//     struct Bar {};
//   }  // namespace
//   TyenameAsString<foo::Bar>();  // returns "foo::Bar"
//
// Note: Only "simple" class types in global namespace or a "normal" namespaces are guaranteed to
// work. Templates, lambdas and anonymous namespaces are not guaranteed to work as expected.
template <typename>
const char* TypenameAsString() {
  // Ideally the typename string would be computed at compile time, however this does not seem to
  // be possible. Thus the typename string is computed the first time this function is evaluated
  // and cached for future function evaluations.
  constexpr int32_t kMaxNameLength = sizeof(__PRETTY_FUNCTION__);  // Use upper bound to be safe.
  static char s_name[kMaxNameLength] = {0};  // Initialize with 0 to get a null-terminated string.
  static char* result = s_name;
  if (s_name[0] == 0 && result != nullptr) {  // Check for first invokation of this function.
    result = TypenameAsStringImpl(__PRETTY_FUNCTION__, s_name, kMaxNameLength);
  }
  return result;
}

}  // namespace nvidia

#endif  // NVIDIA_COMMON_TYPE_NAME_HPP_
