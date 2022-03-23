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
#ifndef NVIDIA_GXF_CORE_EXPECTED_HPP
#define NVIDIA_GXF_CORE_EXPECTED_HPP

#include <type_traits>
#include <utility>

#include "common/expected.hpp"
#include "gxf/core/gxf.h"

namespace nvidia {
namespace gxf {

// Expected type for GXF which uses gxf_result_t as error code.
template <typename T>
using Expected = Expected<T, gxf_result_t>;

// Unexpected type for GXF which uses gxf_result_t as error code.
using Unexpected = Unexpected<gxf_result_t>;

// Special value which can be used instead of {} to return from a function which returns an
// Expected<void>.
const Expected<void> Success{};

// Extracts the error code as an unexpected.
template <typename T>
Unexpected ForwardError(const Expected<T>& expected) {
  return Unexpected{expected.error()};
}

// Extracts the error code as an unexpected.
template <typename T>
Unexpected ForwardError(Expected<T>&& expected) {
  return Unexpected{std::move(expected.error())};
}

// Interprets an expected as a result code. Returns GXF_SUCESS if the result has a value and the
// result's error code otherwise.
template <typename T>
gxf_result_t ToResultCode(const Expected<T>& result) {
  return result ? GXF_SUCCESS : result.error();
}

// If the result code is GXF_SUCCESS the function returns Success, otherwise it returns an
// unexpected with the given error code.
inline
Expected<void> ExpectedOrCode(gxf_result_t code) {
  if (code == GXF_SUCCESS) {
    return Success;
  } else {
    return Unexpected{code};
  }
}

// If the result code is GXF_SUCCESS the function returns the given value, otherwise it returns an
// unexpected with the given error code.
template <typename T>
Expected<std::remove_cv_t<std::remove_reference_t<T>>>
ExpectedOrCode(gxf_result_t code, T&& value) {
  if (code == GXF_SUCCESS) {
    return std::forward<T>(value);
  } else {
    return Unexpected{code};
  }
}

template <typename S, typename T>
Expected<std::remove_cv_t<std::remove_reference_t<T>>>
ExpectedOrError(const Expected<S>& code, T&& value) {
  if (code) {
    return std::forward<T>(value);
  } else {
    return ForwardError(code);
  }
}

inline
gxf_result_t AccumulateError(gxf_result_t previous, gxf_result_t current) {
  return current != GXF_SUCCESS ? current : previous;
}

inline
Expected<void> AccumulateError(Expected<void> previous, Expected<void> current) {
  return !current ? current : previous;
}

}  // namespace gxf
}  // namespace nvidia

#endif
