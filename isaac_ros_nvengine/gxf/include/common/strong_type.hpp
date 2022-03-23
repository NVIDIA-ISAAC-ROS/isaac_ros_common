/*
Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#ifndef NVIDIA_GXF_COMMON_STRONG_TYPE_HPP_
#define NVIDIA_GXF_COMMON_STRONG_TYPE_HPP_

#include <utility>

#include "common/type_utils.hpp"

namespace nvidia {

/// The base type for a
/// [strong type](https://www.fluentcpp.com/2016/12/08/strong-types-for-strong-interfaces).
/// It is useful for making lightweight types which are distinct. Unlike type aliases
/// (@c typedef or @c using), two @c StrongType types are not substitutable, even if the
/// backing type is the same.
///
/// The primary use case for this type is numeric identifiers. For example, imagine a sensor
/// system where the sensor is identified by a @c uint32_t. Beyond being a 32-bit integer, this
/// value shares almost no properties with the C type. Arithmetic and bit shifts do not make a
/// lot of sense for this value -- there is no meaning to the result of multiplication between
/// two sensor IDs. Comparing a sensor ID to any non-sensor ID value is probably a bug. By
/// wrapping ID values into a dedicated @c SensorId @c StrongType, this behavior is prevented.
///
/// @code
/// // Create two types: Foo and Bar, both wrapping a uint64_t
/// using Foo = StrongType<struct foo_t, uint64_t>;
/// using Bar = StrongType<struct bar_t, uint64_t>;
///
/// void fooOnly(const Foo&);
/// void barOnly(const Bar&);
///
/// int main() {
///   // Values must be constructed explicitly
///   auto foo = Foo(106);
///   auto bar = Bar(314);
///
///   fooOnly(foo);         // <- legal
///   fooOnly(bar);         // <- illegal: a Bar type is distinct from Foo
///   fooOnly(foo.value()); // <- illegal: no implicit construction
///   barOnly(foo);         // <- illegal: foo is not a Bar
/// }
/// @endcode
template <typename TName, typename TValue>
class StrongType {
 public:
  /// The underlying type of this type.
  using value_type = TValue;

  /// Default-construct an instance. This overload is only enabled if
  /// @ref value_type is default-constructible.
  constexpr StrongType() = default;

  /// Construct an instance from @a value.
  ///
  /// @param value The value to convert from.
  template <typename UValue>
  constexpr explicit
  StrongType(UValue&& value) noexcept(IsNothrowConstructible_v<value_type, UValue>)
      : value_(std::forward<UValue>(value)) {}

  constexpr explicit operator value_type() const& noexcept { return value_; }
  constexpr explicit operator value_type() &      noexcept { return value_; }
  constexpr explicit operator value_type() &&     noexcept { return value_; }

  constexpr value_type const& value() const &    noexcept { return value_; }
  constexpr value_type&       value() &          noexcept { return value_; }
  constexpr value_type&&      value() &&         noexcept { return std::move(value_); }
  constexpr value_type const* operator->() const noexcept { return &value_; }
  constexpr value_type*       operator->()       noexcept { return &value_; }

  /// Comparison operatorions for the underlying type.
  friend constexpr bool operator==(const StrongType& lhs, const StrongType& rhs) {
    return lhs.value() == rhs.value();
  }
  friend constexpr bool operator!=(const StrongType& lhs, const StrongType& rhs) {
    return lhs.value() != rhs.value();
  }
  friend constexpr bool operator<=(const StrongType& lhs, const StrongType& rhs) {
    return lhs.value() <= rhs.value();
  }
  friend constexpr bool operator>=(const StrongType& lhs, const StrongType& rhs) {
    return lhs.value() >= rhs.value();
  }
  friend constexpr bool operator<(const StrongType& lhs, const StrongType& rhs) {
    return lhs.value() < rhs.value();
  }
  friend constexpr bool operator>(const StrongType& lhs, const StrongType& rhs) {
    return lhs.value() > rhs.value();
  }

 private:
  /// The underlying value of this type.
  value_type value_;
};

}  // namespace nvidia

#endif  // NVIDIA_GXF_COMMON_STRONG_TYPE_HPP_
