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
#ifndef NVIDIA_COMMON_TYPE_UTILS_HPP_
#define NVIDIA_COMMON_TYPE_UTILS_HPP_

#include <cstdint>

// GXF definitions for common <type_traits> class definitions. These should behave identically
// ISO standard definitions. They are defined locally to minimize dependencies of core GXF code
// on any stl implementations.
namespace nvidia {

// https://en.cppreference.com/w/cpp/types/void_t
template <class...> using void_t = void;

// https://en.cppreference.com/w/cpp/types/type_identity
template <class T> struct TypeIdentity { using type = T; };

template <class T> using TypeIdentity_t = typename TypeIdentity<T>::type;

// https://en.cppreference.com/w/cpp/types/integral_constant
template <class T, T v> struct IntegralConstant { static constexpr T value = v; };
template <bool B>       struct BoolConstant : IntegralConstant<bool, B> {};

struct TrueType  : BoolConstant<true>  {};
struct FalseType : BoolConstant<false> {};

// https://en.cppreference.com/w/cpp/types/conditional
template <bool B, class T, class F> struct Conditional              : TypeIdentity<T> {};
template<class T, class F>          struct Conditional<false, T, F> : TypeIdentity<F> {};

template <bool B, class T, class F> using Conditional_t = typename Conditional<B, T, F>::type;

// https://en.cppreference.com/w/cpp/types/conjunction
template <class...>
struct Conjunction : TrueType {};

template <class B, class... Bs>
struct Conjunction<B, Bs...> : Conditional<B::value, Conjunction<Bs...>, B>::type {};

template <class... Bs> constexpr bool Conjunction_v = Conjunction<Bs...>::value;

// https://en.cppreference.com/w/cpp/types/disjunction
template <class...>
struct Disjunction : FalseType {};

template <class B, class... Bs>
struct Disjunction<B, Bs...> : Conditional<B::value, B, Disjunction<Bs...>>::type {};

template <class... Bs> constexpr bool Disjunction_v = Disjunction<Bs...>::value;

// https://en.cppreference.com/w/cpp/types/negation
template <class B> struct Negation : BoolConstant<!static_cast<bool>(B::value)> {};

template <class B> constexpr bool Negation_v = Negation<B>::value;

// https://en.cppreference.com/w/cpp/types/remove_reference
template <class T> struct RemoveReference      : TypeIdentity<T> {};
template <class T> struct RemoveReference<T&>  : TypeIdentity<T> {};
template <class T> struct RemoveReference<T&&> : TypeIdentity<T> {};

template <class T> using RemoveReference_t = typename RemoveReference<T>::type;

// https://en.cppreference.com/w/cpp/types/remove_cv
template <class T> struct RemoveCV                   : TypeIdentity<T> {};
template <class T> struct RemoveCV<const T>          : TypeIdentity<T> {};
template <class T> struct RemoveCV<volatile T>       : TypeIdentity<T> {};
template <class T> struct RemoveCV<const volatile T> : TypeIdentity<T> {};

template <class T> using RemoveCV_t = typename RemoveCV<T>::type;

template <class T> struct RemoveConst                : TypeIdentity<T> {};
template <class T> struct RemoveConst<const T>       : TypeIdentity<T> {};

template <class T> using RemoveConst_t = typename RemoveConst<T>::type;

template <class T> struct RemoveVolatile             : TypeIdentity<T> {};
template <class T> struct RemoveVolatile<volatile T> : TypeIdentity<T> {};

template <class T> using RemoveVolatile_t = typename RemoveVolatile<T>::type;

// https://en.cppreference.com/w/cpp/types/remove_cvref
template <class T> struct RemoveCVRef : RemoveCV<typename RemoveReference<T>::type>{};

template <class T> using RemoveCVRef_t = typename RemoveCVRef<T>::type;

// https://en.cppreference.com/w/cpp/types/add_reference
template <class T>
struct AddLvalueReference {
 private:
  template <class U> static constexpr TypeIdentity<U&> Test(void*);
  template <class U> static constexpr TypeIdentity<U>  Test(...);
 public:
    using type = typename decltype(Test<T>(nullptr))::type;
};

template <class T>
struct AddRvalueReference {
 private:
  template <class U> static constexpr TypeIdentity<U&&> Test(void*);
  template <class U> static constexpr TypeIdentity<U>   Test(...);
 public:
    using type = typename decltype(Test<T>(nullptr))::type;
};

template <class T> using AddLvalueReference_t = typename AddLvalueReference<T>::type;
template <class T> using AddRvalueReference_t = typename AddRvalueReference<T>::type;

// https://en.cppreference.com/w/cpp/utility/declval
template <class T>
AddRvalueReference_t<T> Declval() noexcept {
  static_assert(Conditional_t<false, T, FalseType>::value,
      "Declval() cannot be used in an evaluated context.");
}

// https://en.cppreference.com/w/cpp/types/decay
template <class T>
struct Decay {
 private:
  template <class U> static constexpr auto Id(U u) noexcept { return u; }
  template <class U> static constexpr decltype(Id(Declval<U>())) Test(void*);
  template <class>   static constexpr void Test(...) {}
 public:
    using type = decltype(Test<T>(nullptr));
};

template <class T> using Decay_t = typename Decay<T>::type;

// https://en.cppreference.com/w/cpp/types/enable_if
template <bool, class = void> struct EnableIf {};
template <class T>            struct EnableIf<true, T> : TypeIdentity<T> {};

template <bool B, class T = void> using EnableIf_t = typename EnableIf<B, T>::type;

// https://en.cppreference.com/w/cpp/types/is_same
template <class, class> struct IsSame       : FalseType {};
template <class T>      struct IsSame<T, T> : TrueType  {};

template <class T, class U> constexpr bool IsSame_v = IsSame<T, U>::value;

// https://en.cppreference.com/w/cpp/types/is_void
template <class T> struct IsVoid : IsSame<void, typename RemoveCV<T>::type> {};

template <class T> constexpr bool IsVoid_v = IsVoid<T>::value;

// // https://en.cppreference.com/w/cpp/types/is_const
template <class T> struct IsConst          : FalseType {};
template <class T> struct IsConst<const T> : TrueType {};

template <class T> constexpr bool IsConst_v = IsConst<T>::value;

// https://en.cppreference.com/w/cpp/types/is_reference
template <class>   struct IsReference      : FalseType {};
template <class T> struct IsReference<T&>  : TrueType  {};
template <class T> struct IsReference<T&&> : TrueType  {};

template <class T> constexpr bool IsReference_v = IsReference<T>::value;

// https://en.cppreference.com/w/cpp/types/is_constructible
template <class T, class... Args>
struct IsConstructible {
 private:
  template<class V, class = decltype(static_cast<T>(Declval<V>()))>
  static constexpr bool TestCast(void*) { return true; }

  template<class...>
  static constexpr bool TestCast(...) { return false; }

  template <int, class, class = void>
  struct Test : FalseType {};

  template <class U>
  struct Test<1, U, void> : BoolConstant<TestCast<Args...>(nullptr)> {};

  template <int I, class U>
  struct Test<I, U, void_t<decltype(new U(Declval<Args>()...))>> : TrueType {};

 public:
  static constexpr bool value = Test<sizeof...(Args), T>::value;
};

template <class T, class... Args>
constexpr bool IsConstructible_v = IsConstructible<T, Args...>::value;

// https://en.cppreference.com/w/cpp/types/is_default_constructible
template <class T>
using IsDefaultConstructible = IsConstructible<T>;

template <class T>
constexpr bool IsDefaultConstructible_v = IsDefaultConstructible<T>::value;

// https://en.cppreference.com/w/cpp/types/is_nothrow_constructible
template <class T, class... Args>
struct IsNothrowConstructible {
 private:
  template<class V, bool Result = noexcept(static_cast<T>(Declval<V>()))>
  static constexpr bool TestCast(void*) { return Result; }

  template<class...>
  static constexpr bool TestCast(...) { return false; }

  template<int, class, class = void>
  struct Test : FalseType {};

  template<class U>
  struct Test<0, U, void_t<decltype(U())>> : BoolConstant<noexcept(U())> {};

  template<class U>
  struct Test<1, U, void> : BoolConstant<TestCast<Args...>(nullptr)> {};

  template<int I, class U>
  struct Test<I, U, void_t<decltype(new U(Declval<Args>()...))>>
      : BoolConstant<noexcept(new U(Declval<Args>()...))> {};

 public:
  static constexpr bool value = Test<sizeof...(Args), T>::value;
};

template <class T, class U>
constexpr bool IsNothrowConstructible_v = IsNothrowConstructible<T, U>::value;

// https://en.cppreference.com/w/cpp/types/is_nothrow_default_constructible
template <class T>
using IsNothrowDefaultConstructible = IsNothrowConstructible<T>;

template <class T>
constexpr bool IsNothrowDefaultConstructible_v = IsNothrowDefaultConstructible<T>::value;

// https://en.cppreference.com/w/cpp/types/is_convertible
template <class From, class To>
struct IsConvertible {
 private:
  template <class F, class T, class = decltype(Declval<T(&)(T)>()(Declval<F>()))>
  static constexpr bool Test(void*) { return true; }
  template <class, class>
  static constexpr bool Test(...) { return false; }
 public:
  static constexpr bool value = (IsVoid_v<From> && IsVoid_v<To>) || Test<From, To>(nullptr);
};

template <class From, class To>
constexpr bool IsConvertible_v = IsConvertible<From, To>::value;

// https://en.cppreference.com/w/cpp/types/is_convertible
template <class From, class To>
struct IsNothrowConvertible {
 private:
  template <class T>
  static constexpr T DecayFunc(T) noexcept;
  template <class F, class T, bool Result = noexcept(DecayFunc<T>(Declval<F>()))>
  static constexpr bool Test(void*) { return Result; }
  template <class, class>
  static constexpr bool Test(...) { return false; }
 public:
  static constexpr bool value = (IsVoid_v<From> && IsVoid_v<To>) || Test<From, To>(nullptr);
};

template <class From, class To>
constexpr bool IsNothrowConvertible_v = IsNothrowConvertible<From, To>::value;

// https://en.cppreference.com/w/cpp/types/is_assignable
template <class, class, class = void>
struct IsAssignable : FalseType {};

template <class T, class U>
struct IsAssignable<T, U, void_t<decltype(Declval<T>() = Declval<U>())>> : TrueType{};

template <class T, class U> constexpr bool IsAssignable_v = IsAssignable<T, U>::value;

// https://en.cppreference.com/w/cpp/types/is_integral
template <class T>
struct IsIntegral : BoolConstant<
    IsSame_v<bool,     typename RemoveCV<T>::type>  ||
    IsSame_v<int8_t,   typename RemoveCV<T>::type>  ||
    IsSame_v<uint8_t,  typename RemoveCV<T>::type>  ||
    IsSame_v<int16_t,  typename RemoveCV<T>::type>  ||
    IsSame_v<uint16_t, typename RemoveCV<T>::type>  ||
    IsSame_v<int32_t,  typename RemoveCV<T>::type>  ||
    IsSame_v<uint32_t, typename RemoveCV<T>::type>  ||
    IsSame_v<int64_t,  typename RemoveCV<T>::type>  ||
    IsSame_v<uint64_t, typename RemoveCV<T>::type>> {};

template <class T> constexpr bool IsIntegral_v = IsIntegral<T>::value;

// https://en.cppreference.com/w/cpp/types/is_floating_point
template <class T>
struct IsFloatingPoint : BoolConstant<
    IsSame_v<float,       typename RemoveCV<T>::type>  ||
    IsSame_v<double,      typename RemoveCV<T>::type>  ||
    IsSame_v<long double, typename RemoveCV<T>::type>> {};

template <class T> constexpr bool IsFloatingPoint_v = IsFloatingPoint<T>::value;

// https://en.cppreference.com/w/cpp/types/is_arithmetic
template <class T>
struct IsArithmetic : BoolConstant<IsIntegral_v<T> || IsFloatingPoint_v<T>> {};

template <class T> constexpr bool IsArithmetic_v = IsArithmetic<T>::value;

// https://en.cppreference.com/w/cpp/types/is_signed
template <class T, bool = IsArithmetic_v<T>> struct IsSigned : BoolConstant<(T(-1) < T(0))> {};
template <class T> struct IsSigned<T, false> : FalseType {};
template <class T> constexpr bool IsSigned_v = IsSigned<T>::value;

// https://en.cppreference.com/w/cpp/types/is_unsigned
template <class T, bool = IsArithmetic_v<T>> struct IsUnsigned : BoolConstant<(T(0) < T(-1))> {};
template <class T> struct IsUnsigned<T, false> : FalseType {};
template <class T> constexpr bool IsUnsigned_v = IsUnsigned<T>::value;

// https://en.cppreference.com/w/cpp/types/make_signed
template <class T, class = EnableIf_t<IsIntegral_v<T>>> struct MakeSigned : TypeIdentity<T> {};
template <> struct MakeSigned<uint8_t>  : TypeIdentity<int8_t>  {};
template <> struct MakeSigned<uint16_t> : TypeIdentity<int16_t> {};
template <> struct MakeSigned<uint32_t> : TypeIdentity<int32_t> {};
template <> struct MakeSigned<uint64_t> : TypeIdentity<int64_t> {};
template <class T> using MakeSigned_t = typename MakeSigned<T>::type;

// https://en.cppreference.com/w/cpp/types/make_unsigned
template <class T, class = EnableIf_t<IsIntegral_v<T>>> struct MakeUnsigned : TypeIdentity<T> {};
template <> struct MakeUnsigned<int8_t>  : TypeIdentity<uint8_t>  {};
template <> struct MakeUnsigned<int16_t> : TypeIdentity<uint16_t> {};
template <> struct MakeUnsigned<int32_t> : TypeIdentity<uint32_t> {};
template <> struct MakeUnsigned<int64_t> : TypeIdentity<uint64_t> {};
template <class T> using MakeUnsigned_t = typename MakeUnsigned<T>::type;

}  // namespace nvidia

#endif  // NVIDIA_COMMON_TYPE_UTILS_HPP_
