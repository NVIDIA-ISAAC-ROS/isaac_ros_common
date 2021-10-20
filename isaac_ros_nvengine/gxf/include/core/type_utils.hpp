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

#ifndef NVIDIA_CORE_TYPE_UTILS_HPP_
#define NVIDIA_CORE_TYPE_UTILS_HPP_

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
template <class T> struct AddLvalueReference       : TypeIdentity<T&>   {};
template <class T> struct AddLvalueReference<T&>   : TypeIdentity<T&>   {};
template <class T> struct AddLvalueReference<T&&>  : TypeIdentity<T&>   {};
template <>        struct AddLvalueReference<void> : TypeIdentity<void> {};

template <class T> struct AddRvalueReference       : TypeIdentity<T&&>  {};
template <class T> struct AddRvalueReference<T&>   : TypeIdentity<T&>   {};
template <class T> struct AddRvalueReference<T&&>  : TypeIdentity<T&&>  {};
template <>        struct AddRvalueReference<void> : TypeIdentity<void> {};

template <class T> using AddLvalueReference_t = typename AddLvalueReference<T>::type;
template <class T> using AddRvalueReference_t = typename AddRvalueReference<T>::type;

// https://en.cppreference.com/w/cpp/utility/declval
template <class T>
AddRvalueReference_t<T> Declval() {
  static_assert([]() { return false; } (), "Declval() cannot be used in an evaluated context.");
}

// https://en.cppreference.com/w/cpp/types/enable_if
template <bool, class = void> struct EnableIf {};
template <class T>            struct EnableIf<true, T> : TypeIdentity<T> {};

template <bool B, class T = void> using EnableIf_t = typename EnableIf<B, T>::type;

// https://en.cppreference.com/w/cpp/types/is_same
template <class, class> struct IsSame       : FalseType {};
template <class T>      struct IsSame<T, T> : TrueType  {};

template <class T, class U> constexpr bool IsSame_v = IsSame<T, U>::value;

// https://en.cppreference.com/w/cpp/types/is_reference
template <class>   struct IsReference      : FalseType {};
template <class T> struct IsReference<T&>  : TrueType  {};
template <class T> struct IsReference<T&&> : TrueType  {};

template <class T> constexpr bool IsReference_v = IsReference<T>::value;

// https://en.cppreference.com/w/cpp/types/is_constructible
template <class, class, class = void>
struct IsConstructible : FalseType {};

template <class T, class U>
struct IsConstructible<T, U, void_t<decltype(T(Declval<U>()))>> : TrueType {};

template <class T, class U> constexpr bool IsConstructible_v = IsConstructible<T, U>::value;

// https://en.cppreference.com/w/cpp/types/is_assignable
template <class, class, class = void>
struct IsAssignable : FalseType {};

template <class T, class U>
struct IsAssignable<T, U, void_t<decltype(Declval<T>() = Declval<U>())>> : TrueType{};

template <class T, class U> constexpr bool IsAssignable_v = IsAssignable<T, U>::value;

// https://en.cppreference.com/w/cpp/types/is_void
template <class T> struct IsVoid : IsSame<void, typename RemoveCV<T>::type> {};

template <class T> constexpr bool IsVoid_v = IsVoid<T>::value;
}  // namespace nvidia

#endif  // NVIDIA_CORE_TYPE_UTILS_HPP_
