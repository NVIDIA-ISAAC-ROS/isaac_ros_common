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
#ifndef NVIDIA_COMMON_EXPECTED_HPP_
#define NVIDIA_COMMON_EXPECTED_HPP_

#include <utility>

#include "common/assert.hpp"
#include "common/memory_utils.hpp"
#include "common/strong_type.hpp"
#include "common/type_utils.hpp"

namespace nvidia {

template <class E> using Unexpected = StrongType<struct Unexpected_t, E>;
template <class, class> class Expected;

///-------------------------------------------------------------------------------------------------
///  Local Helper Functions
///-------------------------------------------------------------------------------------------------
namespace detail {

template <class>
struct IsExpectedHelper : FalseType {};

template <class T, class E>
struct IsExpectedHelper<Expected<T, E>> : TrueType {};

template <class T>
using IsExpected = IsExpectedHelper<RemoveCVRef_t<T>>;

template <class T>
constexpr bool IsExpected_v = IsExpected<T>::value;

// Extracts the error type from a pack of Expecteds if they all contain the same Error .
template <class...>
struct ErrorTypeHelper : TypeIdentity<void> {};

template <class... Ts, class E>
struct ErrorTypeHelper<Expected<Ts, E>...> : TypeIdentity<E> {};

template <class... Ts>
using ErrorType = typename ErrorTypeHelper<RemoveCVRef_t<Ts>...>::type;

template <class E, class... Ts>
using ErrorTypeOr = Conditional_t<sizeof...(Ts) == 0, E, ErrorType<Ts...>>;

// Extracts the error type from a pack of Expecteds if they all contain the same Error .
template <class...>
struct ValueTypeHelper {};

template <class T, class E>
struct ValueTypeHelper<Expected<T, E>> { using type = T; };

template <class T>
using ValueType = typename ValueTypeHelper<RemoveCVRef_t<T>>::type;

// Extracts the type from a Unexpected if provided. Otherwise, returns the type T.
template <class T>
struct UnexpectedTypeHelper : TypeIdentity<T> {};

template <class T>
struct UnexpectedTypeHelper<Unexpected<T>> : TypeIdentity<T> {};

template <class T>
using UnexpectedType = typename UnexpectedTypeHelper<RemoveCVRef_t<T>>::type;

// Finds the result type of invoking a function with the wrapped types of a pack of Expecteds.
template <class F, class... Es>
using ResultType = decltype(Declval<F>()(Declval<Es>().value()...));

template <class F, class... Es>
using ErrorResultType = UnexpectedType<decltype(Declval<F>()(Declval<Es>().error()...))>;

// Empty stuct for representing an object with only one possible value (itself). This can be used
// for speciallizing on void types where void would not compile because it is not a valid type.
struct Unit {};

// If T is already Expected, keep as Expected, otherwise wrap as Expected<T, E>
template <class T, class E>
struct FlattenExpectedHelper : TypeIdentity<Expected<T, E>> {};

template <class T, class E>
struct FlattenExpectedHelper<Expected<T, E>, E> : TypeIdentity<Expected<T, E>> {};

template <class T, class E>
struct FlattenExpectedHelper<Expected<T, E>&, E>
: TypeIdentity<Expected<AddLvalueReference_t<T>, E>> {};

template <class T, class E>
struct FlattenExpectedHelper<const Expected<T, E>&, E>
: TypeIdentity<Expected<const AddLvalueReference_t<T>, E>> {};

template <class T, class E>
using FlattenExpected = typename FlattenExpectedHelper<T, E>::type;

// Helper for specializing on variadic parameter packs
template <class...> struct Pack {};

// Checks if a functor is callable with an Expected value.
template <class, class, class = void>
struct IsCallableHelper : FalseType {};

template <class F, class... Es>
struct IsCallableHelper<F, Pack<Es...>, void_t<ResultType<F, Es...>>> : TrueType {};

template <class F, class... Es>
struct IsCallable : IsCallableHelper<F, Pack<Es...>> {};

template <class F, class... Es>
constexpr bool IsCallable_v = IsCallable<F, Es...>::value;

// Checks if a functor is callable with an Expected value.
template <class, class, class = void>
struct IsErrorCallableHelper : FalseType {};

template <class F, class... Es>
struct IsErrorCallableHelper<F, Pack<Es...>, void_t<ErrorResultType<F, Es...>>> : TrueType {};

template <class F, class... Es>
struct IsErrorCallable : IsErrorCallableHelper<F, Pack<Es...>> {};

template <class F, class... Es>
constexpr bool IsErrorCallable_v = IsErrorCallable<F, Es...>::value;

// Maps the values with the given functor. If F returns an Expected, it returns the Expected type
// directly, otherwise it constructs a new Expected.
template <class E = void, class F, class... Args>
auto FunctorMap(F&& func, Args&&... expected) ->
    EnableIf_t<!IsVoid_v<ResultType<F, Args...>>,
               FlattenExpected<ResultType<F, Args...>, ErrorTypeOr<E, Args...>>> {
  return std::forward<F>(func)(std::forward<Args>(expected).value()...);
}

// Maps the values with the given void functor. Returns Expected<void, Es...>.
template <class E = void, class F, class... Args>
auto FunctorMap(F&& func, Args&&... expected) ->
    EnableIf_t<IsVoid_v<ResultType<F, Args...>>, Expected<void, ErrorTypeOr<E, Args...>>> {
  std::forward<F>(func)(std::forward<Args>(expected).value()...);
  return {};
}

// Maps the error values with the given function. If F returns an Unexpected, it returns the
// Unexpected type directly, otherwise it constructs a new Unexpected.
template <class F, class E>
Expected<ValueType<E>, ErrorResultType<F, E>> FunctorMapError(F&& func, E&& expected) {
  using R = Unexpected<ErrorResultType<F, E>>;
  return R{std::forward<F>(func)(std::forward<E>(expected).error())};
}

template <class T, class F>
Expected<T, ErrorResultType<F>> FunctorMapError(F&& func) {
  using R = Unexpected<ErrorResultType<F>>;
  return R{std::forward<F>(func)()};
}

// Base class for all operations of Expected, not directly related to setting or creating the value
// type T into the underlying byte buffer.
template <class T, class E, class Derived>
class ExpectedBase {
 private:
  static_assert(!IsVoid_v<T>, "ExpectedBase cannot wrap void types");
  static_assert(!IsVoid_v<E>, "ExpectedBase cannot wrap void types");
  static_assert(!IsReference_v<T>, "ExpectedBase cannot wrap reference types");
  static_assert(!IsReference_v<E>, "ExpectedBase cannot wrap reference types");

  template <class F, class U>
  using EnableIfMappable_t =
      EnableIf_t<IsCallable_v<F, U>, decltype(FunctorMap(Declval<F>(), Declval<U>()))>;

  template <class F, class U>
  using EnableIfNonaryMappable_t =
      EnableIf_t<IsCallable_v<F> && !IsCallable_v<F, U>, FlattenExpected<ResultType<F>, E>>;

  template <class F, class U>
  using EnableIfMappableError_t =
      EnableIf_t<IsErrorCallable_v<F, U>, decltype(FunctorMapError(Declval<F>(), Declval<U>()))>;

  template <class F, class U>
  using EnableIfNonaryMappableError_t =
      EnableIf_t<IsErrorCallable_v<F> && !IsErrorCallable_v<F, U>,
                 Expected<T, ErrorResultType<F>>>;

 public:
  ///-----------------------------------------------------------------------------------------------
  /// Constructors
  ///-----------------------------------------------------------------------------------------------
  constexpr ExpectedBase(const ExpectedBase& other) { constructFrom(other); }
  constexpr ExpectedBase(ExpectedBase&& other) { constructFrom(std::move(other)); }

  // Construction from convertible error types
  template <class G, class C>
  explicit constexpr ExpectedBase(const ExpectedBase<T, G, C>& other) {
    static_assert(IsConvertible_v<G, E>,
        "Cannot construct Expected from type with unconvertible error type.");
    constructFrom(other);
  }

  template <class G, class C>
  explicit constexpr ExpectedBase(ExpectedBase<T, G, C>&& other) {
    static_assert(IsConvertible_v<G, E>,
        "Cannot construct Expected from type with unconvertible error type.");
    constructFrom(std::move(other));
  }

  template <class G>
  constexpr ExpectedBase(const Unexpected<G>& error) : is_error_{true} {
    static_assert(IsConvertible_v<G, E>,
        "Cannot construct Unexpected from type with unconvertible error type.");
    InplaceConstruct<Unexpected<E>>(buffer_, error.value());
  }

  template <class G>
  constexpr ExpectedBase(Unexpected<G>&& error) : is_error_{true} {
    static_assert(IsConvertible_v<G, E>,
        "Cannot construct Unexpected from type with unconvertible error type.");
    InplaceConstruct<Unexpected<E>>(buffer_, std::move(error.value()));
  }

  ///-----------------------------------------------------------------------------------------------
  /// Destructor
  ///-----------------------------------------------------------------------------------------------
  ~ExpectedBase() { destruct(); }

  ///-----------------------------------------------------------------------------------------------
  /// Assignment
  ///-----------------------------------------------------------------------------------------------
  ExpectedBase& operator=(const ExpectedBase& other) {
    destruct();
    constructFrom(other);
    return *this;
  }

  ExpectedBase& operator=(ExpectedBase&& other) {
    destruct();
    constructFrom(std::move(other));
    return *this;
  }

  template <class G>
  ExpectedBase& operator=(const Unexpected<G>& error) {
    destruct();
    is_error_ = true;
    InplaceConstruct<Unexpected<E>>(buffer_, error);
    return *this;
  }

  template <class G>
  ExpectedBase& operator=(Unexpected<G>&& error) {
    destruct();
    is_error_ = true;
    InplaceConstruct<Unexpected<E>>(buffer_, std::move(error));
    return *this;
  }

  ///-----------------------------------------------------------------------------------------------
  /// Observers
  ///-----------------------------------------------------------------------------------------------
  template <class V = ValueType<Derived>, class D = const Derived>
  constexpr EnableIf_t<!IsVoid_v<V>, RemoveReference_t<decltype(Declval<D*>()->value())>*>
  operator->() const  { return &(derived()->value()); }

  template <class V = ValueType<Derived>, class D = Derived>
  constexpr EnableIf_t<!IsVoid_v<V>, RemoveReference_t<decltype(Declval<D*>()->value())>*>
  operator->()        { return &(derived()->value()); }

  template <class V = ValueType<Derived>>
  constexpr EnableIf_t<!IsVoid_v<V>, const V&>
  operator*() const&  { return derived()->value(); }

  template <class V = ValueType<Derived>>
  constexpr EnableIf_t<!IsVoid_v<V>, V&>
  operator*() &       { return derived()->value(); }

  template <class V = ValueType<Derived>, EnableIf_t<!IsVoid_v<V>, void*> = nullptr>
  constexpr decltype(auto) operator*() const&& { return std::move(derived())->value(); }

  template <class V = ValueType<Derived>, EnableIf_t<!IsVoid_v<V>, void*> = nullptr>
  constexpr decltype(auto) operator*() && { return std::move(derived())->value(); }

  template <class U, class V = ValueType<Derived>, EnableIf_t<!IsVoid_v<V>, void*> = nullptr>
  constexpr V value_or(U&& default_value) const& {
    return has_value() ? derived()->value() : std::forward<U>(default_value);
  }

  template <class U, class V = ValueType<Derived>, EnableIf_t<!IsVoid_v<V>, void*> = nullptr>
  constexpr V value_or(U&& default_value) && {
    return has_value() ? std::move(derived())->value() : std::forward<U>(default_value);
  }

  template <class U, class V = ValueType<Derived>, EnableIf_t<!IsVoid_v<V>, void*> = nullptr>
  constexpr V value_or(U&& default_value) const&& {
    return has_value() ? std::move(derived()->value()) : std::forward<U>(default_value);
  }

  constexpr          bool     has_value() const { return !is_error_; }
  constexpr explicit operator bool()      const { return !is_error_; }

  constexpr const E&  error() const& { return unexpected().value(); }
  constexpr       E&  error() &      { return unexpected().value(); }
  constexpr       E&& error() &&     { return std::move(unexpected()).value(); }

  ///-----------------------------------------------------------------------------------------------
  /// Logging
  ///-----------------------------------------------------------------------------------------------
  // Logging helper functions if this contains an error value.
  template <class... Args>
  constexpr const Derived& log_error(Args&&... args) const& {
    if (is_error_) { GXF_LOG_ERROR(std::forward<Args>(args)...); }
    return static_cast<const Derived&>(*this);
  }
  template <class... Args>
  constexpr Derived& log_error(Args&&... args) & {
    if (is_error_) { GXF_LOG_ERROR(std::forward<Args>(args)...); }
    return static_cast<Derived&>(*this);
  }
  template <class... Args>
  constexpr Derived&& log_error(Args&&... args) const&& {
    if (is_error_) { GXF_LOG_ERROR(std::forward<Args>(args)...); }
    return static_cast<const Derived&&>(std::move(*this));
  }
  template <class... Args>
  constexpr Derived&& log_error(Args&&... args) && {
    if (is_error_) { GXF_LOG_ERROR(std::forward<Args>(args)...); }
    return static_cast<Derived&&>(std::move(*this));
  }

  template <class... Args>
  constexpr const Derived& log_warning(Args&&... args) const& {
    if (is_error_) { GXF_LOG_WARNING(std::forward<Args>(args)...); }
    return static_cast<const Derived&>(*this);
  }
  template <class... Args>
  constexpr Derived& log_warning(Args&&... args) & {
    if (is_error_) { GXF_LOG_WARNING(std::forward<Args>(args)...); }
    return static_cast<Derived&>(*this);
  }
  template <class... Args>
  constexpr Derived&& log_warning(Args&&... args) const&& {
    if (is_error_) { GXF_LOG_WARNING(std::forward<Args>(args)...); }
    return static_cast<const Derived&&>(std::move(*this));
  }
  template <class... Args>
  constexpr Derived&& log_warning(Args&&... args) && {
    if (is_error_) { GXF_LOG_WARNING(std::forward<Args>(args)...); }
    return static_cast<Derived&&>(std::move(*this));
  }

  template <class... Args>
  constexpr const Derived& log_info(Args&&... args) const& {
    if (is_error_) { GXF_LOG_INFO(std::forward<Args>(args)...); }
    return static_cast<const Derived&>(*this);
  }
  template <class... Args>
  constexpr Derived& log_info(Args&&... args) & {
    if (is_error_) { GXF_LOG_INFO(std::forward<Args>(args)...); }
    return static_cast<Derived&>(*this);
  }
  template <class... Args>
  constexpr Derived&& log_info(Args&&... args) const&& {
    if (is_error_) { GXF_LOG_INFO(std::forward<Args>(args)...); }
    return static_cast<const Derived&&>(std::move(*this));
  }
  template <class... Args>
  constexpr Derived&& log_info(Args&&... args) && {
    if (is_error_) { GXF_LOG_INFO(std::forward<Args>(args)...); }
    return static_cast<Derived&&>(std::move(*this));
  }

  template <class... Args>
  constexpr const Derived& log_debug(Args&&... args) const& {
    if (is_error_) { GXF_LOG_DEBUG(std::forward<Args>(args)...); }
    return static_cast<const Derived&>(*this);
  }
  template <class... Args>
  constexpr Derived& log_debug(Args&&... args) & {
    if (is_error_) { GXF_LOG_DEBUG(std::forward<Args>(args)...); }
    return static_cast<Derived&>(*this);
  }
  template <class... Args>
  constexpr Derived&& log_debug(Args&&... args) const&& {
    if (is_error_) { GXF_LOG_DEBUG(std::forward<Args>(args)...); }
    return static_cast<const Derived&&>(std::move(*this));
  }
  template <class... Args>
  constexpr Derived&& log_debug(Args&&... args) && {
    if (is_error_) { GXF_LOG_DEBUG(std::forward<Args>(args)...); }
    return static_cast<Derived&&>(std::move(*this));
  }

  ///-----------------------------------------------------------------------------------------------
  /// Substitute
  ///-----------------------------------------------------------------------------------------------
  // When this expected does not contain an error an expected with the given value 'next' is
  // created and returned, otherwise an unexpected with the error is returned.
  template <class U>
  constexpr FlattenExpected<U, E> substitute(U next) const& {
    return has_value() ? FlattenExpected<U, E>{std::forward<U>(next)} : unexpected();
  }

  template <class U>
  constexpr FlattenExpected<U, E> substitute(U next) && {
    return has_value() ? FlattenExpected<U, E>{std::forward<U>(next)} : std::move(unexpected());
  }

  template <class U>
  constexpr FlattenExpected<U, E> substitute(U next) const&& {
    return has_value() ? FlattenExpected<U, E>{std::forward<U>(next)} : std::move(unexpected());
  }

  ///-----------------------------------------------------------------------------------------------
  /// Substitute Error
  ///-----------------------------------------------------------------------------------------------
  template <class G>
  constexpr Expected<ValueType<Derived>, G> substitute_error(G new_error) const& {
    using R = Expected<ValueType<Derived>, G>;
    return is_error_ ? R{Unexpected<G>{new_error}} : R{*ValuePointer<T>(buffer_)};
  }

  template <class G>
  constexpr Expected<ValueType<Derived>, G> substitute_error(G new_error) && {
    using R = Expected<ValueType<Derived>, G>;
    return is_error_ ? R{Unexpected<G>{new_error}} : R{std::move(*ValuePointer<T>(buffer_))};
  }

  template <class G>
  constexpr Expected<ValueType<Derived>, G> substitute_error(G new_error) const&& {
    using R = Expected<ValueType<Derived>, G>;
    return is_error_ ? R{Unexpected<G>{new_error}} : R{std::move(*ValuePointer<T>(buffer_))};
  }

  ///-----------------------------------------------------------------------------------------------
  /// Unary Free Functor Map
  ///-----------------------------------------------------------------------------------------------
  // If in error the error is returned, otherwise the value is mapped with the given functor and
  // the result is returned.
  template <class F>
  constexpr EnableIfMappable_t<F, Derived&> map(F&& func) & {
    return has_value() ? FunctorMap(std::forward<F>(func), *derived()) : unexpected();
  }

  template <class F>
  constexpr EnableIfMappable_t<F, const Derived&> map(F&& func) const& {
    return has_value() ? FunctorMap(std::forward<F>(func), *derived()) : unexpected();
  }

  template <class F>
  constexpr EnableIfMappable_t<F, Derived&&> map(F&& func) && {
    return has_value() ? FunctorMap(std::forward<F>(func), std::move(*derived()))
                       : std::move(unexpected());
  }

  template <class F>
  constexpr EnableIfMappable_t<F, const Derived&&> map(F&& func) const&& {
    return has_value() ? FunctorMap(std::forward<F>(func), std::move(*derived()))
                       : std::move(unexpected());
  }

  ///-----------------------------------------------------------------------------------------------
  /// Unary Free Functor Map Error
  ///-----------------------------------------------------------------------------------------------
  // If has value the value is returned, otherwise the error is mapped with the given functor and
  // the result is returned.
  template <class F>
  constexpr EnableIfMappableError_t<F, Derived&> map_error(F&& func) & {
    using R = Expected<ValueType<Derived>, ErrorResultType<F, Derived&>>;
    return is_error_ ? FunctorMapError(std::forward<F>(func), *derived())
                     : R{*ValuePointer<T>(buffer_)};
  }

  template <class F>
  constexpr EnableIfMappableError_t<F, const Derived&> map_error(F&& func) const& {
    using R = Expected<ValueType<Derived>, ErrorResultType<F, const Derived&>>;
    return is_error_ ? FunctorMapError(std::forward<F>(func), *derived())
                     : R{*ValuePointer<T>(buffer_)};
  }

  template <class F>
  constexpr EnableIfMappableError_t<F, Derived&&> map_error(F&& func) && {
    using R = Expected<ValueType<Derived>, ErrorResultType<F, Derived&&>>;
    return is_error_ ? FunctorMapError(std::forward<F>(func), std::move(*derived()))
                     : R{std::move(*ValuePointer<T>(buffer_))};
  }

  template <class F>
  constexpr EnableIfMappableError_t<F, const Derived&&> map_error(F&& func) const&& {
    using R = Expected<ValueType<Derived>, ErrorResultType<F, const Derived&&>>;
    return is_error_ ? FunctorMapError(std::forward<F>(func), std::move(*derived()))
                     : R{std::move(*ValuePointer<T>(buffer_))};
  }

  ///-----------------------------------------------------------------------------------------------
  /// Nonary Free Functor Map
  ///-----------------------------------------------------------------------------------------------
  // If in error the error is returned, otherwise the given functor is called with no arguments and
  // the result is returned.
  template <class F>
  constexpr EnableIfNonaryMappable_t<F, const Derived&> map(F&& func) const& {
    return has_value() ? FunctorMap<E>(std::forward<F>(func)) : unexpected();
  }

  template <class F>
  constexpr EnableIfNonaryMappable_t<F, const Derived&&> map(F&& func) const&& {
    return has_value() ? FunctorMap<E>(std::forward<F>(func)) : std::move(unexpected());
  }

  template <class F>
  constexpr FlattenExpected<ResultType<F>, E> and_then(F&& func) const& {
    return has_value() ? FunctorMap<E>(std::forward<F>(func)) : unexpected();
  }
  template <class F>
  constexpr FlattenExpected<ResultType<F>, E> and_then(F&& func) const&& {
    return has_value() ? FunctorMap<E>(std::forward<F>(func)) : std::move(unexpected());
  }

  ///-----------------------------------------------------------------------------------------------
  /// Nonary Free Functor Map Error
  ///-----------------------------------------------------------------------------------------------
  // If has value the value is returned, otherwise the given functor is called with no arguments and
  // the result is returned.
  template <class F>
  constexpr EnableIfNonaryMappableError_t<F, const Derived&> map_error(F&& func) const& {
    static_assert(IsErrorCallable_v<F>);
    static_assert(!IsErrorCallable_v<F, const Derived&>);
    using R = Expected<ValueType<Derived>, ErrorResultType<F>>;
    return is_error_ ? FunctorMapError<T>(std::forward<F>(func)) : R{*ValuePointer<T>(buffer_)};
  }

  template <class F>
  constexpr EnableIfNonaryMappableError_t<F, const Derived&&> map_error(F&& func) const&& {
    using R = Expected<ValueType<Derived>, ErrorResultType<F>>;
    return is_error_ ? FunctorMapError<T>(std::forward<F>(func))
                     : R{std::move(*ValuePointer<T>(buffer_))};
  }

  template <class F>
  constexpr Expected<ValueType<Derived>, ErrorResultType<F>> and_then_error(F&& func) const& {
    using R = Expected<ValueType<Derived>, ErrorResultType<F>>;
    return is_error_ ? FunctorMapError<T>(std::forward<F>(func)) : R{*ValuePointer<T>(buffer_)};
  }

  template <class F>
  constexpr Expected<ValueType<Derived>, ErrorResultType<F>> and_then_error(F&& func) const&& {
    using R = Expected<ValueType<Derived>, ErrorResultType<F>>;
    return is_error_ ? FunctorMapError<T>(std::forward<F>(func))
                     : R{std::move(*ValuePointer<T>(buffer_))};
  }

  ///-----------------------------------------------------------------------------------------------
  /// Member Variable Map
  ///-----------------------------------------------------------------------------------------------
  template <class R, class U>
  constexpr FlattenExpected<R&, E> map(R U::* ptr) & {
    static_assert(IsSame_v<U, RemoveCVRef_t<ValueType<Derived>>>,
        "received pointer-to-member that is not of type T.");
    // NOTE: We explicitly declare the return type of the lambda to allow capturing of member by
    // mutable reference, otherwise a lambda will always return a copy.
    auto func = [&](ValueType<Derived>& a) -> decltype(auto) { return a.*ptr; };
    return has_value() ? FunctorMap(func, *derived()) : unexpected();
  }

  template <class R, class U>
  constexpr FlattenExpected<const R&, E> map(R U::* ptr) const& {
    static_assert(IsSame_v<U, RemoveCVRef_t<ValueType<Derived>>>,
        "received pointer-to-member that is not of type T.");
    auto func = [&](const ValueType<Derived>& a) -> decltype(auto) { return a.*ptr; };
    return has_value() ? FunctorMap(func, *derived()) : unexpected();
  }

  template <class R, class U>
  constexpr FlattenExpected<Decay_t<R>, E> map(R U::* ptr) && {
    static_assert(IsSame_v<U, RemoveCVRef_t<ValueType<Derived>>>,
        "received pointer-to-member that is not of type T.");
    // NOTE: there is a subtle distinction here. If we want to get access to a member variable
    // from an rvalue context, we MUST make a copy, since we cannot guarrantee the reference will
    // still be valid after destruction of the temporary.
    auto func = [&](ValueType<Derived>&& a) { return a.*ptr; };
    return has_value() ? FunctorMap(func, std::move(*derived())) : unexpected();
  }

  template <class R, class U>
  constexpr FlattenExpected<Decay_t<R>, E> map(R U::* ptr) const&& {
    static_assert(IsSame_v<U, RemoveCVRef_t<ValueType<Derived>>>,
        "received pointer-to-member that is not of type T.");
    auto func = [&](const ValueType<Derived>&& a) { return a.*ptr; };
    return has_value() ? FunctorMap(func, std::move(*derived())) : unexpected();
  }

  ///-----------------------------------------------------------------------------------------------
  /// Member Function Map
  ///-----------------------------------------------------------------------------------------------
  template <class R, class U, class... Args>
  constexpr FlattenExpected<R, E> map(R (U::* ptr)(Args...), Args&&... args) {
    static_assert(IsSame_v<U, RemoveCVRef_t<ValueType<Derived>>>,
        "received pointer-to-member that is not of type T.");
    auto func = [&](U& a) { return (a.*ptr)(std::forward<Args>(args)...); };
    return has_value() ? FunctorMap(func, *derived()) : unexpected();
  }

  template <class R, class U, class... Args>
  constexpr FlattenExpected<R, E> map(R (U::* ptr)(Args...) & , Args&&... args) & {
    static_assert(IsSame_v<U, RemoveCVRef_t<ValueType<Derived>>>,
        "received pointer-to-member that is not of type T.");
    auto func = [&](U& a) { return (a.*ptr)(std::forward<Args>(args)...); };
    return has_value() ? FunctorMap(func, *derived()) : unexpected();
  }

  template <class R, class U, class... Args>
  constexpr FlattenExpected<R, E> map(R (U::* ptr)(Args...) &&, Args&&... args) && {
    static_assert(IsSame_v<U, RemoveCVRef_t<ValueType<Derived>>>,
        "received pointer-to-member that is not of type T.");
    auto func = [&](U&& a) { return (std::move(a).*ptr)(std::forward<Args>(args)...); };
    return has_value() ? FunctorMap(func, std::move(*derived())) : unexpected();
  }

  template <class R, class U, class... Args>
  constexpr FlattenExpected<R, E> map(R (U::* ptr)(Args...) const, Args&&... args) const {
    static_assert(IsSame_v<U, RemoveCVRef_t<ValueType<Derived>>>,
        "received pointer-to-member that is not of type T.");
    auto func = [&](const U& a) { return (a.*ptr)(std::forward<Args>(args)...); };
    return has_value() ? FunctorMap(func, *derived()) : unexpected();
  }

  template <class R, class U, class... Args>
  constexpr FlattenExpected<R, E> map(R (U::* ptr)(Args...) const&, Args&&... args) const& {
    static_assert(IsSame_v<U, RemoveCVRef_t<ValueType<Derived>>>,
        "received pointer-to-member that is not of type T.");
    auto func = [&](const U& a) { return (a.*ptr)(std::forward<Args>(args)...); };
    return has_value() ? FunctorMap(func, *derived()) : unexpected();
  }

  template <class R, class U, class... Args>
  constexpr FlattenExpected<R, E> map(R (U::* ptr)(Args...) const&&, Args&&... args) const&& {
    static_assert(IsSame_v<U, RemoveCVRef_t<ValueType<Derived>>>,
        "received pointer-to-member that is not of type T.");
    auto func = [&](const U&& a) { return (std::move(a).*ptr)(std::forward<Args>(args)...); };
    return has_value() ? FunctorMap(func, std::move(*derived())) : unexpected();
  }

  ///-----------------------------------------------------------------------------------------------
  /// Assign Value
  ///-----------------------------------------------------------------------------------------------
  // Assigns the value of an expected directly to a concrete instance with perfect forwarding.
  template <class U>
  constexpr Expected<void, E> assign_to(U& value) const& {
    static_assert(IsAssignable_v<U&, const ValueType<Derived>&>,
                  "Argument Type is not assignable with Expected Type (const T&)");
    auto assign = [&](const ValueType<Derived>& derived) { value = derived; };
    return has_value() ? FunctorMap(assign, *derived()) : unexpected();
  }

  template <class U>
  constexpr Expected<void, E> assign_to(U& value) && {
    static_assert(IsAssignable_v<U&, ValueType<Derived>&&>,
                  "Argument Type is not assignable with Expected Type (T&&)");
    auto move_assign = [&](ValueType<Derived>&& derived) { value = std::move(derived); };
    return has_value() ? FunctorMap(move_assign, std::move(*derived())) : std::move(unexpected());
  }

  template <class U>
  constexpr Expected<void, E> assign_to(U& value) const&& {
    static_assert(IsAssignable_v<U&, const ValueType<Derived>&&>,
                  "Argument Type is not assignable with Expected Type (const T&&)");
    auto move_assign = [&](const ValueType<Derived>&& derived) { value = std::move(derived); };
    return has_value() ? FunctorMap(move_assign, std::move(*derived())) : std::move(unexpected());
  }

  ///-----------------------------------------------------------------------------------------------
  ///  Guard
  ///-----------------------------------------------------------------------------------------------

  template <class P>
  constexpr Expected<ValueType<Derived>, E> guard(P&& pred, E&& new_error) const& {
    if (is_error_) { return unexpected(); }
    return pred(derived()->value()) ? *derived() : Unexpected<E>{new_error};
  }

  template <class P>
  constexpr Expected<ValueType<Derived>, E> guard(P&& pred, E&& new_error) && {
    if (is_error_) { return std::move(unexpected()); }
    return pred(derived()->value()) ? std::move(*derived()) : Unexpected<E>{new_error};
  }

  template <class P>
  constexpr Expected<ValueType<Derived>, E> guard(P&& pred, E&& new_error) const&& {
    if (is_error_) { return std::move(unexpected()); }
    return pred(derived()->value()) ? std::move(*derived()) : Unexpected<E>{new_error};
  }

  ///-----------------------------------------------------------------------------------------------
  /// Ignore Error
  ///-----------------------------------------------------------------------------------------------

  template <class U>
  constexpr Expected<ValueType<Derived>, E> ignore_error(U&& next) const& {
    return has_value() ? *derived() : std::forward<U>(next);
  }

  template <class U>
  constexpr Expected<ValueType<Derived>, E> ignore_error(U&& next) && {
    return has_value() ? std::move(*derived()) : std::forward<U>(next);
  }

  template <class U>
  constexpr Expected<ValueType<Derived>, E> ignore_error(U&& next) const&& {
    return has_value() ? std::move(*derived()) : std::forward<U>(next);
  }

  constexpr Expected<void, E> ignore_error() const { return {}; }

 protected:
  // Allow other Expected classes access to `unexpected` getters.
  template <class, class, class>
  friend class ExpectedBase;

  // Protected default constructor so derived types can write custom initializers.
  ExpectedBase() = default;

  // Protected Value constructor so Derived types can convert to stored types
  template <class... Args, EnableIf_t<IsConstructible_v<T, Args...>, void*> = nullptr>
  constexpr ExpectedBase(Args&&... args) {
    constructValueFrom(std::forward<Args>(args)...);
  }

  // Unexpected accessors to skip extra construction calls when not needed.
  constexpr const Unexpected<E>& unexpected() const& {
    GXF_ASSERT(is_error_, "Expected does not have an error. Check before accessing.");
    return *ValuePointer<Unexpected<E>>(buffer_);
  }

  constexpr Unexpected<E>& unexpected() & {
    GXF_ASSERT(is_error_, "Expected does not have an error. Check before accessing.");
    return *ValuePointer<Unexpected<E>>(buffer_);
  }

  constexpr Unexpected<E>&& unexpected() && {
    GXF_ASSERT(is_error_, "Expected does not have an error. Check before accessing.");
    return std::move(*ValuePointer<Unexpected<E>>(buffer_));
  }

  // Returns a pointer to the dervied type for accessing value result in supported specializations.
  constexpr const Derived* derived() const { return static_cast<const Derived*>(this); }
  constexpr       Derived* derived()       { return static_cast<Derived*>(this); }

  ///-----------------------------------------------------------------------------------------------
  /// Construction Helpers
  ///-----------------------------------------------------------------------------------------------
  // Constuct a new object in allocated buffer.
  template <class U, class G, class D>
  constexpr void constructFrom(const ExpectedBase<U, G, D>& other) {
    other.has_value() ? constructValueFrom(other) : constructErrorFrom(other);
  }

  // Move constuct a new object in allocated buffer.
  template <class U, class G, class D>
  constexpr void constructFrom(ExpectedBase<U, G, D>&& other) {
    other.has_value() ? constructValueFrom(std::move(other)) : constructErrorFrom(std::move(other));
  }

  template <class U, class G, class D>
  constexpr void constructErrorFrom(const ExpectedBase<U, G, D>& other) {
    is_error_ = true;
    InplaceCopyConstruct(buffer_, *ValuePointer<Unexpected<G>>(other.buffer_));
  }

  template <class U, class G, class D>
  constexpr void constructErrorFrom(ExpectedBase<U, G, D>&& other) {
    is_error_ = true;
    InplaceMoveConstruct(buffer_, std::move(*ValuePointer<Unexpected<G>>(other.buffer_)));
  }

  template <class U, class G, class D>
  constexpr void constructValueFrom(const ExpectedBase<U, G, D>& other) {
    is_error_ = false;
    InplaceCopyConstruct<T>(buffer_, *ValuePointer<U>(other.buffer_));
  }

  template <class U, class G, class D>
  constexpr void constructValueFrom(ExpectedBase<U, G, D>&& other) {
    is_error_ = false;
    InplaceConstruct<T>(buffer_, std::move(*ValuePointer<U>(other.buffer_)));
  }

  template <class... Args>
  void constructValueFrom(Args&&... args) {
    is_error_ = false;
    InplaceConstruct<T>(buffer_, std::forward<Args>(args)...);
  }

  // Call the destructor for the current object explicitly
  constexpr void destruct() {
    has_value() ? Destruct<T>(buffer_) : Destruct<Unexpected<E>>(buffer_);
  }

  bool is_error_ = true;
  static constexpr uint32_t kAlign =
      (alignof(Unexpected<E>) < alignof(T)) ? alignof(T) : alignof(Unexpected<E>);
  static constexpr uint32_t kSize =
      (sizeof(Unexpected<E>) < sizeof(T)) ? sizeof(T) : sizeof(Unexpected<E>);
  alignas(kAlign) byte buffer_[kSize] = {0};
};
}  // namespace detail

///-------------------------------------------------------------------------------------------------
//  Expected Implementation
///-------------------------------------------------------------------------------------------------
// Default instance for Expected types.
template <class T, class E>
class Expected : public detail::ExpectedBase<T, E, Expected<T, E>> {
 public:
  using detail::ExpectedBase<T, E, Expected>::ExpectedBase;

  // Enable implicit construction of Move-Only types
  constexpr Expected(T&& value) : detail::ExpectedBase<T, E, Expected>(std::move(value)) {}

  template <class U, EnableIf_t<IsConstructible_v<T, U> && IsConvertible_v<U, T>>* = nullptr>
  constexpr Expected(U&& value) : detail::ExpectedBase<T, E, Expected>(std::forward<U>(value)) {}

  template <class U, EnableIf_t<IsConstructible_v<T, U> && !IsConvertible_v<U, T>>* = nullptr>
  explicit constexpr Expected(U&& value)
      : detail::ExpectedBase<T, E, Expected>(std::forward<U>(value)) {}

  template <class... Args, EnableIf_t<IsConstructible_v<T, Args...>, void*> = nullptr>
  constexpr Expected(Args&&... args)
      : detail::ExpectedBase<T, E, Expected>(std::forward<Args>(args)...) {}

  template <class U, EnableIf_t<IsConstructible_v<T, U> && IsConvertible_v<U, T>>* = nullptr>
  constexpr Expected(const Expected<U, E>& other) {
    other.has_value() ? this->constructValueFrom(other.value()) : this->constructErrorFrom(other);
  }

  template <class U, EnableIf_t<IsConstructible_v<T, U> && !IsConvertible_v<U, T>>* = nullptr>
  explicit constexpr Expected(const Expected<U, E>& other) {
    other.has_value() ? this->constructValueFrom(other.value()) : this->constructErrorFrom(other);
  }

  template <class U, EnableIf_t<IsConstructible_v<T, U> && IsConvertible_v<U, T>>* = nullptr>
  constexpr Expected(Expected<U, E>&& other) {
    other.has_value() ? this->constructValueFrom(std::move(other.value()))
                      : this->constructErrorFrom(std::move(other));
  }

  template <class U, EnableIf_t<IsConstructible_v<T, U> && !IsConvertible_v<U, T>>* = nullptr>
  explicit constexpr Expected(Expected<U, E>&& other) {
    other.has_value() ? this->constructValueFrom(std::move(other.value()))
                      : this->constructErrorFrom(std::move(other));
  }

  template <class U = T, EnableIf_t<IsConstructible_v<T, U> && IsConvertible_v<U, T>>* = nullptr>
  constexpr Expected& operator=(U&& value) {
    this->destruct();
    this->constructValueFrom(std::forward<U>(value));
    return *this;
  }

  template <class... Args>
  T& replace(Args&&... args) {
    static_assert(IsConstructible_v<T, Args...>,
        "T cannot be constructed with the provide argument types.");
    this->destruct();
    this->constructValueFrom(std::forward<Args>(args)...);
    return value();
  }

  constexpr const T& value() const& {
    GXF_ASSERT(this->has_value(), "Expected does not have a value. Check before accessing.");
    return *ValuePointer<T>(this->buffer_);
  }

  constexpr T& value() & {
    GXF_ASSERT(this->has_value(), "Expected does not have a value. Check before accessing.");
    return *ValuePointer<T>(this->buffer_);
  }

  constexpr const T&& value() const&& {
    GXF_ASSERT(this->has_value(), "Expected does not have a value. Check before accessing.");
    return std::move(*ValuePointer<T>(this->buffer_));
  }

  constexpr T&& value() && {
    GXF_ASSERT(this->has_value(), "Expected does not have a value. Check before accessing.");
    return std::move(*ValuePointer<T>(this->buffer_));
  }
};

///-------------------------------------------------------------------------------------------------
///  Expected of Reference Implementation
///-------------------------------------------------------------------------------------------------
// Specialization of Expected for references.
template <class T, class E>
class Expected<T&, E> : public detail::ExpectedBase<T*, E, Expected<T&, E>> {
 private:
  template <class, class = void>
  struct IsExpectedConvertible : FalseType {};

  template <class U>
  struct IsExpectedConvertible<U, void_t<decltype(Declval<U>().value())>>
      : IsConvertible<decltype(Declval<U>().value()), T&> {};

  template <class U>
  static constexpr bool IsExpectedConvertible_v =
      detail::IsExpected_v<U> && IsExpectedConvertible<U>::value;

 public:
  using detail::ExpectedBase<T*, E, Expected>::ExpectedBase;

  template <class U, EnableIf_t<IsConvertible_v<U, T&>, void*> = nullptr>
  constexpr Expected(U&& value) : detail::ExpectedBase<T*, E, Expected>(&std::forward<U>(value)) {}

  template <class U, EnableIf_t<IsExpectedConvertible_v<U>, void*> = nullptr>
  constexpr Expected(U&& other) {
    other.has_value() ? this->constructValueFrom(&std::forward<U>(other).value())
                      : this->constructErrorFrom(std::forward<U>(other));
  }

  constexpr T& value() const {
    GXF_ASSERT(this->has_value(), "Expected does not have a value. Check before accessing.");
    return **ValuePointer<T*>(this->buffer_);
  }
};
///-------------------------------------------------------------------------------------------------
///  Expected of Void Implementation
///-------------------------------------------------------------------------------------------------
// Specialization of Expected for void. This is essentially an std::optional<E>.
template <class E>
class Expected<void, E> : public detail::ExpectedBase<detail::Unit, E, Expected<void, E>> {
 public:
  using detail::ExpectedBase<detail::Unit, E, Expected>::ExpectedBase;

  constexpr Expected() : detail::ExpectedBase<detail::Unit, E, Expected>(detail::Unit{}) {}

  // Constructor that allows assigning an Expected with the same error type to an Expected<void>
  template <class U>
  explicit constexpr Expected(const Expected<U, E>& other) {
    other.has_value() ? this->constructValueFrom() : this->constructErrorFrom(other);
  }

  // Constructor that allows assigning an Expected with the same error type to an Expected<void>
  template <class U>
  explicit constexpr Expected(Expected<U, E>&& other) {
    other.has_value() ? this->constructValueFrom() : this->constructErrorFrom(std::move(other));
  }

  // Combines two expected value taking the first error.
  Expected& operator&=(const Expected& other) {
    *this = *this & other;
    return *this;
  }

  // Combines two expected value taking the last success.
  Expected& operator|=(const Expected& other) {
    *this = *this | other;
    return *this;
  }
};

///-------------------------------------------------------------------------------------------------
///  Utility Functions
///-------------------------------------------------------------------------------------------------
// Checks if all Expected values are not in error. If in error, return the first error found.
template <class E, class T>
Expected<void, E> AllOf(const Expected<T, E>& expected) {
  return Expected<void, E>{expected};
}

// Checks if all Expected values are not in error. If in error, return the first error found.
template <class E, class T, class... Ts>
Expected<void, E> AllOf(const Expected<T, E>& expected, const Expected<Ts, E>&... others) {
  return expected & AllOf(others...);
}

// Checks if all the Arguments are of type Expected<T, E>. If so, unwraps them and calls
// the function object F with the unwrapped arguments. Otherwise, return an Unexpected with
// the first error value found in the argument list.
template <class F, class... Args>
auto Apply(F&& func, Args&&... args) -> EnableIf_t<
    Conjunction_v<detail::IsExpected<Args>...>, decltype(detail::FunctorMap(func, args...)) > {
  const auto all_valid = AllOf(std::forward<Args>(args)...);
  if (!all_valid) { return Unexpected<detail::ErrorType<Args...>>(all_valid.error()); }
  return detail::FunctorMap(std::forward<F>(func), std::forward<Args>(args)...);
}

///-------------------------------------------------------------------------------------------------
///  Comparison Operators
///-------------------------------------------------------------------------------------------------
template <class T, class E, class U, class G>
constexpr bool operator==(const Expected<T, E>& lhs, const Expected<U, G>& rhs) {
  return ( lhs &&  rhs && (lhs.value() == rhs.value())) ||
         (!lhs && !rhs && (lhs.error() == rhs.error()));
}
template <class T, class E, class U>
constexpr bool operator==(const Expected<T, E>& lhs, const U& rhs) {
  return lhs && (*lhs == rhs);
}
template <class T, class E, class U>
constexpr bool operator==(const U& lhs, const Expected<T, E>& rhs) {
  return rhs && (lhs == *rhs);
}
template <class T, class E>
constexpr bool operator==(const Expected<T, E>& lhs, const Unexpected<E>& rhs) {
  return !lhs && (lhs.error() == rhs.value());
}
template <class T, class E>
constexpr bool operator==(const Unexpected<E>& lhs, const Expected<T, E>& rhs) {
  return !rhs && (lhs.value() == rhs.error());
}
template <class E>
constexpr bool operator==(const Unexpected<E>& lhs, const Unexpected<E>& rhs) {
  return lhs.value() == rhs.value();
}

// All inequality operators are defined in terms of negated equality
template <class T, class E, class U, class G>
constexpr bool operator!=(const Expected<T, E>& lhs, const Expected<U, G>& rhs) {
  return !(lhs == rhs);
}
template <class T, class E, class U>
constexpr bool operator!=(const Expected<T, E>& lhs, const U& rhs) {
  return !(lhs == rhs);
}
template <class T, class E, class U>
constexpr bool operator!=(const U& lhs, const Expected<T, E>& rhs) {
  return !(lhs == rhs);
}
template <class E>
constexpr bool operator!=(const Unexpected<E>& lhs, const Unexpected<E>& rhs) {
  return !(lhs == rhs);
}

///-------------------------------------------------------------------------------------------------
///  Binary Logical Operators
///-------------------------------------------------------------------------------------------------
// Combines two expected value taking the first error.
template <class E, class T, class U>
Expected<void, E> operator&(const Expected<T, E>& lhs, const Expected<U, E>& rhs) {
  return !lhs ? Expected<void, E>{lhs} : Expected<void, E>{rhs};
}

// Combines two expected value taking the last success.
template <class E, class T, class U>
Expected<void, E> operator|(const Expected<T, E>& lhs, const Expected<U, E>& rhs) {
  return rhs ? Expected<void, E>{rhs} : Expected<void, E>{lhs};
}

///-------------------------------------------------------------------------------------------------
///  Binary Arithmetic Operators
///-------------------------------------------------------------------------------------------------
// Arithmetic operator+ for values stored in expected values. Returns either the computed value
// or the first unexpected among lhs and rhs.
template <class V, class E>
Expected<V, E> operator+(const Expected<V, E>& lhs, const Expected<V, E>& rhs) {
  return Apply([](const V& lhs, const V& rhs) { return lhs + rhs; }, lhs, rhs);
}

template <class V, class E>
Expected<V, E> operator+(const Expected<V, E>& lhs, const V& rhs) {
  return lhs.map([&](const V& value) { return value + rhs; });
}

template <class V, class E>
Expected<V, E> operator+(const V& lhs, const Expected<V, E>& rhs) {
  return rhs.map([&](const V& value) { return lhs + value; });
}

// Arithmetic operator- for values stored in expected values. Returns either the computed value
// or the first unexpected among lhs and rhs.
template <class V, class E>
Expected<V, E> operator-(const Expected<V, E>& lhs, const Expected<V, E>& rhs) {
  return Apply([](const V& lhs, const V& rhs) { return lhs - rhs; }, lhs, rhs);
}

template <class V, class E>
Expected<V, E> operator-(const Expected<V, E>& lhs, const V& rhs) {
  return lhs.map([&](const V& value) { return value - rhs; });
}

template <class V, class E>
Expected<V, E> operator-(const V& lhs, const Expected<V, E>& rhs) {
  return rhs.map([&](const V& value) { return lhs - value; });
}

// Arithmetic operator* for values stored in expected values. Returns either the computed value
// or the first unexpected among lhs and rhs.
template <class V, class E>
Expected<V, E> operator*(const Expected<V, E>& lhs, const Expected<V, E>& rhs) {
  return Apply([](const V& lhs, const V& rhs) { return lhs * rhs; }, lhs, rhs);
}

template <class V, class E>
Expected<V, E> operator*(const Expected<V, E>& lhs, const V& rhs) {
  return lhs.map([&](const V& value) { return value * rhs; });
}

template <class V, class E>
Expected<V, E> operator*(const V& lhs, const Expected<V, E>& rhs) {
  return rhs.map([&](const V& value) { return lhs * value; });
}

// Arithmetic operator/ for values stored in expected values. Returns either the computed value
// or the first unexpected among lhs and rhs.
template <class V, class E>
Expected<V, E> operator/(const Expected<V, E>& lhs, const Expected<V, E>& rhs) {
  return Apply([](const V& lhs, const V& rhs) { return lhs / rhs; }, lhs, rhs);
}

template <class V, class E>
Expected<V, E> operator/(const Expected<V, E>& lhs, const V& rhs) {
  return lhs.map([&](const V& value) { return value / rhs; });
}

template <class V, class E>
Expected<V, E> operator/(const V& lhs, const Expected<V, E>& rhs) {
  return rhs.map([&](const V& value) { return lhs / value; });
}

// Arithmetic operator% for values stored in expected values. Returns either the computed value
// or the first unexpected among lhs and rhs.
template <class V, class E>
Expected<V, E> operator%(const Expected<V, E>& lhs, const Expected<V, E>& rhs) {
  return Apply([](const V& lhs, const V& rhs) { return lhs % rhs; }, lhs, rhs);
}

template <class V, class E>
Expected<V, E> operator%(const Expected<V, E>& lhs, const V& rhs) {
  return lhs.map([&](const V& value) { return value % rhs; });
}

template <class V, class E>
Expected<V, E> operator%(const V& lhs, const Expected<V, E>& rhs) {
  return rhs.map([&](const V& value) { return lhs % value; });
}

}  // namespace nvidia

#endif  // NVIDIA_COMMON_EXPECTED_HPP_
