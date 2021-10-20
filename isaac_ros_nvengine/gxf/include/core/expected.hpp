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
#ifndef NVIDIA_CORE_EXPECTED_HPP_
#define NVIDIA_CORE_EXPECTED_HPP_

#include <utility>

#include "core/assert.hpp"
#include "core/memory_utils.hpp"
#include "core/type_utils.hpp"

namespace nvidia {

// Loosely implemented after P0323R7
template <typename E>
class Unexpected {
 public:
  constexpr Unexpected() = delete;
  ~Unexpected() = default;
  constexpr Unexpected(const Unexpected&) = default;
  constexpr Unexpected(Unexpected&&) = default;

  // template<class... Args>
  // constexpr explicit Unexpected(in_place_t, Args&&...);
  // template<class U, class... Args>
  // constexpr explicit Unexpected(in_place_t, initializer_list<U>, Args&&...);
  template <class Err = E>
  constexpr explicit Unexpected(Err&& value) : value_{std::forward<Err>(value)} {
    // When debugging is enabled we rais a signal when an error is created.
#ifndef NDEBUG
    std::abort();
#endif
  }
  // template<class Err>
  // constexpr explicit Unexpected(const Unexpected<Err>&) : value_{value} {}
  // template<class Err>
  // constexpr explicit Unexpected(Unexpected<Err>&&) : value_{std::move(value)} {}

  constexpr Unexpected& operator=(const Unexpected&) = default;
  constexpr Unexpected& operator=(Unexpected&&) = default;
  // template<class Err = E>
  // constexpr Unexpected& operator=(const Unexpected<Err>& other) { value_ = other; }
  // template<class Err = E>
  // constexpr Unexpected& operator=(Unexpected<Err>&& other) { value_ = std::move(other); }

  constexpr const E&  value() const& noexcept  { return value_; }
  constexpr E&        value() & noexcept       { return value_; }
  constexpr const E&& value() const&& noexcept { return std::move(value_); }
  constexpr E&&       value() && noexcept      { return std::move(value_); }

  // void swap(Unexpected& other) noexcept(see bellow);

  // template<class E1, class E2>
  // friend constexpr bool operator==(const Unexpected<E1>& lhs, const Unexpected<E2>& rhs) {
  //   return lhs.value() == rhs.value();
  // }
  // template<class E1, class E2>
  // friend constexpr bool operator!=(const Unexpected<E1>& lhs, const Unexpected<E2>& rhs) {
  //   return lhs.value() != rhs.value();
  // }

  // template<class E1>
  // friend void swap(Unexpected<E1>& x, Unexpected<E1>& y) noexcept(noexcept(x.swap(y)));

 private:
  E value_;
};

// Forward delcarations for using with ExpectedBase
template <typename, typename> class Expected;

namespace detail {

template <typename>
struct IsExpected : FalseType {};

template <typename S, typename E>
struct IsExpected<Expected<S, E>> : TrueType {};

// Extracts the error type from a pack of Expecteds if they all contain the same Error .
template <typename...>
struct ErrorTypeHelper {};

template <typename... Ts, typename E>
struct ErrorTypeHelper<Expected<Ts, E>...> { using type = E; };

template <typename... Ts>
using ErrorType = typename ErrorTypeHelper<RemoveCVRef_t<Ts>...>::type;

// Finds the result type of invoking a function with the wrapped types of a pack of Expecteds.
template <typename F, typename... Es>
using ResultType = decltype(Declval<F>()(Declval<Es>().value()...));

// Empty stuct for representing an object with only one possible value (itself). This can be used
// for speciallizing on void types where void would not compile because it is not a valid type.
struct Unit {};

// Maps the values with the given functor. If F returns an Expected, it returns the Expected type
// directly, otherwise it constructs a new Expected.
template <typename F, typename... Es>
auto FunctorMap(F&& func, Es&&... expected) ->
    EnableIf_t<IsVoid<ResultType<F, Es...>>::value, Expected<void, ErrorType<Es...>>> {
  std::forward<F>(func)(std::forward<Es>(expected).value()...);
  return Expected<void, ErrorType<Es...>>{};
}


template <typename F, typename... Es>
auto FunctorMap(F&& func, Es&&... expected) ->
    EnableIf_t<IsExpected<ResultType<F, Es...>>::value, ResultType<F, Es...>> {
  return std::forward<F>(func)(std::forward<Es>(expected).value()...);
}

template <typename F, typename... Es>
auto FunctorMap(F&& func, Es&&... expected) ->
    EnableIf_t<!IsVoid<ResultType<F, Es...>>::value && !IsExpected<ResultType<F, Es...>>::value,
               Expected<ResultType<F, Es...>, ErrorType<Es...>>> {
  return std::forward<F>(func)(std::forward<Es>(expected).value()...);
}

// Base class for all operations of Expected that does not rely on handling of the left type T.
// Loosely specified after P0323R7.
template <typename T, typename E>
class ExpectedBase {
  static_assert(!IsVoid<T>::value, "ExpectedBase cannot wrap void types");
  static_assert(!IsVoid<E>::value, "ExpectedBase cannot wrap void types");
  static_assert(!IsReference<T>::value, "ExpectedBase cannot wrap reference types");
  static_assert(!IsReference<E>::value, "ExpectedBase cannot wrap reference types");

 public:
  // template<class U>
  // using rebind = Expected<U, error_type>;

  // x.x.4.1, constructors
  constexpr ExpectedBase(const ExpectedBase& other) { constructFrom(other); }
  constexpr ExpectedBase(ExpectedBase&& other) { constructFrom(std::forward<ExpectedBase>(other)); }

  // template<class U, class G>
  // explicit constexpr Expected(const Expected<U, G>& other)
  //     : ok_{other.ok_} {
  //   if (ok_) {
  //     value_ = other.value_;
  //   } else {
  //     error_ = other.error_;
  //   }
  // }
  // template<class U, class G>
  // explicit constexpr Expected(Expected<U, G>&& other)
  //     : ok_{other.ok_} {
  //   if (ok_) {
  //     value_ = std::move(other.value_);
  //   } else {
  //     error_ = std::move(other.error_);
  //   }
  // }

  template <class G = E>
  constexpr ExpectedBase(const Unexpected<G>& error) : is_error_{true} {
    InplaceConstruct<Unexpected<E>>(buffer_, error);
  }

  template <class G = E>
  constexpr ExpectedBase(Unexpected<G>&& error) : is_error_{true} {
    InplaceConstruct<Unexpected<E>>(buffer_, std::forward<Unexpected<G>>(error));
  }

  // template<class... Args>
  // constexpr explicit Expected(in_place_t, Args&&...);
  // template<class U, class... Args>
  // constexpr explicit Expected(in_place_t, initializer_list<U>, Args&&...);
  // template<class... Args>
  // constexpr explicit Expected(unexpect_t, Args&&...);
  // template<class U, class... Args>
  // constexpr explicit Expected(unexpect_t, initializer_list<U>, Args&&...);

  // x.x.4.2, destructor
  ~ExpectedBase() { destruct(); }

  // x.x.4.3, assignment
  ExpectedBase& operator=(const ExpectedBase& other) {
    destruct();
    constructFrom(other);
    return *this;
  }

  ExpectedBase& operator=(ExpectedBase&& other) {
    destruct();
    constructFrom(std::forward<ExpectedBase>(other));
    return *this;
  }

  template <class G = E>
  ExpectedBase& operator=(const Unexpected<G>& error) {
    destruct();
    is_error_ = true;
    InplaceConstruct<Unexpected<E>>(buffer_, error);
    return *this;
  }

  template <class G = E>
  ExpectedBase& operator=(Unexpected<G>&& error) {
    destruct();
    is_error_ = true;
    InplaceConstruct<Unexpected<E>>(buffer_, std::forward<Unexpected<G>>(error));
    return *this;
  }

  // x.x.4.4, modifiers
  // template<class... Args>
  //     T& emplace(Args&&...);
  // template<class U, class... Args>
  //     T& emplace(initializer_list<U>, Args&&...);

  // // x.x.4.5, swap
  // void swap(Expected&) noexcept(see below);

  // x.x.4.6, observers
  // constexpr const T* operator->() const;
  // constexpr T* operator->();
  // constexpr const T& operator*() const&;
  // constexpr T& operator*() &;
  // constexpr const T&& operator*() const&&;
  // constexpr T&& operator*() &&;

  constexpr bool has_value() const noexcept { return !is_error_; }

  constexpr explicit operator bool() const noexcept { return has_value(); }

  constexpr const E& error() const& {
    GXF_ASSERT(!has_value(), "Expected does not have an error. Check before accessing.");
    return ValuePointer<Unexpected<E>>(buffer_)->value();
  }
  constexpr E& error() & {
    GXF_ASSERT(!has_value(), "Expected does not have an error. Check before accessing.");
    return ValuePointer<Unexpected<E>>(buffer_)->value();
  }

  constexpr E&& error() && {
    GXF_ASSERT(!has_value(), "Expected does not have an error. Check before accessing.");
    return std::move(ValuePointer<Unexpected<E>>(buffer_)->value());
  }

  // // x.x.4.7, Expected equality operators
  // template<class T1, class E1, class T2, class E2>
  //     friend constexpr bool operator==(const Expected<T1, E1>& x, const Expected<T2, E2>& y);
  // template<class T1, class E1, class T2, class E2>
  //     friend constexpr bool operator!=(const Expected<T1, E1>& x, const Expected<T2, E2>& y);

  // // x.x.4.8, Comparison with T
  // template<class T1, class E1, class T2>
  //     friend constexpr bool operator==(const Expected<T1, E1>&, const T2&);
  // template<class T1, class E1, class T2>
  //     friend constexpr bool operator==(const T2&, const Expected<T1, E1>&);
  // template<class T1, class E1, class T2>
  //     friend constexpr bool operator!=(const Expected<T1, E1>&, const T2&);
  // template<class T1, class E1, class T2>
  //     friend constexpr bool operator!=(const T2&, const Expected<T1, E1>&);

  // // x.x.4.9, Comparison with Unexpected<E>
  // template<class T1, class E1, class E2>
  //     friend constexpr bool operator==(const Expected<T1, E1>&, const Unexpected<E2>&);
  // template<class T1, class E1, class E2>
  //     friend constexpr bool operator==(const Unexpected<E2>&, const Expected<T1, E1>&);
  // template<class T1, class E1, class E2>
  //     friend constexpr bool operator!=(const Expected<T1, E1>&, const Unexpected<E2>&);
  // template<class T1, class E1, class E2>
  //     friend constexpr bool operator!=(const Unexpected<E2>&, const Expected<T1, E1>&);

  // // x.x.4.10, Specialized algorithms
  // template<class T1, class E1>
  //     friend void swap(Expected<T1, E1>&, Expected<T1, E1>&) noexcept(see below);

  // When this expected does not contain an error an expected with the given value 'next' is
  // created and returned, otherwise an unexpected with the error is returned.
  template <class U>
  constexpr Expected<U, E> substitute(U next) const {
    return has_value() ? Expected<U, E>{std::forward<U>(next)} : Expected<U, E>{unexpected()};
  }

  // If in error the error is returned, otherwise the value is mapped with the given functor and
  // the result is returned.
  template <typename F>
  auto and_then(F&& func) const -> EnableIf_t<IsVoid<decltype(func())>::value, Expected<void, E>> {
    if (has_value()) {
      std::forward<F>(func)();
      return Expected<void, E>{};
    } else {
      return unexpected();
    }
  }

  template <typename F>
  auto and_then(F&& func) const ->
      EnableIf_t<IsExpected<decltype(func())>::value, decltype(func())> {
    if (has_value()) {
      return std::forward<F>(func)();
    } else {
      return unexpected();
    }
  }

  template <typename F>
  auto and_then(F&& func) const ->
      EnableIf_t<!IsVoid<decltype(func())>::value && !IsExpected<decltype(func())>::value,
      Expected<decltype(func()), E>> {
    if (has_value()) {
      return std::forward<F>(func)();
    } else {
      return unexpected();
    }
  }

 protected:
  // Allow other Expecteds access to `unexpected` getters.
  template <typename U, typename G>
  friend class Expected;

  // Protected default constructor so derived types can write custom initializers.
  ExpectedBase() = default;

  // Unexpected accessors to skip extra construction calls when not needed.
  constexpr const Unexpected<E>& unexpected() const& {
    GXF_ASSERT(!has_value(), "Expected does not have an error. Check before accessing.");
    return *ValuePointer<Unexpected<E>>(buffer_);
  }

  constexpr Unexpected<E>& unexpected() & {
    GXF_ASSERT(!has_value(), "Expected does not have an error. Check before accessing.");
    return *ValuePointer<Unexpected<E>>(buffer_);
  }

  constexpr Unexpected<E>&& unexpected() && {
    GXF_ASSERT(!has_value(), "Expected does not have an error. Check before accessing.");
    return std::move(*ValuePointer<Unexpected<E>>(buffer_));
  }

  // Constuct a new object in allocated buffer.
  void constructFrom(const ExpectedBase& other) {
    is_error_ = other.is_error_;
    if (other.has_value()) {
      InplaceCopyConstruct(buffer_, *ValuePointer<T>(other.buffer_));
    } else {
      InplaceCopyConstruct(buffer_, other.unexpected());
    }
  }

  // Move constuct a new object in allocated buffer.
  void constructFrom(ExpectedBase&& other) {
    is_error_ = other.is_error_;
    if (other.has_value()) {
      InplaceMoveConstruct(buffer_, std::forward<T>(*ValuePointer<T>(other.buffer_)));
    } else {
      InplaceMoveConstruct(buffer_, std::forward<Unexpected<E>>(other.unexpected()));
    }
  }

  // Call the destructor for the current object explicitly
  void destruct() {
    if (has_value()) {
      Destruct<T>(buffer_);
    } else {
      Destruct<Unexpected<E>>(buffer_);
    }
  }

  bool is_error_ = true;
  static constexpr uint32_t kAlign =
      (alignof(Unexpected<E>) < alignof(T)) ? alignof(T) : alignof(Unexpected<E>);
  static constexpr uint32_t kSize =
      (sizeof(Unexpected<E>) < sizeof(T)) ? sizeof(T) : sizeof(Unexpected<E>);
  alignas(kAlign) byte buffer_[kSize] = {0};
};
}  // namespace detail

// Default instance for Expected types.
template <typename T, typename E>
class Expected : public detail::ExpectedBase<T, E> {
 public:
  using detail::ExpectedBase<T, E>::ExpectedBase;
  template <typename U, typename G> friend class Expected;

  constexpr Expected(T&& value) {
    this->is_error_ = false;
    InplaceMoveConstruct(this->buffer_, std::forward<T>(value));
  }

  template <class U = T, typename = EnableIf_t<IsConstructible<T, U&&>::value>>
  constexpr Expected(U&& value) {
    this->is_error_ = false;
    InplaceConstruct<T>(this->buffer_, std::forward<U>(value));
  }

  template <class U = T, typename = EnableIf_t<IsConstructible<T, U&&>::value>>
  Expected& operator=(U&& value) {
    this->destruct();
    this->is_error_ = false;
    InplaceConstruct<T>(this->buffer_, std::forward<U>(value));
    return *this;
  }

  const T*  operator->() const  { return &value(); }
  T*        operator->()        { return &value(); }
  const T&  operator*() const&  { return value(); }
  T&        operator*() &       { return value(); }
  const T&& operator*() const&& { return std::move(value()); }
  T&&       operator*() &&      { return std::move(value()); }

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

  template <class U>
  constexpr T value_or(U&& default_value) const& {
    if (this->has_value()) {
      return *ValuePointer<T>(this->buffer_);
    } else {
      return std::forward<U>(default_value);
    }
  }

  template <class U>
  constexpr T value_or(U&& default_value) && {
    if (this->has_value()) {
      return std::move(*ValuePointer<T>(this->buffer_));
    } else {
      return std::forward<U>(default_value);
    }
  }

  // If in error the error is returned, otherwise the value is mapped with the given functor and
  // the result is returned.
  template <typename F>
  auto map(F&& func) & {
    return this->has_value() ? detail::FunctorMap(std::forward<F>(func), *this)
                             : this->unexpected();
  }

  template <typename F>
  auto map(F&& func) const& {
    return this->has_value() ? detail::FunctorMap(std::forward<F>(func), *this)
                             : this->unexpected();
  }

  template <typename F>
  auto map(F&& func) && {
    return this->has_value() ? detail::FunctorMap(std::forward<F>(func), std::move(*this))
                             : this->unexpected();
  }

  template <typename F>
  auto map(F&& func) const&& {
    return this->has_value() ? detail::FunctorMap(std::forward<F>(func), std::move(*this))
                             : this->unexpected();
  }

  // Assigns the value of an expected directly to a concrete instance with perfect forwarding.
  template <typename U>
  Expected<void, E> assign_to(U& value) const& {
    static_assert(IsAssignable_v<U&, const T&>,
                  "Argument Type is not assignable with Expected Type (const T&)");
    auto assign = [&](const T& expected_value) { value = expected_value; };
    return this->has_value() ? detail::FunctorMap(assign, std::move(*this)) : this->unexpected();
  }

  template <typename U>
  Expected<void, E> assign_to(U& value) && {
    static_assert(IsAssignable_v<U&, T&&>,
                  "Argument Type is not assignable with Expected Type (T&&)");
    auto move_assign = [&](T&& expected_value) { value = std::move(expected_value); };
    return this->has_value() ? detail::FunctorMap(move_assign, std::move(*this))
                             : this->unexpected();
  }

  template <typename U>
  Expected<void, E> assign_to(U& value) const&& {
    static_assert(IsAssignable_v<U&, const T&&>,
                  "Argument Type is not assignable with Expected Type (const T&&)");
    auto move_assign = [&](const T&& expected_value) { value = std::move(expected_value); };
    return this->has_value() ? detail::FunctorMap(move_assign, std::move(*this))
                             : this->unexpected();
  }
};

// Specialization of Expected for references.
template <typename T, typename E>
class Expected<T&, E> : public detail::ExpectedBase<T*, E> {
 public:
  using detail::ExpectedBase<T*, E>::ExpectedBase;

  constexpr Expected(T& value) {
    this->is_error_ = false;
    InplaceCopyConstruct(this->buffer_, &value);
  }

  Expected& operator=(T& value) {
    this->destruct();
    this->is_error_ = false;
    InplaceCopyConstruct(this->buffer_, &value);
    return *this;
  }

  T*  operator->() const { return &value(); }
  T&  operator*()  const { return value(); }

  constexpr T& value() const {
    GXF_ASSERT(this->has_value(), "Expected does not have a value. Check before accessing.");
    return **ValuePointer<T*>(this->buffer_);
  }

  constexpr T& value_or(T& default_value) const {
    return (this->has_value()) ? value() : default_value;
  }

  // If in error the error is returned, otherwise the value is mapped with the given functor and
  // the result is returned.
  template <typename F>
  auto map(F&& func) const {
    return (this->has_value()) ? detail::FunctorMap(std::forward<F>(func), *this)
                               : this->unexpected();
  }

  // Assigns the value of an expected directly to a concrete instance with perfect forwarding.
  template <typename U>
  Expected<void, E> assign_to(U& value) const {
    static_assert(IsAssignable_v<U&, T&>,
                  "Argument Type is not assignable with Expected Type (T&)");
    auto assign = [&](T& expected_value) { value = expected_value; };
    return this->has_value() ? detail::FunctorMap(assign, std::move(*this)) : this->unexpected();
  }
};

// Specialization of Expected for void. This is essentially an std::optional<E>.
template <typename E>
class Expected<void, E> : public detail::ExpectedBase<detail::Unit, E> {
 public:
  using detail::ExpectedBase<detail::Unit, E>::ExpectedBase;
  constexpr Expected() { this->is_error_ = false; }

  // Constructor that allows assigning an Expected with the same error type to an Expected<void>
  template<typename U>
  explicit constexpr Expected(const Expected<U, E>& other) {
    this->is_error_ = !other.has_value();
    if (!other) {
      InplaceConstruct<Unexpected<E>>(this->buffer_, other.unexpected());
    }
  }

  // Constructor that allows assigning an Expected with the same error type to an Expected<void>
  template<typename U>
  explicit constexpr Expected(Expected<U, E>&& other) {
    this->is_error_ = !other.has_value();
    if (!other) {
      InplaceConstruct<Unexpected<E>>(this->buffer_, std::move(other.unexpected()));
    }
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

  // If in error the error is returned, otherwise the value is mapped with the given functor and
  // the result is returned.
  template <typename F>
  auto map(F&& func) const { return this->and_then(std::forward<F>(func)); }
};

// Checks if all Expected values are not in error. If in error, return the first error found.
template <typename E, typename T>
Expected<void, E> AllOf(const Expected<T, E>& expected) {
  return Expected<void, E>{expected};
}

// Checks if all Expected values are not in error. If in error, return the first error found.
template <typename E, typename T, typename... Ts>
Expected<void, E> AllOf(const Expected<T, E>& expected, const Expected<Ts, E>&... others) {
  return expected & AllOf(others...);
}

// Checks if all the Arguments are of type Expected<T, E>. If so, unwraps them and calls
// the function object F with the unwrapped arguments. Otherwise, return an Unexpected with
// the first error value found in the argument list.
template <typename F, typename... Args>
auto Apply(F&& func, Args&&... args) -> EnableIf_t<
    Conjunction<detail::IsExpected<RemoveCVRef_t<Args>>...>::value,
    decltype(detail::FunctorMap(Declval<F>(), Declval<Args>()...)) > {
  const auto all_valid = AllOf(args...);
  if (!all_valid) { return Unexpected<detail::ErrorType<Args...>>(all_valid.error()); }
  return detail::FunctorMap(std::forward<F>(func), std::forward<Args>(args)...);
}

// Combines two expected value taking the first error.
template <typename E, typename T, typename U>
Expected<void, E> operator&(const Expected<T, E>& lhs, const Expected<U, E>& rhs) {
  return !lhs ? Expected<void, E>{lhs} : Expected<void, E>{rhs};
}

// Combines two expected value taking the last success.
template <typename E, typename T, typename U>
Expected<void, E> operator|(const Expected<T, E>& lhs, const Expected<U, E>& rhs) {
  return rhs ? Expected<void, E>{rhs} : Expected<void, E>{lhs};
}

// Arithmetic operator+ for values stored in expected values. Returns either the computed value
// or the first unexpected among lhs and rhs.
template <typename V, typename E>
Expected<V, E> operator+(const Expected<V, E>& lhs, const Expected<V, E>& rhs) {
  return Apply([](const V& lhs, const V& rhs) { return lhs + rhs; }, lhs, rhs);
}

template <typename V, typename E>
Expected<V, E> operator+(const Expected<V, E>& lhs, const V& rhs) {
  return lhs.map([&](const V& value) { return value + rhs; });
}

template <typename V, typename E>
Expected<V, E> operator+(const V& lhs, const Expected<V, E>& rhs) {
  return rhs.map([&](const V& value) { return lhs + value; });
}

// Arithmetic operator- for values stored in expected values. Returns either the computed value
// or the first unexpected among lhs and rhs.
template <typename V, typename E>
Expected<V, E> operator-(const Expected<V, E>& lhs, const Expected<V, E>& rhs) {
  return Apply([](const V& lhs, const V& rhs) { return lhs - rhs; }, lhs, rhs);
}

template <typename V, typename E>
Expected<V, E> operator-(const Expected<V, E>& lhs, const V& rhs) {
  return lhs.map([&](const V& value) { return value - rhs; });
}

template <typename V, typename E>
Expected<V, E> operator-(const V& lhs, const Expected<V, E>& rhs) {
  return rhs.map([&](const V& value) { return lhs - value; });
}

// Arithmetic operator* for values stored in expected values. Returns either the computed value
// or the first unexpected among lhs and rhs.
template <typename V, typename E>
Expected<V, E> operator*(const Expected<V, E>& lhs, const Expected<V, E>& rhs) {
  return Apply([](const V& lhs, const V& rhs) { return lhs * rhs; }, lhs, rhs);
}

template <typename V, typename E>
Expected<V, E> operator*(const Expected<V, E>& lhs, const V& rhs) {
  return lhs.map([&](const V& value) { return value * rhs; });
}

template <typename V, typename E>
Expected<V, E> operator*(const V& lhs, const Expected<V, E>& rhs) {
  return rhs.map([&](const V& value) { return lhs * value; });
}

// Arithmetic operator/ for values stored in expected values. Returns either the computed value
// or the first unexpected among lhs and rhs.
template <typename V, typename E>
Expected<V, E> operator/(const Expected<V, E>& lhs, const Expected<V, E>& rhs) {
  return Apply([](const V& lhs, const V& rhs) { return lhs / rhs; }, lhs, rhs);
}

template <typename V, typename E>
Expected<V, E> operator/(const Expected<V, E>& lhs, const V& rhs) {
  return lhs.map([&](const V& value) { return value / rhs; });
}

template <typename V, typename E>
Expected<V, E> operator/(const V& lhs, const Expected<V, E>& rhs) {
  return rhs.map([&](const V& value) { return lhs / value; });
}

}  // namespace nvidia

#endif  // NVIDIA_CORE_EXPECTED_HPP_
