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
#ifndef NVIDIA_GXF_COMMON_ITERATOR_HPP_
#define NVIDIA_GXF_COMMON_ITERATOR_HPP_

#include <iterator>

#include "common/expected.hpp"
#include "common/type_utils.hpp"

namespace nvidia {

namespace detail {

/// Type traits to determine if a type has `data()`
template <class, class = void> struct HasData                               : FalseType {};
template <class T> struct HasData<T, void_t<decltype(Declval<T>().data())>> : TrueType  {};
template <class T> constexpr bool HasData_v = HasData<T>::value;

/// Type traits to determine if a type has `size()`
template <class, class = void> struct HasSize                               : FalseType {};
template <class T> struct HasSize<T, void_t<decltype(Declval<T>().size())>> : TrueType  {};
template <class T> constexpr bool HasSize_v = HasSize<T>::value;

}  // namespace detail

/// Random-access iterator for containers with elements in contiguous memory.
/// A pointer to the container is stored so the iterator does not get invalidated if elements are
/// added and/or removed from the container. Incrementing/decrementing operations will saturate
/// if the iterator reaches the end/beginning of the container. Element access is wrapped in
/// an `Expected` to prevent dereferencing invalid memory.
template <typename TContainer, typename TValue = typename TContainer::value_type>
class RandomAccessIterator : public std::iterator<std::random_access_iterator_tag, TValue> {
  static_assert(detail::HasData_v<TContainer>, "TContainer must have data()");
  static_assert(detail::HasSize_v<TContainer>, "TContainer must have size()");
  static_assert(IsIntegral_v<decltype(Declval<TContainer>().size())>,
                "size() must return an integral type");

 public:
  enum struct Error {
    kArgumentOutOfRange,
    kInvalidIterator,
  };

  template <typename U>
  using Expected = Expected<U, Error>;

  using typename std::iterator<std::random_access_iterator_tag, TValue>::difference_type;

  constexpr RandomAccessIterator() : container_{nullptr}, index_{-1} {}
  constexpr RandomAccessIterator(TContainer& container, size_t start)
      : container_{&container}, index_{0} {
    *this += start;
  }

  constexpr RandomAccessIterator(const RandomAccessIterator& other) = default;
  constexpr RandomAccessIterator(RandomAccessIterator&& other) = default;
  constexpr RandomAccessIterator& operator=(const RandomAccessIterator& other) = default;
  constexpr RandomAccessIterator& operator=(RandomAccessIterator&& other) = default;

  constexpr Expected<TValue&> operator*() const {
    return 0 <= index_ && index_ < static_cast<difference_type>(container_->size())
        ? Expected<TValue&>{container_->data()[index_]}
        : Unexpected<Error>{Error::kInvalidIterator};
  }
  constexpr Expected<TValue&> operator[](difference_type offset) const {
    if (container_ == nullptr) {
      return Unexpected<Error>{Error::kInvalidIterator};
    }
    difference_type index = index_ + offset;
    return 0 <= index && index < static_cast<difference_type>(container_->size())
        ? Expected<TValue&>{container_->data()[index]}
        : Unexpected<Error>{Error::kArgumentOutOfRange};
  }

  constexpr RandomAccessIterator& operator+=(difference_type offset) {
    if (container_ == nullptr) {
      return *this;
    }
    difference_type index = index_ + offset;
    if (index < 0) {
      index = 0;
    } else if (index > static_cast<difference_type>(container_->size())) {
      index = container_->size();
    }
    index_ = index;
    return *this;
  }
  constexpr RandomAccessIterator& operator++() {
    *this += 1;
    return *this;
  }
  constexpr RandomAccessIterator operator++(int) {
    RandomAccessIterator iter = *this;
    ++(*this);
    return iter;
  }
  constexpr RandomAccessIterator& operator-=(difference_type offset) {
    *this += -offset;
    return *this;
  }
  constexpr RandomAccessIterator& operator--() {
    *this -= 1;
    return *this;
  }
  constexpr RandomAccessIterator operator--(int) {
    RandomAccessIterator iter = *this;
    --(*this);
    return iter;
  }

  friend constexpr RandomAccessIterator operator+(RandomAccessIterator a, difference_type n) {
    a += n;
    return a;
  }
  friend constexpr RandomAccessIterator operator+(difference_type n, RandomAccessIterator a) {
    return a + n;
  }
  friend constexpr RandomAccessIterator operator-(RandomAccessIterator a, difference_type n) {
    a -= n;
    return a;
  }
  friend constexpr difference_type operator-(const RandomAccessIterator& a,
                                             const RandomAccessIterator& b) {
    return a.index_ - b.index_;
  }
  friend constexpr bool operator==(const RandomAccessIterator& a, const RandomAccessIterator& b) {
    return a.container_ == b.container_ && a.index_ == b.index_;
  }
  friend constexpr bool operator!=(const RandomAccessIterator& a, const RandomAccessIterator& b) {
    return !(a == b);
  }

 private:
  /// Container pointer
  TContainer* container_;
  /// Iterator index
  difference_type index_;
};

/// Constant Random-access iterator
template <typename TContainer>
using ConstRandomAccessIterator = RandomAccessIterator<const TContainer,
                                                       const typename TContainer::value_type>;

/// Reverse iterator
template <typename TIterator>
class ReverseIterator
    : public std::iterator<std::random_access_iterator_tag, typename TIterator::value_type> {
 public:
  using TValue = typename TIterator::value_type;
  using Error = typename TIterator::Error;

  template <typename U>
  using Expected = Expected<U, Error>;

  using difference_type = typename TIterator::difference_type;

  constexpr explicit ReverseIterator() : iter_{} {}
  constexpr explicit ReverseIterator(TIterator iter) : iter_{iter} {}

  constexpr ReverseIterator(const ReverseIterator& other) = default;
  constexpr ReverseIterator(ReverseIterator&& other) = default;
  constexpr ReverseIterator& operator=(const ReverseIterator& other) = default;
  constexpr ReverseIterator& operator=(ReverseIterator&& other) = default;

  constexpr TIterator base() const { return iter_; }

  constexpr Expected<TValue&> operator*() const { return *std::prev(iter_); }
  constexpr Expected<TValue&> operator[](difference_type offset) const {
    return std::prev(iter_)[-offset];
  }

  constexpr ReverseIterator& operator+=(difference_type offset) {
    iter_ -= offset;
    return *this;
  }
  constexpr ReverseIterator& operator++() {
    iter_ -= 1;
    return *this;
  }
  constexpr ReverseIterator operator++(int) {
    ReverseIterator iter = *this;
    ++(*this);
    return iter;
  }
  constexpr ReverseIterator& operator-=(difference_type offset) {
    iter_ += offset;
    return *this;
  }
  constexpr ReverseIterator& operator--() {
    iter_ += 1;
    return *this;
  }
  constexpr ReverseIterator operator--(int) {
    ReverseIterator iter = *this;
    --(*this);
    return iter;
  }

  friend constexpr ReverseIterator operator+(ReverseIterator a, difference_type n) {
    return ReverseIterator(a.iter_ - n);
  }
  friend constexpr ReverseIterator operator+(difference_type n, ReverseIterator a) {
    return ReverseIterator(a.iter_ - n);
  }
  friend constexpr ReverseIterator operator-(ReverseIterator a, difference_type n) {
    return ReverseIterator(a.iter_ + n);
  }
  friend constexpr difference_type operator-(const ReverseIterator& a, const ReverseIterator& b) {
    return b.iter_ - a.iter_;
  }
  friend constexpr bool operator==(const ReverseIterator& a, const ReverseIterator& b) {
    return a.iter_ == b.iter_;
  }
  friend constexpr bool operator!=(const ReverseIterator& a, const ReverseIterator& b) {
    return !(a == b);
  }

 private:
  /// Random-access iterator
  TIterator iter_;
};

}  // namespace nvidia

#endif  // NVIDIA_GXF_COMMON_ITERATOR_HPP_
