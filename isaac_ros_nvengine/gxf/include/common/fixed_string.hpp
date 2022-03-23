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
#ifndef NVIDIA_GXF_COMMON_FIXED_STRING_HPP_
#define NVIDIA_GXF_COMMON_FIXED_STRING_HPP_

#include <algorithm>
#include <cstring>

#include "common/expected.hpp"
#include "common/iterator.hpp"
#include "common/type_utils.hpp"

namespace nvidia {

// Safety container that substitutes std::string and uses Expected for error handling.
// Uses a fixed-size buffer allocated on the stack for storage.
template <size_t N>
class FixedString {
 public:
  using value_type             = char;
  using size_type              = size_t;
  using const_iterator         = ConstRandomAccessIterator<FixedString>;
  using const_reverse_iterator = ReverseIterator<const_iterator>;

  // Custom error codes for container
  enum struct Error {
    kArgumentNull,
    kExceedingPreallocatedSize,
  };

  // Expected type which uses class specific errors
  template <typename U>
  using Expected = Expected<U, Error>;
  // Special value for returning a success
  const Expected<void> Success{};

  // Constructors
  constexpr FixedString() { clear(); }
  template <size_t M>
  constexpr FixedString(const char (&str)[M]) { copy(str); }
  template <size_t M>
  constexpr FixedString(const FixedString<M>& str) { copy(str); }

  // Assignment operators
  template <size_t M>
  constexpr FixedString& operator=(const char (&other)[M]) {
    copy(other);
    return *this;
  }
  template <size_t M>
  constexpr FixedString& operator=(const FixedString<M>& other) {
    copy(other);
    return *this;
  }
  // ISO standards state the assignment operator for M = N cannot be a template
  constexpr FixedString& operator=(const FixedString& other) {
    copy(other);
    return *this;
  }

  constexpr const_iterator         begin()  const { return cbegin(); }
  constexpr const_iterator         end()    const { return cend(); }
  constexpr const_reverse_iterator rbegin() const { return crbegin(); }
  constexpr const_reverse_iterator rend()   const { return crend(); }

  constexpr const_iterator         cbegin()  const { return const_iterator(*this, 0); }
  constexpr const_iterator         cend()    const { return const_iterator(*this, size_); }
  constexpr const_reverse_iterator crbegin() const { return const_reverse_iterator(cend()); }
  constexpr const_reverse_iterator crend()   const { return const_reverse_iterator(cbegin()); }

  // Comparison operators
  template <size_t M>
  constexpr bool operator==(const char (&other)[M]) const { return compare(other) == 0; }
  template <size_t M>
  constexpr bool operator==(const FixedString<M>& other) const { return compare(other) == 0; }
  template <size_t M>
  constexpr bool operator!=(const char (&other)[M]) const { return !(*this == other); }
  template <size_t M>
  constexpr bool operator!=(const FixedString<M>& other) const { return !(*this == other); }
  template <size_t M>
  constexpr bool operator<(const char (&other)[M]) const { return compare(other) < 0; }
  template <size_t M>
  constexpr bool operator<(const FixedString<M>& other) const { return compare(other) < 0; }
  template <size_t M>
  constexpr bool operator<=(const char (&other)[M]) const { return compare(other) <= 0; }
  template <size_t M>
  constexpr bool operator<=(const FixedString<M>& other) const { return compare(other) <= 0; }
  template <size_t M>
  constexpr bool operator>(const char (&other)[M]) const { return !(*this <= other); }
  template <size_t M>
  constexpr bool operator>(const FixedString<M> other) const { return !(*this <= other); }
  template <size_t M>
  constexpr bool operator>=(const char (&other)[M]) const { return !(*this < other); }
  template <size_t M>
  constexpr bool operator>=(const FixedString<M> other) const { return !(*this < other); }

  // Returns the length of the string excluding the null terminator
  constexpr size_t   size() const { return size_; }
  constexpr size_t length() const { return size_; }

  // Returns the number of characters the string can store excluding the null terminator
  constexpr size_t max_size() const { return kMaxSize; }

  // Returns the number of characters the string can store including the null terminator
  constexpr size_t capacity() const { return kCapacity; }

  // Returns true if the string is empty
  constexpr bool empty() const { return size_ == 0; }

  // Returns true if the string is full
  constexpr bool full() const { return size_ == kMaxSize; }

  // Returns a pointer to a null-terminated array of characters
  constexpr AddLvalueReference_t<const char[N + 1]>  data() const { return data_; }
  constexpr AddLvalueReference_t<const char[N + 1]> c_str() const { return data_; }

  // Clears the string
  constexpr void clear() {
    size_ = 0;
    data_[size_] = '\0';
  }

  // Appends a C-string to the end of the string
  constexpr Expected<void> append(const char* str, size_t size) {
    if (str == nullptr) {
      return Unexpected<Error>{Error::kArgumentNull};
    }
    const size_t length = strnlen(str, size);
    if (size_ + length > kMaxSize) {
      return Unexpected<Error>{Error::kExceedingPreallocatedSize};
    }
    std::memcpy(&data_[size_], str, length);
    size_ += length;
    data_[size_] = '\0';
    return Success;
  }

  // Appends a character array to the end of the string
  template <size_t M>
  constexpr Expected<void> append(const char (&str)[M]) { return append(str, M); }

  // Appends a fixed string to the end of the string
  template <size_t M>
  constexpr Expected<void> append(const FixedString<M>& str) {
    return append(str.data(), str.size());
  }

  // Appends a character to the end of the string
  constexpr Expected<void> append(char c) { return append(&c, 1); }

  // Copies a C-string to the string
  constexpr Expected<void> copy(const char* str, size_t size) {
    clear();
    return append(str, size);
  }

  // Copies a character array to the string
  template <size_t M>
  constexpr void copy(const char (&str)[M]) {
    static_assert(M <= kMaxSize, "Exceeding container capacity");
    copy(str, M);
  }

  // Copies a fixed string to the string
  template <size_t M>
  constexpr void copy(const FixedString<M>& str) {
    static_assert(M <= kMaxSize, "Exceeding container capacity");
    copy(str.data(), str.size());
  }

  // Comapres string to a character array
  template <size_t M>
  constexpr int compare(const char (&str)[M]) const {
    const size_t length = strnlen(str, M);
    const int result = std::strncmp(data_, str, std::min(size_, length));
    if (result != 0) {
      return result;
    }
    if (length > size_) {
      return -1;
    }
    if (length < size_) {
      return 1;
    }
    return 0;
  }

  // Compares string to a fixed string
  template <size_t M>
  constexpr int compare(const FixedString<M>& str) const { return compare(str.data()); }

 private:
  // Storage capacity
  static constexpr size_t kCapacity = N + 1;
  // Maximum length of the string
  static constexpr size_t kMaxSize = N;

  // String length
  size_t size_;
  // Data buffer
  char data_[kCapacity];
};

}  // namespace nvidia

#endif  // NVIDIA_GXF_COMMON_FIXED_STRING_HPP_
