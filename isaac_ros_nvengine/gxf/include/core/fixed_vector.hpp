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
#ifndef NVIDIA_GXF_CORE_FIXED_VECTOR_HPP_
#define NVIDIA_GXF_CORE_FIXED_VECTOR_HPP_

#include <cstring>
#include <utility>

#include "core/byte.hpp"
#include "core/expected.hpp"
#include "core/memory_utils.hpp"

namespace nvidia {

// Data structure that provides similar functionality to std::vector but does not dynamically
// reallocate memory and uses Expected type for error handling instead of exceptions.
// This container is not thread-safe.
//
// This container supports allocating memory on the stack or on the heap. If the template argument
// N is specified and greater than 0, a container with a capacity of N will be created on the stack.
// The stack allocated container cannot be resized after it is initialized. The heap allocated
// container must use the reserve function to allocate memory before use. It does not support
// copy assignment or construction.

// Base implementation
template <typename T>
class FixedVectorBase {
 public:
  virtual ~FixedVectorBase() = default;

  // Custom error codes for vector
  enum struct Error {
    kOutOfMemory,         // Memory allocation failed
    kArgumentOutOfRange,  // Argument is out of valid range
    kContainerEmpty,      // Container is empty
    kContainerFull,       // Container is fixed and reached max capacity
  };

  // Expected type which uses class specific errors
  template <typename U>
  using Expected = Expected<U, Error>;
  // Special value for returning a success
  const Expected<void> Success{};

  bool operator==(const FixedVectorBase& other) const {
    if (size_ != other.size_) {
      return false;
    }
    for (size_t i = 0; i < size_; i ++) {
      if (data_[i] != other.data_[i]) {
        return false;
      }
    }
    return true;
  }
  bool operator!=(const FixedVectorBase& other) const { return !(*this == other); }

  // Returns a pointer to data
  T* data() { return data_; }
  // Returns a read-only pointer to data
  const T* data() const { return data_; }
  // Returns the number of elements the vector can currently hold
  virtual size_t capacity() const = 0;
  // Returns the number of elements in the vector
  size_t size() const { return size_; }
  // Returns true if the vector contains no elements
  bool empty() const { return size_ == 0; }
  // Returns true if the vector reached capacity
  bool full() const { return size_ == capacity(); }

  // Returns a reference to the element at the given index
  Expected<T&> at(size_t index) {
    if (index >= size_) {
      return Unexpected<Error>{Error::kArgumentOutOfRange};
    }
    return data_[index];
  }

  // Returns a read-only reference to the element at the given index
  Expected<const T&> at(size_t index) const {
    if (index >= size_) {
      return Unexpected<Error>{Error::kArgumentOutOfRange};
    }
    return data_[index];
  }

  // Returns a reference to the first element
  Expected<T&> front() {
    if (empty()) {
      return Unexpected<Error>{Error::kContainerEmpty};
    }
    return data_[0];
  }

  // Returns a read-only reference to the first element
  Expected<const T&> front() const {
    if (empty()) {
      return Unexpected<Error>{Error::kContainerEmpty};
    }
    return data_[0];
  }

  // Returns a reference to the last element
  Expected<T&> back() {
    if (empty()) {
      return Unexpected<Error>{Error::kContainerEmpty};
    }
    return data_[size_ - 1];
  }

  // Returns a read-only reference to the last element
  Expected<const T&> back() const {
    if (empty()) {
      return Unexpected<Error>{Error::kContainerEmpty};
    }
    return data_[size_ - 1];
  }

  // Creates a new object with the provided arguments and adds it at the specified index
  template <typename... Args>
  Expected<void> emplace(size_t index, Args&&... args) {
    if (index > size_) {
      return Unexpected<Error>{Error::kArgumentOutOfRange};
    }
    if (full()) {
      return Unexpected<Error>{Error::kContainerFull};
    }
    if (index < size_) {
      ArrayMoveConstruct(BytePointer(&data_[index + 1]), &data_[index], size_ - index);
    }
    InplaceConstruct<T>(BytePointer(&data_[index]), std::forward<Args>(args)...);
    size_++;
    return Success;
  }

  // Creates a new object with the provided arguments and adds it to the end of the vector
  template <typename... Args>
  Expected<void> emplace_back(Args&&... args) {
    return emplace(size_, std::forward<Args>(args)...);
  }

  // Copies the object to the specified index
  Expected<void> insert(size_t index, const T& obj) { return emplace(index, obj); }
  // Moves the object to the specified index
  Expected<void> insert(size_t index, T&& obj) { return emplace(index, std::forward<T>(obj)); }
  // Copies the object to the end of the vector
  Expected<void> push_back(const T& obj) { return emplace_back(obj); }
  // Moves the object to the end of the vector
  Expected<void> push_back(T&& obj) { return emplace_back(std::forward<T>(obj)); }

  // Removes the object at the specified index and destroys it
  Expected<void> erase(size_t index) {
    if (index >= size_) {
      return Unexpected<Error>{Error::kArgumentOutOfRange};
    }
    if (empty()) {
      return Unexpected<Error>{Error::kContainerEmpty};
    }
    Destruct<T>(BytePointer(&data_[index]));
    size_--;
    if (index < size_) {
      ArrayMoveConstruct(BytePointer(&data_[index]), &data_[index + 1], size_ - index);
    }
    return Success;
  }

  // Removes the object at the end of the vector and destroys it
  Expected<void> pop_back() { return erase(size_ - 1); }

  // Removes all objects from the vector
  void clear() {
    while (size_ > 0) {
      Destruct<T>(BytePointer(&data_[--size_]));
    }
  }

  // Resizes the vector by removing objects from the end if shrinking
  // or by adding default objects to the end if expanding
  Expected<void> resize(size_t count) {
    if (count > capacity()) {
      return Unexpected<Error>{Error::kArgumentOutOfRange};
    }
    while (count > size_) {
      push_back(T());
    }
    while (count < size_) {
      pop_back();
    }
    return Success;
  }

  // Resizes the vector by removing objects from the end if shrinking
  // or by adding copies of the given object to the end if expanding
  Expected<void> resize(size_t count, const T& obj) {
    if (count > capacity()) {
      return Unexpected<Error>{Error::kArgumentOutOfRange};
    }
    while (count > size_) {
      push_back(obj);
    }
    while (count < size_) {
      pop_back();
    }
    return Success;
  }

 protected:
  FixedVectorBase() = default;

  // Pointer to an array of objects
  T* data_;
  // Number of objects stored
  size_t size_;
};

// Vector with stack memory allocation
template <typename T, size_t N = 0>
class FixedVector : public FixedVectorBase<T> {
 public:
  FixedVector() {
    data_ = ValuePointer<T>(pool_);
    size_ = 0;
  }
  FixedVector(const FixedVector& other) { *this = other; }
  FixedVector(FixedVector&& other) { *this = std::move(other); }
  FixedVector& operator=(const FixedVector& other) {
    data_ = ValuePointer<T>(pool_);
    size_ = other.size_;
    ArrayCopyConstruct(BytePointer(data_), other.data_, size_);
    return *this;
  }
  FixedVector& operator=(FixedVector&& other) {
    data_ = ValuePointer<T>(pool_);
    size_ = other.size_;
    ArrayMoveConstruct(BytePointer(data_), other.data_, size_);
    other.size_ = 0;
    return *this;
  }
  ~FixedVector() { FixedVectorBase<T>::clear(); }

  size_t capacity() const override { return N; }

 private:
  using FixedVectorBase<T>::data_;
  using FixedVectorBase<T>::size_;

  // Size of memory pool for stack allocation
  static constexpr size_t kPoolSize = N * sizeof(T);
  // Memory pool for storing objects on the stack
  byte pool_[kPoolSize];
};

// Vector with heap memory allocation
template <typename T>
class FixedVector<T, 0> : public FixedVectorBase<T> {
 public:
  using Error = typename FixedVectorBase<T>::Error;

  template<typename U>
  using Expected = typename FixedVectorBase<T>::template Expected<U>;
  using FixedVectorBase<T>::Success;

  FixedVector() { reset(); }
  FixedVector(const FixedVector& other) = delete;
  FixedVector(FixedVector&& other) { *this = std::move(other); }
  FixedVector& operator=(const FixedVector& other) = delete;
  FixedVector& operator=(FixedVector&& other) {
    data_ = other.data_;
    capacity_ = other.capacity_;
    size_ = other.size_;
    other.reset();
    return *this;
  }
  ~FixedVector() {
    FixedVectorBase<T>::clear();
    DeallocateArray<T>(data_);
    reset();
  }

  size_t capacity() const override { return capacity_; }

  // Copies the contents of the given vector
  // Current contents are discarded
  // Fails if given vector is larger than current capacity
  Expected<void> copy_from(const FixedVector& other) {
    if (other.size_ > capacity_) {
      return Unexpected<Error>{Error::kArgumentOutOfRange};
    }
    FixedVectorBase<T>::clear();
    size_ = other.size_;
    ArrayCopyConstruct(BytePointer(data_), other.data_, size_);
    return Success;
  }

  // Allocates memory to hold the specified number of elements
  Expected<void> reserve(size_t capacity) {
    if (capacity > capacity_) {
      T* data = AllocateArray<T>(capacity);
      if (!data) {
        return Unexpected<Error>{Error::kOutOfMemory};
      }
      ArrayMoveConstruct(BytePointer(data), data_, size_);
      DeallocateArray<T>(data_);
      data_ = data;
      capacity_ = capacity;
    }
    return Success;
  }

  // Shrinks memory allocation to fit current number of elements
  Expected<void> shrink_to_fit() {
    if (size_ < capacity_) {
      T* data = AllocateArray<T>(size_);
      if (!data) {
        return Unexpected<Error>(Error::kOutOfMemory);
      }
      ArrayMoveConstruct(BytePointer(data), data_, size_);
      DeallocateArray<T>(data_);
      data_ = data;
      capacity_ = size_;
    }
    return Success;
  }

 private:
  using FixedVectorBase<T>::data_;
  using FixedVectorBase<T>::size_;

  // Resets class members to default values
  void reset() {
    data_ = nullptr;
    capacity_ = 0;
    size_ = 0;
  }

  // Number of objects that can be stored
  size_t capacity_;
};

}  // namespace nvidia

#endif  // NVIDIA_GXF_CORE_FIXED_VECTOR_HPP_
