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
#ifndef NVIDIA_COMMON_MEMORY_UTILS_HPP_
#define NVIDIA_COMMON_MEMORY_UTILS_HPP_

#include <cstring>
#include <memory>
#include <new>
#include <type_traits>
#include <utility>

#include "common/byte.hpp"

namespace nvidia {

// Convert a raw byte pointer `src` to a concrete type T
template <typename T>
T* ValuePointer(byte* src) {
  return reinterpret_cast<T*>(src);
}

// Convert a const raw byte pointer `src` to a concrete type T
template <typename T>
const T* ValuePointer(const byte* src) {
  return reinterpret_cast<const T*>(src);
}

// Convert a type T pointer `src` to a raw byte pointer
template <typename T>
byte* BytePointer(T* src) {
  return reinterpret_cast<byte*>(src);
}

// Convert a const type T pointer `src` to a raw byte pointer
template <typename T>
const byte* BytePointer(const T* src) {
  return reinterpret_cast<const byte*>(src);
}

// Call the destructor for type T on the object located at `src`.
template <typename T>
void Destruct(byte* src) {
  reinterpret_cast<T*>(src)->~T();
}

// Construct an object of type T with the provided arguments into the memory at `dst` with
// placement new operator.
template <typename T, typename... Args>
T* InplaceConstruct(byte* dst, Args&&... args) {
  return new (dst) T{std::forward<Args>(args)...};
}

// Move construct an object of type T into the memory at `dst` with placement new operator.
template <typename T>
T* InplaceMoveConstruct(byte* dst, T&& other) {
  return new (dst) T{std::forward<T>(other)};
}

// Copy construct an object of type T into the memory at `dst` with placement new operator.
template <typename T>
T* InplaceCopyConstruct(byte* dst, const T& other) {
  return new (dst) T{other};
}

// Move constructs an array of objects of type T into the memory at `dst`
// Used if object is movable
template <typename T, std::enable_if_t<std::is_move_constructible<T>::value>* = nullptr>
void ArrayMoveConstruct(byte* dst, T* src, size_t count) {
  // Reverse move direction to handle overlapping memory segments
  const bool reverse = dst > BytePointer(src) && dst < BytePointer(src + count);
  for (size_t i = 0; i < count; i++) {
    const size_t index = reverse ? count - 1 - i : i;
    InplaceMoveConstruct<T>(dst + sizeof(T) * index, std::move(src[index]));
  }
}

// Move constructs an array of objects of type T into the memory at `dst`
// Used if object is not movable but trivially copyable
template <typename T, std::enable_if_t<!std::is_move_constructible<T>::value &&
                                       std::is_trivially_copyable<T>::value>* = nullptr>
void ArrayMoveConstruct(byte* dst, T* src, size_t count) {
  std::memmove(dst, src, sizeof(T) * count);
}

// Move constructs an array of objects of type T into the memory at `dst`
// Used if object is not movable and not trivially copyable
template <typename T, std::enable_if_t<!std::is_move_constructible<T>::value &&
                                       !std::is_trivially_copyable<T>::value>* = nullptr>
void ArrayMoveConstruct(byte* dst, T* src, size_t count) {
  // Reverse move direction to handle overlapping memory segments
  const bool reverse = dst > BytePointer(src) && dst < BytePointer(src + count);
  for (size_t i = 0; i < count; i++) {
    const size_t index = reverse ? count - 1 - i : i;
    InplaceCopyConstruct<T>(dst + sizeof(T) * index, src[index]);
  }
}

// Copy constructs an array of objects of type T into the memory at `dst`
// Used if object is trivially copyable
template <typename T, std::enable_if_t<std::is_trivially_copyable<T>::value>* = nullptr>
void ArrayCopyConstruct(byte* dst, T* src, size_t count) {
  std::memmove(dst, src, sizeof(T) * count);
}

// Copy constructs an array of objects of type T into the memory at `dst`
// Used if object is not trivially copyable
template <typename T, std::enable_if_t<!std::is_trivially_copyable<T>::value>* = nullptr>
void ArrayCopyConstruct(byte* dst, T* src, size_t count) {
  // Reverse move direction to handle overlapping memory segments
  const bool reverse = dst > BytePointer(src) && dst < BytePointer(src + count);
  for (size_t i = 0; i < count; i++) {
    const size_t index = reverse ? count - 1 - i : i;
    InplaceCopyConstruct<T>(dst + sizeof(T) * index, src[index]);
  }
}

// Allocates an array with `size` number of objects of type T
// Does not call object constructor
template <typename T>
T* AllocateArray(size_t size) {
  return static_cast<T*>(::operator new(size * sizeof(T), std::nothrow));
}

// Deallocates the array `data` of objects of type T
// Does not call object destructor
template <typename T>
void DeallocateArray(T* data) {
  ::operator delete(data);
}

// Construct an object of type T with the provided arguments and returns a unique pointer to it
// Null pointer is returned on error
template <typename T, typename... Args>
std::unique_ptr<T> MakeUniqueNoThrow(Args&&... args) {
  return std::unique_ptr<T>(new(std::nothrow) T{std::forward<Args>(args)...});
}

// Construct an object of type T with the provided arguments and returns a shared pointer to it
// Null pointer is returned on error
template <typename T, typename... Args>
std::shared_ptr<T> MakeSharedNoThrow(Args&&... args) {
  return std::shared_ptr<T>(new(std::nothrow) T{std::forward<Args>(args)...});
}

}  // namespace nvidia

#endif  // NVIDIA_COMMON_MEMORY_UTILS_HPP_
