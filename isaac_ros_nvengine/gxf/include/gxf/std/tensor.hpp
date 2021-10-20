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
#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "core/byte.hpp"
#include "gxf/core/component.hpp"
#include "gxf/core/expected.hpp"
#include "gxf/std/allocator.hpp"
#include "gxf/std/memory_buffer.hpp"

namespace nvidia {
namespace gxf {

// Type of parameters and other primitives
enum class PrimitiveType {
  kCustom,
  kInt8,
  kUnsigned8,
  kInt16,
  kUnsigned16,
  kInt32,
  kUnsigned32,
  kInt64,
  kUnsigned64,
  kFloat32,
  kFloat64,
};

// Returns the size of each element of specific PrimitiveType as number of bytes.
// Returns 0 for kCustom.
uint64_t PrimitiveTypeSize(PrimitiveType primitive);

template <typename T>
struct PrimitiveTypeTraits;

#define GXF_PRIMITIVE_TYPE_TRAITS(TYPE, ENUM)                                                      \
  template <> struct PrimitiveTypeTraits<TYPE> {                                                   \
    static constexpr PrimitiveType value = PrimitiveType::ENUM;                                    \
    static constexpr size_t size = sizeof(TYPE);                                                   \
  };                                                                                               \

GXF_PRIMITIVE_TYPE_TRAITS(int8_t, kInt8);
GXF_PRIMITIVE_TYPE_TRAITS(uint8_t, kUnsigned8);
GXF_PRIMITIVE_TYPE_TRAITS(int16_t, kInt16);
GXF_PRIMITIVE_TYPE_TRAITS(uint16_t, kUnsigned16);
GXF_PRIMITIVE_TYPE_TRAITS(int32_t, kInt32);
GXF_PRIMITIVE_TYPE_TRAITS(uint32_t, kUnsigned32);
GXF_PRIMITIVE_TYPE_TRAITS(int64_t, kInt64);
GXF_PRIMITIVE_TYPE_TRAITS(uint64_t, kUnsigned64);
GXF_PRIMITIVE_TYPE_TRAITS(float, kFloat32);
GXF_PRIMITIVE_TYPE_TRAITS(double, kFloat64);

// Type to hold the shape of a tensor
class Shape {
 public:
  // The maximum possible rank of the tensor.
  static constexpr uint32_t kMaxRank = 8;

  // Intializes an empty rank-0 tensor.
  Shape() : rank_(0) {}

  // Initializes a shape object with the given dimensions.
  Shape(std::initializer_list<int32_t> dimensions)
      : rank_(0) {
    for (int32_t dimension : dimensions) {
      if (rank_ == kMaxRank) {
        return;  // FIXME(v1)
      }
      dimensions_[rank_++] = dimension;
    }
  }

  // Creates shape from array
  Shape(const std::array<int32_t, Shape::kMaxRank>& dims, uint32_t rank)
    : rank_(rank), dimensions_(dims) {
  }

  // Creates shape from array with correct rank
  template <size_t N>
  Shape(const std::array<int32_t, N>& dims) : rank_(N) {
    static_assert(N < kMaxRank, "Invalid rank");
    for (size_t i = 0; i < N; i++) {
      dimensions_[i] = dims[i];
    }
  }

  // The rank of the tensor
  uint32_t rank() const { return rank_; }

  // The total number of elements in the tensor. Note: this is not the same as the number of bytes
  // required in memory.
  uint64_t size() const {
    uint64_t element_count = 1;
    for (size_t i = 0; i < rank_; i++) {
      element_count *= dimensions_[i];
    }
    return rank_ == 0 ? 0 : element_count;
  }

  // Gets the i-th dimension of the tensor.
  // Special cases:
  //  If the rank is 0 the function always returns 0.
  //  If 'index' is greater or equal than the rank the function returns 1.
  int32_t dimension(uint32_t index) const {
    if (rank_ == 0) {
      return 0;
    } else if (index >= rank_) {
      return 1;
    } else {
      return dimensions_[index];
    }
  }

  bool operator== (const Shape& other) const {
    if (rank_ != other.rank_) { return false; }
    for (uint32_t i = 0; i < rank_; ++i) {
      if (dimensions_[i] != other.dimensions_[i]) { return false; }
    }
    return true;
  }

  bool operator!= (const Shape& other) const {
    return !(*this == other);
  }

  // Check whether shape is valid
  bool valid() const {
    for (uint32_t i = 0; i < rank_; ++i) {
      if (dimensions_[i] <= 0) { return false; }
    }
    return true;
  }

 private:
  uint32_t rank_ = 0;
  std::array<int32_t, kMaxRank> dimensions_;
};

// A component which holds a single tensor. Multiple tensors can be added to one
// entity to create a map of tensors. The component name can be used as key.
class Tensor {
 public:
  typedef std::array<uint64_t, Shape::kMaxRank> stride_array_t;

  Tensor() = default;

  ~Tensor() {
    memory_buffer_.freeBuffer();  // FIXME(V2) error code?
  }

  Tensor(const Tensor&) = delete;

  Tensor(Tensor&& other) {
    *this = std::move(other);
  }

  Tensor& operator=(const Tensor&) = delete;

  Tensor& operator=(Tensor&& other) {
    shape_ = other.shape_;
    element_count_ = other.element_count_;
    element_type_ = other.element_type_;
    bytes_per_element_ = other.bytes_per_element_;
    strides_ = std::move(other.strides_);
    memory_buffer_ = std::move(other.memory_buffer_);

    return *this;
  }

  // The type of memory where the tensor data is stored.
  MemoryStorageType storage_type() const { return memory_buffer_.storage_type(); }

  // The shape of the dimensions holds the rank and the dimensions.
  const Shape& shape() const { return shape_; }

  // The rank of the tensor.
  uint32_t rank() const { return shape_.rank(); }

  // The scalar type of elements stored in the tensor
  PrimitiveType element_type() const { return element_type_; }

  // Number of bytes stored per element
  uint64_t bytes_per_element() const { return bytes_per_element_; }

  // Total number of elements stored in the tensor.
  uint64_t element_count() const { return element_count_; }

  // Size of tensor contents in bytes
  size_t size() const { return memory_buffer_.size(); }

  // Raw pointer to the first byte of elements stored in the tensor.
  byte* pointer() const { return memory_buffer_.pointer(); }

  // Gets a pointer to the first element stored in this tensor. Requested type must match the
  // tensor element type.
  template <typename T>
  Expected<T*> data() {
    if (element_type_ != PrimitiveType::kCustom && PrimitiveTypeTraits<T>::value != element_type_) {
      return Unexpected{GXF_INVALID_DATA_FORMAT};
    }
    return reinterpret_cast<T*>(memory_buffer_.pointer());
  }

  // Gets a pointer to the first element stored in this tensor. Requested type must match the
  // tensor element type.
  template <typename T>
  Expected<const T*> data() const {
    if (element_type_ != PrimitiveType::kCustom && PrimitiveTypeTraits<T>::value != element_type_) {
      return Unexpected{GXF_INVALID_DATA_FORMAT};
    }
    return reinterpret_cast<const T*>(memory_buffer_.pointer());
  }

  // Changes the shape and type of the tensor. Uses a primitive type and dense memory layot.
  // Memory will be allocated with the given allocator.
  template <typename T>
  Expected<void> reshape(const Shape& shape, MemoryStorageType storage_type,
                         Handle<Allocator> allocator) {
    return reshapeCustom(shape, PrimitiveTypeTraits<T>::value, PrimitiveTypeTraits<T>::size,
                         Unexpected{GXF_UNINITIALIZED_VALUE}, storage_type, allocator);
  }
  // Changes the shape and type of the tensor. Memory will be allocated with the given allocator
  // strides: The number of bytes that each slide takes for each dimension (alignment).
  //          Use ComputeStrides() to calculate it.
  Expected<void> reshapeCustom(const Shape& shape,
                               PrimitiveType element_type, uint64_t bytes_per_element,
                               Expected<stride_array_t> strides,
                               MemoryStorageType storage_type, Handle<Allocator> allocator);

  // Type of the callback function to release memory passed to the tensor using the
  // wrapMemory method
  using release_function_t = std::function<Expected<void> (void* pointer)>;

  // Wrap existing memory inside the tensor. A callback function of type release_function_t
  // may be passed that will be called when the Tensor wants to release the memory.
  Expected<void> wrapMemory(const Shape& shape,
                            PrimitiveType element_type, uint64_t bytes_per_element,
                            Expected<stride_array_t> strides,
                            MemoryStorageType storage_type, void* pointer,
                            release_function_t release_func);

  // The size of data in bytes
  uint64_t bytes_size() { return shape_.dimension(0) * strides_[0]; }

  // The stride of specified rank in bytes
  uint64_t stride(uint32_t index) const {
    if (index >= shape_.rank()) {
      return 0;
    }
    return strides_[index];
  }

 private:
  Shape shape_;
  uint64_t element_count_ = 0;
  PrimitiveType element_type_ = PrimitiveType::kUnsigned8;
  uint64_t bytes_per_element_ = 1;
  stride_array_t strides_;
  MemoryBuffer memory_buffer_;
};

// Helper function to compute strides from Tensor shape, element size and non-trivial
// alignment step size for row dimension.
// The third rank from the end is assumed to be the row dimension.
Expected<Tensor::stride_array_t> ComputeRowStrides(const Shape& shape, uint32_t row_step_size,
                                                   const uint32_t bytes_per_element);

// Helper function to compute trivial strides from Tensor shape and element size
Tensor::stride_array_t ComputeTrivialStrides(const Shape& shape, const uint32_t bytes_per_element);

// Helper function to compute strides from steps (minimal number of bytes per slice on each rank)
Tensor::stride_array_t ComputeStrides(const Shape& shape,
                                      const Tensor::stride_array_t& stride_steps);

// Type to description a tensor used by 'CreateTensorMap'
struct TensorDescription {
  std::string name;
  MemoryStorageType storage_type;
  Shape shape;
  PrimitiveType element_type;
  uint64_t bytes_per_element;
  // array providing number of bytes for each slice on each rank
  Expected<Tensor::stride_array_t> strides = Unexpected{GXF_UNINITIALIZED_VALUE};
};

// Creates a new entity with a collection of named tensors
Expected<Entity> CreateTensorMap(gxf_context_t context, Handle<Allocator> pool,
                                 std::initializer_list<TensorDescription> descriptions,
                                 bool activate = true);

}  // namespace gxf
}  // namespace nvidia
