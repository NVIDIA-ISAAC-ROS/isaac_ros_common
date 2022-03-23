/*
Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#ifndef NVIDIA_GXF_SERIALIZATION_ENDPOINT_HPP_
#define NVIDIA_GXF_SERIALIZATION_ENDPOINT_HPP_

#include "gxf/core/component.hpp"

namespace nvidia {
namespace gxf {

// Interface for exchanging data external to an application graph
class Endpoint : public Component {
 public:
  virtual ~Endpoint() = default;

  // Writes data to the endpoint and returns the number of bytes written
  virtual gxf_result_t write_abi(const void* data, size_t size, size_t* bytes_written) = 0;
  // Reads data from the endpoint and returns the number of bytes read
  virtual gxf_result_t read_abi(void* data, size_t size, size_t* bytes_read) = 0;

  // C++ API wrappers
  Expected<size_t> write(const void* data, size_t size);
  Expected<size_t> read(void* data, size_t size);

  // Writes an object of type T to the endpoint
  template <typename T>
  Expected<size_t> writeTrivialType(const T* object) { return write(object, sizeof(T)); }

  // Reads an object of type T from the endpoint
  template <typename T>
  Expected<size_t> readTrivialType(T* object) { return read(object, sizeof(T)); }
};

}  // namespace gxf
}  // namespace nvidia

#endif  // NVIDIA_GXF_SERIALIZATION_ENDPOINT_HPP_
