/*
Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#ifndef NVIDIA_GXF_SERIALIZATION_SERIALIZATION_BUFFER_HPP_
#define NVIDIA_GXF_SERIALIZATION_SERIALIZATION_BUFFER_HPP_

#include <mutex>

#include "gxf/serialization/endpoint.hpp"
#include "gxf/std/allocator.hpp"
#include "gxf/std/memory_buffer.hpp"

namespace nvidia {
namespace gxf {

// Buffer to hold serialized data
class SerializationBuffer : public Endpoint {
 public:
  gxf_result_t registerInterface(Registrar* registrar) override;
  gxf_result_t initialize() override;
  gxf_result_t deinitialize() override { return ToResultCode(buffer_.freeBuffer()); }

  gxf_result_t write_abi(const void* data, size_t size, size_t* bytes_written) override;
  gxf_result_t read_abi(void* data, size_t size, size_t* bytes_read) override;

  // Returns a read-only pointer to buffer data
  const byte* data() const { return buffer_.pointer(); }
  // Returns the capacity of the buffer
  size_t capacity() const { return buffer_.size(); }
  // Returns the number of bytes written to the buffer
  size_t size() const;
  // Resets buffer for sequential access
  void reset();

 private:
  Parameter<Handle<Allocator>> allocator_;
  Parameter<size_t> buffer_size_;

  // Data buffer
  MemoryBuffer buffer_;
  // Offset for sequential writes
  size_t write_offset_;
  // Offset for sequential reads
  size_t read_offset_;
  // Mutex to guard concurrent access
  mutable std::mutex mutex_;
};

}  // namespace gxf
}  // namespace nvidia

#endif  // NVIDIA_GXF_SERIALIZATION_SERIALIZATION_BUFFER_HPP_
