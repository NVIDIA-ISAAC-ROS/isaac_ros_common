/*
Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#ifndef NVIDIA_GXF_MULTIMEDIA_AUDIO_HPP_
#define NVIDIA_GXF_MULTIMEDIA_AUDIO_HPP_

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "gxf/core/expected.hpp"
#include "gxf/core/handle.hpp"
#include "gxf/std/allocator.hpp"
#include "gxf/std/memory_buffer.hpp"

namespace nvidia {
namespace gxf {

// Supported raw audio media types
enum class AudioFormat {
  GXF_AUDIO_FORMAT_CUSTOM = 0,
  GXF_AUDIO_FORMAT_S16LE,  // 16-bit signed PCM audio
  GXF_AUDIO_FORMAT_F32LE,  // 32-bit floating-point audio
};

// Supported channel layouts
enum class AudioLayout {
  GXF_AUDIO_LAYOUT_CUSTOM = 0,
  GXF_AUDIO_LAYOUT_INTERLEAVED,      // Data from all the channels to be interleaved LRLRLR
  GXF_AUDIO_LAYOUT_NON_INTERLEAVED,  // Data from all the channels not to be interleaved LLLRRR
};

template <AudioFormat>
struct AudioTypeTraits;

#define GXF_AUDIO_TYPE_TRAITS(ENUM, WIDTH)                        \
  template <>                                                     \
  struct AudioTypeTraits<AudioFormat::ENUM> {                     \
    static constexpr const char* name = #ENUM;                    \
    static constexpr const AudioFormat value = AudioFormat::ENUM; \
    static constexpr const int8_t width = WIDTH;                  \
  };

GXF_AUDIO_TYPE_TRAITS(GXF_AUDIO_FORMAT_CUSTOM, 0);
GXF_AUDIO_TYPE_TRAITS(GXF_AUDIO_FORMAT_S16LE, 16);
GXF_AUDIO_TYPE_TRAITS(GXF_AUDIO_FORMAT_F32LE, 32);

// Descriptor for an AudioBuffer
struct AudioBufferInfo {
  // Number of channels in an audio frame
  uint32_t channels;
  // Number of samples in an audio frame
  uint32_t samples;
  // sampling rate in Hz
  uint32_t sampling_rate;
  // Number of bytes required per sample
  uint32_t bytes_per_sample;
  // AudioFormat of an audio frame
  AudioFormat audio_format;
  // AudioLayout of an audio frame
  AudioLayout audio_layout;
};

// A media data type which stores information corresponding to an audio frame
// resize(...) function is used to allocate memory for the audio frame based on
// the audio frame info
class AudioBuffer {
 public:
  AudioBuffer() = default;

  ~AudioBuffer() { memory_buffer_.freeBuffer(); }

  AudioBuffer(const AudioBuffer&) = delete;

  AudioBuffer(AudioBuffer&& other) { *this = std::move(other); }

  AudioBuffer& operator=(const AudioBuffer&) = delete;

  AudioBuffer& operator=(AudioBuffer&& other) {
    buffer_info_ = other.buffer_info_;
    memory_buffer_ = std::move(other.memory_buffer_);

    return *this;
  }

  template <AudioFormat A>
  Expected<void> resize(uint32_t channels, uint32_t samples, uint32_t sampling_rate,
                        AudioLayout layout, MemoryStorageType storage_type,
                        Handle<Allocator> allocator) {
    AudioTypeTraits<A> audio_type;
    uint32_t bytes_per_sample = std::ceil(audio_type.width / 8);
    AudioBufferInfo buffer_info{channels,         samples,          sampling_rate,
                                bytes_per_sample, audio_type.value, layout};
    return resizeCustom(buffer_info, storage_type, allocator);
  }

  // Type of the callback function to release memory passed to the AudioFrame using the
  // wrapMemory method
  using release_function_t = MemoryBuffer::release_function_t;

  // Wrap existing memory inside the AudioBuffer. A callback function of type release_function_t
  // may be passed that will be called when the AudioBuffer wants to release the memory.
  Expected<void> wrapMemory(AudioBufferInfo buffer_info, uint64_t size,
                            MemoryStorageType storage_type, void* pointer,
                            release_function_t release_func);

  // AudioBufferInfo of the AudioBuffer
  AudioBufferInfo audio_buffer_info() const { return buffer_info_; }

  // The type of memory where the frame data is stored.
  MemoryStorageType storage_type() const { return memory_buffer_.storage_type(); }

  // Size of the audio frame in bytes
  uint64_t size() const { return memory_buffer_.size(); }

  // Raw pointer to the first byte of the audio frame
  byte* pointer() const { return memory_buffer_.pointer(); }

  // Resizes the audio frame and allocates the corresponding memory with the allocator provided
  // Any data previously stored in the frame would be freed
  Expected<void> resizeCustom(AudioBufferInfo buffer_info, MemoryStorageType storage_type,
                              Handle<Allocator> allocator);

 private:
  AudioBufferInfo buffer_info_;
  MemoryBuffer memory_buffer_;
};

}  // namespace gxf
}  // namespace nvidia

#endif  // NVIDIA_GXF_MULTIMEDIA_AUDIO_HPP_
