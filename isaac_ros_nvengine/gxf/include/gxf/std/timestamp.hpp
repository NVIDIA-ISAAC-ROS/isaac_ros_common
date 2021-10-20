/*
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#ifndef NVIDIA_GXF_STD_TIMESTAMP_HPP
#define NVIDIA_GXF_STD_TIMESTAMP_HPP

#include <cstdint>

namespace nvidia {
namespace gxf {

// Contains timing information for the data in a message. All times are relative to the global GXF
// clock and in nanoseconds.
struct Timestamp {
  // The timestamp in nanoseconds at which the message was published into the system.
  int64_t pubtime;
  // The timestamp in nanoseconds at the message was acquired. This usually refers to the timestamp
  // of the original sensor data which created the message.
  int64_t acqtime;
};

}  // namespace gxf
}  // namespace nvidia

#endif  // NVIDIA_GXF_STD_TIMESTAMP_HPP
