/*
Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#ifndef NVIDIA_GXF_CUDA_CUDA_STREAM_ID_HPP_
#define NVIDIA_GXF_CUDA_CUDA_STREAM_ID_HPP_

#include "gxf/core/gxf.h"


namespace nvidia {
namespace gxf {

// The Structure indicates cuda stream component ID.
// Message entity carrying CudaStreamId indicates that Tensors will be or
// has been proccessed by corresponding cuda stream. The handle could
// be deduced by Handle<CudaStream>::Create(context, stream_cid).
struct CudaStreamId {
  // component id of CudaStream
  gxf_uid_t stream_cid;
};


}  // namespace gxf
}  // namespace nvidia

#endif  // NVIDIA_GXF_CUDA_CUDA_STREAM_ID_HPP_
