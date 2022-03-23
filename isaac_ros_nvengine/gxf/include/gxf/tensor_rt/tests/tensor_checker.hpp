/*
Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#ifndef NVIDIA_GXF_EXTENSIONS_TENSOR_RT_TEST_TENSOR_CHECKER_HPP_
#define NVIDIA_GXF_EXTENSIONS_TENSOR_RT_TEST_TENSOR_CHECKER_HPP_

#include <string>
#include <vector>

#include "gxf/std/allocator.hpp"
#include "gxf/std/codelet.hpp"
#include "gxf/std/receiver.hpp"
#include "gxf/std/tensor.hpp"

namespace nvidia {
namespace gxf {

// Verifies the content of result tensor for TensorRTInference codelet.
// Checks expected dimensions, tensor name, and where the max value presents.
class TensorChecker : public gxf::Codelet {
 public:
  gxf_result_t start() override;

  gxf_result_t tick() override;

  gxf_result_t stop() override;

  gxf_result_t registerInterface(gxf::Registrar* registrar) override;

 private:
  gxf::Parameter<gxf::Handle<gxf::Receiver>> rx_;
  gxf::Parameter<std::string> tensor_name_;
  gxf::Parameter<std::vector<int32_t>> dimensions_;
  gxf::Parameter<uint64_t> max_element_offset_;
};

}  // namespace gxf
}  // namespace nvidia

#endif  // NVIDIA_GXF_EXTENSIONS_TENSOR_RT_TEST_TENSOR_CHECKER_HPP_
