/*
Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#ifndef NVIDIA_GXF_TEST_EXTENSIONS_TENSOR_GENERATOR_HPP_
#define NVIDIA_GXF_TEST_EXTENSIONS_TENSOR_GENERATOR_HPP_

#include <random>
#include <string>
#include <vector>

#include "gxf/std/allocator.hpp"
#include "gxf/std/codelet.hpp"
#include "gxf/std/transmitter.hpp"

namespace nvidia {
namespace gxf {
namespace test {

// Creates a random tensor message with a timestamp
class TensorGenerator : public Codelet {
 public:
  // Type for tensor contents
  using DataType = float;

  gxf_result_t registerInterface(Registrar* registrar) override;
  gxf_result_t initialize() override { return GXF_SUCCESS; }
  gxf_result_t deinitialize() override { return GXF_SUCCESS; }

  gxf_result_t start() override { return GXF_SUCCESS; }
  gxf_result_t tick() override;
  gxf_result_t stop() override { return GXF_SUCCESS; }

 private:
  // Output message
  Parameter<Handle<Transmitter>> output_;
  // Memory allocator
  Parameter<Handle<Allocator>> allocator_;
  // Tensor shape
  Parameter<std::vector<int32_t>> shape_;
  // Tensor storage type
  Parameter<int32_t> storage_type_;
  // Flag to enable timestamps
  Parameter<bool> enable_timestamps_;
  // Name of the tensor in the generated message
  Parameter<std::string> tensor_name_;
  // Name of the timestamp in the generated message
  Parameter<std::string> timestamp_name_;
  // Random number generator
  std::default_random_engine generator_;
};

}  // namespace test
}  // namespace gxf
}  // namespace nvidia

#endif  // NVIDIA_GXF_TEST_EXTENSIONS_TENSOR_GENERATOR_HPP_
