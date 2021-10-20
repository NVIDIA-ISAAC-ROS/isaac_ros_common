/*
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#ifndef NVIDIA_GXF_NPP_NPPI_SET_HPP
#define NVIDIA_GXF_NPP_NPPI_SET_HPP

#include <array>
#include <vector>

#include "gxf/std/allocator.hpp"
#include "gxf/std/codelet.hpp"
#include "gxf/std/parameter_parser_std.hpp"
#include "gxf/std/transmitter.hpp"

namespace nvidia {
namespace gxf {

// Creates a CUDA tensor with constant values using NPP.
class NppiSet : public Codelet {
 public:
  virtual ~NppiSet() = default;

  gxf_result_t registerInterface(Registrar* registrar) override;
  gxf_result_t start() override { return GXF_SUCCESS; }
  gxf_result_t tick() override;
  gxf_result_t stop() override { return GXF_SUCCESS; }

 private:
  Parameter<int32_t> rows_;
  Parameter<int32_t> columns_;
  Parameter<int32_t> channels_;
  Parameter<Handle<Allocator>> pool_;
  Parameter<std::vector<double>> value_;
  Parameter<Handle<Transmitter>> out_;
};

}  // namespace gxf
}  // namespace nvidia

#endif
