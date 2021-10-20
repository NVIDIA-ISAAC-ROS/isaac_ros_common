/*
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#ifndef NVIDIA_GXF_NPP_NPPI_MUL_C_HPP
#define NVIDIA_GXF_NPP_NPPI_MUL_C_HPP

#include <vector>

#include "gxf/std/allocator.hpp"
#include "gxf/std/codelet.hpp"
#include "gxf/std/parameter_parser_std.hpp"
#include "gxf/std/receiver.hpp"
#include "gxf/std/transmitter.hpp"

namespace nvidia {
namespace gxf {

// Multiplies a CUDA tensor with a constant factor using NPP.
class NppiMulC : public Codelet {
 public:
  virtual ~NppiMulC() = default;

  gxf_result_t registerInterface(Registrar* registrar) override;
  gxf_result_t start() override { return GXF_SUCCESS; }
  gxf_result_t tick() override;
  gxf_result_t stop() override { return GXF_SUCCESS; }

 private:
  Parameter<Handle<Receiver>> in_;
  Parameter<std::vector<double>> factor_;
  Parameter<Handle<Allocator>> pool_;
  Parameter<Handle<Transmitter>> out_;
};

}  // namespace gxf
}  // namespace nvidia

#endif
