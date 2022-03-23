/*
Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#ifndef NVIDIA_GXF_EXTENSIONS_TENSOR_RT_TENSOR_RT_INFERENCE_HPP_
#define NVIDIA_GXF_EXTENSIONS_TENSOR_RT_TENSOR_RT_INFERENCE_HPP_

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "cuda_runtime.h"

#include "NvInfer.h"

#include "gxf/core/entity.hpp"
#include "gxf/core/gxf.h"
#include "gxf/core/parameter.hpp"
#include "gxf/cuda/cuda_stream.hpp"
#include "gxf/cuda/cuda_stream_pool.hpp"
#include "gxf/std/allocator.hpp"
#include "gxf/std/clock.hpp"
#include "gxf/std/codelet.hpp"
#include "gxf/std/parameter_parser_std.hpp"
#include "gxf/std/receiver.hpp"
#include "gxf/std/tensor.hpp"
#include "gxf/std/transmitter.hpp"

namespace nvidia {
namespace gxf {

// Logger for TensorRT to redirect logging into gxf console spew.
class TensorRTInferenceLogger : public nvinfer1::ILogger {
 public:
  void log(ILogger::Severity severity, const char* msg) throw() override;
  // Sets verbose flag for logging
  void setVerbose(bool verbose);

 private:
  bool verbose_;
};

// Loads ONNX model, takes input tensors and run inference against them with TensorRT.
// It takes input from all receivers provided and try to locate Tensor component with specified name
// on them one by one. The first occurence would be used. Only takes gpu memory tensor.
// Supports dynamic batch as first dimension.
// Requires gxf::CudaStream to run load on specific CUDA stream.
class TensorRtInference : public gxf::Codelet {
 public:
  gxf_result_t start() override;
  gxf_result_t tick() override;
  gxf_result_t stop() override;
  gxf_result_t registerInterface(gxf::Registrar* registrar) override;

 private:
  // Helper deleter to call destroy while destroying the cuda objects
  template <typename T>
  struct DeleteFunctor {
    inline void operator()(void* ptr) { reinterpret_cast<T*>(ptr)->destroy(); }
  };
  // unique_ptr using custom Delete Functor above
  template <typename T>
  using NvInferHandle = std::unique_ptr<T, DeleteFunctor<T>>;

  // To cache binding info for tensors
  typedef struct {
    int32_t index;
    uint32_t rank;
    std::string binding_name;
    gxf::PrimitiveType element_type;
    std::array<int32_t, gxf::Shape::kMaxRank> dimensions;
  } BindingInfo;
  std::unordered_map<std::string, BindingInfo> binding_infos_;

  // Converts loaded model to engine plan
  gxf::Expected<std::vector<char>> convertModelToEngine();

  gxf::Parameter<std::string> model_file_path_;
  gxf::Parameter<std::string> engine_file_path_;
  gxf::Parameter<std::string> plugins_lib_namespace_;
  gxf::Parameter<bool> force_engine_update_;
  gxf::Parameter<std::vector<std::string>> input_tensor_names_;
  gxf::Parameter<std::vector<std::string>> input_binding_names_;
  gxf::Parameter<std::vector<std::string>> output_tensor_names_;
  gxf::Parameter<std::vector<std::string>> output_binding_names_;
  gxf::Parameter<gxf::Handle<gxf::Allocator>> pool_;
  gxf::Parameter<gxf::Handle<gxf::CudaStreamPool>> cuda_stream_pool_;
  gxf::Parameter<int64_t> max_workspace_size_;
  gxf::Parameter<int64_t> dla_core_;
  gxf::Parameter<int32_t> max_batch_size_;
  gxf::Parameter<bool> enable_fp16_;
  gxf::Parameter<bool> relaxed_dimension_check_;
  gxf::Parameter<bool> verbose_;
  gxf::Parameter<gxf::Handle<gxf::Clock>> clock_;

  gxf::Parameter<std::vector<gxf::Handle<gxf::Receiver>>> rx_;
  gxf::Parameter<gxf::Handle<gxf::Transmitter>> tx_;

  // Logger instance for TensorRT
  TensorRTInferenceLogger cuda_logger_;

  NvInferHandle<nvinfer1::IExecutionContext> cuda_execution_ctx_;
  NvInferHandle<nvinfer1::ICudaEngine> cuda_engine_;

  gxf::Handle<gxf::CudaStream> cuda_stream_;
  std::vector<void*> cuda_buffers_;
  cudaStream_t cached_cuda_stream_;
  cudaEvent_t cuda_event_consumed_;
  cudaEvent_t cuda_event_done_;
};

}  // namespace gxf
}  // namespace nvidia

#endif  // NVIDIA_GXF_EXTENSIONS_TENSOR_RT_TENSOR_RT_INFERENCE_HPP_
