/*
Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#ifndef NVIDIA_GXF_CUDA_TESTS_TEST_CUDA_HELPER_HPP
#define NVIDIA_GXF_CUDA_TESTS_TEST_CUDA_HELPER_HPP

#include <cublas_v2.h>
#include <limits>
#include <numeric>
#include <string>
#include <utility>

#include "common/assert.hpp"
#include "gxf/cuda/cuda_common.hpp"
#include "gxf/cuda/cuda_event.hpp"
#include "gxf/cuda/cuda_stream.hpp"
#include "gxf/cuda/cuda_stream_id.hpp"
#include "gxf/cuda/cuda_stream_pool.hpp"
#include "gxf/std/allocator.hpp"
#include "gxf/std/codelet.hpp"
#include "gxf/std/receiver.hpp"
#include "gxf/std/tensor.hpp"
#include "gxf/std/transmitter.hpp"

#define CHECK_CUBLUS_ERROR(cu_result, fmt, ...)                         \
  do {                                                                  \
    cublasStatus_t err = (cu_result);                                   \
    if (err != CUBLAS_STATUS_SUCCESS) {                                 \
      GXF_LOG_ERROR(fmt ", cublas_error: %d", ##__VA_ARGS__, (int)err); \
      return Unexpected{GXF_FAILURE};                                   \
    }                                                                   \
  } while (0)

namespace nvidia {
namespace gxf {
namespace test {
namespace cuda {

constexpr static const char* kStreamName0 = "CudaStream0";
constexpr static int kDefaultDevId = 0;
static const Shape kInitTensorShape{1024, 2048};

// The base class with cuda stream utils
class StreamBasedOps : public Codelet {
 public:
  static Expected<Handle<CudaStream>> getStream(Entity& message) {
    auto stream_id = message.get<CudaStreamId>();
    GXF_ASSERT(stream_id, "failed to find cudastreamid");
    auto stream =
        Handle<CudaStream>::Create(stream_id.value().context(), stream_id.value()->stream_cid);
    GXF_ASSERT(stream, "create cudastream from cid failed");
    GXF_ASSERT(stream.value(), "cudastream handle is null");
    return stream;
  }

  static Expected<void> addStream(Entity& message, Handle<CudaStream>& stream,
                                  const char* name = nullptr) {
    auto stream_id = message.add<CudaStreamId>(name);
    GXF_ASSERT(stream_id, "failed to add cudastreamid");
    stream_id.value()->stream_cid = stream.cid();
    GXF_ASSERT(stream_id.value()->stream_cid != kNullUid, "stream_cid is null");
    return Success;
  }

  static Expected<Handle<Tensor>> addTensor(Entity& message, Handle<Allocator> pool,
                                            const TensorDescription& description) {
    GXF_ASSERT(pool, "pool is not set");
    auto tensor = message.add<Tensor>(description.name.c_str());
    GXF_ASSERT(tensor, "failed to add message tensor");
    const uint64_t bytes_per_element = description.element_type == PrimitiveType::kCustom
                                           ? description.bytes_per_element
                                           : PrimitiveTypeSize(description.element_type);

    auto result = tensor.value()->reshapeCustom(description.shape, description.element_type,
                                                bytes_per_element, description.strides,
                                                description.storage_type, pool);
    GXF_ASSERT(result, "reshape tensor:%s failed", description.name.c_str());
    return tensor;
  }

  Expected<Handle<CudaEvent>> addNewEvent(Entity& message, const char* name = nullptr) const {
    GXF_ASSERT(opsEvent(), "cuda steram ops event is not initialized");

    auto mabe_event = message.add<CudaEvent>(name);
    GXF_ASSERT(mabe_event, "failed to add cudaevent");
    auto& event = mabe_event.value();
    auto ret = event->initWithEvent(ops_event_->event().value(), ops_event_->dev_id());
    GXF_ASSERT(ret, "failed to init with event into message");
    GXF_ASSERT(event->event(), "stream_cid is null");
    return mabe_event;
  }

  Expected<void> initOpsEvent() {
    if (ops_event_) { return Success; }
    auto ops_event_entity = Entity::New(context());
    GXF_ASSERT(ops_event_entity, "New event entity failed");
    ops_event_holder_ = ops_event_entity.value();
    auto maybe_event = ops_event_holder_.add<CudaEvent>("ops_event");
    GXF_ASSERT(maybe_event, "failed to init ops cudaevent");
    ops_event_ = maybe_event.value();
    auto ret = ops_event_->init(0, kDefaultDevId);
    if (!ret) {
      GXF_LOG_ERROR("failed to init Ops event");
      return ForwardError(ret);
    }
    return ret;
  }

  const Handle<CudaEvent>& opsEvent() const { return ops_event_; }

 private:
  Handle<CudaEvent> ops_event_;
  Entity ops_event_holder_;
};

// Generate cuda tensor map with cudastreams for transmitter cuda_tx,
// Generate host tensor map for transmitter host_tx
class StreamTensorGenerator : public StreamBasedOps {
 public:
  gxf_result_t initialize() override {
    GXF_ASSERT(stream_pool_.get(), "stream pool is not set");
    auto stream = stream_pool_->allocateStream();
    GXF_ASSERT(stream, "allocating stream failed");
    stream_ = std::move(stream.value());
    GXF_ASSERT(stream_->stream(), "allocated stream is not initialized.");
    return GXF_SUCCESS;
  }
  gxf_result_t start() override { return GXF_SUCCESS; }

  gxf_result_t tick() override {
    Expected<Entity> maybe_dev_msg = Entity::New(context());
    GXF_ASSERT(maybe_dev_msg, "New dev message failed");
    auto& dev_msg = maybe_dev_msg.value();

    Expected<Entity> maybe_host_msg = Entity::New(context());
    GXF_ASSERT(maybe_host_msg, "New host message failed");
    auto& host_msg = maybe_host_msg.value();

    auto ret = createTensors(dev_msg, host_msg);
    GXF_ASSERT(ret, "creating tensors failed");
    GXF_ASSERT(stream_, "stream is not allocated");

    ret = addStream(dev_msg, stream_, kStreamName0);
    GXF_ASSERT(ret, "stream tensor generator adding stream failed");

    ret = cuda_tx_->publish(dev_msg);
    if (!ret) {
      GXF_LOG_ERROR("stream tensor generator publishing cuda tensors failed");
      return ToResultCode(ret);
    }

    if (host_tx_.get()) {
      ret = host_tx_->publish(host_msg);
      if (!ret) {
        GXF_LOG_ERROR("stream tensor generator publishing cuda tensors failed");
        return ToResultCode(ret);
      }
    }

    return GXF_SUCCESS;
  }

  gxf_result_t stop() override { return GXF_SUCCESS; }

  gxf_result_t registerInterface(Registrar* registrar) override {
    Expected<void> result;
    result &= registrar->parameter(cuda_tx_, "cuda_tx", "transmitter of cuda tensors", "");
    result &= registrar->parameter(host_tx_, "host_tx", "transmitter of host tensors", "",
                                   Handle<Transmitter>());
    result &= registrar->parameter(cuda_tensor_pool_, "cuda_tensor_pool", "Cuda Tensor Pool", "");
    result &= registrar->parameter(host_tensor_pool_, "host_tensor_pool", "Host Tensor Pool", "");
    result &= registrar->parameter(stream_pool_, "stream_pool", "Cuda Stream Pool", "");
    return ToResultCode(result);
  }

 private:
  Expected<void> createTensors(Entity& dev_msg, Entity& host_msg) {
    Shape shape = kInitTensorShape;
    for (size_t i = 0; i < 2; ++i) {
      TensorDescription dev_desc{"cuda_tensor", MemoryStorageType::kDevice, shape,
                                 PrimitiveType::kFloat32};
      auto cuda_tensor_ret = addTensor(dev_msg, cuda_tensor_pool_, dev_desc);
      GXF_ASSERT(cuda_tensor_ret, "Generator dev message adding tensor failed.");
      auto& cuda_tensor = cuda_tensor_ret.value();
      Expected<float*> cuda_data = cuda_tensor->data<float>();
      if (!cuda_data) { return ForwardError(cuda_data); }

      TensorDescription host_desc{"host_tensor", MemoryStorageType::kHost, shape,
                                  PrimitiveType::kFloat32};
      auto host_tensor_ret = addTensor(host_msg, host_tensor_pool_, host_desc);
      GXF_ASSERT(host_tensor_ret, "Generator host message adding tensor failed.");
      auto& host_tensor = host_tensor_ret.value();
      Expected<float*> host_data = host_tensor->data<float>();
      if (!host_data) { return ForwardError(host_data); }

      for (size_t j = 0; j < shape.size(); ++j) {
        host_data.value()[j] = (j % 100) + 1.0f;
      }

      cudaError_t error = cudaMemcpy(cuda_data.value(), host_data.value(), cuda_tensor->size(),
                                     cudaMemcpyHostToDevice);
      CHECK_CUDA_ERROR(error, "StreamTensorGenerator cuda memory cpy H2D failed.");
    }

    return Success;
  }

  Parameter<Handle<Transmitter>> cuda_tx_;
  Parameter<Handle<Transmitter>> host_tx_;
  Parameter<Handle<Allocator>> cuda_tensor_pool_;
  Parameter<Handle<Allocator>> host_tensor_pool_;
  Parameter<Handle<CudaStreamPool>> stream_pool_;

  Handle<CudaStream> stream_;
};

// Dot product execution base class
class DotProductExe {
 private:
  Handle<Receiver> rx_;
  Handle<Transmitter> tx_;
  Handle<Allocator> tensor_pool_;

 public:
  DotProductExe() = default;
  virtual ~DotProductExe() = default;
  void setEnv(const Handle<Receiver>& rx, const Handle<Transmitter>& tx,
              const Handle<Allocator>& pool) {
    rx_ = rx;
    tx_ = tx;
    tensor_pool_ = pool;
  }

  virtual Expected<void> dotproduct_i(float* in0, float* in1, float* out, int32_t row,
                                      int32_t column, Entity& in_msg, Entity& out_msg) = 0;

  Expected<void> execute(const char* out_tensor_name = "") {
    GXF_ASSERT(rx_ && tx_ && tensor_pool_, "dotproduct received empty in_msg");

    Expected<Entity> in_msg = rx_->receive();
    GXF_ASSERT(in_msg, "dotproduct received empty in_msg");

    // get tensors
    auto in_tensors = in_msg.value().findAll<Tensor>();
    GXF_ASSERT(in_tensors.size() == 2, "doesn't find Tensors in in_msg");
    GXF_ASSERT(in_tensors[0]->rank() == 2, "Input tensor rank is not 2");
    int32_t column = in_tensors[0]->shape().dimension(1);
    int32_t row = in_tensors[0]->shape().dimension(0);
    MemoryStorageType mem_type = in_tensors[0]->storage_type();
    PrimitiveType data_type = in_tensors[0]->element_type();
    float* in_data[2] = {nullptr};
    for (size_t i = 0; i < in_tensors.size(); ++i) {
      GXF_ASSERT(in_tensors[i], "Input Tensor Handle is empty");
      GXF_ASSERT(in_tensors[i]->rank() == 2, "Input tensor rank is not 2");
      in_data[i] = ValuePointer<float>(in_tensors[i]->pointer());
    }

    Expected<Entity> output = Entity::New(in_msg->context());
    GXF_ASSERT(output, "Creating dotproduct output tensor failed.");
    Shape out_shape{row};
    GXF_ASSERT(out_shape.rank() == 1 && out_shape.size() == static_cast<uint64_t>(row),
               "output_shape is not correct");
    TensorDescription outDesc{out_tensor_name, mem_type, out_shape, data_type};
    auto out_tensor = StreamBasedOps::addTensor(output.value(), tensor_pool_, outDesc);
    GXF_ASSERT(out_tensor && out_tensor.value(), "cuda dotproduct output tensor is not found");
    float* out_data = ValuePointer<float>(out_tensor.value()->pointer());

    auto ret =
        dotproduct_i(in_data[0], in_data[1], out_data, row, column, in_msg.value(), output.value());
    GXF_ASSERT(ret, "dotproduct execute with implementation failed");

    ret = tx_->publish(output.value());
    GXF_ASSERT(ret, "dotproduct publishing tensors failed");
    return ret;
  }
};

// Cublas Dot product Operators
class CublasDotProduct : public StreamBasedOps {
 public:
  // Culblas dot production execution class
  class CublasDotProductExe : public DotProductExe {
   private:
    cublasHandle_t handle_ = nullptr;
    CublasDotProduct* codelet_ = nullptr;

   public:
    CublasDotProductExe(CublasDotProduct* codelet) : codelet_(codelet) {}
    ~CublasDotProductExe() {
      if (handle_) { cublasDestroy(handle_); }
    }
    Expected<void> dotproduct_i(float* in0, float* in1, float* out, int32_t row, int32_t column,
                                Entity& in_msg, Entity& out_msg) override {
      // locate stream
      auto maybe_stream = StreamBasedOps::getStream(in_msg);
      GXF_ASSERT(maybe_stream && maybe_stream.value(), "get stream from in_msg failed");
      auto& stream = maybe_stream.value();
      auto ret = StreamBasedOps::addStream(out_msg, stream);
      GXF_ASSERT(ret, "adding cudastream into dotproduct output message failed.");

      int gpu_id = stream->dev_id();
      if (gpu_id >= 0) {
        CHECK_CUDA_ERROR(cudaSetDevice(gpu_id), "failed to set deviceid: %d", gpu_id);
      }

      if (!handle_) {
        CHECK_CUBLUS_ERROR(cublasCreate(&handle_), "failed to create cublas handle");
      }
      auto custream_id = stream->stream();
      GXF_ASSERT(custream_id, "cudastream id is invalid");
      CHECK_CUBLUS_ERROR(cublasSetStream(handle_, custream_id.value()), "cublas set stream failed");

      for (int i = 0; i < row; ++i) {
        CHECK_CUBLUS_ERROR(
            cublasSdot(handle_, column, in0 + column * i, 1, in1 + column * i, 1, out + i),
            "cublasSdot failed on row :%d", i);
      }

      auto maybe_event = codelet_->addNewEvent(out_msg, "cudotproduct_event");
      GXF_ASSERT(maybe_event, "failed to add cublas dot product event");
      ret = stream->record(maybe_event.value(), in_msg,
                           []() { GXF_LOG_DEBUG("cublas dotproduct event synced"); });
      GXF_ASSERT(ret, "cublas dotproduct record event failed");
      return ret;
    }
  };

  CublasDotProduct() : exec_(this) {}

  gxf_result_t initialize() override {
    GXF_ASSERT(tensor_pool_.get() && rx_.get() && tx_.get(), "params not set");
    exec_.setEnv(rx_.get(), tx_.get(), tensor_pool_.get());
    return GXF_SUCCESS;
  }

  gxf_result_t start() override { return ToResultCode(initOpsEvent()); }
  gxf_result_t stop() override { return GXF_SUCCESS; }

  gxf_result_t tick() override {
    auto ret = exec_.execute("cublasdotproduct_tensor");
    return ToResultCode(ret);
  }

  gxf_result_t registerInterface(Registrar* registrar) override {
    Expected<void> result;
    result &= registrar->parameter(tx_, "tx", "transmitter of tensors", "");
    result &= registrar->parameter(rx_, "rx", "receiver of tensors", "");
    result &= registrar->parameter(tensor_pool_, "tensor_pool", "Tensor Pool", "");
    return ToResultCode(result);
  }

 private:
  Parameter<Handle<Transmitter>> tx_;
  Parameter<Handle<Receiver>> rx_;
  Parameter<Handle<Allocator>> tensor_pool_;

  CublasDotProductExe exec_;
};

// CPU Dot product Operators
class HostDotProduct : public Codelet {
 public:
  // CPU dot production execution class
  class HostDotProductExe : public DotProductExe {
   public:
    HostDotProductExe() = default;
    ~HostDotProductExe() = default;
    Expected<void> dotproduct_i(float* in0, float* in1, float* out, int32_t row, int32_t column,
                                Entity& in_msg, Entity& out_msg) override {
      for (int i = 0; i < row; ++i) {
        float sum = 0.0;
        float* x = in0 + column * i;
        float* y = in1 + column * i;
        for (int j = 0; j < column; ++j) { sum += x[j] * y[j]; }
        out[i] = sum;
      }
      return Success;
    }
  };

  gxf_result_t initialize() override {
    GXF_ASSERT(tensor_pool_.get() && rx_.get() && tx_.get(), "params not set");
    exec_.setEnv(rx_.get(), tx_.get(), tensor_pool_.get());
    return GXF_SUCCESS;
  }

  gxf_result_t start() override { return GXF_SUCCESS; }
  gxf_result_t stop() override { return GXF_SUCCESS; }

  gxf_result_t tick() override {
    auto ret = exec_.execute("host_dotproduct_tensor");
    return ToResultCode(ret);
  }

  gxf_result_t registerInterface(Registrar* registrar) override {
    Expected<void> result;
    result &= registrar->parameter(tx_, "tx", "transmitter of tensors", "");
    result &= registrar->parameter(rx_, "rx", "receiver of tensors", "");
    result &= registrar->parameter(tensor_pool_, "tensor_pool", "Tensor Pool", "");
    return ToResultCode(result);
  }

 private:
  Parameter<Handle<Transmitter>> tx_;
  Parameter<Handle<Receiver>> rx_;
  Parameter<Handle<Allocator>> tensor_pool_;

  HostDotProductExe exec_;
};

// Stream based Memory copy from device to host
class MemCpy2Host : public StreamBasedOps {
 public:
  gxf_result_t start() override { return ToResultCode(initOpsEvent()); }
  gxf_result_t stop() override { return GXF_SUCCESS; }

  gxf_result_t tick() override {
    auto in = rx_->receive();
    GXF_ASSERT(in, "rx received empty message");
    auto maybe_tensor = in.value().get<Tensor>();
    GXF_ASSERT(maybe_tensor, "tensor not found");
    auto& in_tensor = maybe_tensor.value();
    byte* in_data = in_tensor->pointer();

    Expected<Entity> out_msg = Entity::New(context());
    TensorDescription out_desc{in_tensor.name(), MemoryStorageType::kHost, in_tensor->shape(),
                               in_tensor->element_type()};
    auto maybe_out_tensor = addTensor(out_msg.value(), tensor_pool_, out_desc);
    GXF_ASSERT(maybe_out_tensor, "Memcpy host message adding tensor failed.");
    auto& out_tensor = maybe_out_tensor.value();
    byte* out_data = out_tensor->pointer();

    auto maybe_stream = getStream(in.value());
    GXF_ASSERT(maybe_stream && maybe_stream.value(), "get stream from in failed");
    auto& stream = maybe_stream.value();
    auto ret = addStream(out_msg.value(), stream);
    GXF_ASSERT(ret, "adding cudastream into memcpy output message failed.");

    // wrap cuda operations since CHECK_CUDA_ERROR return Expected<void>
    ret = [&, this]() -> Expected<void> {
      int gpu_id = stream->dev_id();
      if (gpu_id >= 0) {
        CHECK_CUDA_ERROR(cudaSetDevice(gpu_id), "failed to set deviceid: %d", gpu_id);
      }
      cudaError_t error = cudaMemcpyAsync(out_data, in_data, in_tensor->size(),
                                          cudaMemcpyDeviceToHost, stream->stream().value());
      CHECK_CUDA_ERROR(error, "CUDA memory cpy to host failed.");
      return Success;
    }();
    GXF_ASSERT(ret, "CUDA memory cpy to host failed.");

    auto maybe_event = addNewEvent(out_msg.value(), "memcpy_event");
    GXF_ASSERT(maybe_event, "failed to add memcpy_event");
    ret = stream->record(maybe_event.value(), in.value(),
                         []() { GXF_LOG_DEBUG("memcpy_to_host event synced"); });
    GXF_ASSERT(ret, "memcpy_to_host record event failed");

    ret = tx_->publish(out_msg.value());
    GXF_ASSERT(ret, "memcpy_to_host publishing tensors failed");

    return ToResultCode(ret);
  }

  gxf_result_t registerInterface(Registrar* registrar) override {
    Expected<void> result;
    result &= registrar->parameter(tx_, "tx", "transmitter of tensors", "");
    result &= registrar->parameter(rx_, "rx", "receiver of tensors", "");
    result &= registrar->parameter(tensor_pool_, "tensor_pool", "Tensor output pool", "");
    return ToResultCode(result);
  }

 private:
  Parameter<Handle<Transmitter>> tx_;
  Parameter<Handle<Receiver>> rx_;
  Parameter<Handle<Allocator>> tensor_pool_;
};

// Equal verification
class VerifyEqual : public Codelet {
 public:
  gxf_result_t start() override { return GXF_SUCCESS; }
  gxf_result_t stop() override { return GXF_SUCCESS; }

  gxf_result_t tick() override {
    GXF_LOG_DEBUG("verifying frame: %d", count_++);
    auto in0 = rx0_->receive();
    GXF_ASSERT(in0, "rx0 received empty message");
    auto in1 = rx1_->receive();
    GXF_ASSERT(in1, "rx1 received empty message");

    // get tensors
    auto maybe_tensor0 = in0.value().get<Tensor>();
    GXF_ASSERT(maybe_tensor0, "tensor0 not found");
    auto maybe_tensor1 = in1.value().get<Tensor>();
    GXF_ASSERT(maybe_tensor1, "tensor1 not found");
    auto& tensor0 = maybe_tensor0.value();
    auto& tensor1 = maybe_tensor1.value();
    GXF_ASSERT(std::string(tensor0.name()) != std::string(tensor1.name()),
               "2 tensor name should not same");
    GXF_ASSERT(tensor0->shape() == tensor1->shape(), "2 tensors' shape not matched");
    GXF_ASSERT(tensor0->element_type() == tensor1->element_type(),
               "2 tensor's element type not matched");
    GXF_ASSERT(tensor0->storage_type() == tensor1->storage_type(),
               "2 tensor's storage_type not matched");
    GXF_ASSERT(tensor0->storage_type() == MemoryStorageType::kHost,
               "very tensor storage_type is not from host");

    float* data0 = tensor0->data<float>().value();
    float* data1 = tensor1->data<float>().value();
    uint64_t count = tensor0->element_count();
    for (uint64_t i = 0; i < count; ++i) {
      // printf("i:%d, [%f], [%f]\n", static_cast<int>(i), data0[i], data1[i]);
      GXF_ASSERT(fequal(data0[i], data1[i]), "data0[%d]: %f but data1: %f.", static_cast<int>(i),
                 data0[i], data1[i]);
    }

    return GXF_SUCCESS;
  }

  gxf_result_t registerInterface(Registrar* registrar) override {
    Expected<void> result;
    result &= registrar->parameter(rx0_, "rx0", "receiver0 of tensors", "");
    result &= registrar->parameter(rx1_, "rx1", "receiver1 of tensors", "");
    return ToResultCode(result);
  }

 private:
  bool fequal(float a, float b) {
    if (fabs(a - b) <= std::numeric_limits<float>::epsilon()) return true;
    return false;
  }
  Parameter<Handle<Receiver>> rx0_;
  Parameter<Handle<Receiver>> rx1_;
  int32_t count_ = 0;
};

}  // namespace cuda

}  // namespace test

}  // namespace gxf

}  // namespace nvidia

#endif  // NVIDIA_GXF_CUDA_TESTS_TEST_CUDA_HELPER_HPP
