/*
 * Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto. Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "NvInfer.h"
#include "NvInferRuntimeBase.h"
#include "NvOnnxParser.h"  // For ONNX parsing
#include "NvInferRuntimeCommon.h"  // Important for runtime components

// A simple logger class for TensorRT
class Logger : public nvinfer1::ILogger
{
public:
  void log(Severity severity, const char * msg) noexcept override
  {
    // Suppress info and verbose messages for brevity unless it's an error/warning
    if (severity == Severity::kINFO && strstr(msg, "successfully") == nullptr &&
      strstr(msg, "passed") == nullptr && strstr(msg, "destroy") == nullptr)
    {
      // return; // Keep some INFO for key steps
    }
    if (severity == Severity::kVERBOSE) {
      return;
    }

    switch (severity) {
      case Severity::kINTERNAL_ERROR: std::cerr << "TRT INTERNAL_ERROR: "; break;
      case Severity::kERROR: std::cerr << "TRT ERROR: "; break;
      case Severity::kWARNING: std::cerr << "TRT WARNING: "; break;
      case Severity::kINFO: std::cerr << "TRT INFO: "; break;
      case Severity::kVERBOSE: std::cerr << "TRT VERBOSE: "; break;
      default: std::cerr << "TRT UNKNOWN: "; break;
    }
    std::cerr << msg << std::endl;
  }
};

int main(int argc, char ** argv)
{
  Logger logger;

  logger.log(nvinfer1::ILogger::Severity::kINFO, "Starting TensorRT ONNX loading test...");

  const char * onnxModelPath = "dummy_model.onnx";

  // Check if model file exists
  std::ifstream modelFile(onnxModelPath, std::ios::binary);
  if (!modelFile) {
    logger.log(
      nvinfer1::ILogger::Severity::kERROR,
      (std::string("ONNX model file not found at: ") + onnxModelPath).c_str());
    return 1;
  }
  modelFile.close();
  logger.log(
    nvinfer1::ILogger::Severity::kINFO,
    (std::string("Found ONNX model: ") + onnxModelPath).c_str());


  // 1. Create a TensorRT builder
  nvinfer1::IBuilder * builder = nvinfer1::createInferBuilder(logger);
  if (!builder) {
    logger.log(nvinfer1::ILogger::Severity::kERROR, "Failed to create TensorRT Builder!");
    return 1;
  }
  logger.log(nvinfer1::ILogger::Severity::kINFO, "TensorRT Builder created successfully.");

  // 2. Create Network Definition
  // Explicit batch flag is required for ONNX parsing
  uint32_t explicitBatch = 1U <<
    static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
  nvinfer1::INetworkDefinition * network = builder->createNetworkV2(explicitBatch);
  if (!network) {
    logger.log(nvinfer1::ILogger::Severity::kERROR, "Failed to create INetworkDefinition!");
    delete builder;
    return 1;
  }
  logger.log(nvinfer1::ILogger::Severity::kINFO, "INetworkDefinition created successfully.");

  // 3. Create ONNX Parser
  nvonnxparser::IParser * parser = nvonnxparser::createParser(*network, logger);
  if (!parser) {
    logger.log(nvinfer1::ILogger::Severity::kERROR, "Failed to create ONNX Parser!");
    delete network;
    delete builder;
    return 1;
  }
  logger.log(nvinfer1::ILogger::Severity::kINFO, "ONNX Parser created successfully.");

  // 4. Parse ONNX model
  logger.log(
    nvinfer1::ILogger::Severity::kINFO,
    (std::string("Attempting to parse ONNX model: ") + onnxModelPath).c_str());
  bool parsed = parser->parseFromFile(
    onnxModelPath,
    static_cast<int32_t>(nvinfer1::ILogger::Severity::kWARNING));
  if (!parsed) {
    logger.log(
      nvinfer1::ILogger::Severity::kERROR,
      (std::string("Failed to parse ONNX model: ") + onnxModelPath).c_str());
    // Parser automatically prints errors, so we don't need to repeat them here.
    delete parser;
    delete network;
    delete builder;
    return 1;
  }
  logger.log(nvinfer1::ILogger::Severity::kINFO, "ONNX model parsed successfully.");

  // 5. Create Builder Config
  nvinfer1::IBuilderConfig * config = builder->createBuilderConfig();
  if (!config) {
    logger.log(nvinfer1::ILogger::Severity::kERROR, "Failed to create IBuilderConfig!");
    delete parser;
    delete network;
    delete builder;
    return 1;
  }
  logger.log(nvinfer1::ILogger::Severity::kINFO, "IBuilderConfig created successfully.");

  // Set max workspace size (example: 256 MB). The error log mentioned "275742720" bytes.
  // 275742720 bytes is ~263 MB. Let's try 512MB to be safe.
  size_t workspaceSize = 1ULL << 29;  // 512 MB
  config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, workspaceSize);
  logger.log(
    nvinfer1::ILogger::Severity::kINFO,
    (std::string("Set max workspace size to: ") + std::to_string(workspaceSize / (1024 * 1024)) +
    " MB").c_str());

  // 6. Build Serialized Network (Engine Plan)
  logger.log(
    nvinfer1::ILogger::Severity::kINFO,
    "Attempting to build serialized network (engine plan)...");
  nvinfer1::IHostMemory * plan = builder->buildSerializedNetwork(*network, *config);

  if (!plan) {
    logger.log(
      nvinfer1::ILogger::Severity::kERROR,
      "Failed to build serialized network (engine plan).");
    // Builder automatically prints errors from TRT.
    delete config;
    delete parser;
    delete network;
    delete builder;
    return 1;
  }
  logger.log(
    nvinfer1::ILogger::Severity::kINFO,
    "Serialized network (engine plan) built successfully.");


  // Clean up
  logger.log(nvinfer1::ILogger::Severity::kINFO, "Cleaning up resources...");
  delete plan;
  delete config;
  delete parser;
  delete network;
  delete builder;

  logger.log(
    nvinfer1::ILogger::Severity::kINFO,
    "TensorRT ONNX loading and engine building test passed!");
  return 0;
}
