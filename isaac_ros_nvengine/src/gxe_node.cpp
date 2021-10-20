/**
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "isaac_ros_nvengine/gxe_node.hpp"

#include <ament_index_cpp/get_package_share_directory.hpp>

#include <string>
#include <vector>
#include <memory>

#include "gxf/core/gxf.h"

#include "rclcpp/logger.hpp"
#include "rclcpp/rclcpp.hpp"

namespace isaac_ros
{
namespace nvengine
{

const char * nvengine_extensions[] = {
  "gxf/std/libgxf_std.so",
  "gxf/libgxf_ros_bridge.so"
};

namespace
{
// Split strings
std::vector<std::string> SplitStrings(const std::string & list_of_files)
{
  std::vector<std::string> filenames;

  for (size_t begin = 0;; ) {
    const size_t end = list_of_files.find(',', begin);
    if (end == std::string::npos) {
      if (begin == 0 && !list_of_files.empty()) {
        filenames.push_back(list_of_files);
      } else if (!list_of_files.substr(begin).empty()) {
        filenames.push_back(list_of_files.substr(begin));
      }
      break;
    } else {
      filenames.push_back(list_of_files.substr(begin, end - begin));
      begin = end + 1;
    }
  }

  return filenames;
}

}  // namespace


struct GXENode::GXENodeImpl
{
  explicit GXENodeImpl(const GXENode & node)
  : node_(node) {}

  ~GXENodeImpl() {}

  // Loads extension manifest(s)
  gxf_result_t LoadExtensionManifest(
    const std::string & base_dir,
    const char * const * extensions,
    const uint32_t extensions_length)
  {
    const GxfLoadExtensionsInfo load_extension_info{
      extensions, extensions_length, nullptr, 0, base_dir.c_str()};

    return GxfLoadExtensions(context_, &load_extension_info);
  }

// Loads application graph file(s)
  gxf_result_t LoadApplication(const std::string & list_of_files)
  {
    const auto filenames = SplitStrings(list_of_files);

    if (filenames.empty()) {
      RCLCPP_ERROR(
        node_.get_logger(), "An NVEngine application file has to be specified");
      return GXF_FILE_NOT_FOUND;
    }

    for (const auto & filename : filenames) {
      RCLCPP_INFO(
        node_.get_logger(), "Loading app: '%s'", filename.c_str());
      const gxf_result_t code = GxfGraphLoadFile(context_, filename.c_str());
      if (code != GXF_SUCCESS) {return code;}
    }

    return GXF_SUCCESS;
  }

  gxf_result_t GetCid(std::string group_name, std::string codelet_type, gxf_uid_t & cid)
  {
    gxf_result_t code;
    gxf_uid_t eid;

    code = GxfEntityFind(context_, group_name.c_str(), &eid);
    if (code != GXF_SUCCESS) {
      RCLCPP_ERROR(node_.get_logger(), "GXF Entity find failed");
      return GXF_FAILURE;
    }

    gxf_tid_t tid;

    code = GxfComponentTypeId(context_, codelet_type.c_str(), &tid);
    if (code != GXF_SUCCESS) {
      RCLCPP_ERROR(node_.get_logger(), "GXF Component Type ID get Failed");
      return GXF_FAILURE;
    }

    code = GxfComponentFind(context_, eid, tid, nullptr, nullptr, &cid);
    if (code != GXF_SUCCESS) {
      RCLCPP_ERROR(node_.get_logger(), "GXF Component find failed");
      return GXF_FAILURE;
    }

    return GXF_SUCCESS;
  }

  const GXENode & node_;
  gxf_context_t context_;
};

GXENode::GXENode(
  const rclcpp::NodeOptions & options,
  std::string app_yaml_filename,
  const char * const * extensions,
  uint32_t extensions_length,
  std::string package_name)
: Node("GXENode", options),
  impl_(std::make_unique<GXENodeImpl>(*this))
{
  const std::string package_share_directory = ament_index_cpp::get_package_share_directory(
    package_name);

  const std::string nvengine_package_share_directory = ament_index_cpp::get_package_share_directory(
    "isaac_ros_nvengine");

  RCLCPP_INFO(get_logger(), "%s", package_share_directory.c_str());

  // FLAGS
  const int32_t FLAGS_severity = 2;

  gxf_result_t code;

  RCLCPP_INFO(get_logger(), "Creating context");
  code = GxfContextCreate(&impl_->context_);

  if (code != GXF_SUCCESS) {
    RCLCPP_INFO(get_logger(), "GxfContextCreate Error: %s", GxfResultStr(code));
    return;
  }

  code = GxfSetSeverity(&impl_->context_, static_cast<gxf_severity_t>(FLAGS_severity));
  if (code != GXF_SUCCESS) {
    RCLCPP_ERROR(get_logger(), "GxfSetSeverity Error: %s", GxfResultStr(code));
    return;
  }

  code = impl_->LoadExtensionManifest(
    nvengine_package_share_directory, extensions, extensions_length);

  if (code != GXF_SUCCESS) {
    RCLCPP_ERROR(get_logger(), "LoadExtensionManifest Error: %s", GxfResultStr(code));
    return;
  }

  code = impl_->LoadApplication(package_share_directory + "/" + app_yaml_filename);
  if (code != GXF_SUCCESS) {
    RCLCPP_ERROR(get_logger(), "LoadApplication Error: %s", GxfResultStr(code));
    return;
  }
}

GXENode::GXENode(const rclcpp::NodeOptions & options)
: GXENode(options, "config/test_tensor.yaml", nvengine_extensions, 2, "isaac_ros_nvengine")
{
  SetParameterInt64(
    "tx", "nvidia::isaac_ros::RosBridgeTensorSubscriber", "node_address",
    reinterpret_cast<int64_t>(this));

  SetParameterInt64(
    "rx", "nvidia::isaac_ros::RosBridgeTensorPublisher", "node_address",
    reinterpret_cast<int64_t>(this));

  RunGraph();
}

void GXENode::RunGraph()
{
  gxf_result_t code;

  RCLCPP_INFO(get_logger(), "Initializing...");
  code = GxfGraphActivate(impl_->context_);
  if (code != GXF_SUCCESS) {
    RCLCPP_INFO(get_logger(), "GxfGraphActivate Error: %s", GxfResultStr(code));
    return;
  }

  RCLCPP_INFO(get_logger(), "Running...");
  code = GxfGraphRunAsync(impl_->context_);
  if (code != GXF_SUCCESS) {
    RCLCPP_INFO(get_logger(), "GxfGraphRunError: %s", GxfResultStr(code));
    return;
  }
}

std::vector<char *> GXENode::toCStringArray(const std::vector<std::string> & strings)
{
  std::vector<char *> cstrings;
  cstrings.reserve(strings.size());


  for (size_t i = 0; i < strings.size(); ++i) {
    cstrings.push_back(const_cast<char *>(strings[i].c_str()));
  }

  return cstrings;
}

void GXENode::SetParameterInt64(
  const std::string & group_name, const std::string & codelet_type,
  const std::string & parameter_name, const int64_t parameter_value)
{
  gxf_uid_t cid;
  if (impl_->GetCid(group_name, codelet_type, cid) != GXF_SUCCESS) {return;}

  GxfParameterSetInt64(impl_->context_, cid, parameter_name.c_str(), parameter_value);
}

void GXENode::SetParameterInt32(
  const std::string & group_name, const std::string & codelet_type,
  const std::string & parameter_name, const int32_t parameter_value)
{
  gxf_uid_t cid;
  if (impl_->GetCid(group_name, codelet_type, cid) != GXF_SUCCESS) {return;}

  GxfParameterSetInt32(impl_->context_, cid, parameter_name.c_str(), parameter_value);
}

void GXENode::SetParameterUInt32(
  const std::string & group_name, const std::string & codelet_type,
  const std::string & parameter_name, const uint32_t parameter_value)
{
  gxf_uid_t cid;
  if (impl_->GetCid(group_name, codelet_type, cid) != GXF_SUCCESS) {return;}

  GxfParameterSetUInt32(impl_->context_, cid, parameter_name.c_str(), parameter_value);
}

void GXENode::SetParameterUInt16(
  const std::string & group_name, const std::string & codelet_type,
  const std::string & parameter_name, const uint16_t parameter_value)
{
  gxf_uid_t cid;
  if (impl_->GetCid(group_name, codelet_type, cid) != GXF_SUCCESS) {return;}

  GxfParameterSetUInt16(impl_->context_, cid, parameter_name.c_str(), parameter_value);
}

void GXENode::SetParameterStr(
  const std::string & group_name, const std::string & codelet_type,
  const std::string & parameter_name,
  const std::string & parameter_value)
{
  gxf_uid_t cid;
  if (impl_->GetCid(group_name, codelet_type, cid) != GXF_SUCCESS) {return;}

  GxfParameterSetStr(impl_->context_, cid, parameter_name.c_str(), parameter_value.c_str());
}

void GXENode::SetParameterBool(
  const std::string & group_name, const std::string & codelet_type,
  const std::string & parameter_name, const bool parameter_value)
{
  gxf_uid_t cid;
  if (impl_->GetCid(group_name, codelet_type, cid) != GXF_SUCCESS) {return;}

  GxfParameterSetBool(impl_->context_, cid, parameter_name.c_str(), parameter_value);
}

void GXENode::SetParameter1DStrVector(
  const std::string & group_name, const std::string & codelet_type,
  const std::string & parameter_name, const std::vector<std::string> & parameter_value)
{
  gxf_uid_t cid;
  if (impl_->GetCid(group_name, codelet_type, cid) != GXF_SUCCESS) {return;}

  std::vector<char *> parameter_value_cstring = toCStringArray(parameter_value);

  GxfParameterSet1DStrVector(
    impl_->context_, cid, parameter_name.c_str(),
    (const char **) &parameter_value_cstring[0], parameter_value_cstring.size());
}

GXENode::~GXENode()
{
  gxf_result_t code;

  RCLCPP_INFO(get_logger(), "Interrupting GXF...");
  code = GxfGraphInterrupt(impl_->context_);
  if (code != GXF_SUCCESS) {
    RCLCPP_INFO(get_logger(), "GxfGraphInterrupt Error: %s", GxfResultStr(code));
  }

  RCLCPP_INFO(get_logger(), "Waiting on GXF...");
  code = GxfGraphWait(impl_->context_);
  if (code != GXF_SUCCESS) {
    RCLCPP_ERROR(get_logger(), "GxfGraphWait Error: %s", GxfResultStr(code));
    return;
  }

  RCLCPP_INFO(get_logger(), "Deinitializing...");
  code = GxfGraphDeactivate(impl_->context_);
  if (code != GXF_SUCCESS) {
    RCLCPP_ERROR(get_logger(), "GxfGraphDeactivate Error: %s", GxfResultStr(code));
    return;
  }

  RCLCPP_INFO(get_logger(), "Destroying context");
  code = GxfContextDestroy(impl_->context_);
  if (code != GXF_SUCCESS) {
    RCLCPP_ERROR(get_logger(), "GxfContextDestroy Error: %s", GxfResultStr(code));
    return;
  }

  RCLCPP_INFO(get_logger(), "Done.");
}


}  // namespace nvengine
}  // namespace isaac_ros

// Register as a component
#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(isaac_ros::nvengine::GXENode)
