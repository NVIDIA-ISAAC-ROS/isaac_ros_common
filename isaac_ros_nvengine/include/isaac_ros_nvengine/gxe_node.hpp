/**
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#ifndef ISAAC_ROS_NVENGINE__GXE_NODE_HPP_
#define ISAAC_ROS_NVENGINE__GXE_NODE_HPP_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "rclcpp/rclcpp.hpp"

namespace isaac_ros
{
namespace nvengine
{

class GXENode : public rclcpp::Node
{
public:
  // Constructor brings up functional ROS Node which only runs GXF test ping
  explicit GXENode(const rclcpp::NodeOptions & options);

  ~GXENode();

  GXENode(const GXENode & node) = delete;

  GXENode & operator=(const GXENode & node) = delete;

protected:
  GXENode(
    const rclcpp::NodeOptions & options,
    std::string app_yaml_filename,
    const char * const * extensions,
    uint32_t extensions_length,
    std::string package_name);

  // Activates and asynchronously runs nvengine graph
  void RunGraph();

  void SetParameterInt64(
    const std::string & group_name,
    const std::string & codelet_type,
    const std::string & parameter_name,
    const int64_t parameter_value);

  void SetParameterInt32(
    const std::string & group_name,
    const std::string & codelet_type,
    const std::string & parameter_name,
    const int32_t parameter_value);

  void SetParameterUInt32(
    const std::string & group_name,
    const std::string & codelet_type,
    const std::string & parameter_name,
    const uint32_t parameter_value);

  void SetParameterUInt16(
    const std::string & group_name,
    const std::string & codelet_type,
    const std::string & parameter_name,
    const uint16_t parameter_value);

  void SetParameterStr(
    const std::string & group_name,
    const std::string & codelet_type,
    const std::string & parameter_name,
    const std::string & parameter_value);

  void SetParameterBool(
    const std::string & group_name,
    const std::string & codelet_type,
    const std::string & parameter_name,
    const bool parameter_value);

  void SetParameter1DStrVector(
    const std::string & group_name,
    const std::string & codelet_type,
    const std::string & parameter_name,
    const std::vector<std::string> & parameter_value);

private:
  struct GXENodeImpl;
  std::unique_ptr<GXENodeImpl> impl_;
  std::vector<char *> toCStringArray(const std::vector<std::string> & strings);
};

}  // namespace nvengine
}  // namespace isaac_ros

#endif  // ISAAC_ROS_NVENGINE__GXE_NODE_HPP_
