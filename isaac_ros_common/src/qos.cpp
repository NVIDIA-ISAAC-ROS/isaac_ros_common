// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0

#include "isaac_ros_common/qos.hpp"

#include <algorithm>
#include <string>

#include "rclcpp/rclcpp.hpp"

namespace isaac_ros
{
namespace common
{

rclcpp::QoS AddQosParameter(
  rclcpp::Node & node,
  std::string default_qos,
  std::string parameter_name,
  const int default_depth)
{
  std::string qos_str = node.declare_parameter<std::string>(parameter_name, default_qos);
  const int depth = node.declare_parameter<int>(parameter_name + "_depth", default_depth);
  return ParseQosString(qos_str, depth);
}


rclcpp::QoS ParseQosString(const std::string & str, const int depth)
{
  std::string profile = str;
  // Convert to upper case.
  std::transform(profile.begin(), profile.end(), profile.begin(), ::toupper);

  rmw_qos_profile_t rmw_qos = rmw_qos_profile_default;

  if (profile == "SYSTEM_DEFAULT") {
    rmw_qos = rmw_qos_profile_system_default;
  } else if (profile == "DEFAULT") {
    rmw_qos = rmw_qos_profile_default;
  } else if (profile == "PARAMETER_EVENTS") {
    rmw_qos = rmw_qos_profile_parameter_events;
  } else if (profile == "SERVICES_DEFAULT") {
    rmw_qos = rmw_qos_profile_services_default;
  } else if (profile == "PARAMETERS") {
    rmw_qos = rmw_qos_profile_parameters;
  } else if (profile == "SENSOR_DATA") {
    rmw_qos = rmw_qos_profile_sensor_data;
  } else {
    RCLCPP_WARN_STREAM(
      rclcpp::get_logger("parseQoSString"),
      "Unknown QoS profile: " << profile << ". Returning profile: DEFAULT");
  }
  auto qos_init = depth ==
    0 ? rclcpp::QoSInitialization::from_rmw(rmw_qos_profile_default) : rclcpp::KeepLast(depth);
  return rclcpp::QoS(qos_init, rmw_qos);
}

}  // namespace common
}  // namespace isaac_ros
