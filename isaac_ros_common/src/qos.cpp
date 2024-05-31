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
  std::string parameter_name)
{
  return ParseQosString(node.declare_parameter<std::string>(parameter_name, default_qos));
}

rclcpp::QoS ParseQosString(const std::string & str)
{
  std::string profile = str;
  // Convert to upper case.
  std::transform(profile.begin(), profile.end(), profile.begin(), ::toupper);

  if (profile == "SYSTEM_DEFAULT") {
    return rclcpp::QoS(rclcpp::SystemDefaultsQoS());
  }
  if (profile == "DEFAULT") {
    return rclcpp::QoS(rclcpp::QoSInitialization::from_rmw(rmw_qos_profile_default));
  }
  if (profile == "PARAMETER_EVENTS") {
    return rclcpp::QoS(rclcpp::ParameterEventsQoS());
  }
  if (profile == "SERVICES_DEFAULT") {
    return rclcpp::QoS(rclcpp::ServicesQoS());
  }
  if (profile == "PARAMETERS") {
    return rclcpp::QoS(rclcpp::ParametersQoS());
  }
  if (profile == "SENSOR_DATA") {
    return rclcpp::QoS(rclcpp::SensorDataQoS());
  }
  RCLCPP_WARN_STREAM(
    rclcpp::get_logger("parseQoSString"),
    "Unknown QoS profile: " << profile << ". Returning profile: DEFAULT");
  return rclcpp::QoS(rclcpp::QoSInitialization::from_rmw(rmw_qos_profile_default));
}

}  // namespace common
}  // namespace isaac_ros
