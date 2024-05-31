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

#ifndef ISAAC_ROS_COMMON__QOS_HPP_
#define ISAAC_ROS_COMMON__QOS_HPP_

#include <string>

#include <rclcpp/rclcpp.hpp>

namespace isaac_ros
{
namespace common
{

rclcpp::QoS AddQosParameter(
  rclcpp::Node & node,
  std::string default_qos = "SYSTEM_DEFAULT",
  std::string parameter_name = "qos");

rclcpp::QoS ParseQosString(const std::string & str);

}  // namespace common
}  // namespace isaac_ros

#endif  // ISAAC_ROS_COMMON__QOS_HPP_
