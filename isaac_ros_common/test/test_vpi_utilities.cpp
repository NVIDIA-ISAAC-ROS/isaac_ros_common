// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <gtest/gtest.h>
#include <rclcpp/rclcpp.hpp>

#include "isaac_ros_common/vpi_utilities.hpp"

class VPIUtilitiesTest : public ::testing::Test
{
protected:
  void SetUp()
  {
    setenv("AMENT_PREFIX_PATH", ".", 0);
    setenv("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp", 0);  // Example RMW implementation
    setenv("ROS_LOG_DIR", "/tmp/ros_logs", 0);
    rclcpp::init(0, nullptr);
  }
  void TearDown() {(void)rclcpp::shutdown();}
};

TEST_F(VPIUtilitiesTest, DefaultBackendParameterTest) {
  rclcpp::Node::SharedPtr node;
  node = rclcpp::Node::make_shared("node");
  ASSERT_EQ(
    VPI_BACKEND_CUDA,
    isaac_ros::common::DeclareVPIBackendParameter(node.get(), VPI_BACKEND_CUDA));
}

TEST_F(VPIUtilitiesTest, ArgvBackendParameterTest) {
  rclcpp::NodeOptions options;
  options.arguments(
  {
    "--ros-args",
    "-p", "backends:=CPU",
  });
  rclcpp::Node::SharedPtr node;
  node = rclcpp::Node::make_shared("node", options);
  ASSERT_EQ(
    VPI_BACKEND_CPU,
    isaac_ros::common::DeclareVPIBackendParameter(node.get(), VPI_BACKEND_CUDA));
}
