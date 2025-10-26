// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "isaac_ros_common/qos.hpp"

class QosTest : public ::testing::Test
{
protected:
  rclcpp::Node::SharedPtr node;
  void SetUp() override
  {
    int argc = 0;
    char ** argv = NULL;
    setenv("AMENT_PREFIX_PATH", ".", 1);
    setenv("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp", 1);    // Example RMW implementation
    setenv("ROS_LOG_DIR", "/tmp/ros_logs", 1);
    rclcpp::init(argc, argv);
    node = rclcpp::Node::make_shared("node");
  }
};

TEST_F(QosTest, DefaultQoSProfileTest) {
  isaac_ros::common::AddQosParameter(*node);
}
