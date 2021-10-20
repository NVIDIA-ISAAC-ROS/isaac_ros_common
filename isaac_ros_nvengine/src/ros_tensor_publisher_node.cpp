/**
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "isaac_ros_nvengine/ros_tensor_publisher_node.hpp"

#include <cstring>
#include <chrono>
#include <vector>

#include "isaac_ros_nvengine_interfaces/msg/tensor.hpp"
#include "isaac_ros_nvengine_interfaces/msg/tensor_shape.hpp"

namespace isaac_ros
{
namespace nvengine
{

using namespace std::chrono_literals;

ROSTensorPublisherNode::ROSTensorPublisherNode(const rclcpp::NodeOptions & options)
: Node("tensor_publisher", options)
{
  pub_ = this->create_publisher<isaac_ros_nvengine_interfaces::msg::TensorList>("tensor_pub", 10);
  timer_ =
    this->create_wall_timer(1000ms, std::bind(&ROSTensorPublisherNode::timer_callback, this));
}

void ROSTensorPublisherNode::timer_callback()
{
  auto tensor_1 = isaac_ros_nvengine_interfaces::msg::Tensor();
  tensor_1.name = "Tensor_1";
  tensor_1.data_type = 1;
  tensor_1.strides = {4};
  tensor_1.data = {1, 2, 3, 4};
  auto tensor_1_shape = isaac_ros_nvengine_interfaces::msg::TensorShape();
  tensor_1_shape.rank = 1;
  tensor_1_shape.dims = {4};
  tensor_1.shape = tensor_1_shape;

  auto tensor_2 = isaac_ros_nvengine_interfaces::msg::Tensor();
  tensor_2.name = "Tensor_2";
  tensor_2.data_type = 1;
  tensor_2.strides = {4};
  tensor_2.data = {5, 6, 7, 8};
  auto tensor_2_shape = isaac_ros_nvengine_interfaces::msg::TensorShape();
  tensor_2_shape.rank = 1;
  tensor_2_shape.dims = {4};
  tensor_2.shape = tensor_2_shape;

  auto tensor_list = isaac_ros_nvengine_interfaces::msg::TensorList();
  tensor_list.tensors = {tensor_1, tensor_2};
  pub_->publish(tensor_list);
}

}  // namespace nvengine
}  // namespace isaac_ros
