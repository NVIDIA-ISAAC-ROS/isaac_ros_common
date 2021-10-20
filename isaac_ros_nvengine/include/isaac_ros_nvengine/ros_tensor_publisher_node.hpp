/**
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#ifndef ISAAC_ROS_NVENGINE__ROS_TENSOR_PUBLISHER_NODE_HPP_
#define ISAAC_ROS_NVENGINE__ROS_TENSOR_PUBLISHER_NODE_HPP_

#include <string>
#include <utility>

#include "rclcpp/rclcpp.hpp"

#include "isaac_ros_nvengine_interfaces/msg/tensor_list.hpp"

namespace isaac_ros
{
namespace nvengine
{

class ROSTensorPublisherNode : public rclcpp::Node
{
public:
  // Constructor brings up functional ROS Node which only runs GXF test ping
  explicit ROSTensorPublisherNode(const rclcpp::NodeOptions & options);

private:
  void timer_callback();
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::Publisher<isaac_ros_nvengine_interfaces::msg::TensorList>::SharedPtr pub_;
};

}  // namespace nvengine
}  // namespace isaac_ros

#endif  // ISAAC_ROS_NVENGINE__ROS_TENSOR_PUBLISHER_NODE_HPP_
