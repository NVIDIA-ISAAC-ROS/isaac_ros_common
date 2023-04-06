/**
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#ifndef ISAAC_ROS_COMMON__VPI_UTILITIES_HPP_
#define ISAAC_ROS_COMMON__VPI_UTILITIES_HPP_

#include "rclcpp/rclcpp.hpp"
#include "vpi/VPI.h"

// VPI status check macro
#define CHECK_STATUS(STMT) \
  do { \
    VPIStatus status = (STMT); \
    if (status != VPI_SUCCESS) { \
      char buffer[VPI_MAX_STATUS_MESSAGE_LENGTH]; \
      vpiGetLastStatusMessage(buffer, sizeof(buffer)); \
      std::ostringstream ss; \
      ss << __FILE__ << ":" << __LINE__ << ": " << vpiStatusGetName(status) << ": " << buffer; \
      throw std::runtime_error(ss.str()); \
    } \
  } while (0);

namespace isaac_ros
{
namespace common
{

/**
 * @brief Declare and parse ROS 2 parameter into VPI backend flags
 *
 * @param node The node to declare the parameter with
 * @param default_backends The default backends to use if given invalid input
 * @return uint32_t The resulting VPI backend flags
 */
uint32_t DeclareVPIBackendParameter(rclcpp::Node * node, uint32_t default_backends) noexcept;

}  // namespace common

}  // namespace isaac_ros

#endif  // ISAAC_ROS_COMMON__VPI_UTILITIES_HPP_
