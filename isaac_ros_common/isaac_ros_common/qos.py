# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES',
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import rclpy
from rclpy.node import Node
import rclpy.qos


def add_qos_parameter(node: Node, default_qos='SYSTEM_DEFAULT', parameter_name='qos'):
    return parse_qos_string(
        node.declare_parameter(parameter_name, default_qos).get_parameter_value().string_value)


def parse_qos_string(qos_str: str):
    profile = qos_str.upper()

    if profile == 'SYSTEM_DEFAULT':
        return rclpy.qos.qos_profile_system_default
    if profile == 'DEFAULT':
        return rclpy.qos.QoSProfile(depth=10)
    if profile == 'PARAMETER_EVENTS':
        return rclpy.qos.qos_profile_parameter_events
    if profile == 'SERVICES_DEFAULT':
        return rclpy.qos.qos_profile_services_default
    if profile == 'PARAMETERS':
        return rclpy.qos.qos_profile_parameters
    if profile == 'SENSOR_DATA':
        return rclpy.qos.qos_profile_sensor_data

    Node('parseQoSString').get_logger().warn(
        f'Unknown QoS profile: {profile}. Returning profile: DEFAULT')
    return rclpy.qos.QoSProfile(depth=10)
