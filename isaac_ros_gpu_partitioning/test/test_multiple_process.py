#!/usr/bin/env python3

# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import atexit
import os
import time
import unittest

from isaac_ros_gpu_partitioning.gpu_partitioning import GpuPartition, stop_cuda_mps_daemon
import launch
from launch.actions import TimerAction
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
import launch_testing
import pytest
import rclpy
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import Int32MultiArray


@pytest.mark.launch_test
def generate_test_description():
    """Generate a launch description for the gpu partition test node."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(current_dir, 'config/cuda_mps_multiple_process.yaml')
    # Safety net for runs that never reach the post-shutdown cleanup.
    atexit.register(stop_cuda_mps_daemon)
    global gpu_partition
    gpu_partition = GpuPartition(
        config_file=config_file, clear_partitions=True, start_server=True)
    global sm_count_1, sm_count_2
    result_1 = gpu_partition.get_partition_id(name='process_1')
    if result_1 is None:
        raise ValueError('Partition ID not found for name: process_1')
    partition_id_1, sm_count_1 = result_1
    result_2 = gpu_partition.get_partition_id(name='process_2')
    if result_2 is None:
        raise ValueError('Partition ID not found for name: process_2')
    partition_id_2, sm_count_2 = result_2

    print(f'Partition process_2: {partition_id_2} with SM Count: {sm_count_2}')
    print(f'Partition process_1: {partition_id_1} with SM Count: {sm_count_1}')

    node_1 = ComposableNode(
        name='test_node',
        package='isaac_ros_gpu_partitioning',
        namespace='isaac_ros_gpu_partitioning',
        plugin='nvidia::GpuPartitionTestNode',
    )
    node_2 = ComposableNode(
        name='test_node_2',
        package='isaac_ros_gpu_partitioning',
        namespace='isaac_ros_gpu_partitioning',
        plugin='nvidia::GpuPartitionTestNode',
    )
    container_1 = ComposableNodeContainer(
        name='gpu_partition_test_container',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[node_1],
        namespace='',
        output='screen',
    )
    container_2 = ComposableNodeContainer(
        name='gpu_partition_test_container_2',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[node_2],
        namespace='',
        output='screen',
    )
    action_1 = gpu_partition.gpu_partition_launch_action(name='process_1')
    action_2 = gpu_partition.gpu_partition_launch_action(name='process_2')

    delayed_action_2 = TimerAction(period=0.2, actions=[action_2, container_2])
    launch_description = launch.LaunchDescription([action_1, container_1, delayed_action_2])
    read_to_test = launch_testing.actions.ReadyToTest()

    return (
        launch.LaunchDescription([
            launch_description,
            read_to_test,
        ]),
        {}
    )


class TestMultipleProcess(unittest.TestCase):
    """Unit test to check the config file is loaded and SM count matches."""

    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        """Shut down rclpy; TestCudaMpsCleanup stops the MPS daemon afterwards."""
        rclpy.shutdown()

    def setUp(self) -> None:
        """Set up before each test method."""
        self.node = rclpy.create_node(
            'test_config_file_node', namespace='isaac_ros_gpu_partitioning')

    def tearDown(self) -> None:
        self.node.destroy_node()

    def test_gpu_partition_info_callback(self):
        """Test that the SM count is positive."""
        received_messages = []

        def gpu_partition_info_callback(msg):
            received_messages.append(msg)

        sub = self.node.create_subscription(
            Int32MultiArray,
            'gpu_partition_info',
            gpu_partition_info_callback,
            QoSProfile(depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        )

        TIMEOUT = 30  # seconds
        end_time = time.time() + TIMEOUT
        found = False

        while time.time() < end_time:
            rclpy.spin_once(self.node, timeout_sec=0.1)
            if len(received_messages) > 1:
                found = True
                break

        self.assertTrue(found, "Didn't receive message on gpu_partition_info topic!")

        received_sm_count1 = received_messages[0].data[2]

        global sm_count_1, sm_count_2
        self.assertEqual(
            received_sm_count1, sm_count_1,
            'sm_count reported by GpuPartitionTestNode ({received_sm_count1}) != '
            'sm_count from config ({sm_count_1})'
        )
        received_sm_count2 = received_messages[1].data[2]
        self.assertEqual(
            received_sm_count2, sm_count_2,
            'sm_count reported by GpuPartitionTestNode ({received_sm_count2}) != '
            'sm_count from config ({sm_count_2})'
        )
        self.node.destroy_subscription(sub)
        print('Test complete, shutting down.')


@launch_testing.post_shutdown_test()
class TestCudaMpsCleanup(unittest.TestCase):
    """Stop the CUDA MPS daemon once the launched containers have exited."""

    def test_cuda_mps_daemon_stopped(self):
        """Quit the MPS daemon now that the launched processes have exited."""
        stop_cuda_mps_daemon()
