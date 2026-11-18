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

from isaac_ros_gpu_partitioning.gpu_partitioning import GpuPartition, gpuPartitionConfig

from launch import LaunchDescription
from launch.actions import OpaqueFunction, RegisterEventHandler
from launch.event_handlers import OnShutdown
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode


gpu_partition_config = [
    gpuPartitionConfig(name='node_1', device='0', chunks=3),
    gpuPartitionConfig(name='node_2', device='0', chunks=4),
    gpuPartitionConfig(name='default', device='0', chunks=0),
]


def generate_launch_description():
    """Generate a launch description for the gpu partition test node."""
    gpu_partition = GpuPartition(gpu_partitions=gpu_partition_config,
                                 clear_partitions=True, start_server=True)
    result1 = gpu_partition.get_partition_id(name='node_1')
    if result1 is None:
        raise ValueError('Partition ID not found for name: node_1')
    partition_id1, sm_count1 = result1
    result2 = gpu_partition.get_partition_id(name='node_2')
    if result2 is None:
        raise ValueError('Partition ID not found for name: node_2')
    partition_id2, sm_count2 = result2
    default_result = gpu_partition.get_partition_id(name='default')
    if default_result is None:
        raise ValueError('Partition ID not found for name: default')
    partition_id_default, sm_count_default = default_result
    print(f'Partition {gpu_partition_config[0].name}: {partition_id1} with SM Count: {sm_count1}')
    print(f'Partition {gpu_partition_config[1].name}: {partition_id2} with SM Count: {sm_count2}')
    print(f'Partition {gpu_partition_config[2].name}: {partition_id_default} \
            with SM Count: {str(sm_count_default)}')

    node = ComposableNode(
        name='test_node',
        package='isaac_ros_gpu_partitioning',
        plugin='nvidia::GpuPartitionTestNode',
    )

    container = ComposableNodeContainer(
        name='gpu_partition_test_container',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[node],
        namespace='',
        output='screen',
        additional_env={'CUDA_MPS_SM_PARTITION': str(partition_id1)}
    )

    def stop_server(_):
        gpu_partition.stop_cuda_mps_server()
        return []

    shutdown_handler = RegisterEventHandler(
        OnShutdown(
            on_shutdown=[OpaqueFunction(function=stop_server)]
        )
    )

    return LaunchDescription([shutdown_handler, container])
