# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
Basic Proof-Of-Life test for the Isaac ROS NVEngine package.

This test ensures that an NVEngine graph of bridge codelets can successfully
round trip a message from ROS through GXF and back.
"""
import time

from isaac_ros_nvengine_interfaces.msg import Tensor, TensorList, TensorShape
from isaac_ros_test import IsaacROSBaseTest

import launch_ros
import pytest
import rclpy
from std_msgs.msg import Header


@pytest.mark.rostest
def generate_test_description():
    """Generate launch description with GXENode nodes for testing."""
    nvengine_exe = launch_ros.actions.Node(
        package='isaac_ros_nvengine',
        executable='test_tensor',
        name='nvengine_exe',
        namespace=IsaacROSNVEngineTest.generate_namespace()
    )

    return IsaacROSNVEngineTest.generate_test_description([
        nvengine_exe
    ])


class IsaacROSNVEngineTest(IsaacROSBaseTest):
    SUBSCRIBER_CHANNEL = 'ping'
    # The amount of seconds to allow GXENode node to run before verifying received tensors
    PYTHON_SUBSCRIBER_WAIT_SEC = 5.0

    def test_tensor_rt_node(self) -> None:
        self.node._logger.info('Starting Isaac ROS NVEngine POL Test')

        received_messages = {}

        self.generate_namespace_lookup(['tensor_pub'])

        subscriber_topic_namespace = self.generate_namespace(self.SUBSCRIBER_CHANNEL)
        test_subscribers = [
            (subscriber_topic_namespace, TensorList)
        ]

        subs = self.create_logging_subscribers(
            subscription_requests=test_subscribers,
            received_messages=received_messages,
            use_namespace_lookup=False,
            accept_multiple_messages=True,
            add_received_message_timestamps=True
        )

        tensor_pub = self.node.create_publisher(TensorList, self.namespaces['tensor_pub'],
                                                self.DEFAULT_QOS)

        # Create test tensor to publish and receive back.
        pub_header = Header()
        pub_header.stamp.sec = 1111
        pub_header.stamp.nanosec = 7777

        pub_tensor_shape = TensorShape()
        pub_tensor_shape.rank = 1
        pub_tensor_shape.dims = [1]

        pub_tensor = Tensor()
        pub_tensor.name = 'Test'
        pub_tensor.shape = pub_tensor_shape
        pub_tensor.data_type = 1  # uint8
        pub_tensor.strides = [1]
        pub_tensor.data = [1]
        pub_tensor_list = TensorList()
        pub_tensor_list.tensors.append(pub_tensor)
        pub_tensor_list.header = pub_header

        try:
            end_time = time.time() + self.PYTHON_SUBSCRIBER_WAIT_SEC
            while time.time() < end_time:
                tensor_pub.publish(pub_tensor_list)
                rclpy.spin_once(self.node, timeout_sec=0.1)

            # Verify received tensors and log total number of tensors received
            num_tensors_received = len(received_messages[subscriber_topic_namespace])
            self.assertGreater(num_tensors_received, 0)
            self.node._logger.info(
                f'Received {num_tensors_received} tensors in '
                f'{self.PYTHON_SUBSCRIBER_WAIT_SEC} seconds')

            # Log properties of last received tensor
            tensor_list, _ = received_messages[subscriber_topic_namespace][-1]
            tensor = tensor_list.tensors[0]
            shape = tensor.shape
            length = len(tensor.data.tolist())
            strides = tensor.strides.tolist()
            dimensions = shape.dims.tolist()

            self.node._logger.info(
                f'Received Tensor Properties:\n'
                f'Name: {tensor.name}\n'
                f'Data Type: {tensor.data_type}\n'
                f'Strides: {strides}\n'
                f'Byte Length: {length}\n'
                f'Rank: {shape.rank}\n'
                f'Dimensions: {dimensions}'
            )

            self.assertEqual(pub_tensor_list, tensor_list,
                             'Received tensor_list does not match published tensor_list')

            self.node._logger.info('Finished Isaac ROS NVEngine POL Test')
        finally:
            [self.node.destroy_subscription(sub) for sub in subs]
            self.node.destroy_publisher(tensor_pub)
