# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Example to show how an integration test can be written with isaac_ros_test."""

import time

from isaac_ros_test import IsaacROSBaseTest
import launch_ros
import pytest
import rclpy
from std_msgs.msg import String


@pytest.mark.rostest
def generate_test_description():
    """Generate launch description with all ROS 2 nodes for testing."""
    nodes = [
        launch_ros.actions.Node(package='demo_nodes_cpp',
                                executable='talker',
                                namespace=IsaacROSTalkerTest.generate_namespace())
    ]

    return IsaacROSTalkerTest.generate_test_description(nodes)


class IsaacROSTalkerTest(IsaacROSBaseTest):
    """Tests for demo_nodes_cpp's talker node."""

    def test_messages_received(self) -> None:
        """Expect the node to send messages to output topic."""
        TIMEOUT = 10  # Seconds before this test times out
        # Minimum number of messages that must be received
        MESSAGES_RECEIVED_COUNT = 5

        received_messages = []

        sub = self.node.create_subscription(
            String,
            self.generate_namespace('chatter'),
            lambda msg: received_messages.append(msg),
            10,
        )

        try:
            # Wait until the node publishes messages over the ROS topic
            end_time = time.time() + TIMEOUT
            done = False
            while time.time() < end_time:
                rclpy.spin_once(self.node, timeout_sec=0.1)

                if len(received_messages) >= MESSAGES_RECEIVED_COUNT:
                    done = True
                    break

            self.assertTrue(done,
                            f'Expected {MESSAGES_RECEIVED_COUNT} messages but received'
                            f'{len(received_messages)} messages on topic.'
                            )

        finally:
            self.node.destroy_subscription(sub)
