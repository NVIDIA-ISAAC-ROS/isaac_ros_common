# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Example to show how a comparison test can be written with isaac_ros_test."""

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
        # Normally, we will compare our custom implementation to a reference implementation,
        # but for the sake of example here we will compare two separate reference implementations.
        launch_ros.actions.Node(
            package='demo_nodes_cpp',
            executable='talker_loaned_message',
            namespace=IsaacROSTalkerComparisonTest.generate_namespace('custom')
        ),
        launch_ros.actions.Node(
            package='demo_nodes_cpp',
            executable='talker',
            namespace=IsaacROSTalkerComparisonTest.generate_namespace(
                'reference')
        )
    ]

    return IsaacROSTalkerComparisonTest.generate_test_description(nodes)


class IsaacROSTalkerComparisonTest(IsaacROSBaseTest):
    """Comparison tests for demo_nodes_cpp's talker node."""

    def test_messages_match(self) -> None:
        """Expect the messages sent to output topics to match."""
        TIMEOUT = 10  # Seconds before this test times out
        # Minimum number of messages that must be received
        MESSAGES_RECEIVED_COUNT = 5

        received_messages = {}

        custom_namespace = self.generate_namespace('custom', 'chatter')
        reference_namespace = self.generate_namespace('reference', 'chatter')

        custom_sub, reference_sub = self.create_logging_subscribers(
            [(custom_namespace, String),
             (reference_namespace, String)],
            received_messages,
            use_namespace_lookup=False,
            accept_multiple_messages=True
        )

        try:
            # Wait until the nodes publish messages over the ROS topic
            end_time = time.time() + TIMEOUT
            done = False
            while time.time() < end_time:
                rclpy.spin_once(self.node, timeout_sec=0.1)

                if all(
                    len(messages) >= MESSAGES_RECEIVED_COUNT
                    for messages in received_messages.values()
                ):
                    done = True
                    break

            self.assertTrue(done,
                            f'Expected {MESSAGES_RECEIVED_COUNT} messages but received'
                            f'{len(received_messages)} messages on topic.'
                            )

            for custom_msg, reference_msg in zip(
                    received_messages[custom_namespace], received_messages[reference_namespace]
            ):
                # We ignore the message number typically included in the talker's output to
                # avoid synchronization errors
                self.assertEqual(custom_msg.data[:-1], reference_msg.data[:-1])

        finally:
            self.node.destroy_subscription(custom_sub)
            self.node.destroy_subscription(reference_sub)
