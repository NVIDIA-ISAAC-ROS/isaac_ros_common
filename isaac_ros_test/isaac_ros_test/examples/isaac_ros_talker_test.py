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
