# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Base test class for all Isaac ROS tests."""

import functools
from pathlib import Path
import time
from typing import Any, Callable, Dict, Iterable, Tuple
import unittest

import cv2  # noqa: F401
from cv_bridge import CvBridge
import launch
import launch_testing.actions
import numpy as np
import rclpy
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from rclpy.subscription import Subscription
from sensor_msgs.msg import CameraInfo, Image


class IsaacROSBaseTest(unittest.TestCase):
    """Base class for all Isaac ROS integration tests."""

    DEFAULT_NAMESPACE = 'isaac_ros_test'
    DEFAULT_QOS = 10
    DEFAULT_BUFFER_QOS = QoSProfile(
        depth=100,
        reliability=QoSReliabilityPolicy.RELIABLE,
        durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)

    def for_each_test_case(subfolder: Path = '') -> Callable:
        """
        Create a decorator to run a test function on each of several test case folders.

        Parameters
        ----------
        subfolder : Path, optional
            Subfolder under test_cases/ to iterate through, by default ''

        Returns
        -------
        Callable
            Decorator that will iterate in the specified folder or subfolder

        """
        def test_case_decorator(test_fn: Callable[[Path], None]) -> Callable[[], None]:
            """
            Decorate a test function to run on each folder under a specified path.

            Parameters
            ----------
            test_fn : Callable[[Path], None]
                The test function to run on each case, with the case's path passed in

            Returns
            -------
            Callable[[], None]
                The wrapped function that iterates over all test cases

            """
            @functools.wraps(test_fn)
            def wrapper(self):
                for test_folder in (self.filepath / 'test_cases' / subfolder).iterdir():
                    if (test_folder / 'SKIP').exists():  # Skip folders with SKIP file
                        self.node.get_logger().info(
                            f'Skipping folder: {test_folder}')
                    else:
                        self.node.get_logger().info(
                            f'Starting test for case: {test_folder}')
                        test_fn(self, test_folder)
            return wrapper

        return test_case_decorator

    @classmethod
    def generate_namespace(cls, *tokens: Iterable[str], absolute=True) -> str:
        """
        Generate a namespace with an optional list of tokens.

        This function is a utility for producing namespaced topic and service names in
        such a way that there are no collisions between 'dummy' nodes running for testing
        and 'normal' nodes running on the same machine.

        Parameters
        ----------
        tokens : Iterable[str]
            List of tokens to include in the namespace. Often used to generate
            separate namespaces for Isaac ROS and reference implementations.

        absolute: bool
            Whether or not to generate an absolute namespace, by default True.

        Returns
        -------
        str
            The generated namespace as a slash-delimited string

        """
        return ('/' if absolute else '') + '/'.join([cls.DEFAULT_NAMESPACE, *tokens])

    def generate_namespace_lookup(
        self, topic_names: Iterable[str], *tokens: Iterable[str], absolute: bool = True
    ) -> None:
        """
        Save a lookup dictionary mapping topics from friendly names to namespaced names.

        Parameters
        ----------
        topic_names : Iterable[str]
            The friendly topic names to produce namespaced names for
        tokens : Iterable[str]
            List of tokens to include in the namespace.
            Passed directly to generate_namespace
        absolute : bool, optional
            Whether or not to generate an absolute namespace, by default True.
            Passed directly to generate_namespace

        """
        self.namespaces = {
            topic: self.generate_namespace(*tokens, topic, absolute=absolute)
            for topic in topic_names
        }

    @classmethod
    def generate_test_description(
        cls, nodes: Iterable[launch.Action], node_startup_delay: float = 2.0
    ) -> launch.LaunchDescription:
        """
        Generate a test launch description.

        The nodes included in this launch description will be launched as a test fixture
        immediately before the first test in the test class runs. Note that the graph is
        NOT shut down or re-launched between tests within the same class.

        Parameters
        ----------
        nodes : Iterable[launch.Action]
            List of Actions to launch before running the test.
        node_startup_delay : float, optional
            Seconds to delay by to account for node startup, by default 2.0

        Returns
        -------
        launch.LaunchDescription
            The LaunchDescription object to launch before running the test

        """
        return launch.LaunchDescription(
            nodes + [
                # Start tests after a fixed delay for node startup
                launch.actions.TimerAction(
                    period=node_startup_delay, actions=[launch_testing.actions.ReadyToTest()])
            ]
        )

    def assertImagesEqual(
            self, actual: np.ndarray, expected: np.ndarray, threshold_fraction: float = 0.01
    ) -> None:
        """
        Assert that two images are equal within tolerance.

        Parameters
        ----------
        actual : np.ndarray
            Actual image received
        expected : np.ndarray
            Expected image to match against
        threshold_fraction : float, optional
            The fraction of allowable variation between the images, by default 0.01.
            A value of 0 means a pixel-perfect match is required. A value of 1 means
            that even the biggest possible difference (full-white against full-black)
            will count as a match.

        """
        self.assertTupleEqual(actual.shape, expected.shape)
        difference = np.linalg.norm(actual - expected)
        threshold_pixels = threshold_fraction * actual.size * 255
        self.assertLessEqual(
            difference, threshold_pixels,
            f'Image difference of {difference} pixels is larger than '
            f'threshold of {threshold_pixels} pixels!'
        )

    def create_logging_subscribers(
        self,
        subscription_requests: Iterable[Tuple[str, Any]],
        received_messages: Dict[str, Iterable],
        use_namespace_lookup: bool = True,
        accept_multiple_messages: bool = False,
        add_received_message_timestamps: bool = False,
        qos_profile: QoSProfile = DEFAULT_QOS
    ) -> Iterable[Subscription]:
        """
        Create subscribers that log any messages received to the passed-in dictionary.

        Parameters
        ----------
        subscription_requests : Iterable[Tuple[str, Any]]
            List of topic names and topic types to subscribe to.

        received_messages : Dict[str, Iterable]
            Output dictionary mapping topic name to list of messages received

        use_namespace_lookup : bool
            Whether the object's namespace dictionary should be used for topic
            namespace remapping, by default True

        accept_multiple_messages : bool
            Whether the generated subscription callbacks should accept multiple messages,
            by default False

        add_received_message_timestamps : bool
            Whether the generated subscription callbacks should add a timestamp to the messages,
            by default False

        qos_profile : QoSProfile
            What Quality of Service policy to use for all subscribers

        Returns
        -------
        Iterable[Subscription]
            List of subscribers, passing the unsubscribing responsibility to the caller

        """
        received_messages.clear()
        if accept_multiple_messages:
            for topic, _ in subscription_requests:
                received_messages[topic] = []

        def make_callback(topic):
            def callback(msg):
                if accept_multiple_messages:
                    if add_received_message_timestamps:
                        received_messages[topic].append((msg, time.time()))
                    else:
                        received_messages[topic].append(msg)
                else:
                    self.assertTrue(topic not in received_messages,
                                    f'Already received a message on topic {topic}!')
                    received_messages[topic] = msg

            return callback

        subscriptions = [self.node.create_subscription(
            msg_type,
            self.namespaces[topic] if use_namespace_lookup else topic,
            make_callback(topic),
            qos_profile,
        ) for topic, msg_type in subscription_requests]

        return subscriptions

    def synchronize_timestamps(
        self,
        image: Image,
        camera_info: CameraInfo
    ) -> Tuple[Image, CameraInfo]:
        """
        Create subscribers that log any messages received to the passed-in dictionary.

        Parameters
        ----------
        image : Image
            Image message to synchronize timestamp with camera_info

        camera_info : CameraInfo
            CameraInfo to synchronize timestamp with image

        Returns
        -------
        Tuple[Image, CameraInfo]
            Same input image and camera info but now with equal timestamps

        """
        timestamp = self.node.get_clock().now().to_msg()
        image.header.stamp = timestamp
        camera_info.header.stamp = timestamp

        return image, camera_info

    @classmethod
    def setUpClass(cls) -> None:
        """Set up before first test method."""
        # Initialize the ROS context for the test node
        rclpy.init()

    @classmethod
    def tearDownClass(cls) -> None:
        """Tear down after last test method."""
        # Shutdown the ROS context
        rclpy.shutdown()

    def setUp(self) -> None:
        """Set up before each test method."""
        # Create a ROS node for tests
        self.node = rclpy.create_node(
            'isaac_ros_base_test_node', namespace=self.generate_namespace())
        self.bridge = CvBridge()

    def tearDown(self) -> None:
        """Tear down after each test method."""
        self.node.destroy_node()
