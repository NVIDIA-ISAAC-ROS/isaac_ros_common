# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import time
from typing import List
import unittest

import isaac_ros_launch_utils as lu
from isaac_ros_launch_utils.all_types import Action
import launch
from launch import LaunchDescription
import launch_testing
from launch_testing import post_shutdown_test
import pytest
import rclpy
from rclpy.node import Node


def generate_parameterized_launchfile_test(args: lu.ArgumentContainer) -> List[Action]:
    # The launch arguments are passed as a single string, in order to separate them
    # from the arguments used by the test infrastructure. So we now split them up
    # to be passed to the underlying lauchfile.
    launch_arguments = {}
    if len(args.launch_file_arguments) > 0:
        for arg_string in args.launch_file_arguments.split(' '):
            arg_name, arg_value = arg_string.split(':=')
            launch_arguments[arg_name] = arg_value
    # The launchfile we've been asked to test.
    actions = []
    actions.append(
        lu.include(
            args.package_under_test,
            args.launch_file_under_test,
            launch_arguments=launch_arguments,
        ))
    return actions


class TimeoutStore:
    """This class just provides access to a single shared variable 'timeout_s'."""

    timeout_s = None
    check_exit_code = None

    @staticmethod
    def save_timeout(args: lu.ArgumentContainer):
        """Store the timeout. Called as isaac_ros_launch_utils opaque function."""
        TimeoutStore.timeout_s = float(args.timeout)

    @staticmethod
    def get_timeout() -> int:
        """Get the stored value."""
        return TimeoutStore.timeout_s

    @staticmethod
    def save_check_exit_code(args: lu.ArgumentContainer):
        """Store the flag. Called as isaac_ros_launch_utils opaque function."""
        TimeoutStore.check_exit_code = lu.is_true(args.check_exit_code)

    @staticmethod
    def get_check_exit_code() -> bool:
        """Get the flag."""
        return TimeoutStore.check_exit_code


@pytest.mark.rostest
def generate_test_description():

    args = lu.ArgumentContainer()
    args.add_arg('package_under_test',
                 cli=True,
                 description='The package containing the launch file to test.')
    args.add_arg('launch_file_under_test',
                 cli=True,
                 description='The path within the package to the launch file to test.')
    args.add_arg('timeout',
                 cli=True,
                 description='The time after which we declare a non-crashed graph a test success.')
    args.add_arg('check_exit_code',
                 default=True,
                 cli=True,
                 description='Whether or not to check the error code completing the dry-run.')
    args.add_arg('launch_file_arguments',
                 default=None,
                 cli=True,
                 description='The arguments to be passed to the launch file under test.')

    # Launch the test launchfile.
    actions = args.get_launch_actions()
    actions.append(args.add_opaque_function(generate_parameterized_launchfile_test))

    # Save the timeout during startup
    # NOTE(alexmillane): This is the best way I could think of to get an evaluated launch
    # parameter into the test fixture. If someone can think of something better, please update.
    actions.append(args.add_opaque_function(TimeoutStore.save_timeout))
    actions.append(args.add_opaque_function(TimeoutStore.save_check_exit_code))

    # NOTE(alexmillane): We trigger the ready-to-test action 1 second after the graph
    # starts coming up in order to ensure that the timeout parameter is evaluated.
    ready_to_test_time = 1.0

    # Required for ROS launch testing.
    actions.append(launch_testing.util.KeepAliveProc())
    actions.append(
        launch.actions.TimerAction(period=ready_to_test_time,
                                   actions=[launch_testing.actions.ReadyToTest()]))

    return LaunchDescription(actions)


class DummyTest(unittest.TestCase):
    """This test does nothing, except keep the test alive until the timeout is elapsed."""

    def test_graph_startup_test(self):
        rclpy.init()

        # Create a Node for logging
        node = Node('test_node')

        # Get the timeout requested.
        assert TimeoutStore.get_timeout() is not None, 'Need to increase ready to test time.'
        timeout_s = TimeoutStore.get_timeout()

        # Loop until the timeout
        loop_period_s = 0.1
        log_period_s = 1.0
        start_time = time.time()
        node.get_logger().info(f'Start test. Waiting for {timeout_s} seconds.')
        while rclpy.ok() and ((time.time() - start_time) < timeout_s):
            already_waited_time_s = time.time() - start_time
            node.get_logger().info(
                f'Waited for {already_waited_time_s:0.2f} out of'
                f' {timeout_s} seconds.',
                throttle_duration_sec=log_period_s)
            time.sleep(loop_period_s)
        # Check if we stopped looping because of an error (fail) or because the test timed
        # out (pass).
        if rclpy.ok():
            node.get_logger().info('Test success. Shutting down test node.')
        else:
            elapsed_time_s = time.time() - start_time
            node.get_logger().info(f'Test terminated early after {elapsed_time_s}.')
        rclpy.shutdown()


@post_shutdown_test()
class TestAfterShutdown(unittest.TestCase):

    disallowed_phrases_in_log = [
        # This phrase appears in the log if a launchfile tries to add a Component
        # which doesn't exist in the workspace.
        'Failed to find class'
    ]

    def test_exit_code(self, proc_info):
        if TimeoutStore.get_check_exit_code():
            launch_testing.asserts.assertExitCodes(proc_info)

    def test_error_message(self, proc_output):
        for proc in proc_output:
            for disallowed_phrase in self.disallowed_phrases_in_log:
                assert disallowed_phrase not in str(
                    proc.text), f'Found disallowed phrase \"{disallowed_phrase}\"'
