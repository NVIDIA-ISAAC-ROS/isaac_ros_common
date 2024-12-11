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

from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch.actions import TimerAction


def generate_launch_description():
    # This launchfile creates a timer action. The graph terminates after completion.
    # The timeout should be chosen to be more than the graph_startup_test test time.
    test_time_s = 100.0
    return LaunchDescription([
        TimerAction(period=test_time_s,
                    actions=[ExecuteProcess(cmd=['echo', '"Hello test"'], output='screen')])
    ])
