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


import sys

import pytest


def test_smoke():
    try:
        from isaac_ros_launch_utils import NovaRobot
        assert NovaRobot.NOVA_CARTER.value == 1, 'Could not verify NOVA_CARTER'
    except ImportError:
        # Print the module resolution path
        for path in sys.path:
            print(path)
        assert False, 'Could not import isaac_ros_launch_utils'


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__]))
