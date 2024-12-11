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

import importlib.util
from pathlib import Path
import sys

from ament_index_python.packages import get_resource
from setuptools import setup

ISAAC_ROS_COMMON_PATH = get_resource(
    'isaac_ros_common_scripts_path',
    'isaac_ros_common'
)[0]

ISAAC_ROS_COMMON_VERSION_INFO = Path(ISAAC_ROS_COMMON_PATH) / 'isaac_ros_common-version-info.py'

spec = importlib.util.spec_from_file_location(
    'isaac_ros_common_version_info',
    ISAAC_ROS_COMMON_VERSION_INFO
)

isaac_ros_common_version_info = importlib.util.module_from_spec(spec)
sys.modules['isaac_ros_common_version_info'] = isaac_ros_common_version_info
spec.loader.exec_module(isaac_ros_common_version_info)

from isaac_ros_common_version_info import GenerateVersionInfoCommand  # noqa: E402, I100

PACKAGE_NAME = 'isaac_ros_rosbag_utils'

LICENSE = """
Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto.  Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""

setup(
    name=PACKAGE_NAME,
    version='3.0.1',
    packages=[PACKAGE_NAME, f'{PACKAGE_NAME}.scripts'],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + PACKAGE_NAME]),
        ('share/' + PACKAGE_NAME, ['package.xml']),
        ('share/' + PACKAGE_NAME, ['requirements.txt']),
        ('share/' + PACKAGE_NAME + '/config', ['config/edex_extraction_nova.yaml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Isaac ROS Maintainers',
    maintainer_email='isaac-ros-maintainers@nvidia.com',
    description='Utilities for working with ROS bags',
    license=LICENSE,
    # Use scripts/extract_urdf.py main as entrypoint:
    entry_points={
        'console_scripts': [
            'extract_urdf = isaac_ros_rosbag_utils.scripts.extract_urdf:main',
            'extract_edex = isaac_ros_rosbag_utils.scripts.extract_edex:main',
        ],
    },
    cmdclass={
        'build_py': GenerateVersionInfoCommand,
    },
    tests_require=['pytest'],
)
