# setup.py
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import subprocess
import sys

from setuptools.command.build_py import build_py


class GenerateVersionInfoCommand(build_py):
    """Generate version_info.yaml before building."""

    def run(self):
        project_name = self.distribution.get_name()

        # Get the current working directory where setup.py is executed
        project_path = os.getcwd()

        # Log the project path for debugging
        print(f'Project path for {project_name}: {project_path}')

        # Call generate_version_info with the correct project_path
        output_path, install_destination = generate_version_info(project_name, project_path)

        # Add the generated file to the package data
        if self.distribution.data_files is None:
            self.distribution.data_files = []
        self.distribution.data_files.append((install_destination, [output_path]))

        # Continue with the build process
        super().run()


def generate_version_info(project_name, source_dir):
    from ament_index_python.packages import get_resource

    # Determine the script path
    if project_name == 'isaac_ros_common':
        # Use relative pathing
        script_path = os.path.join(os.path.dirname(__file__), 'scripts',
                                   'isaac_ros_version_embed.py')
    else:
        # Use the package path resolution
        try:
            script_path = os.path.join(
                get_resource(
                    'isaac_ros_common_scripts_path',
                    'isaac_ros_common'
                )[0],
                'isaac_ros_version_embed.py'
            )
        except ImportError:
            print('Error: isaac_ros_common package not found.')
            sys.exit(1)
        except Exception as e:
            print(f'Error finding isaac_ros_version_embed.py: {e}')
            sys.exit(1)

    # Output path for the version_info.yaml file
    build_dir = os.path.join(os.getcwd(), 'build')
    os.makedirs(build_dir, exist_ok=True)
    output_path = os.path.join(build_dir, 'version_info.yaml')

    # Install destination for the generated YAML file
    install_destination = os.path.join('share', project_name)

    # Run the script to generate the version info YAML file
    command = [
        sys.executable, script_path,
        '--output', output_path,
        '--source-dir', source_dir
    ]
    try:
        subprocess.check_call(command, cwd=source_dir)
        print('Generating version information as YAML')
    except subprocess.CalledProcessError as e:
        print(f'Error generating version information: {e}')
        sys.exit(1)

    return output_path, install_destination
