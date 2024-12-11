#!/usr/bin/env python3
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
import argparse


def main(isaac_package):
    """
    Generate a bug report for the specified `isaac_package`.

    Args
    ----
        isaac_package (str): The name of the `isaac_package`
            for which the bug report is being generated.

    Returns
    -------
        None

    Raises
    ------
        FileNotFoundError: If the `version_info.yaml` file is not found in the package share path.

    """
    from ament_index_python.packages import get_package_share_path

    print(f'\nGenerating bug report for isaac_package: {isaac_package}')
    package_share_path = get_package_share_path(isaac_package)

    print(f'\nPackage share path: {package_share_path}')

    with open(f'{package_share_path}/version_info.yaml', 'r') as file:
        package_yaml = file.read()

    print(package_yaml)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate bug report for given isaac_package')
    parser.add_argument('isaac_package',
                        help='isaac_package for which bug report is to be generated')

    args = parser.parse_args()

    main(args.isaac_package)
