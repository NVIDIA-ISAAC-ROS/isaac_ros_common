#!/usr/bin/env python3
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import argparse
import datetime
import os
import subprocess
import xml.etree.ElementTree as ET

import yaml  # Requires PyYAML to be installed


def get_git_info(repo_dir):
    """
    Retrieve git information for a given repository directory.

    Args
    ----
        repo_dir (str): The path to the repository directory.

    Returns
    -------
        dict: A dictionary containing the following git information:
            - commit_hash (str): The commit hash.
            - commit_date (str): The commit date.
            - commit_message (str): The commit message.
            - is_dirty (str): 'Yes' if the repository is dirty, 'No' otherwise.

    Raises
    ------
        subprocess.CalledProcessError: If there is an error executing the git commands.

    """
    git_info = {
        'commit_hash': 'N/A',
        'commit_date': 'N/A',
        'commit_message': 'N/A',
        'git_branch': 'N/A',
        'is_dirty': 'N/A'
    }

    try:
        # Check if inside a git repository
        subprocess.check_output(['git', 'rev-parse', '--is-inside-work-tree'],
                                cwd=repo_dir, stderr=subprocess.DEVNULL)

        # Get commit hash
        commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                              cwd=repo_dir).decode().strip()
        git_info['commit_hash'] = commit_hash

        # Get commit date
        commit_date = subprocess.check_output(['git', 'show', '-s', '--format=%ci', 'HEAD'],
                                              cwd=repo_dir).decode().strip()
        git_info['commit_date'] = commit_date

        # Get commit message
        commit_message = subprocess.check_output(['git', 'log', '-1', '--pretty=%B'],
                                                 cwd=repo_dir).decode().strip()
        git_info['commit_message'] = commit_message

        # Get git branch
        git_branch = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                                             cwd=repo_dir).decode().strip()
        git_info['git_branch'] = git_branch

        # Check if workspace is dirty
        status = subprocess.check_output(['git', 'status', '--porcelain'],
                                         cwd=repo_dir).decode().strip()
        git_info['is_dirty'] = 'Yes' if status else 'No'

    except subprocess.CalledProcessError:
        pass  # Not in a git repository

    return git_info


def get_version_from_package_xml(package_xml_path):
    """
    Retrieve the version from a package.xml file.

    Args
    ----
        package_xml_path (str): The path to the package.xml file.

    Returns
    -------
        str: The version number extracted from the package.xml file.
              If an error occurs, returns 'N/A'.

    Raises
    ------
        Exception: If an error occurs while parsing the package.xml file.

    """
    try:
        tree = ET.parse(package_xml_path)
        root = tree.getroot()
        version = root.find('version').text.strip()
        return version
    except:  # noqa: E722
        return 'N/A'


def main():
    parser = argparse.ArgumentParser(description='Generate version information as a YAML file.')
    parser.add_argument('--output', '-o', required=True, help='Output YAML file path.')
    parser.add_argument('--source-dir', required=True, help='Source directory of the package.')
    args = parser.parse_args()

    # Get the directory of the package being built
    package_dir = os.path.abspath(args.source_dir)

    # Get version from package.xml
    package_xml_path = os.path.join(package_dir, 'package.xml')
    version = get_version_from_package_xml(package_xml_path)

    # Get git info
    git_info = get_git_info(package_dir)

    # Get current datetime
    build_datetime = datetime.datetime.now().isoformat()

    # Prepare data
    data = {
        'version': version,
        'build_datetime': build_datetime,
        'git_branch': git_info['git_branch'],
        'git_commit_hash': git_info['commit_hash'],
        'git_commit_date': git_info['commit_date'],
        'git_commit_message': git_info['commit_message'],
        'git_workspace_dirty': git_info['is_dirty']
    }

    # Write YAML data to the specified output path
    output_path = args.output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        yaml.dump(data, f)


if __name__ == '__main__':
    main()
