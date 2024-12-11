#!/usr/bin/env python3
import argparse
import logging
import os
import pathlib

import yaml

from isaac_ros_rosbag_utils import rosbag_edex_extraction


def main():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c',
        '--config_path',
        type=pathlib.Path,
        required=True,
        help='Path to config file.',
    )
    parser.add_argument(
        '--rosbag_path',
        type=pathlib.Path,
        required=True,
        help='Path to rosbag file. Can be used to override the rosbag path from the config file.',
    )
    parser.add_argument(
        '--rosbag_name',
        type=str,
        required=False,
        help='Name of the rosbag. If not provided will be inferred automatically.',
    )
    parser.add_argument(
        '--edex_path',
        type=pathlib.Path,
        required=True,
        help=('Path where edex is generated. Can be used to override the rosbag path from ' +
              'the config file.'),
    )
    parser.add_argument(
        '--extract_only_video',
        action='store_true',
        help='If set will only extract the video and not the images.',
    )

    args = parser.parse_args()

    assert args.config_path.exists(), f"Config path '{args.config_path}' does not exist."
    yaml_string = args.config_path.read_text()
    yaml_dict = yaml.safe_load(yaml_string)

    # We allow to override some arguments from the CLI.
    if args.rosbag_path is not None:
        yaml_dict['rosbag_path'] = args.rosbag_path.absolute()
    if args.rosbag_name is not None:
        yaml_dict['rosbag_name'] = args.rosbag_name
    if args.edex_path is not None:
        yaml_dict['edex_path'] = args.edex_path.absolute()

    config = rosbag_edex_extraction.Config(**yaml_dict)

    # Change working dir s.t. we resolve relative paths relative to the config file directory.
    os.chdir(args.config_path.parent)

    if args.extract_only_video:
        rosbag_edex_extraction.extract_videos(config)
    else:
        rosbag_edex_extraction.extract_edex(config)


if __name__ == '__main__':
    main()
