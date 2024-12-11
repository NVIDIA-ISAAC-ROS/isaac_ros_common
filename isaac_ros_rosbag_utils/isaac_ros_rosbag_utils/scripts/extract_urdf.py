"""
Use this script to extract a URDF from the /tf_static topic in a rosbag.

The generated URDF is minimal and only contains transforms. Physical parameters like mass, inertia
etc. are not contained.
"""

import argparse
import pathlib

from isaac_ros_rosbag_utils import rosbag_urdf_extraction

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-n', '--name', type=str, required=True)
    parser.add_argument('-r', '--rosbag_path', type=pathlib.Path, required=True)
    parser.add_argument('-o', '--output_path', type=pathlib.Path, required=True)
    args = parser.parse_args()
    rosbag_urdf_extraction.extract_urdf(args.name, args.rosbag_path, args.output_path)

if __name__ == '__main__':
    main()
