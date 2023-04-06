# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Utilities to convert ROS 2 messages from PCD files."""

from pathlib import Path
from typing import Tuple

import numpy as np
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header


class PCDLoader:
    """Class for PC Dloader utilities."""

    @staticmethod
    def get_relevant_properties_from_ascii_pcd(pcd_path: Path) -> Tuple:
        """
        Generate the PCD fields and the PCD data from a ASCII PCD file.

        This function serves a utility function for GeneratePointCloud2FromASCIIPCD.

        Parameters
        ----------
        pcd_path : Path
            The filepath to the PCD file that should be read

        Returns
        -------
        [Tuple][fields, data]
            The first element gives the fields that the PCD file has
            The second element gives the PCD data, in the order of first element.
            For example, if fields was 'x,y,z,rgb' the data will be formatted as: [[x,y,z,rgb]]

        """
        # Read the PCD file
        pcd_lines = []
        with open(pcd_path, 'r') as pcd_file:
            for pcd_line in pcd_file.readlines():
                pcd_lines.append(pcd_line.strip().split(' '))

        # Assign relevant data to the PCD file
        # This can be hardcoded since it won't vary
        fields = pcd_lines[2]
        sizes = pcd_lines[3]
        types = pcd_lines[4]
        counts = pcd_lines[5]
        data_type = pcd_lines[10]
        data = pcd_lines[11:]

        # Ensure our assumptions about the PCD file is valid
        assert data_type[1] == 'ascii'
        for i in range(1, len(fields)):
            assert fields[i] == 'x' or fields[i] == 'y' \
                or fields[i] == 'z' or fields[i] == 'rgb'
        for i in range(1, len(sizes)):
            assert int(sizes[i]) == 4
        for i in range(1, len(types)):
            assert types[i] == 'F' or types[i] == 'U'
        for i in range(1, len(counts)):
            assert int(counts[i]) == 1

        # Get rid of header
        types = types[1:]

        # Convert the data to the necessary format
        for i in range(len(data)):
            for j in range(len(data[i])):
                if types[j] == 'F':
                    data[i][j] = np.float32(data[i][j])
                if types[j] == 'U':
                    data[i][j] = np.uintc(data[i][j])

        # Get rid of header
        fields = fields[1:]

        return fields, data

    def generate_pointcloud2_from_pcd_file(pcd_path: Path, cloud_frame: str) -> PointCloud2:
        """
        Generate a PointCloud2 message from a PCD file path and a coordinate frame.

        Note: this does not fill the timestamp of the message

        Parameters
        ----------
        pcd_path : Path
            The filepath to the PCD file that should be read
        cloud_frame : str
            The coordinate frame that the PointCloud is in
            Note: this does not create the necessary transform on the tf2 tree

        Returns
        -------
        [sensor_msgs.msg.PointCloud2]
            The PointCloud2 message that was generated

        """
        fields, data = PCDLoader.get_relevant_properties_from_ascii_pcd(
            pcd_path)
        point_fields = []

        # This can be hardcoded because of the above function
        # Note: we reinterpret a uint32 datatype as a float
        # This is fine since the bits are what's important
        size_of_float = 4
        float_data_type = 7

        # Add the point fields to message
        for i in range(len(fields)):
            point_fields.append(PointField(name=fields[i],
                                           offset=size_of_float * i,
                                           datatype=float_data_type, count=1))

        # Create an empty header and set the frame id
        cloud_header = Header()
        cloud_header.frame_id = cloud_frame
        return point_cloud2.create_cloud(cloud_header, point_fields, data)
