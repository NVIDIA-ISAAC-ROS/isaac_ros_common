# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Utilities to convert ROS 2 messages to and from human-readable JSON."""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose, PoseArray
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import CameraInfo, Image


class JSONConversion:
    """Class for JSON conversion utilities."""

    @staticmethod
    def load_from_json(json_filepath: Path) -> Dict:
        """
        Load a dictionary from a JSON filepath.

        Parameters
        ----------
        json_filepath : Path
            The path to a JSON file containing the object

        Returns
        -------
        Dict
            Generated dictionary containing object fields

        """
        with open(json_filepath) as json_file:
            return json.load(json_file)

    @staticmethod
    def save_to_json(obj: Dict, json_filepath: Path) -> None:
        """
        Save an arbitrary object to a JSON filepath.

        Parameters
        ----------
        obj : Dict
            The object to save, as a dictionary

        json_filepath : Path
            The path to the JSON file to save the object to

        """
        with open(json_filepath, 'w+') as json_file:
            json.dump(obj, json_file, indent=2)

    @staticmethod
    def load_camera_info_from_json(json_filepath: Path,
                                   desired_size: Tuple[int] = None) -> CameraInfo:
        """
        Load a CameraInfo message from a JSON filepath.

        Parameters
        ----------
        json_filepath : Path
            The path to a JSON file containing the CameraInfo fields
        desired_size: Tuple[int]
            The desired dimension of the CameraInfo

        Returns
        -------
        CameraInfo
            Generated CameraInfo message

        """
        camera_info_json = JSONConversion.load_from_json(json_filepath)

        camera_info = CameraInfo()
        camera_info.header.frame_id = camera_info_json['header']['frame_id']
        camera_info.width = camera_info_json['width']
        camera_info.height = camera_info_json['height']
        camera_info.distortion_model = camera_info_json['distortion_model']
        camera_info.d = camera_info_json['D']
        camera_info.k = camera_info_json['K']
        camera_info.r = camera_info_json['R']
        camera_info.p = camera_info_json['P']

        if(desired_size):
            camera_info.width = desired_size[0]
            camera_info.height = desired_size[1]
        return camera_info

    @staticmethod
    def save_camera_info_to_json(camera_info: CameraInfo, json_filepath: Path) -> None:
        """
        Save a CameraInfo message to a JSON filepath.

        Parameters
        ----------
        camera_info : CameraInfo
            The message to save to JSON

        json_filepath : Path
            The path to save the JSON file

        """
        camera_info_json = {}

        camera_info_json['header'] = {}
        camera_info_json['header']['frame_id'] = camera_info.header.frame_id

        camera_info_json['width'] = camera_info.width
        camera_info_json['height'] = camera_info.height
        camera_info_json['distortion_model'] = camera_info.distortion_model

        camera_info_json['D'] = camera_info.d.tolist()
        camera_info_json['K'] = camera_info.k.tolist()
        camera_info_json['R'] = camera_info.r.tolist()
        camera_info_json['P'] = camera_info.p.tolist()

        JSONConversion.save_to_json(camera_info_json, json_filepath)

    @staticmethod
    def load_occupancy_grid_from_json(json_filepath: Path) -> OccupancyGrid:
        """
        Load a OccupancyGrid message from a JSON filepath.

        Parameters
        ----------
        json_filepath : Path
            The path to a JSON file containing the OccupancyGrid fields

        Returns
        -------
        OccupancyGrid
            Generated OccupancyGrid message

        """
        occupancy_grid_json = JSONConversion.load_from_json(json_filepath)

        occupancy_grid = OccupancyGrid()
        occupancy_grid.header.frame_id = occupancy_grid_json['header']['frame_id']

        occupancy_grid.info.resolution = occupancy_grid_json['info']['resolution']
        occupancy_grid.info.width = occupancy_grid_json['info']['width']
        occupancy_grid.info.height = occupancy_grid_json['info']['height']

        occupancy_grid.info.origin.position.x = occupancy_grid_json[
            'info']['origin']['position']['x']
        occupancy_grid.info.origin.position.y = occupancy_grid_json[
            'info']['origin']['position']['y']
        occupancy_grid.info.origin.position.z = occupancy_grid_json[
            'info']['origin']['position']['z']

        occupancy_grid.info.origin.orientation.x = occupancy_grid_json[
            'info']['origin']['orientation']['x']
        occupancy_grid.info.origin.orientation.y = occupancy_grid_json[
            'info']['origin']['orientation']['y']
        occupancy_grid.info.origin.orientation.z = occupancy_grid_json[
            'info']['origin']['orientation']['z']
        occupancy_grid.info.origin.orientation.w = occupancy_grid_json[
            'info']['origin']['orientation']['w']

        occupancy_grid.data = occupancy_grid_json['data']

        return occupancy_grid

    @staticmethod
    def load_pose_array_from_json(json_filepath: Path) -> PoseArray:
        """
        Load a PoseArray message from a JSON filepath.

        Parameters
        ----------
        json_filepath : Path
            The path to a JSON file containing the PoseArray fields

        Returns
        -------
        PoseArray
            Generated PoseArray message

        """
        pose_array_json = JSONConversion.load_from_json(json_filepath)

        pose_array = PoseArray()
        pose_array.header.frame_id = pose_array_json['header']['frame_id']

        for pose_json in pose_array_json['poses']:
            pose = Pose()

            pose.position.x = pose_json['position']['x']
            pose.position.y = pose_json['position']['y']
            pose.position.z = pose_json['position']['z']

            pose.orientation.x = pose_json['orientation']['x']
            pose.orientation.y = pose_json['orientation']['y']
            pose.orientation.z = pose_json['orientation']['z']
            pose.orientation.w = pose_json['orientation']['w']

            pose_array.poses.append(pose)

        return pose_array

    @staticmethod
    def load_image_from_json(json_filepath: Path) -> Image:
        """
        Load an Image message from a JSON filepath.

        Parameters
        ----------
        json_filepath : Path
            The path to a JSON file containing the Image fields

        Returns
        -------
        Image
            Generated Image message

        """
        image_json = JSONConversion.load_from_json(json_filepath)

        # Load the main image data from a JSON-specified image file
        image = CvBridge().cv2_to_imgmsg(cv2.imread(
            str(json_filepath.parent / image_json['image'])))

        image.encoding = image_json['encoding']

        return image

    @staticmethod
    def save_image_to_json(
            image: Image, json_filepath: Path, image_filename: str = 'image_raw.jpg') -> None:
        """
        Save an Image message to a JSON filepath.

        Parameters
        ----------
        image : Image
            The message to save to JSON

        json_filepath : Path
            The path to save the JSON file

        image_filename : str
            The filename to save the image data to, by default image_raw.jpg

        """
        # Load the main image data from a JSON-specified image file
        cv2.imwrite(str(json_filepath.parent /
                    image_filename), CvBridge().imgmsg_to_cv2(image))

        image_json = {}
        image_json['image'] = image_filename
        image_json['encoding'] = image.encoding

        JSONConversion.save_to_json(image_json, json_filepath)

    @staticmethod
    def load_chessboard_image_from_json(json_filepath: Path) -> Tuple[Image, Tuple[int, int]]:
        """
        Load a chessboard Image message from a JSON filepath.

        Parameters
        ----------
        json_filepath : Path
            The path to a JSON file containing the Image fields

        Returns
        -------
        Tuple[Image, Tuple[int, int]]
            Generated Image message and tuple of chessboard dimensions as (width, height)

        """
        image_json = JSONConversion.load_from_json(json_filepath)

        # Load the chessboard dimensions from JSON
        chessboard_dimensions = (
            image_json['chessboard']['width'], image_json['chessboard']['height'])

        # Return the loaded image along with the dimensions
        return JSONConversion.load_image_from_json(json_filepath), \
            chessboard_dimensions

    @staticmethod
    def load_ground_truth_pose_list_from_json(json_filepath: Path) -> List[float]:
        """
        Load a ground truth list from a JSON filepath.

        Parameters
        ----------
        json_filepath : Path
            The path to a JSON file to read from

        Returns
        -------
        List
            The ground truth pose of the object location.
            Note: the expected format is [x_p, y_p, z_p, x_o, y_o, z_o, w_o]
            where the subscript p represents position and the subscript o represents orientation

        """
        pose_json = JSONConversion.load_from_json(json_filepath)
        position = pose_json['ground_truth']['position']
        orientation = pose_json['ground_truth']['orientation']
        ground_truth = position + orientation
        return ground_truth
