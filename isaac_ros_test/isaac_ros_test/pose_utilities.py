# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Utilities to work with Poses for Pose Estimation tests."""

from typing import List

from geometry_msgs.msg import Pose
import numpy as np
from scipy.spatial.transform import Rotation as R


class PoseUtilities:
    """Class for Pose Utilities."""

    @staticmethod
    def generate_random_pose_offset_by_list(offset: List[float],
                                            translation_scale: int, rotation_scale: int,
                                            seed: int) -> List[float]:
        """
        Generate a random Pose message that is offset slightly from a list.

        Parameters
        ----------
        offset : list
            The list that the random pose will be offset from.
            Note: the expected format is [x_p, y_p, z_p, x_o, y_o, z_o, w_o]
            where the subscript p represents position and the subscript o represents orientation
        translation_scale: int
            The amount to scale the randomization for each position term by.
            If set to 1, the offset will be +/- 1.
        rotation_scale: int
            The amount to scale the randomization for each rotation term by.
            If set to 1, the offset will be +/- 1.
        seed: int
            The seed for randomization.

        Returns
        -------
        [geometry_msgs.msg.Pose]
            The randomized pose msg that was generated

        """
        pose = Pose()

        # Generate random position
        np.random.seed(seed)
        pose.position.x = offset[0] + \
            np.random.random_sample() / translation_scale
        pose.position.y = offset[1] + \
            np.random.random_sample() / translation_scale
        pose.position.z = offset[2] + \
            np.random.random_sample() / translation_scale

        # Convert to Euler angles, then randomize, then convert back to quarternion
        r = R.from_quat([offset[3], offset[4], offset[5], offset[6]])
        euler_r = r.as_euler('xyz')
        euler_r[0] += np.random.random_sample() / rotation_scale
        euler_r[1] += np.random.random_sample() / rotation_scale
        euler_r[2] += np.random.random_sample() / rotation_scale
        quat_r = R.from_euler(
            'xyz', [euler_r[0], euler_r[1], euler_r[2]]).as_quat()

        # Normalize the quarternion
        normalization_factor = np.linalg.norm(quat_r)
        quat_r = quat_r / normalization_factor

        pose.orientation.x = quat_r[0]
        pose.orientation.y = quat_r[1]
        pose.orientation.z = quat_r[2]
        pose.orientation.w = quat_r[3]
        return pose

    @staticmethod
    def calculate_MSE_between_pose_and_list(pose: Pose,
                                            ground_truth: List[float]) -> float:
        """
        Calculate the mean squared error (MSE) between a Pose msg and ground truth pose.

        Parameters
        ----------
        pose : geometry_msgs.msg.Pose
            The predicted pose.
        ground_truth : List
            The ground truth pose.
            Note: the expected format is [x_p, y_p, z_p, x_o, y_o, z_o, w_o]
            where the subscript p represents position and the subscript o represents orientation

        Returns
        -------
        float
            The computed mean squared error between the pose and ground truth

        """
        # Convert to numpy array
        ground_truth_np = np.array(ground_truth)
        estimated_pose_np = np.array([pose.position.x, pose.position.y, pose.position.z,
                                      pose.orientation.x, pose.orientation.y, pose.orientation.z,
                                      pose.orientation.w])
        # Compute mean squared error
        return (np.square(ground_truth_np - estimated_pose_np)).mean()

    @staticmethod
    def print_pose(pose: Pose) -> None:
        """
        Print the pose of a pose message in the [position, orientation] format.

        Note: the orientation is given as a quarternion.

        Parameters
        ----------
        pose : [geometry_msgs.msg.Pose]
            The pose that will be printed

        """
        pose_array = [pose.position.x, pose.position.y, pose.position.z]
        pose_array += [pose.orientation.x, pose.orientation.y,
                       pose.orientation.z, pose.orientation.z]
        print(pose_array)
