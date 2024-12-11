# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import pathlib
import xml.etree.ElementTree as ET
from typing import Literal

import numpy as np
import pydantic
from pytransform3d import transformations
from pytransform3d import rotations
from pytransform3d import transform_manager

from isaac_ros_rosbag_utils import rosbag_tf_extraction


class Translation(pydantic.BaseModel):
    """ Representation of a translation. """
    translation: list[float]

    def to_urdf(self) -> str:
        """Returns the translation element in the URDF format."""
        return " ".join(map(str, self.translation))


class Rotation(pydantic.BaseModel):
    """ Representation of a rotation. """
    rotation: list[float]

    def to_euler(self, order: Literal['xyz', 'xzy', 'yxz', 'yzx', 'zxy', 'zyx']) -> np.ndarray:
        """Returns the euler representation of the rotation vector."""
        idx_from_axis = {'x': 0, 'y': 1, 'z': 2}
        i = idx_from_axis[order[0]]
        j = idx_from_axis[order[1]]
        k = idx_from_axis[order[2]]
        return rotations.euler_from_quaternion(self.rotation, i, j, k, extrinsic=False)

    def to_urdf(self) -> str:
        """Returns the rotation element in the URDF format."""
        return " ".join(map(str, self.to_euler('xyz')))


class Transform(pydantic.BaseModel):
    """ Representation of a transform. """
    translation: Translation
    rotation: Rotation

    @staticmethod
    def from_homogenous_matrix(tf: np.ndarray) -> 'Transform':
        """Returns the transform from the given tf_name in the given TransformManager."""
        pq = transformations.pq_from_transform(tf)
        return Transform(translation=Translation(translation=pq[:3]),
                         rotation=Rotation(rotation=pq[3:]))

    def to_urdf(self) -> ET.Element:
        """Returns the transform element in the URDF format."""
        return ET.Element("origin", xyz=self.translation.to_urdf(), rpy=self.rotation.to_urdf())


class Link(pydantic.BaseModel):
    """Representation of a link."""
    name: str

    def to_urdf(self) -> ET.Element:
        """Returns the link element in the URDF format."""
        return ET.Element("link", name=self.name)


class Joint(pydantic.BaseModel):
    """Representation of a joint."""
    name: str
    type: Literal['fixed']
    parent: Link
    child: Link
    transform: Transform

    def to_urdf(self) -> ET.Element:
        """Returns the joint element in the URDF format."""
        element = ET.Element("joint", name=self.name, type=self.type)
        ET.SubElement(element, "parent", link=self.parent.name)
        ET.SubElement(element, "child", link=self.child.name)
        element.append(self.transform.to_urdf())
        return element


class Robot:
    """Representation of a robot."""

    def __init__(self, name: str):
        self._name = name
        self._links = {}
        self._joints = []

    def add_link(self, link: Link):
        """Add a link to the robot."""
        self._links[link.name] = link

    def get_link(self, name: str) -> Link:
        """Get a link from the robot."""
        return self._links[name]

    def add_joint(self, joint: Joint):
        """Add a joint to the robot."""
        self._joints.append(joint)

    def to_urdf(self) -> ET.Element:
        """Returns the robot element in the URDF format."""
        root = ET.Element("robot", name=self._name)
        for link in self._links.values():
            root.append(link.to_urdf())
        for joint in self._joints:
            root.append(joint.to_urdf())
        return root


def get_urdf_from_tf_manager(name: str, tf_manager: transform_manager.TransformManager) -> str:
    """Create a URDF (in string format) from a TransformManager.)"""
    robot = Robot(name)

    # We potentially add some links twice, but we don't care as long as we haven't made joints yet.
    for parent_frame, child_frame in tf_manager.transforms:
        robot.add_link(Link(name=parent_frame))
        robot.add_link(Link(name=child_frame))

    for parent_frame, child_frame in tf_manager.transforms:
        tf = tf_manager.get_transform(parent_frame, child_frame)
        robot.add_joint(
            Joint(
                name=f"{parent_frame}_T_{child_frame}",
                type='fixed',
                parent=robot.get_link(parent_frame),
                child=robot.get_link(child_frame),
                transform=Transform.from_homogenous_matrix(tf),
            ))

    urdf_xml = robot.to_urdf()
    # Intent is needed to format the XML nicely.
    ET.indent(urdf_xml, 4 * ' ')
    urdf_string = ET.tostring(urdf_xml, encoding='unicode')
    return urdf_string


def extract_urdf(name: str, rosbag_path: pathlib.Path, output_path: pathlib.Path) -> None:
    """Extracts the URDF from a rosbag and saves it to the output path."""
    tf_manager = rosbag_tf_extraction.get_static_transform_manager_from_bag(rosbag_path)
    urdf_content = get_urdf_from_tf_manager(name, tf_manager)
    output_path.write_text(urdf_content)
