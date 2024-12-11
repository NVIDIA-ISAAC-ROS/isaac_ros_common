# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# flake8: noqa

import pathlib
import shutil
import subprocess

# This is a WAR until we can specify pip deps in package.xml
REQUIREMENTS_FILE = pathlib.Path(__file__).parent.parent / 'requirements.txt'
subprocess.call(['python3', '-m', 'pip', 'install', '-r', str(REQUIREMENTS_FILE)])

import numpy as np
from pytransform3d import rotations
from pytransform3d import transformations
import rosbags.rosbag2
import rosbags.typesys
from rosbags.typesys.stores.ros2_humble import builtin_interfaces__msg__Time as Time
from rosbags.typesys.stores.ros2_humble import geometry_msgs__msg__Quaternion as Quaternion
from rosbags.typesys.stores.ros2_humble import geometry_msgs__msg__Transform as Transform
from rosbags.typesys.stores.ros2_humble import geometry_msgs__msg__TransformStamped as \
    TransformStamped
from rosbags.typesys.stores.ros2_humble import geometry_msgs__msg__Vector3 as Vector3
from rosbags.typesys.stores.ros2_humble import std_msgs__msg__Header as Header
from rosbags.typesys.stores.ros2_humble import tf2_msgs__msg__TFMessage as TFMessage

from isaac_ros_rosbag_utils import rosbag_tf_extraction


def sec_to_ns(s: float) -> int:
    return int(s * 1e9)


def nanoseconds_to_msg(total_ns: int) -> Time:
    ns = int(total_ns % 1e9)
    s = int(total_ns // 1e9)
    return Time(sec=s, nanosec=ns)


def tf_message_from_pq(pq: np.ndarray, parent_frame: str, child_frame: str,
                       stamp_ns: int) -> TFMessage:
    msg = TFMessage(transforms=[
        TransformStamped(
            header=Header(
                stamp=nanoseconds_to_msg(stamp_ns),
                frame_id=parent_frame,
            ),
            child_frame_id=child_frame,
            transform=Transform(
                translation=Vector3(x=pq[0], y=pq[1], z=pq[2]),
                rotation=Quaternion(w=pq[3], x=pq[4], y=pq[5], z=pq[6]),
            ),
        )
    ])
    return msg


def write_tf_messages_to_bag(tf_messages: list[TFMessage], stamps_ns: list[int],
                             path: pathlib.Path):
    assert len(tf_messages) == len(stamps_ns)
    shutil.rmtree(path, ignore_errors=True)
    with rosbags.rosbag2.Writer(path, version=9) as writer:
        typestore = rosbags.typesys.get_typestore(rosbags.typesys.Stores.ROS2_FOXY)
        connection = writer.add_connection('/tf', TFMessage.__msgtype__, typestore=typestore)
        for stamp_ns, msg in zip(stamps_ns, tf_messages):
            serialized_msg = typestore.serialize_cdr(msg, connection.msgtype)
            writer.write(connection, stamp_ns, serialized_msg)


def get_linear_motion(distance: float,
                      axis: int,
                      stamp1_ns: int,
                      stamp2_ns: int,
                      frame1: str = '1',
                      frame2: str = '2') -> tuple[list[TFMessage], list[int]]:
    assert axis < 3
    # Generate the test motion
    p1 = np.array([0, 0, 0])
    q1 = np.array([1, 0, 0, 0])
    pq1 = np.hstack([p1, q1])
    msg1 = tf_message_from_pq(pq1, frame1, frame2, stamp1_ns)

    p2 = np.array([0, 0, 0])
    # Set the requested motion
    p2[axis] = distance
    q2 = np.array([1, 0, 0, 0])
    pq2 = np.hstack([p2, q2])
    msg2 = tf_message_from_pq(pq2, frame1, frame2, stamp2_ns)

    tf_msgs = [msg1, msg2]
    stamps_ns = [stamp1_ns, stamp2_ns]
    return tf_msgs, stamps_ns


def get_angular_motion(angle_rad: float,
                       axis: int,
                       stamp1_ns: int,
                       stamp2_ns: int,
                       frame1: str = '1',
                       frame2: str = '2') -> tuple[list[TFMessage], list[int]]:
    assert axis < 3
    # Generate the test motion
    p1 = np.array([0, 0, 0])
    q1 = np.array([1, 0, 0, 0])
    pq1 = np.hstack([p1, q1])
    msg1 = tf_message_from_pq(pq1, frame1, frame2, stamp1_ns)

    p2 = np.array([0, 0, 0])
    q2 = rotations.quaternion_from_angle(0, angle_rad)
    pq2 = np.hstack([p2, q2])
    msg2 = tf_message_from_pq(pq2, frame1, frame2, stamp2_ns)

    tf_msgs = [msg1, msg2]
    stamps_ns = [stamp1_ns, stamp2_ns]
    return tf_msgs, stamps_ns


def assert_quaternion_close(q1: np.ndarray, q2: np.ndarray):
    q2_inv = rotations.q_conj(q2)
    q_diff = rotations.concatenate_quaternions(q1, q2_inv)
    np.testing.assert_allclose(np.abs(q_diff), np.array([1, 0, 0, 0]), rtol=1e-5, atol=1e-8)


def test_x_motion(tmp_path: pathlib.Path):
    test_bag_path = tmp_path / 'tf_test_bag'

    # Build the test motion
    tf_msgs, stamps_ns = get_linear_motion(
        distance=1.0,
        axis=0,
        stamp1_ns=0,
        stamp2_ns=sec_to_ns(1),
    )

    # Write test data
    write_tf_messages_to_bag(tf_msgs, stamps_ns, test_bag_path)
    temporal_transform_manager = rosbag_tf_extraction.get_transform_manager_from_bag(test_bag_path)

    # Test
    transform = temporal_transform_manager.get_transform_at_time('2', '1', 0.5)
    pq = transformations.pq_from_transform(transform)
    np.testing.assert_allclose(pq, np.array([0.5, 0, 0, 1, 0, 0, 0]))

    # NOTE(alexmillane): Turns out the pytransforms3d extrapolates automatically.
    transform = temporal_transform_manager.get_transform_at_time('2', '1', 2.0)
    pq = transformations.pq_from_transform(transform)
    np.testing.assert_allclose(pq, np.array([2.0, 0, 0, 1, 0, 0, 0]))


def test_xy_motion(tmp_path: pathlib.Path):
    test_bag_path = tmp_path / 'tf_test_bag'

    # Build the test motion
    tf_msgs_1, stamps_ns_1 = get_linear_motion(
        distance=1.0,
        axis=0,
        stamp1_ns=sec_to_ns(0),
        stamp2_ns=sec_to_ns(4),
        frame1='1',
        frame2='2',
    )
    tf_msgs_2, stamps_ns_2 = get_linear_motion(
        distance=1.0,
        axis=1,
        stamp1_ns=sec_to_ns(1),
        stamp2_ns=sec_to_ns(3),
        frame1='2',
        frame2='3',
    )

    tf_msgs = tf_msgs_1 + tf_msgs_2
    stamps_ns = stamps_ns_1 + stamps_ns_2

    # Write test data
    write_tf_messages_to_bag(tf_msgs, stamps_ns, test_bag_path)
    temporal_transform_manager = rosbag_tf_extraction.get_transform_manager_from_bag(test_bag_path)

    # Test
    transform = temporal_transform_manager.get_transform_at_time('3', '1', 2.0)
    pq = transformations.pq_from_transform(transform)
    np.testing.assert_allclose(pq, np.array([0.5, 0.5, 0, 1, 0, 0, 0]))

    # NOTE(alexmillane): Turns out the pytransforms3d extrapolates automatically.
    transform = temporal_transform_manager.get_transform_at_time('3', '1', 3.0)
    pq = transformations.pq_from_transform(transform)
    np.testing.assert_allclose(pq, np.array([0.75, 1.0, 0, 1, 0, 0, 0]))


def test_x_rot_motion(tmp_path):
    test_bag_path = tmp_path / 'tf_test_bag'

    # Build the test motion
    tf_msgs, stamps_ns = get_angular_motion(
        angle_rad=np.pi,
        axis=0,
        stamp1_ns=sec_to_ns(0),
        stamp2_ns=sec_to_ns(1),
        frame1='1',
        frame2='2',
    )

    # Write test data
    write_tf_messages_to_bag(tf_msgs, stamps_ns, test_bag_path)
    temporal_transform_manager = rosbag_tf_extraction.get_transform_manager_from_bag(test_bag_path)

    # Test
    transform = temporal_transform_manager.get_transform_at_time('2', '1', 0.5)
    pq = transformations.pq_from_transform(transform)
    q_expected = rotations.quaternion_from_angle(0, 0.5 * np.pi)
    pq_expected = np.hstack([np.array([0, 0, 0]), q_expected])
    np.testing.assert_allclose(pq, pq_expected)

    # Extrapolate
    transform = temporal_transform_manager.get_transform_at_time('2', '1', 1.5)
    pq = transformations.pq_from_transform(transform)
    q_expected = rotations.quaternion_from_angle(0, 1.5 * np.pi)
    assert_quaternion_close(pq[3:], q_expected)


def test_xy_rot_motion(tmp_path: pathlib.Path):
    test_bag_path = tmp_path / 'tf_test_bag'

    # Build the test motion
    tf_msgs_1, stamps_ns_1 = get_angular_motion(
        angle_rad=np.pi,
        axis=0,
        stamp1_ns=sec_to_ns(0),
        stamp2_ns=sec_to_ns(4),
        frame1='1',
        frame2='2',
    )
    tf_msgs_2, stamps_ns_2 = get_angular_motion(
        angle_rad=np.pi,
        axis=1,
        stamp1_ns=sec_to_ns(1),
        stamp2_ns=sec_to_ns(3),
        frame1='2',
        frame2='3',
    )

    tf_msgs = tf_msgs_1 + tf_msgs_2
    stamps_ns = stamps_ns_1 + stamps_ns_2

    # Write test data
    write_tf_messages_to_bag(tf_msgs, stamps_ns, test_bag_path)
    temporal_transform_manager = rosbag_tf_extraction.get_transform_manager_from_bag(test_bag_path)

    # Test
    transform = temporal_transform_manager.get_transform_at_time('3', '1', 2.0)
    pq = transformations.pq_from_transform(transform)
    q_expected = rotations.quaternion_from_angle(0, np.pi)
    pq_expected = np.hstack([np.array([0, 0, 0]), q_expected])
    np.testing.assert_allclose(pq, pq_expected, atol=1e-6)
