# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import collections
import pathlib

import pandas as pd
from pytransform3d import trajectories
from pytransform3d import transform_manager

from rosbags import highlevel


def _extract_tf_dataframe_from_bag(rosbag_path: pathlib.Path) -> pd.DataFrame:
    """
    Read a bag and return a pandas dataframe containing the transforms from /tf and /tf_static.
    """
    data = collections.defaultdict(list)
    with highlevel.AnyReader([rosbag_path]) as reader:
        connections = [x for x in reader.connections if x.topic in ['/tf_static', '/tf']]
        for connection, timestamp, rawdata in reader.messages(connections=connections):
            msg = reader.deserialize(rawdata, connection.msgtype)
            for tf in msg.transforms:
                data['topic'].append(connection.topic)
                data['sec'].append(tf.header.stamp.sec)
                data['nanosec'].append(tf.header.stamp.nanosec)
                data['parent'].append(tf.header.frame_id)
                data['child'].append(tf.child_frame_id)
                data['x'].append(tf.transform.translation.x)
                data['y'].append(tf.transform.translation.y)
                data['z'].append(tf.transform.translation.z)
                data['qw'].append(tf.transform.rotation.w)
                data['qx'].append(tf.transform.rotation.x)
                data['qy'].append(tf.transform.rotation.y)
                data['qz'].append(tf.transform.rotation.z)
    df = pd.DataFrame(data)
    df['time_s'] = df['sec'] + df['nanosec'] * 10**9
    return df


def get_transform_manager_from_bag(
        rosbag_path: pathlib.Path) -> transform_manager.TemporalTransformManager:
    """Reads a bag and returns a TemporalTransformManager containing all transforms inside."""
    df = _extract_tf_dataframe_from_bag(rosbag_path)

    tf_manager = transform_manager.TemporalTransformManager()
    for (parent, child), group in df.groupby(['parent', 'child']):
        timestamps = group['time_s'].to_numpy()
        pqs = group[['x', 'y', 'z', 'qw', 'qx', 'qy', 'qz']].to_numpy()
        tf_series = transform_manager.NumpyTimeseriesTransform(timestamps, pqs)
        tf_manager.add_transform(child, parent, tf_series)

    return tf_manager


def get_static_transform_manager_from_bag(
        rosbag_path: pathlib.Path) -> transform_manager.TransformManager:
    """Reads a bag and returns a TransformManager containing the static transforms only."""
    df = _extract_tf_dataframe_from_bag(rosbag_path)
    # Filter out any non-static tfs.
    df_static = df[df['topic'] == '/tf_static'].reset_index()
    transforms = trajectories.transforms_from_pqs(df_static[['x', 'y', 'z', 'qw', 'qx', 'qy',
                                                             'qz']])
    tf_manager = transform_manager.TransformManager()
    for index, row in df_static.iterrows():
        tf_manager.add_transform(row['child'], row['parent'], transforms[index])
    return tf_manager
