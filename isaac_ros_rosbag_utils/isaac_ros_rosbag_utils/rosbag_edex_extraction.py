import json
import logging
import math
import os
import pathlib
import shutil
from typing import Any

import numpy as np
import pandas as pd
import pydantic

from rosbags import highlevel
from pytransform3d import transform_manager

from isaac_ros_rosbag_utils import rosbag_image_extraction
from isaac_ros_rosbag_utils import rosbag_video_extraction
from isaac_ros_rosbag_utils import rosbag_tf_extraction
from isaac_ros_rosbag_utils import rosbag_urdf_extraction

CV_T_ROS = np.array([
    [00, -1, 0, 0],
    [00, 00, 1, 0],
    [-1, 00, 0, 0],
    [00, 00, 0, 1],
])
ROS_T_CV = np.linalg.inv(CV_T_ROS)
CAMERA_OPTICAL_ROS_T_CAMERA_OPTICAL_CV = np.array([
    [1, 00, 00, 0],
    [0, -1, 00, 0],
    [0, 00, -1, 0],
    [0, 00, 00, 1],
])


class Config(pydantic.BaseModel):
    """Configuration for the bag to edex converter."""
    # Path of the rosbag used for extraction.
    rosbag_path: pathlib.Path
    # Path of the generated edex.
    edex_path: pathlib.Path
    # Topics used to get the camera's intrinsics (and extrinsics if frames are not set explicitly).
    camera_info_topics: list[str]
    # Topics used to extract images. Must be the same length as camera_info_topics.
    image_topics: list[str]
    # Topic used to get IMU measurements.
    imu_topic: str | None = None
    # Frames used to acquire the extrinsics. If not set the frames from the messages will be used:
    rig_frame: str
    camera_optical_frames: list[str] | None = None
    imu_frame: str | None = None
    # Number of workers used in image extraction.
    num_workers: int | None = None
    # Threshold used for syncing images in the same frame.
    sync_threshold_ns: int = int(0.001 * 10**9)
    # Width and height used to resize the extracted images.
    output_width: int | None = None
    output_height: int | None = None
    output_format: str | None = None

    @pydantic.model_validator(mode='after')
    def check_fields(self):
        """ Preprocess the values and then validate that all members are valid. """
        if not self.rosbag_path.exists():
            raise ValueError(f"Path '{self.rosbag_path}' does not exist")
        if len(self.image_topics) != len(self.camera_info_topics):
            raise ValueError("Need same number of image topics as camera info topics.")
        if self.camera_optical_frames:
            if len(self.camera_optical_frames) != len(self.camera_info_topics):
                raise ValueError(
                    "Need same number of camera optical frames as camera info topics.")
        return self


def to_edex_format(pose_matrix: np.array) -> list[list[float]]:
    """ Convert a 4x4 pose matrix to the 3x4 format that edex expects. """
    # Do some safety checks to make sure that the matrix is a valid pose matrix.
    assert pose_matrix.shape == (4, 4)
    assert math.isclose(np.linalg.det(pose_matrix), 1.0)
    assert pose_matrix[3, 3] == 1.0
    return pose_matrix[0:3, :].tolist()


def log_rosbag_info(reader: highlevel.AnyReader):
    """Log the topics and message types of all message channels in the rosbag."""
    logs = [f'\t- {c.topic}: {c.msgtype}' for c in reader.connections]
    logs = sorted(logs)
    # pylint: disable=logging-not-lazy
    logging.info('Found the following topics in rosbag:\n' + '\n'.join(logs))


def get_first_message(reader: highlevel.AnyReader, topics: list[str]) -> list[object]:
    """ Get the first message of every topic. """
    connections = [c for c in reader.connections if c.topic in topics]
    topic_and_first_msg = {}
    for connection, _, rawdata in reader.messages(connections):
        msg = reader.deserialize(rawdata, connection.msgtype)
        topic_and_first_msg[connection.topic] = msg
        if len(topic_and_first_msg) == len(topics):
            break

    # Generate the list in the same order as the input topics.
    return [topic_and_first_msg[topic] for topic in topics]


def synchronize_images(timestamps_df: pd.DataFrame, images_base_path: pathlib.Path,
                       sync_threshold_ns: int) -> pd.DataFrame:
    """
    Synchronize the images based on their timestamp. This will also modify/move the images on
    disk. Returns a dataframe with the synchronized timestamps.
    """
    # Our strategy is to iterate through the timestamps from the front. If all front stamps are
    # inside of the threshold we have a match. Else we increment the index of the earliest
    # timestamp in the front set.

    # Setup helper objects.
    topics = timestamps_df.columns
    front_idx = {topic: 0 for topic in topics}
    frame_idx = 0
    synced_timestamps: dict[str, list[int]] = {topic: [] for topic in topics}

    # Iterate until we reach the end of one image stream.
    while all(front_idx[topic] < timestamps_df.shape[0] for topic in topics):
        # Update the front values list.
        front = [timestamps_df[topic][idx] for topic, idx in front_idx.items()]

        # Values are nan if we reached the end of an image stream.
        if any(np.isnan(front)):
            break

        argmin = np.argmin(front)
        argmax = np.argmax(front)
        if front[argmax] - front[argmin] < sync_threshold_ns:
            # Rename images on disk.
            for topic, old_frame_idx in front_idx.items():
                old_path = rosbag_image_extraction.get_image_path(images_base_path, topic,
                                                                  old_frame_idx)
                new_path = rosbag_image_extraction.get_image_path(images_base_path, topic,
                                                                  frame_idx)
                logging.debug(f'Renaming {old_path} to {new_path}.')
                os.rename(old_path, new_path)
                synced_timestamps[topic].append(timestamps_df[topic][old_frame_idx])
            # Bump all frame indices.
            front_idx = {topic: front_idx[topic] + 1 for topic in topics}
            frame_idx += 1
        else:
            front_idx[topics[argmin]] += 1

    # Remove the leftover images.
    for topic in topics:
        if np.isnan(front_idx[topic]):
            continue
        for old_frame_idx in range(front_idx[topic], timestamps_df.shape[0]):
            old_path = rosbag_image_extraction.get_image_path(images_base_path, topic,
                                                              old_frame_idx)
            old_path.unlink(missing_ok=True)

    # Store the synchronized timestamps.
    synced_timestamp_df = pd.DataFrame(synced_timestamps)
    synced_timestamp_df.to_csv(images_base_path / 'synced_timestamps.csv')
    return synced_timestamp_df


def extract_images(reader: highlevel.AnyReader, config: Config) -> pd.DataFrame:
    """ Extract the images from the bag deterministically. """
    timestamps_df = rosbag_image_extraction.extract_images(
        reader=reader,
        topics=config.image_topics,
        width=config.output_width,
        height=config.output_height,
        format=config.output_format,
        images_base_path=config.edex_path / 'images',
    )
    synced_timestamps_df = synchronize_images(timestamps_df, config.edex_path / 'images',
                                              config.sync_threshold_ns)
    return synced_timestamps_df


def extract_imu_stream(reader: highlevel.AnyReader, config: Config):
    """ Extract all imu messages from the bag and store to disk. """
    imu_path = config.edex_path / 'imu.jsonl'
    logging.info(f"Writing imu data '{imu_path}'.")
    with open(imu_path, 'w', encoding='utf-8') as file:
        connections = [c for c in reader.connections if c.topic == config.imu_topic]
        for connection, _, rawdata in reader.messages(connections):
            msg = reader.deserialize(rawdata, connection.msgtype)

            imu_data = {
                'timestamp': msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec,
                'AngularVelocityX': msg.angular_velocity.x,
                'AngularVelocityY': msg.angular_velocity.y,
                'AngularVelocityZ': msg.angular_velocity.z,
                'LinearAccelerationX': msg.linear_acceleration.x,
                'LinearAccelerationY': msg.linear_acceleration.y,
                'LinearAccelerationZ': msg.linear_acceleration.z,
            }

            json.dump(imu_data, file)
            file.write('\n')


def extract_frame_metadata(synced_timestamps_df: pd.DataFrame, config: Config) -> int:
    """ Extract all frame metadata and store to disk. Returns the number of found frames. """
    topics = synced_timestamps_df.columns
    num_frames = synced_timestamps_df.shape[0]

    with (config.edex_path / 'frame_metadata.jsonl').open('w') as outfile:
        for frame_idx in range(num_frames):
            timestamps = synced_timestamps_df.iloc[frame_idx]

            cams_list = []
            for camera_idx, topic in enumerate(topics):
                path = rosbag_image_extraction.get_image_path(pathlib.Path('images'), topic,
                                                              frame_idx)
                cams_list.append({
                    'id': camera_idx,
                    'filename': str(path),
                    'timestamp': int(timestamps.iloc[camera_idx]),
                })

            out_line = {'frame_id': frame_idx, 'cams': cams_list}
            json.dump(out_line, outfile)
            outfile.write('\n')

    return num_frames


def get_imu_metadata(imu_msg: Any, tf_manager: transform_manager.TransformManager,
                     config: Config) -> dict:
    """ Create the imu metadata needed for the stereo.edex file. """
    if not config.imu_frame:
        config.imu_frame = imu_msg.header.frame_id

    rig_ros_T_imu_ros = tf_manager.get_transform(config.imu_frame, config.rig_frame)
    rig_cv_T_imu_cv = CV_T_ROS @ rig_ros_T_imu_ros @ ROS_T_CV
    imu_metadata = {
        'g': [0.0, -9.81, 0.0],
        'measurements': 'imu.jsonl',
        'transform': to_edex_format(rig_cv_T_imu_cv),
    }
    return imu_metadata


def get_camera_metadata(camera_idx: int, camera_msg: Any,
                        tf_manager: transform_manager.TransformManager, config: Config) -> dict:
    """ Create the camera metadata needed for the stereo.edex file. """
    assert camera_msg.distortion_model == 'rational_polynomial', \
        ('Bag to edex converter cannot (yet) handle other distortion models than ' +
         "'rational_polynomial'")

    if config.camera_optical_frames:
        camera_optical_frame = config.camera_optical_frames[camera_idx]
    else:
        camera_optical_frame = camera_msg.header.frame_id

    rig_ros_T_camera_optical_ros = tf_manager.get_transform(camera_optical_frame, config.rig_frame)
    rig_cv_T_camera_optical_cv = \
        CV_T_ROS @ rig_ros_T_camera_optical_ros @ CAMERA_OPTICAL_ROS_T_CAMERA_OPTICAL_CV

    width_ratio = config.output_width / camera_msg.width if config.output_width else 1.0
    height_ratio = config.output_height / camera_msg.height if config.output_height else 1.0
    if width_ratio != height_ratio:
        logging.warning('The resized images do not have the same aspect ratio. This may lead '
                        'to incorrect results.')

    sx, sy = int(width_ratio * camera_msg.width), int(height_ratio * camera_msg.height)
    # Focal length and principal point of the raw camera.
    #     [fx  0 cx]
    # K = [ 0 fy cy]
    #     [ 0  0  1]
    fx, fy = (width_ratio * camera_msg.k[0]), (height_ratio * camera_msg.k[4])  # noqa: F841
    cx, cy = (width_ratio * camera_msg.k[2]), (height_ratio * camera_msg.k[5])  # noqa: F841
    # Focal length and principal point of the rectified camera.
    #     [fx'  0  cx' Tx]
    # P = [ 0  fy' cy' Ty]
    #     [ 0   0   1   0]
    # pylint: disable=unused-variable
    fx_, fy_ = (width_ratio * camera_msg.p[0]), (height_ratio * camera_msg.p[5])  # noqa: F841
    cx_, cy_ = (width_ratio * camera_msg.p[2]), (height_ratio * camera_msg.p[6])  # noqa: F841
    tx_, ty_ = camera_msg.p[3], camera_msg.p[7]  # noqa: F841

    camera_metadata = {
        'transform': to_edex_format(rig_cv_T_camera_optical_cv),
        'intrinsics': {
            'distortion_model': 'polynomial',
            'distortion_params': camera_msg.d.tolist(),
            'focal': [fx, fy],
            'principal': [cx, cy],
            'size': [sx, sy],
        },
    }
    return camera_metadata


def extract_edex_metadata(
    reader: highlevel.AnyReader,
    tf_manager: transform_manager.TransformManager,
    config: Config,
    num_frames: int,
):
    """ Create the stereo.edex metadata file. """
    camera_info_msgs = get_first_message(reader, config.camera_info_topics)
    cameras_metadata = []
    for idx, msg in enumerate(camera_info_msgs):
        camera_metadata = get_camera_metadata(idx, msg, tf_manager, config)
        cameras_metadata.append(camera_metadata)

    a_metadata = {
        'version': '0.9',
        'frame_start': 0,
        'frame_end': num_frames,
        'cameras': cameras_metadata,
    }

    if config.imu_topic:
        imu_msg = get_first_message(reader, [config.imu_topic])[0]
        imu_metadata = get_imu_metadata(imu_msg, tf_manager, config)
        a_metadata['imu'] = imu_metadata

    sequence_paths = [
        rosbag_image_extraction.get_image_path(pathlib.Path('images'), topic, 0)
        for topic in config.image_topics
    ]
    sequence_paths = [str(path).lstrip('/') for path in sequence_paths]
    b_metadata = {
        'frame_metadata': 'frame_metadata.jsonl',
        'sequence': sequence_paths,
    }
    edex_metadata = [a_metadata, b_metadata]

    edex_metadata_path = config.edex_path / 'stereo.edex'
    logging.info(f"Writing edex metadata to '{edex_metadata_path}'.")
    json_string = json.dumps(edex_metadata, indent=2)
    edex_metadata_path.write_text(json_string)


def extract_edex(config: Config):
    """ Extract the entire edex using the config. """
    # Create edex path.
    shutil.rmtree(config.edex_path, ignore_errors=True)
    config.edex_path.mkdir(parents=True)

    # Extract the URDF from the rosbag.
    tf_manager = rosbag_tf_extraction.get_static_transform_manager_from_bag(config.rosbag_path)
    urdf_content = rosbag_urdf_extraction.get_urdf_from_tf_manager('robot', tf_manager)
    (config.edex_path / 'robot.urdf').write_text(urdf_content)

    with highlevel.AnyReader([config.rosbag_path]) as reader:
        log_rosbag_info(reader)

        # Do some quick checks that all the required data is present in the rosbag.
        # If not we ignore the inexistent topics.
        bag_topics = [c.topic for c in reader.connections]
        image_topics = []
        camera_info_topics = []
        camera_optical_frames = []
        for idx, (image_topic, info_topic) in enumerate(zip(
                config.image_topics, config.camera_info_topics)):
            if image_topic not in bag_topics:
                logging.warning(f"Could not find topic '{image_topic}' in rosbag. Ignoring it.")
            elif info_topic not in bag_topics:
                logging.warning(f"Could not find topic '{info_topic}' in rosbag. Ignoring it.")
            else:
                image_topics.append(image_topic)
                camera_info_topics.append(info_topic)
                if config.camera_optical_frames:
                    camera_optical_frames.append(config.camera_optical_frames[idx])

        config.image_topics = image_topics
        config.camera_info_topics = camera_info_topics
        config.camera_optical_frames = camera_optical_frames or None

        # Extract all data and store to disk.
        synced_timestamps_df = extract_images(reader, config)
        num_frames = extract_frame_metadata(synced_timestamps_df, config)
        extract_edex_metadata(reader, tf_manager, config, num_frames)

        if config.imu_topic:
            extract_imu_stream(reader, config)

    logging.info(f"Finished extracting edex to '{config.edex_path}'.")


def extract_videos(config: Config):
    """ Extract only the videos using the config. """
    # Create edex path.
    shutil.rmtree(config.edex_path, ignore_errors=True)
    config.edex_path.mkdir(parents=True)

    with highlevel.AnyReader([config.rosbag_path]) as reader:
        log_rosbag_info(reader)

        # Do some quick checks that all the required data is present in the rosbag.
        bag_topics = [c.topic for c in reader.connections]
        for topic in config.image_topics:
            assert topic in bag_topics, f"Could not find topic '{topic}' in rosbag."

        rosbag_video_extraction.extract_videos(reader, config.image_topics,
                                               config.edex_path / 'videos')

    logging.info(f"Finished extracting videos to '{config.edex_path}'.")
