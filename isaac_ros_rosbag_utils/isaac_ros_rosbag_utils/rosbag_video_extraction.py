""" Functions to extract videos from rosbags. """
import logging
import pathlib

from rosbags import highlevel


def get_video_path(base_path: pathlib.Path, topic: str) -> pathlib.Path:
    """ Get the path to the video with the given topic. """
    # Remove basename from the topic and remove leading slash.
    last_slash_idx = topic.rfind('/')
    topic = topic[:last_slash_idx]
    topic = topic.lstrip('/')
    topic = topic.replace('/', '_')
    return base_path / f'{topic}.h264'


def extract_video(reader: highlevel.AnyReader, topic: str, video_path: pathlib.Path):
    """ Extract an image topic from a rosbag and store as an h264 encoded video. """
    # Store topic as an h264 encoded video to disk.
    logging.info(f"Writing h264 video to '{video_path}'.")
    video_path.parent.mkdir(parents=True, exist_ok=True)

    connections = [c for c in reader.connections if c.topic == topic]
    with video_path.open('wb') as file:
        for connection, _, rawdata in reader.messages(connections):
            msg = reader.deserialize(rawdata, connection.msgtype)
            file.write(msg.data.tobytes())


def extract_videos(reader: highlevel.AnyReader, topics: list[str],
                   base_video_path: pathlib.Path) -> list[pathlib.Path]:
    """ Extract multiple images topics from a rosbag and store as an h264 encoded video. """
    video_paths = []
    for topic in topics:
        video_path = get_video_path(base_video_path, topic)
        video_paths.append(video_path)
        extract_video(reader, topic, video_path)
    return video_paths
