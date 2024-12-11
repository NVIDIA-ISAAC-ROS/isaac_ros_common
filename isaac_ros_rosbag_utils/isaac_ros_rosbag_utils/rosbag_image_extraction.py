"""
Contains functions to extract the images of multiple h264 encoded videos in parallel.
"""
import concurrent.futures
import logging
import os
import pathlib
import queue
import sys
import threading
import time

import av
import pandas as pd
from rosbags import highlevel

SYNC_THRESHOLD_NS = 0.001 * 10**9


def progress_bar(iteration: int, total: int, prefix='', suffix='', line_length=80, fill='█'):
    length = line_length - len(prefix) - len(suffix)
    percent = ("{0:.1f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    sys.stdout.write(f'\r{prefix} |{bar}| {percent}% {suffix}')
    sys.stdout.flush()


def pyav_format_from_ros_encoding(encoding: str) -> tuple[str, int]:
    """ Convert a ros encoding to a pyav format string and num color channels. """
    ros_to_pyav = {
        'mono8': ('gray8', 1),
        'bgr8': ('bgr24', 3),
        'rgb8': ('rgb24', 3),
    }
    return ros_to_pyav[encoding]


def get_image_path(base_path: pathlib.Path, topic: str, frame_idx: int) -> pathlib.Path:
    """ Get the path to the image with the given index in the given camera. """
    # Remove basename from the topic and remove leading slash.
    last_slash_idx = topic.rfind('/')
    topic = topic[:last_slash_idx]
    topic = topic.lstrip('/')
    if topic == '':
        return base_path / f'{frame_idx:06d}.png'
    return base_path / f'{topic}/{frame_idx:06d}.png'


def _producer(
    reader: highlevel.AnyReader,
    topics: list[str],
    width: int | None,
    height: int | None,
    format: str | None,
    images_base_path: pathlib.Path,
    frame_queue: queue.Queue,
    shutdown_event: threading.Event,
) -> pd.DataFrame:
    """ A function that fills a queue with frames that should be written to disk. """
    if width or height:
        assert width and height, 'Both width and height must be specified.'

    logging.debug('Started producer thread.')
    logging.info(f"Writing images to '{images_base_path}'.")

    # Setup required helper objects.
    decoders: dict[str, av.CodecContext] = {}
    timestamps: dict[str, list[int]] = {}
    camera_indices: dict[str, int] = {}
    for idx, topic in enumerate(topics):
        decoders[topic] = av.CodecContext.create('h264', 'r')
        timestamps[topic] = []
        camera_indices[topic] = idx
        # If the parent directory of the images does not exist it will fail silently.
        get_image_path(images_base_path, topic, 0).parent.mkdir(parents=True, exist_ok=True)

    # Incrementally extract the images from the h264 streams.
    connections = [c for c in reader.connections if c.topic in topics]
    logging.info('Starting to extract images from rosbag.')

    num_messages = sum(1 for _ in reader.messages(connections))

    for idx, (connection, _, rawdata) in enumerate(reader.messages(connections)):
        if idx % 100 == 0:
            progress_bar(idx, num_messages, prefix='Extracting images...', suffix='Done')

        # Deserialize the ROS message.
        topic = connection.topic
        msg = reader.deserialize(rawdata, connection.msgtype)

        if connection.msgtype == 'sensor_msgs/msg/Image':
            # Directly use the uncompressed frame.
            format, num_color_channels = pyav_format_from_ros_encoding(msg.encoding)
            if num_color_channels == 1:
                decoded_frame = av.VideoFrame.from_ndarray(
                    msg.data.reshape(msg.height, msg.width),
                    format=format,
                )
            else:
                decoded_frame = av.VideoFrame.from_ndarray(
                    msg.data.reshape(msg.height, msg.width, num_color_channels),
                    format=format,
                )
        elif connection.msgtype == 'sensor_msgs/msg/CompressedImage':
            # Decode the h264 frame.
            encoded_frame_bytes = msg.data.tobytes()
            try:
                encoded_packet = av.packet.Packet(encoded_frame_bytes)
                decoded_frames = decoders[topic].decode(encoded_packet)
            except av.error.InvalidDataError:
                continue

            assert len(decoded_frames) == 1
            decoded_frame = decoded_frames[0]
        else:
            raise ValueError(f"Unknown message type '{connection.msgtype}' in topic '{topic}'.")

        decoded_frame = decoded_frame.reformat(
            width=width or decoded_frame.width,
            height=height or decoded_frame.height,
            format=format or decoded_frame.format,
        )

        # Store the timestamp corresponding to the frame.
        frame_idx = len(timestamps[topic])
        timestamps[topic].append(msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec)

        frame_queue.put((images_base_path, topic, frame_idx, decoded_frame))

    # Print new line to finish progress bar.
    print("")

    shutdown_event.set()
    logging.info('Finished extracting images from rosbag.')

    # Append -1 to all timestamps lists to make them the same length.
    max_len = max(len(timestamps_list) for timestamps_list in timestamps.values())
    for timestamps_list in timestamps.values():
        timestamps_list += [-1] * (max_len - len(timestamps_list))

    timestamp_df = pd.DataFrame(timestamps)
    timestamp_df.to_csv(images_base_path / 'raw_timestamps.csv')

    return timestamp_df


def _consumer(thread_id: int, frame_queue: queue.Queue, shutdown_event: threading.Event) -> None:
    """ A function that consumes the next frame from the queue and writes it to disk. """
    logging.info(f'Started consumer thread {thread_id}.')
    while True:
        try:
            images_base_path, topic, frame_idx, frame = frame_queue.get(timeout=1)
            image_path = get_image_path(images_base_path, topic, frame_idx)
            logging.debug(f'Writing frame {topic}/{frame_idx} to {image_path}.')
            frame.to_image().save(str(image_path))
        except queue.Empty:
            # If the queue is empty and the shutdown event is set the producer is done, thus we can
            # stop.
            if shutdown_event.is_set():
                break
    logging.info(f'Finished consumer thread {thread_id}.')


def extract_images(
    reader: highlevel.AnyReader,
    topics: list[str],
    width: int,
    height: int,
    format: str,
    images_base_path: pathlib.Path,
    num_workers: int = -1,
) -> pd.DataFrame:
    """
    Extract all images from a rosbag.
    """
    if num_workers == -1:
        num_workers = 2 * (os.cpu_count() or 1)

    if num_workers < 2:
        logging.warning(f"Need at least 2' workers, but "
                        f'only has {num_workers} workers. Increasing num_workers')
        num_workers = 2

    shutdown_event = threading.Event()
    frame_queue: queue.Queue[tuple] = queue.Queue(maxsize=num_workers * 2)

    start = time.time()
    logging.info(f'Starting thread pool with {num_workers} workers.')
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Start 1 thread for the producer.
        producer_future = executor.submit(_producer, reader, topics, width, height, format,
                                          images_base_path, frame_queue, shutdown_event)

        # Use all remaining resources for the consumers.
        consumer_futures = [
            executor.submit(_consumer, i, frame_queue, shutdown_event)
            for i in range(num_workers - 1)
        ]

        # Wait for the producers and consumers to finish.
        all_futures = [producer_future] + consumer_futures
        for future in concurrent.futures.as_completed(all_futures):
            future.result()
        timestamps_df = producer_future.result()

    end = time.time()
    duration_s = end - start
    logging.info(f'Finished extracting all images. Took {duration_s} seconds.')
    return timestamps_df
