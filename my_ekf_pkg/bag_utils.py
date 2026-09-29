"""
Read rosbag2 bags from Python (used by evaluate.py and run_offline.py).

The ROS imports are done inside the functions so that importing this module
does not require a sourced ROS environment.
"""

import os

import yaml


def resolve_bag_dir(path):
    """Return the bag directory for a bag directory or a file inside it."""
    path = os.path.abspath(os.path.expanduser(path))
    if os.path.isfile(path):
        path = os.path.dirname(path)
    if not os.path.isdir(path):
        raise FileNotFoundError(f'Bag not found: {path}')
    return path


def detect_storage_id(bag_dir):
    """Return the rosbag2 storage id ('sqlite3' or 'mcap') of a bag."""
    metadata = os.path.join(bag_dir, 'metadata.yaml')
    if os.path.exists(metadata):
        with open(metadata, 'r') as f:
            info = yaml.safe_load(f) or {}
        storage = info.get('rosbag2_bagfile_information', {}).get('storage_identifier')
        if storage:
            return storage
    for name in sorted(os.listdir(bag_dir)):
        if name.endswith('.mcap'):
            return 'mcap'
        if name.endswith('.db3'):
            return 'sqlite3'
    raise ValueError(f'Cannot determine the storage format of {bag_dir}')


def _open_reader(path):
    """Open a rosbag2 SequentialReader; return (reader, {topic: type})."""
    import rosbag2_py

    bag_dir = resolve_bag_dir(path)
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=bag_dir, storage_id=detect_storage_id(bag_dir)),
        rosbag2_py.ConverterOptions(input_serialization_format='cdr',
                                    output_serialization_format='cdr'))
    type_names = {info.name: info.type for info in reader.get_all_topics_and_types()}
    return reader, type_names


def list_topics(path):
    """Return {topic: type} of a bag."""
    return _open_reader(path)[1]


def read_messages(path, topics=None):
    """
    Yield (topic, message, receive_time_ns) for every message in a bag.

    :param topics: optional iterable of topic names to read (others skipped).
    """
    from rclpy.serialization import deserialize_message
    import rosbag2_py
    from rosidl_runtime_py.utilities import get_message

    reader, type_names = _open_reader(path)
    if topics is not None:
        topics = [t for t in topics if t in type_names]
        if not topics:
            return
        reader.set_filter(rosbag2_py.StorageFilter(topics=topics))

    msg_types = {}
    while reader.has_next():
        topic, data, t_ns = reader.read_next()
        if topic not in msg_types:
            msg_types[topic] = get_message(type_names[topic])
        yield topic, deserialize_message(data, msg_types[topic]), t_ns
