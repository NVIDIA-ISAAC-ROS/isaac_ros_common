import pathlib
import subprocess

# flake8: noqa

# This is a WAR until we can specify pip deps in package.xml
REQUIREMENTS_FILE = pathlib.Path(__file__).parent.parent / 'requirements.txt'
subprocess.call(['python3', '-m', 'pip', 'install', '-r', str(REQUIREMENTS_FILE)])

import yaml
import ament_index_python.packages

from isaac_ros_rosbag_utils import rosbag_edex_extraction

SCRIPT_DIR = pathlib.Path(__file__).parent


def count_lines(path: pathlib.Path) -> int:
    with path.open('rb') as f:
        return sum(1 for _ in f)


def get_r2b_galileo() -> pathlib.Path:
    return pathlib.Path(
        ament_index_python.packages.get_package_share_directory(
            'isaac_ros_r2b_galileo')) / 'data/r2b_galileo'


def get_config(path: pathlib.Path, rosbag_path: pathlib.Path,
               edex_path: pathlib.Path) -> rosbag_edex_extraction.Config:
    yaml_string = path.read_text()
    yaml_dict = yaml.safe_load(yaml_string)
    yaml_dict['rosbag_path'] = rosbag_path
    yaml_dict['edex_path'] = edex_path
    return rosbag_edex_extraction.Config(**yaml_dict)


def test_edex_extraction(tmp_path: pathlib.Path):
    edex_path = tmp_path / 'edex'
    config = get_config(
        SCRIPT_DIR / '../config/edex_extraction_nova.yaml',
        get_r2b_galileo(),
        edex_path,
    )
    rosbag_edex_extraction.extract_edex(config)

    assert edex_path.is_dir()
    assert (edex_path / 'robot.urdf').is_file()
    assert (edex_path / 'stereo.edex').is_file()
    assert (edex_path / 'frame_metadata.jsonl').is_file()
    assert count_lines(edex_path / 'frame_metadata.jsonl') == 335


def test_video_extraction(tmp_path: pathlib.Path):
    edex_path = tmp_path / 'edex'
    config = get_config(
        SCRIPT_DIR / '../config/edex_extraction_nova.yaml',
        get_r2b_galileo(),
        edex_path,
    )
    rosbag_edex_extraction.extract_videos(config)

    assert edex_path.is_dir()
    assert (edex_path / 'videos/front_stereo_camera_left.h264').is_file()
    assert (edex_path / 'videos/front_stereo_camera_right.h264').is_file()
    assert (edex_path / 'videos/left_stereo_camera_left.h264').is_file()
    assert (edex_path / 'videos/left_stereo_camera_right.h264').is_file()
    assert (edex_path / 'videos/right_stereo_camera_left.h264').is_file()
    assert (edex_path / 'videos/right_stereo_camera_right.h264').is_file()
    assert (edex_path / 'videos/back_stereo_camera_left.h264').is_file()
    assert (edex_path / 'videos/back_stereo_camera_right.h264').is_file()
