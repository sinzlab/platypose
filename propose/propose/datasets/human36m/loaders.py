import os
import pickle
from pathlib import Path
from typing import Union
from xml.dom import minidom

import cdflib
import numpy as np
from propose.cameras import Camera
from propose.poses.utils import load_data_ids

PathType = Union[str, Path]


def load_poses(path: PathType) -> np.ndarray:
    """
    Loads the poses from the .cdf file.
    :param path: Path to the .cdf file.
    :return: A numpy array with the poses. (N, 17, 3)
    """
    file = cdflib.CDF(path)

    poses_3d = file[0].squeeze()
    assert (
        poses_3d.shape[1] == 96 or poses_3d.shape[1] == 64
    ), f"Wrong number of joints, expected 96, got {poses_3d.shape[1]}"
    n_dims = poses_3d.shape[1] // 32

    dirname = os.path.dirname(__file__)
    metadata_path = os.path.join(dirname, "../../poses/metadata/human36m.yaml")
    joints = load_data_ids(metadata_path)

    # Select only the joints we want
    poses_3d = poses_3d.reshape(-1, 32, n_dims)[:, joints]

    # Reformat the data to be compatible with the propose pose format.
    # poses_3d = poses_3d - poses_3d[:, :1]  # center around root
    # data is saved in x, z, y order in the .cdf file, so we need to swap the z, y axes
    # if n_dims == 3:
    #     poses_3d[..., [1, 2]] = poses_3d[..., [2, 1]]
    #     poses_3d[..., 2] = -poses_3d[..., 2]  # flip the z axis
    #
    # if n_dims == 2:
    #     poses_3d[..., 1] = -poses_3d[..., 1]  # flip the z axis

    return poses_3d


def load_camera(cameras_data, subject, camera):
    """Load h36m camera parameters
    Args
      w0: 300-long array read from XML metadata
      subect: int subject id
      camera: int camera id
    Returns
      camera

    Adapted from https://github.com/una-dinosauria/3d-pose-baseline/blob/master/src/cameras.py
    """

    # Get the 15 numbers for this subject and camera
    camera_data = np.zeros(15)
    start = 6 * (camera * 11 + (subject - 1))
    camera_data[:6] = cameras_data[start : start + 6]
    camera_data[6:] = cameras_data[(265 + camera * 9 - 1) : (264 + (camera + 1) * 9)]

    f, c = camera_data[6:8], camera_data[8:10]
    intrinsic_matrix = Camera.construct_intrinsic_matrix(c[0], c[1], f[0], f[1])
    # rotation_matrix = Camera.construct_rotation_matrix(
    #     alpha=camera_data[0], beta=camera_data[2], gamma=camera_data[1]
    # )
    # translation_vector = camera_data[3:6][np.newaxis, :]

    rotation_matrix = np.eye(3)
    translation_vector = np.zeros((1, 3))

    radial_distortion = camera_data[10:13][np.newaxis, :]
    tangential_distortion = camera_data[13:15][np.newaxis, :][::-1]

    return Camera(
        intrinsic_matrix=intrinsic_matrix,
        rotation_matrix=rotation_matrix,
        translation_vector=translation_vector,
        radial_distortion=radial_distortion,
        tangential_distortion=tangential_distortion,
    )


def load_cameras(path: PathType, subjects: list[int] = [1, 5, 6, 7, 8, 9, 11]) -> dict:
    """Loads the cameras of h36m
    Args
    path: path to xml file with h36m camera data
    subjects: List of ints representing the subject IDs for which cameras are requested
    Returns
    rcams: dictionary of 4 tuples per subject ID containing its camera parameters for the 4 h36m cams

    Adapted from https://github.com/una-dinosauria/3d-pose-baseline/blob/master/src/cameras.py
    """

    metadata_doc = minidom.parse(str(path))
    cameras_data = metadata_doc.getElementsByTagName("w0")[0].firstChild.data[1:-1]
    cameras_data = np.array(
        list(map(float, cameras_data.split(" ")))
    )  # Parse camera_data into floats

    assert (
        len(cameras_data) == 300
    ), "The loaded xml file does not contain w0 with 300 numbers. Make sure the metadata.xml file is correct."

    camera_names = ["54138969", "55011271", "58860488", "60457274"]

    cameras = {}
    for subject_idx in subjects:
        subject_key = f"S{subject_idx}"
        cameras[subject_key] = {
            camera_names[camera_idx]: load_camera(cameras_data, subject_idx, camera_idx)
            for camera_idx in range(len(camera_names))
        }

    return cameras


def load_poses_both(path: PathType, subject: str) -> np.ndarray:
    # Load 3D Poses
    path_3d = Path(path) / "3D" / "S1" / "MyPoseFeatures" / "D3_Positions_mono"
    poses_3d = load_poses(path_3d / f"Directions.54138969.cdf")

    # Load 2D Poses
    path_2d = Path(path) / "2D" / "S1" / "MyPoseFeatures" / "D2_Positions"
    poses_2d = load_poses(path_2d / f"Directions.54138969.cdf")

    return poses_3d, poses_2d
