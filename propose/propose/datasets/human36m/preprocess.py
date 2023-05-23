import os
import pickle
from pathlib import Path
from typing import Union

import cdflib
import numpy as np

from propose.datasets.human36m.loaders import load_cameras, load_poses
from propose.poses.human36m import MPII_2_H36M

PathType = Union[str, Path]


def process_pose(pose):
    """
    Reformat the data to be compatible with the propose pose format.
    :param pose: A numpy array with the poses. (N, 17, 3)
    :return: A numpy array with the poses. (N, 17, 3)
    """
    n_dims = pose.shape[-1]

    center = pose[:, :1].copy()
    pose = pose - center  # center around root

    # data is saved in x, z, y order in the .cdf file, so we need to swap the z, y axes
    if n_dims == 3:
        pose[..., [1, 2]] = pose[..., [2, 1]]
        pose[..., 2] = -pose[..., 2]  # flip the z axis

    if n_dims == 2:
        pose[..., 1] = -pose[..., 1]  # flip the z axis

    return pose, center


def pickle_poses(
    input_dir_path: PathType,
    output_dir_path: Union[str, Path],
    test: bool = False,
    universal: bool = False,
):
    """
    Loads the poses from the .cdf file and saves them as a pickle file.
    :param input_dir_path: Path to the directory containing the .cdf files.
    :param output_dir_path: Path to the directory where the pickle files will be saved.
    :param test: Whether to load the test poses or the train poses.
    :param universal: If True, the universal format of poses will be used..
    """
    if isinstance(input_dir_path, str):
        input_dir_path = Path(input_dir_path)

    input_dir_path_3d = input_dir_path / ("3D" + ("_universal" if universal else ""))
    input_dir_path_2d = input_dir_path / "2D"

    # Load the frames used in ProHPE
    if test:
        with open(input_dir_path / "used_frames.pkl", "rb") as f:
            used_frames = pickle.load(f)

    with open("/data/human36m/processed/cameras.pkl", "rb") as f:
        cameras = pickle.load(f)

    subjects = [subject.name for subject in input_dir_path_3d.glob("S*")]

    if len(subjects) == 0:
        print("No subjects found in the input directory.")
        return

    print(f"Found {len(subjects)} subjects. Subjects: {subjects}")

    dataset = {
        "poses3d": [],
        "poses2d": [],
        "center3d": [],
        "center2d": [],
        "actions": [],
        "cameras": [],
        "subjects": [],
        "occlusions": [],
        "image_paths": [],
        "gaussfits": [],
    }

    with open(input_dir_path_2d / "h36m_without_3d.pickle", "rb") as f:
        d2poses = pickle.load(f)

    for subject in subjects:
        print(f" 🔧 Processing subject {subject}")
        pose3d_dir = (
            input_dir_path_3d
            / subject
            / "MyPoseFeatures"
            / ("D3_Positions_mono" + ("_universal" if universal else ""))
        )
        pose2d_dir = input_dir_path_2d / subject / "MyPoseFeatures" / "D2_Positions"

        for path in pose3d_dir.glob("*.cdf"):
            action = path.name.split(".")[0]
            camera = path.name.split(".")[1]

            act, subact = extract_action_info(action, subject)

            if (
                subject == "S5"
                and subact == "1"
                and act == "Waiting"
                and camera == "55011271"
            ):
                continue
            if subject == "S11" and act == "Directions" and camera == "54138969":
                continue  # Apparently this does not exist.

            gaussfit = d2poses[subject][act][subact][camera]["gaussfit"].reshape(
                -1, 16, 7
            )
            occluded_joints = np.logical_or(
                gaussfit[:, :, 3] > 2.5 * 2.0, gaussfit[:, :, 5] > 2.5 * 2.0
            )

            poses3d = load_poses(path)
            # use every 4th frame to avoid missing frames
            indices = np.arange(3, len(poses3d), 4)
            poses3d = poses3d[indices, :]
            max_frames = len(d2poses[subject][act][subact][camera]["imgpath"])

            # poses3d = poses3d[:max_frames, :]

            # poses2d = load_poses(pose2d_dir / path.name)
            # poses2d = poses2d[indices, :]

            poses3d = poses3d[:max_frames, :]
            poses2d = d2poses[subject][act][subact][camera]["2d_hrnet"]
            image_paths = d2poses[subject][act][subact][camera]["imgpath"]
            poses2d = poses2d.reshape(-1, 2, 16)
            poses2d = poses2d[..., MPII_2_H36M]  # Transform MPII to H36M
            poses2d = np.insert(poses2d, 9, 0, axis=-1)
            poses2d = poses2d.swapaxes(1, 2)

            # poses2d = cameras[subject][camera].proj2D(poses3d)

            poses3d, center3d = process_pose(poses3d)
            poses2d, center2d = process_pose(poses2d)

            # if test:
            # uframes = used_frames[subject][act][subact][camera]
            #
            # poses3d = poses3d[:uframes]
            # poses2d = poses2d[:uframes]

            dataset["poses3d"].append(poses3d)
            dataset["poses2d"].append(poses2d)
            dataset["center3d"].append(center3d)
            dataset["center2d"].append(center2d)
            dataset["occlusions"].append(occluded_joints)
            dataset["image_paths"] += image_paths
            dataset["gaussfits"].append(gaussfit)

            n_frames = len(poses3d)
            dataset["actions"] += [action] * n_frames
            dataset["cameras"] += [camera] * n_frames
            dataset["subjects"] += [subject] * n_frames

    dataset["poses3d"] = np.concatenate(dataset["poses3d"])
    dataset["poses2d"] = np.concatenate(dataset["poses2d"])
    dataset["occlusions"] = np.concatenate(dataset["occlusions"])

    assert dataset["poses3d"].shape[0] == dataset["poses2d"].shape[0]

    print(f" 💾 Saving {dataset['poses3d'].shape[0]:,} poses.")

    output_dir_path.mkdir(exist_ok=True)

    file_path = output_dir_path / "poses.pkl"

    with open(file_path, "wb") as f:
        pickle.dump(dataset, f)


def pickle_cameras(input_dir_path: PathType, output_dir_path: PathType):
    cameras = load_cameras(input_dir_path / "Release-v1.2" / "metadata.xml")

    with open(output_dir_path / "cameras.pkl", "wb") as f:
        pickle.dump(cameras, f)


def extract_action_info(action, subject):
    # Correct the action and subaction names
    act_split = action.split(" ")
    if len(act_split) == 2:
        act, subact = act_split
    else:
        act, subact = act_split[0], "0"

    if act == "TakingPhoto":
        act = "Photo"

    if act == "WalkingDog":
        act = "WalkDog"

    if subject == "S1":
        if act == "Eating" and subact == "2":
            subact = "1"
        if act == "Sitting" and subact == "2":
            subact = "0"
        if act == "SittingDown" and subact == "2":
            subact = "1"

    if subject == "S5":
        if act == "Directions" and subact == "2":
            subact = "0"
        if act == "Discussion" and subact == "2":
            subact = "1"
        if act == "Discussion" and subact == "3":
            subact = "0"
        if act == "Greeting" and subact == "2":
            subact = "0"
        if act == "Photo" and subact == "2":
            subact = "1"
        if act == "Waiting" and subact == "2":
            subact = "0"

    if subject == "S6":
        if act == "Eating" and subact == "2":
            subact = "0"
        if act == "Posing" and subact == "2":
            subact = "1"
        if act == "Sitting" and subact == "2":
            subact = "0"
        if act == "Waiting" and subact == "3":
            subact = "1"

    if subject == "S7":
        if act == "Phoning" and subact == "2":
            subact = "1"
        if act == "Waiting" and subact == "2":
            subact = "0"
        if act == "Walking" and subact == "2":
            subact = "0"

    if subject == "S8":
        if act == "WalkTogether" and subact == "2":
            subact = "0"

    if subject == "S9":
        if act == "Discussion" and subact == "2":
            subact = "0"

    if subject == "S11":
        if act == "Discussion" and subact == "2":
            subact = "0"
        if act == "Greeting" and subact == "2":
            subact = "1"
        if act == "Phoning" and subact == "2":
            subact = "1"
        if act == "Phoning" and subact == "3":
            subact = "0"
        if act == "Smoking" and subact == "2":
            subact = "1"

    return act, subact
