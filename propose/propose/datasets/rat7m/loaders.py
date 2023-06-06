from collections import namedtuple

import numpy as np
import propose.datasets.rat7m.transforms as tr
import scipy.io as sio
from neuralpredictors.data.transforms import ScaleInputs, ToTensor
from propose.cameras import Camera
from propose.datasets.rat7m import Rat7mDataset
from propose.poses import Rat7mPose
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

TemporalSplit = namedtuple("TemporalSplit", ["train", "validation", "test"])


def load_cameras(path: str) -> dict:
    """
    Loads the camera parameters for the Rat7M dataset used for mocap.
    :param path: path to the mocap file (e.g. /path/to/mocap-s4-d1.mat)
    :return: dict of cameras.
    """
    data = sio.loadmat(path, struct_as_record=False)
    camera_data = vars(data["cameras"][0][0])

    camera_names = camera_data["_fieldnames"]

    cameras = {}
    for camera_name in camera_names:
        camera_calibration = vars(camera_data[camera_name][0][0])

        camera = Camera(
            intrinsic_matrix=camera_calibration["IntrinsicMatrix"],
            rotation_matrix=camera_calibration["rotationMatrix"],
            translation_vector=camera_calibration["translationVector"],
            tangential_distortion=camera_calibration["TangentialDistortion"],
            radial_distortion=camera_calibration["RadialDistortion"],
            frames=camera_calibration["frame"].squeeze(),
        )

        cameras[camera_name] = camera

    return cameras


def load_mocap(path: str) -> Rat7mPose:
    """
    Loads mocap datafor the Rat7M dataset.
    :param path: path to the mocap file (e.g. /path/to/mocap-s4-d1.mat)
    :return: [Nd array] a Rat7mPose of [frame, joint, xyz]
    """
    d = sio.loadmat(path, struct_as_record=False)
    dataset = vars(d["mocap"][0][0])

    dataset.pop("_fieldnames")

    marker_names = Rat7mPose.marker_names

    n_frames = dataset[marker_names[0]].shape[0]
    n_poses = 20
    n_dims = 3

    poses = np.empty((n_frames, n_poses, n_dims))
    for idx, marker_name in enumerate(marker_names):
        poses[:, idx, :] = dataset[marker_name]

    return Rat7mPose(poses)


def temporal_split_dataset(
    dataset: Rat7mDataset, train_frac: float = 0.6, validation_frac: float = 0.2
) -> TemporalSplit:
    """
    Splits the Rat7mDataset into train and test datasets based on time. i.e. first train_frac% timesteps are used as
    training data
    :param dataset: Dataset for which to get split
    :param train_frac: fraction of timesteps to be used for training
    :param validation_frac: fraction of timesteps to be used for validation
    :return: namedtuple with train and test indexes
    """

    n_frames = len(dataset.poses)
    n_cameras = len(dataset.cameras)

    train_frames = int(n_frames * train_frac)
    val_frames = int(n_frames * validation_frac)

    train = np.concatenate(
        [np.arange(i * n_frames, i * n_frames + train_frames) for i in range(n_cameras)]
    )
    validation = np.concatenate(
        [
            np.arange(
                i * n_frames + train_frames, i * n_frames + train_frames + val_frames
            )
            for i in range(n_cameras)
        ]
    )
    test = np.concatenate(
        [
            np.arange(i * n_frames + train_frames + val_frames, (i + 1) * n_frames)
            for i in range(n_cameras)
        ]
    )

    return TemporalSplit(train=train, validation=validation, test=test)


def static_loader(path: str, batch_size: int, cuda: bool = False) -> dict:
    """
    Constructs train, validation and test DataLoaders.
    :param path: Path the data directory
    :param data_key: data_key of the subject and day e.g. s1-d1
    :param batch_size: Number of samples in each batch.
    :param cuda: Whether cuda should be used
    :return: Dict of DataLoaders.
    """
    transforms = [
        tr.SwitchArmsElbows(),
        tr.CropImageToPose(),
        tr.RotatePoseToCamera(),
        tr.CenterPose(),
        tr.ScalePose(scale=0.03),
        ScaleInputs(scale=0.1, anti_aliasing=True),
        tr.ToGraph(),
        ToTensor(cuda),
    ]

    dat = Rat7mDataset(path, transforms=transforms)

    split_dat = temporal_split_dataset(dat)

    keys = ["train", "validation", "test"]

    dataloaders = {}
    for tier in keys:
        subset_idx = getattr(split_dat, tier)
        sampler = SubsetRandomSampler(subset_idx)
        dataloaders[tier] = DataLoader(dat, sampler=sampler, batch_size=batch_size)

    return dataloaders
