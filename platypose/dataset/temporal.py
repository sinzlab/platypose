import pickle

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from propose.propose.cameras import Camera


def rotate_pose(pose, n_rotations=10):
    """
    Applies a random rotation around the z-axis to the pose.
    :param pose:
    :return:
    """

    angles = np.linspace(-np.pi, np.pi, n_rotations)

    # Create a rotation matrix around the z-axis
    rotation_matrix = np.array(
        [
            [
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1],
            ]
            for angle in angles
        ]
    )

    pose = pose.dot(rotation_matrix).transpose(0, 3, 1, 2, 4)
    return np.concatenate(pose)


def flip_pose(pose):
    """
    Switches the left and the right side of the pose,
    :param pose:
    :return:
    """
    flipped_pose = pose.copy()

    left_joints = [4, 5, 6, 11, 12, 13]
    right_joints = [1, 2, 3, 14, 15, 16]

    flipped_pose[..., :, :, 0] *= -1
    flipped_pose[..., :, left_joints + right_joints, :] = flipped_pose[
                                                          ..., :, right_joints + left_joints, :
                                                          ]

    return np.concatenate([pose, flipped_pose])


def compute_joint_velocity(pose):
    """
    Computes the velocity of each joint and adds it to the state vector. i.e. R3 -> R6
    :param pose:
    :return:
    """
    velocity = np.zeros_like(pose)
    velocity[..., 1:, :] = pose[..., 1:, :] - pose[..., :-1, :]
    return np.concatenate([pose, velocity], axis=-1)


def compute_joint_angles(pose: np.array, bones: list) -> np.array:
    """
    Compute the euler angles of each bone.
    :param pose: pose vector (batch_size, n_frames, 17, 3)
    :param bones: list of bones (parent, child)
    :return: np.array of euler angles (yaw, pitch)
    """
    batch_size, n_frames, n_joints, _ = pose.shape

    # Create an array of bone vectors
    parent_indices, child_indices = zip(*bones)
    parent_joints = pose[:, :, parent_indices]
    child_joints = pose[:, :, child_indices]
    bone_vectors = child_joints - parent_joints

    # Compute yaw and pitch angles
    yaw = np.arctan2(bone_vectors[:, :, :, 1], bone_vectors[:, :, :, 0])  # Yaw angle
    pitch = np.arctan2(
        -bone_vectors[:, :, :, 2],
        np.sqrt(bone_vectors[:, :, :, 0] ** 2 + bone_vectors[:, :, :, 1] ** 2),
    )  # Pitch angle

    # Stack yaw and pitch angles along the last dimension
    euler_angles = np.stack((yaw, pitch), axis=-1)

    return euler_angles


def rotate_joint_angles(joint_angles, yaw_rotation):
    """
    Rotates the joint angles around the yaw axis.
    :param joint_angles: a numpy array of shape (batch_size, n_frames, n_bones, 2)
    :param yaw_rotation: the angle by which the pose was rotated
    :return: new joint angles rotated around the yaw axis
    """
    # Extract the yaw angles from the joint_angles array
    yaw_angles = joint_angles[:, :, :, 0]

    # Add the yaw_rotation angle to the yaw angles
    new_yaw_angles = yaw_angles + yaw_rotation

    # Ensure the new yaw angles are within the range [-pi, pi]
    new_yaw_angles = (new_yaw_angles + np.pi) % (2 * np.pi) - np.pi

    # Create a new joint_angles array with the updated yaw angles
    rotated_joint_angles = np.copy(joint_angles)
    rotated_joint_angles[:, :, :, 0] = new_yaw_angles

    return rotated_joint_angles


def compute_angle_velocities(joint_angles):
    """
    Computes the velocity of each joint angle and adds it to the state vector. i.e. R3 -> R6
    :param joint_angles: (batch_size, n_frames, n_bones, 2)
    :return: joint_angle + joint_angle_velocity
    """
    velocity = np.zeros_like(joint_angles)
    velocity[..., 1:, :, :] = joint_angles[..., 1:, :, :] - joint_angles[..., :-1, :, :]
    return np.concatenate([joint_angles, velocity], axis=-1)


def center_pose(pose):
    """
    Centers the pose around the root joint. i.e. the root joint is at (0, 0, 0). Then puts the translation back into
    at the root joint location. So Hip is removed as it is always (0, 0, 0), instead the translation is stored in the
    first joint.
    :param pose:
    :return:
    """
    centered_pose = pose.copy()
    # centered_pose[..., :2] = centered_pose[..., :2] - centered_pose[..., :1, :1, :2]
    trajectory = centered_pose[..., :1, :].copy()
    centered_pose = centered_pose - trajectory
    centered_pose[..., :1, :] = trajectory
    # centered_pose = centered_pose - centered_pose[..., :1, :1, :]
    return centered_pose


def normalize_2d_pose(poses_2d, window_size=1000):
    """
    Normalize the 2D poses to be in the range [-1, 1]
    :param poses_2d: the 2D poses in pixels
    :param window_size: the size of the window in pixels
    :return: the 2D poses normalized to be in the range [-1, 1]
    """
    return poses_2d / window_size * 2 - 1


bones = [
    (0, 7),
    (7, 8),
    (8, 9),
    (9, 10),
    (0, 1),
    (1, 2),
    (2, 3),
    (0, 4),
    (4, 5),
    (5, 6),
    (8, 11),
    (11, 12),
    (12, 13),
    (8, 14),
    (14, 15),
    (15, 16),
]


class Human36mDataset(Dataset):
    def __init__(self, path, chunk_size=256, stride=256, train=True, augment=True, subjects=None):
        super().__init__()
        print("Loading dataset...")
        self.dataset = pickle.load(open(path, "rb"))
        self.chunk_size = chunk_size
        self.stride = stride

        self.subjects = {
            "train": ["S1", "S5", "S6", "S7", "S8"],
            # 'train': ['S1'],
            # "test": ["S9", "S11"]
            'test': ["TS1", "TS2", "TS3", "TS4", "TS5", "TS6"],
            # 'test': ['a1'],
            # "test": ["S11"]
            # 'test': ['S1']
        }

        if train:
            self.subjects = self.subjects["train"]
        else:
            self.subjects = self.subjects["test"]

        if subjects is not None:
            self.subjects = subjects

        poses = []
        poses_2d = []
        heatmaps_2d = []
        crop_sizes = []
        crop_bb = []
        subjects = []
        cameras = []
        for subject in tqdm(self.subjects):
            actions = list(self.dataset["poses_2d"][subject].keys())
            # actions = ['Walking']

            d3_poses = [self.dataset["poses"][subject][action] for action in actions]
            subject_poses = [
                self.chunk_array(
                    sequence, chunk_size=self.chunk_size, stride=self.stride
                )
                for sequence in d3_poses
            ]
            subject_poses = np.concatenate(subject_poses)
            subject_idx = int(subject[-1:])

            subject_poses_2d_all = []
            # heatmaps = [] #np.zeros((len(subject_poses), 4, 256, 16, 64, 64), dtype=int)
            # crop_sizes_chunks = []
            # crop_bbs_chunks = []

            for camera_idx in range(len(self.dataset["poses_2d"][subject][actions[0]])):
                d2_poses = [self.dataset["poses_2d"][subject][action][camera_idx] for action in actions]

                subject_poses_2d = [
                    self.chunk_array(
                        sequence, chunk_size=self.chunk_size, stride=self.stride
                    )
                    for sequence in d2_poses
                ]

                # d2_heatmaps = [self.dataset["heatmaps"][subject][action][camera_idx]["heatmaps"] for action in actions]
                #
                # crop_size = [np.array(self.dataset["heatmaps"][subject][action][camera_idx]["crop"]["size"]) for action in actions]
                #
                # crop_bb = [np.array(self.dataset["heatmaps"][subject][action][camera_idx]["crop"]["bbox"]) for action in actions]
                #
                # heatmaps_chunks = [
                #     self.chunk_heatmap(
                #         sequence, chunk_size=self.chunk_size, stride=self.stride
                #     )
                #     for sequence in d2_heatmaps
                # ]

                # crop_size_chunks = [np.lib.stride_tricks.as_strided(
                #     arr, shape=(((len(arr) - self.chunk_size) // self.stride) + 1, self.chunk_size), strides=(arr.strides[0] * self.stride,) + arr.strides,
                # ) for arr in crop_size]
                #
                # crop_bb_chunks = [np.lib.stride_tricks.as_strided(
                #     arr, shape=(((len(arr) - self.chunk_size) // self.stride) + 1, self.chunk_size, 4), strides=(arr.strides[0] * self.stride,) + arr.strides,
                # ) for arr in crop_bb]

                subject_poses_2d_all.append(np.concatenate(subject_poses_2d))

                # heatmaps.append(np.concatenate(heatmaps_chunks))
                #
                # crop_sizes_chunks.append(np.concatenate(crop_size_chunks))
                #
                # crop_bbs_chunks.append(np.concatenate(crop_bb_chunks))

            if len(subject_poses_2d_all):
                subject_poses_2d_all = np.stack(subject_poses_2d_all, axis=1)

            # if len(heatmaps):
            #     heatmaps = np.stack(heatmaps, axis=1)
            #
            # if len(crop_sizes_chunks):
            #     crop_sizes_chunks = np.stack(crop_sizes_chunks, axis=1)
            #
            # if len(crop_bbs_chunks):
            #     crop_bbs_chunks = np.stack(crop_bbs_chunks, axis=1)

            subject_cameras = []
            num_cams = 1
            for camera_idx in range(num_cams):
                camera_params = self.dataset["cameras"][
                    (subject_idx, camera_idx + 1)
                ]

                if isinstance(camera_params, tuple):
                    camera_params = [camera_params] * len(subject_poses)

                cam_params_list = []
                for cam_params in camera_params:
                    R, T, f, c, k, p, name = cam_params
                    f = f / 1000 * 2
                    w = 1000
                    h = 1000
                    c = c / w * 2 - [1, h / w]
                    camera = {
                        "intrinsic_matrix": Camera.construct_intrinsic_matrix(
                            c[0], c[1], f[0], f[1]
                        ),
                        "rotation_matrix": torch.Tensor(R),
                        "translation_vector": torch.Tensor(T),
                        "tangential_distortion": torch.Tensor(p),
                        "radial_distortion": torch.Tensor(k),
                    }
                    cam_params_list.append(camera)

                subject_cameras.append(cam_params_list)

            subject_poses = center_pose(subject_poses)
            if augment:
                subject_poses = rotate_pose(subject_poses)
                subject_poses = flip_pose(subject_poses)
                # subject_poses = compute_joint_velocity(subject_poses)
                #
                # joint_angles = compute_joint_angles(subject_poses, bones)
                # joint_angles = compute_angle_velocities(joint_angles)

                subject_poses = subject_poses.reshape(
                    subject_poses.shape[0], subject_poses.shape[1], -1
                )
                # joint_angles = joint_angles.reshape(joint_angles.shape[0], joint_angles.shape[1], -1)

                # subject_poses = np.concatenate([subject_poses, joint_angles], axis=-1)

            poses.append(subject_poses)
            poses_2d.append(subject_poses_2d_all)
            # heatmaps_2d.append(heatmaps)
            # crop_sizes.append(crop_sizes_chunks)
            # crop_bb.append(crop_bbs_chunks)
            subjects.append(np.full(subject_poses.shape[0], subject))
            cameras += list(zip(*subject_cameras))
            # print('length of cams', len(cameras[0]))

        poses = np.concatenate(poses)
        subjects = np.concatenate(subjects)
        poses_2d = np.concatenate(poses_2d)
        # heatmaps_2d = np.concatenate(heatmaps_2d)
        # crop_sizes = crop_sizes_chunks
        # crop_bbs = crop_bbs_chunks

        # if HRNet dont do it
        # poses_2d[..., [0, 1]] = poses_2d[..., [1, 0]]
        # poses_2d = poses_2d * 2

        # do for CPN
        poses_2d = normalize_2d_pose(poses_2d)

        self.data = {"poses": poses, "poses_2d": poses_2d, "subjects": subjects, "cameras": cameras,
                     # "heatmaps": heatmaps_2d, "crop_size": crop_sizes, "crop_bb": crop_bbs
                     }

    def __len__(self):
        return self.data["poses"].shape[0]

    def __getitem__(self, idx):
        return (
            torch.Tensor(self.data["poses"][idx]),
            self.data["poses_2d"][idx],
            self.data["subjects"][idx],
            self.data["cameras"][idx],
            # self.data["heatmaps"][idx],
            # self.data["crop_size"][idx],
            # self.data["crop_bb"][idx],
        )

    @staticmethod
    def chunk_array(arr, chunk_size, stride):
        # Calculate the number of chunks
        n_frames, joints, dims = arr.shape
        num_chunks = ((n_frames - chunk_size) // stride) + 1

        # Calculate the shape of the resulting chunked array
        chunked_shape = (num_chunks, chunk_size, joints, dims)

        # Calculate the strides for the original array to create the chunked view
        strides = (arr.strides[0] * stride,) + arr.strides

        # Use NumPy's stride_tricks to create a view of the chunked array
        chunked_array = np.lib.stride_tricks.as_strided(
            arr, shape=chunked_shape, strides=strides
        )

        return chunked_array

    @staticmethod
    def chunk_heatmap(arr, chunk_size, stride):
        # Calculate the number of chunks
        n_frames, joints, x_dims, y_dims = arr.shape
        num_chunks = ((n_frames - chunk_size) // stride) + 1

        # Calculate the shape of the resulting chunked array
        chunked_shape = (num_chunks, chunk_size, joints, x_dims, y_dims)

        # Calculate the strides for the original array to create the chunked view
        strides = (arr.strides[0] * stride,) + arr.strides

        # Use NumPy's stride_tricks to create a view of the chunked array
        chunked_array = np.lib.stride_tricks.as_strided(
            arr, shape=chunked_shape, strides=strides
        )

        return chunked_array
