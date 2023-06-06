import os
import pickle

import imageio
from neuralpredictors.data.datasets.base import TransformDataset
from propose.poses.rat7m import Rat7mPose

CHUNK_SIZE = 3500


class Rat7mDataset(TransformDataset):
    def __init__(self, dirname: str, transforms=None):
        self.dirname = dirname
        self.data_key = os.path.basename(dirname)

        data_keys = ["poses", "cameras", "images"]

        super().__init__(*data_keys, transforms=transforms)

        self.poses_path = f"{dirname}/poses/{self.data_key}.npy"
        self.cameras_path = f"{dirname}/cameras/{self.data_key}.pickle"
        self.image_dir = f"{dirname}/images"

        self.poses = Rat7mPose.load(self.poses_path)

        with open(self.cameras_path, "rb") as f:
            self.cameras = pickle.load(f)
            self.camera_keys = list(self.cameras.keys())

    def __len__(self):
        return len(self.poses) * len(self.cameras)

    def __getitem__(self, item):
        pose_idx = item % len(self.poses)
        pose = self.poses[pose_idx]

        camera_idx = item // len(self.poses)
        camera_key = self.camera_keys[camera_idx]
        camera = self.cameras[camera_key]

        chunk = camera.frames[pose_idx] // CHUNK_SIZE * CHUNK_SIZE
        image_idx = camera.frames[pose_idx] + 1 - chunk

        image_path = f"{self.image_dir}/{self.data_key}-{camera_key.lower()}-{chunk}/{self.data_key}-{camera_key.lower()}-{image_idx:05d}.jpg"
        image = imageio.imread(image_path)

        data = self.data_point(poses=pose, cameras=camera, images=image)
        data = self.transform(data)

        return data
