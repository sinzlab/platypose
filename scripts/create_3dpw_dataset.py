from pathlib import Path
import numpy as np

data_dir = Path('./dataset/3dpw/pw3d_test.npz')

data = np.load(data_dir, allow_pickle=True)
dataset = dict(poses={}, poses_2d={}, cameras={})

smpl2h36m = [
    14,
    10,
    1,
    3,
    11,
    0,
    2,
    15,
    12,
    16,
    13,
    4,
    6,
    8,
    5,
    7,
    9,
]

subject = 'a1'
action = 'all'
poses = data['keypoints3d17_relative'][..., :3] + data['root_cam'][:, None, :]
poses = poses[..., smpl2h36m, :]
poses[..., [1, 2]] = poses[..., [2, 1]]
poses[..., 2] = -poses[..., 2]

dataset['poses'][subject] = {}
dataset['poses'][subject][action] = poses.copy()

dataset['poses_2d'][subject] = {}
dataset['poses_2d'][subject][action] = poses.copy()[None, ..., :2][..., smpl2h36m, :]

cam_params = data['cam_param'].item()
R = np.array([[
    [1, 0, 0],
    [0, 0, -1],
    [0, 1, 0]
]]).repeat(poses.shape[0], axis=0)
T = np.zeros((poses.shape[0], 3))
f = cam_params['f']
c = cam_params['c']
k = np.zeros((poses.shape[0], 3))
p = np.zeros((poses.shape[0], 2))
name = [subject] * poses.shape[0]

cam_per_frame = list(zip(R, T, f, c, k, p, name))
dataset['cameras'][(1, 1)] = cam_per_frame

import pickle
with open('./dataset/dataset_3dpw.pkl', 'wb') as f:
    pickle.dump(dataset, f)

