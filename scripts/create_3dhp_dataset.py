import h5py
from pathlib import Path
import numpy as np


cameras = [{'cx': 1017.3768231769433,
  'cy': 1043.0617066309674,
  'fx': 1500.0026763683243,
  'fy': 1500.653563770609},
 {'cx': 1015.2332835036037,
  'cy': 1038.6779735645273,
  'fx': 1503.7547333381692,
  'fy': 1501.2960541197708},
 {'cx': 1017.38890576427,
  'cy': 1043.0479217185737,
  'fx': 1499.9948168861915,
  'fy': 1500.5952584161635},
 {'cx': 1017.3629901820193,
  'cy': 1042.9893946483614,
  'fx': 1499.889694845776,
  'fy': 1500.7589012253272},
 {'cx': 939.9366622036999,
  'cy': 560.196743470783,
  'fx': 1683.4033373885632,
  'fy': 1671.9980973522306},
 {'cx': 939.8504013098557,
  'cy': 560.1146111183259,
  'fx': 1683.9052204148456,
  'fy': 1672.674313185811}]

cpm2h36m = [
    14,
    8,
    9,
    10,
    11,
    12,
    13,
    15,
    1,
    16,
    0,
    5,
    6,
    7,
    2,
    3,
    4
]


data_dir = Path('./dataset/mpii3d/test')

subjects = [subject.name for subject in data_dir.glob('TS*')]

dataset = dict(poses={}, poses_2d={}, cameras={})
for subject in subjects:
    subject_idx = int(subject[2:]) - 1

    camera_params = cameras[subject_idx]
    R = np.array([
        [1, 0, 0],
        [0, 0, -1],
        [0, 1, 0]
    ])
    T = np.zeros(3)
    f = np.array([camera_params['fx'], camera_params['fy']])
    c = np.array([camera_params['cx'], camera_params['cy']])
    k = np.array([0, 0, 0])
    p = np.array([0, 0])
    name = subject
    dataset['cameras'][(subject_idx + 1, 1)] = (R, T, f, c, k, p, name)

    dataset['poses'][subject] = dict(all=None)
    dataset['poses_2d'][subject] = dict(all=None)
    subject_file = data_dir / subject / 'annot_data.mat'
    file = h5py.File(subject_file, 'r')
    valid_frames = file['valid_frame'][:, 0].astype(bool)
    print(valid_frames.shape)
    poses_2d = file['annot2'][:, 0][:, cpm2h36m][valid_frames]
    print(poses_2d.shape)
    poses_3d = file['annot3'][:, 0][:, cpm2h36m][valid_frames]
    print(poses_3d.shape)

    poses_3d[..., [1, 2]] = poses_3d[..., [2, 1]]
    poses_3d[..., 2] = -poses_3d[..., 2]
    poses_3d = poses_3d / 1000

    dataset['poses'][subject]['all'] = poses_3d
    dataset['poses_2d'][subject]['all'] = [poses_2d]

import pickle
with open('./dataset/dataset_3dhp.pkl', 'wb') as f:
    pickle.dump(dataset, f)

