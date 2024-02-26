import pickle
from pathlib import Path
import numpy as np

def merge_dicts(dict1, dict2):
    merged = dict1.copy()
    for key, value in dict2.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged

with open('./dataset/dataset_cpn.pkl', 'rb') as f:
    dataset = pickle.load(f)

cameras = ["54138969", "55011271", "58860488", "60457274"]

dataset['poses_2d'] = {}
dataset['heatmaps'] = {}
for subject in ['S11']:
    dataset['poses_2d'][subject] = {}
    dataset['heatmaps'][subject] = {}
    for path in Path(f'./dataset/{subject}/HRNetHeatmaps/').glob('*'):
        path = str(path)
        action, camera, _ = path.split('/')[-1].split('.')

        poses = np.load(path, allow_pickle=True)['arr_0'].item()

        if action not in dataset['poses_2d'][subject]:
            dataset['poses_2d'][subject][action] = {}

        if action not in dataset['heatmaps'][subject]:
            dataset['heatmaps'][subject][action] = {}

        dataset['poses_2d'][subject][action][cameras.index(camera)] = poses['heatmaps'][subject][action][camera]['peaks']
        dataset['heatmaps'][subject][action][cameras.index(camera)] = poses['heatmaps'][subject][action][camera]
        print(dataset['poses_2d'][subject][action].keys())

with open('./dataset/dataset_hrnet.pkl', 'wb') as f:
    pickle.dump(dataset, f)