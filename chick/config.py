import argparse
from typing import Literal

import torch
from yacs.config import CfgNode as CN

Projections = Literal["camera", "dummy"]
Keypoints = Literal["cpn_ft_h36m_dbb", "gt"]
Datasets = Literal["h36m"]

_C = CN()

_C.seed = 1
_C.device = "cuda"  # Overwritten when config is loaded

_C.dataset = CN()
_C.dataset.path = "data_3d_h36m.npz"
_C.dataset.root = "./dataset/"
_C.dataset.full_path = _C.dataset.root + _C.dataset.path  # Overwritten when config is loaded

_C.experiment = CN()
_C.experiment.num_samples = 200
_C.experiment.energy_scale = 30
_C.experiment.num_substeps = 1
_C.experiment.num_repeats = 1
_C.experiment.projection: Projections = "dummy"
_C.experiment.dataset: Datasets = "h36m"
_C.experiment.keypoints: Keypoints = "gt"

_C.model = CN()
_C.model.num_frames = 1
_C.model.name = "sinzlab/chick/MDM_H36m_1_frame_50_steps:latest"
_C.model.short_name = "MDM_H36m_1_frame_50_steps"  # Overwritten when config is loaded

_C.train = CN()
_C.train.batch_size = 64
_C.train.num_steps = 600_000
_C.train.lr = 1e-4
_C.train.weight_decay = 0.0


parser = argparse.ArgumentParser(description="Experiment settings.")
parser.add_argument("--dataset", type=str, help="Dataset to use.")
parser.add_argument("--keypoints", type=str, help="2D detections to use.")
parser.add_argument("--num_samples", type=int, help="Number of samples.")
parser.add_argument("--energy_scale", type=int, help="Energy scale.")
parser.add_argument("--num_frames", type=int, help="Number of frames.")
parser.add_argument("--model", type=str, help="Model to use.")
parser.add_argument("--seed", type=int, help="Random seed.")
parser.add_argument("--projection", type=str, help="Projection to use.")
parser.add_argument("--experiment", type=str, help="Name of the experiment.")


def get_cfg_defaults() -> CN:
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()


def merge_args(cfg: CN, args: argparse.Namespace) -> CN:
    """
    Merge the args into the config
    :param cfg: The config object
    :param args: The args from argparse
    :return:
    """
    for key, value in vars(args).items():
        if value is None:
            continue

        if key in cfg.experiment:
            cfg.experiment[key] = value

        if key in cfg.model:
            cfg.model[key] = value

    return cfg


def get_experiment_config(experiment_name: str = None) -> CN:
    """
    It loads the default config, then merges in the config for the experiment you want to run

    :param experiment_name: The name of the experiment. This is used to create a directory to store the results of the
    experiment
    :return: A config object that is a clone of the default config, but with the values from the experiment config file
    merged in.
    """
    args = parser.parse_args()

    if "experiment" in args:
        experiment_name = args.experiment

    if experiment_name is None:
        raise ValueError(
            "You must provide an experiment_name either as an argument or as a parameter"
        )

    cfg = _C.clone()
    cfg.merge_from_file(f"./experiments/{experiment_name}.yaml")

    cfg = merge_args(cfg, args)

    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.model.short_name = list(filter(lambda x: len(x), cfg.model.name.split('/')))[-1].split(':')[0]
    cfg.dataset.full_path = cfg.dataset.root + cfg.dataset.path

    cfg.freeze()
    return cfg


def cfg_to_dict(cfg):
    if isinstance(cfg, CN):
        return {key: cfg_to_dict(cfg[key]) for key in cfg.keys()}
    else:
        return cfg


if __name__ == "__main__":
    cfg = get_experiment_config()
    print(cfg)
