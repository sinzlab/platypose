from yacs.config import CfgNode as CN
from typing import Literal
import argparse

Projections = Literal['camera', 'dummy']
Keypoints = Literal['cpn_ft_h36m_dbb', 'gt']
Datasets = Literal['h36m']

_C = CN()

_C.experiment = CN()
_C.experiment.num_samples = 200
_C.experiment.energy_scale = 30
_C.experiment.num_frames = 29
_C.experiment.model = "sinzlab/chick/MDM_H36m_1_frame_50_steps:latest"
_C.experiment.seed = 1
_C.experiment.projection: Projections = "dummy"
_C.experiment.dataset: Datasets = "h36m"
_C.experiment.keypoints: Keypoints = "gt"


parser = argparse.ArgumentParser(description='Experiment settings.')
parser.add_argument('--dataset', type=str, help='Dataset to use.')
parser.add_argument('--keypoints', type=str, help='2D detections to use.')
parser.add_argument('--num_samples', type=int, help='Number of samples.')
parser.add_argument('--energy_scale', type=int, help='Energy scale.')
parser.add_argument('--num_frames', type=float, help='Number of frames.')
parser.add_argument('--model', type=str, help='Model to use.')
parser.add_argument('--seed', type=int, help='Random seed.')
parser.add_argument('--projection', type=str, help='Projection to use.')
parser.add_argument('--experiment', type=str, help='Name of the experiment.')

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
        raise ValueError("You must provide an experiment_name either as an argument or as a parameter")

    cfg = _C.clone()
    cfg.merge_from_file(f"./experiments/{experiment_name}.yaml")

    cfg = merge_args(cfg, args)

    cfg.freeze()
    return cfg


if __name__ == "__main__":
    cfg = get_experiment_config()
    print(cfg)
