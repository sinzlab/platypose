from yacs.config import CfgNode as CN

_C = CN()

_C.EXPERIMENT = CN()
_C.EXPERIMENT.N_SAMPLES = 50
_C.EXPERIMENT.ENERGY_SCALE = 30
_C.EXPERIMENT.FRAMES = 29
_C.EXPERIMENT.MODEL_PATH = "./models/model_30_frames.pt"


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()


def get_experiment_config(experiment_name):
    """
    It loads the default config, then merges in the config for the experiment you want to run

    :param experiment_name: The name of the experiment. This is used to create a directory to store the results of the
    experiment
    :return: A config object that is a clone of the default config, but with the values from the experiment config file
    merged in.
    """
    cfg = _C.clone()
    cfg.merge_from_file(f"../experiments/{experiment_name}.yaml")
    cfg.freeze()
    return cfg


if __name__ == "__main__":
    import os

    cfg = get_experiment_config("base")
    print(cfg)
