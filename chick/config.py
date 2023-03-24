from yacs.config import CfgNode as CN

_C = CN()

_C.SYSTEM = CN()
# Number of GPUS to use in the experiment
_C.SYSTEM.NUM_GPUS = 8
# Number of workers for doing things
_C.SYSTEM.NUM_WORKERS = 4

_C.TRAIN = CN()
# A very important hyperparameter
_C.TRAIN.HYPERPARAMETER_1 = 0.1
# The all important scales for the stuff
_C.TRAIN.SCALES = (2, 4, 8, 16)


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
    cfg.merge_from_file(f"../configs/{experiment_name}.yaml")
    cfg.freeze()
    return cfg


if __name__ == "__main__":
    import os

    cfg = get_experiment_config("base")
    print(cfg)
