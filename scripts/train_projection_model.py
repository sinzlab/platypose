import sys
import time

import torch
import wandb

sys.path.append("/src")

from chick.config import cfg_to_dict, get_experiment_config
from chick.dataset.h36m import H36MVideoDataset
from chick.projection import Projection
from chick.platform import platform
from chick.utils.reproducibility import set_random_seed

cfg = get_experiment_config()

if __name__ == "__main__":
    platform.init(project="chick", entity="sinzlab", name=f"train_projection_{time.time()}")
    platform.config.update(cfg_to_dict(cfg))

    set_random_seed(cfg.seed)

    dataset = H36MVideoDataset(
        path=cfg.dataset.full_path,
        root_path=cfg.dataset.root,
        frames=1,
        mode="train",
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )

    # initialize a clean pipeline
    pipe = Projection.pretrain(dataloader)

    torch.save(pipe.model.state_dict(), f"./models/projection.pt")
    artifact = wandb.Artifact(
        name=f"projection_{cfg.experiment.keypoints}",
        type="model",
    )
    artifact.add_file(f"./models/projection.pt", name="model.pt")
    wandb.run.log_artifact(artifact)
