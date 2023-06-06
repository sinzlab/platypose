import sys
import time

import torch
import wandb

sys.path.append("/src")

from chick.config import cfg_to_dict, get_experiment_config
from chick.dataset.h36m import H36MVideoDataset
from chick.pipeline import SkeletonPipeline
from chick.platform import platform
from chick.utils.reproducibility import set_random_seed

cfg = get_experiment_config()

if __name__ == "__main__":
    platform.init(project="chick", entity="sinzlab", name=f"train_{time.time()}")
    platform.config.update(cfg_to_dict(cfg))

    set_random_seed(cfg.seed)

    dataset = H36MVideoDataset(
        path=cfg.dataset.full_path,
        root_path=cfg.dataset.root,
        frames=cfg.model.num_frames,
        mode="train",
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )

    # initialize a clean pipeline
    pipe = SkeletonPipeline.pretrain(dataloader, cfg)

    # save the model
    torch.save(pipe.model.state_dict(), f"./models/{cfg.model.name}.pt")
    artifact = wandb.Artifact(
        name=cfg.model.short_name,
        type="model",
    )
    artifact.add_file(f"./models/{cfg.model.name}.pt", name="model.pt")
    wandb.run.log_artifact(artifact)
