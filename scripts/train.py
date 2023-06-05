import time

import torch
import wandb

from chick.chick import Chick
from chick.config import get_experiment_config
from chick.dataset.h36m import H36MVideoDataset
from chick.platform import platform
from chick.utils.reproducibility import set_random_seed

cfg = get_experiment_config()

if __name__ == "__main__":
    platform.init(project="chick", entity="sinzlab", name=f"train_{time.time()}")
    platform.config.update({key: dict(value) for key, value in cfg.items()})

    set_random_seed(cfg.seed)

    dataset = H36MVideoDataset(
        path=cfg.root_path + cfg.dataset_path,
        root_path=cfg.root_path,
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

    # initialize a clean model
    chick = Chick.train(dataloader, cfg)

    # save the model
    torch.save(chick.model.state_dict(), f"./output/{cfg.model.name}.pt")
    artifact = wandb.Artifact(
        name={cfg.model.name},
        type="model",
    )
    artifact.add_file(f"./output/{cfg.model.name}.pt", name="model.pt")
    wandb.run.log_artifact(artifact)
