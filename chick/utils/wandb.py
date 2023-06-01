import torch
import wandb


def download_wandb_artefact(artefact_name):
    api = wandb.Api()
    artifact = api.artifact(artefact_name, type="model")

    if wandb.run:
        wandb.run.use_artifact(artifact, type="model")

    artifact_dir = artifact.download()

    return torch.load(artifact_dir + "/model.pt", map_location="cpu")
