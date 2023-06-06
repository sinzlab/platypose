import torch
import wandb


def download_wandb_artefact(artifact_name):
    api = wandb.Api()
    artifact = api.artifact(artifact_name, type="model")

    if wandb.run:
        wandb.run.use_artifact(artifact, type="model")

    dirs = "/".join(artifact_name.split("/")[:-1])
    artifact_dir = artifact.download("./models/" + dirs)
    model_name = artifact_name.split("/")[-1]

    # rename the model file to artifact_name and create the necessary directory
    import os

    os.rename(artifact_dir + "/model.pt", artifact_dir + model_name + ".pt")
    os.chmod(artifact_dir + model_name + ".pt", 0o777)

    return torch.load(artifact_dir + model_name + ".pt", map_location="cpu")
