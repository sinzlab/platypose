import os

import torch
import wandb
from torch import nn
from tqdm import tqdm

from chick.utils.wandb import download_wandb_artefact


class Projection(nn.Module):
    """
    Projects a 3D pose (N, 17 * 3) to a 2D pose (N, 17 * 2) using a multi-layer perceptron.
    """

    def __init__(self, num_joints=17):
        super().__init__()
        self.num_joints = num_joints
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = nn.Sequential(
            nn.Linear(num_joints * 3, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_joints * 2),
        )

    def forward(self, x, *args, **kwargs):
        x = x.reshape(x.shape[0], x.shape[1], self.num_joints * 3)

        # add the camera parameters to the input
        # x = torch.cat([x, cam_params.unsqueeze(1).repeat(1, x.shape[1], 1)], dim=-1)

        return self.model(x, *args, **kwargs).reshape(
            x.shape[0], x.shape[1], self.num_joints, 2
        )

    @classmethod
    def pretrain(cls, dataloader, num_epochs=1):
        """
        Train the model on the given dataset.
        """
        self = cls()
        self.cuda()

        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)

        for epoch in range(num_epochs):
            pbar = tqdm(dataloader)
            for (
                batch_cam,
                gt_3D,
                input_2D,
                action,
                subject,
                scale,
                bb_box,
                cam_ind,
            ) in pbar:
                gt_3D[:, :, 0] = 0  # set the root to 0

                # cam_params = []
                # for idx, s in zip(cam_ind, subject):
                #     cam_dict = dataloader.dataset.dataset.cameras()[s][idx]
                #     cam_param = torch.Tensor(cam_dict["intrinsic"][:4])
                #     cam_params.append(cam_param)
                #
                # cam_params = torch.stack(cam_params).cuda()
                # cam_params = cam_dict["intrinsic"][:4]

                # center input_2D around 0
                input_2D = input_2D - input_2D[..., 0:1, :]

                gt_3D = gt_3D.cuda()
                input_2D = input_2D.cuda()

                optimizer.zero_grad()

                output = self(gt_3D)

                input_2D = input_2D.reshape(input_2D.shape[0], -1)
                output = output.reshape(output.shape[0], -1)
                # compute loss
                loss = torch.nn.functional.mse_loss(output, input_2D)
                loss.backward()
                optimizer.step()

                pbar.set_description(f"Epoch {epoch} loss: {loss.item()}")

                if wandb.run is not None:
                    wandb.log({"loss": loss.item()})

        return self

    @staticmethod
    def _get_state_dict(path_or_artefact: str, use_cache: bool = True):
        """
        Checks if path_or_artefact is a path if not tries to download the model from wandb
        :param path_or_artefact: path to model or wandb artifact
        :param use_cache: if true uses the cached model
        :return: state dict of model
        """

        cache_path = "./models/" + path_or_artefact
        # check if ends with .pt
        if not path_or_artefact.endswith(".pt"):
            cache_path = cache_path + ".pt"

        if os.path.exists(cache_path) and use_cache:
            print("Using cached model")
            return torch.load(cache_path, map_location="cpu")
        else:
            return download_wandb_artefact(path_or_artefact)

    @classmethod
    def from_pretrained(
        cls, artefact="sinzlab/chick/projection_cpn_ft_h36m_dbb:latest"
    ):
        """
        Loads a pretrained model from wandb
        :param artefact: wandb artifact name
        :return: None
        """
        chick = cls()

        state_dict = chick._get_state_dict(artefact)

        chick.model.load_state_dict(state_dict)
        chick.requires_grad_(True).eval()
        chick.to(chick.device)

        return chick
