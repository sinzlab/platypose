import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from torch_geometric.loader.dataloader import Collater
from tqdm import tqdm

from propose.poses.human36m import Human36mPose

from .utils import get_x_graph


def prior_trainer(
    dataloader,
    flow,
    optimizer=None,
    epochs=100,
    device="cpu",
    viz_batch=None,
    use_wandb=False,
):
    """
    Train a flow in a supervised way
    :param dataloader: dataloader for the supervised training
    :param flow: flow to be trained
    :param optimizer: optimizer to be used
    :param epochs: number of epochs
    :param device: device to be used
    :param viz_batch: batch to be visualized
    :param use_wandb: whether to use wandb
    :return: None
    """
    if use_wandb:
        import wandb

    if optimizer is None:
        optimizer = torch.optim.Adam(flow.parameters(), lr=0.001, weight_decay=1e-5)

    # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True, threshold=5e-2)

    for epoch in range(epochs):
        pbar = tqdm(
            dataloader,
            desc=f"Epoch: {epoch + 1}/{epochs} | NLLoss: 0 | RecLoss: 0 | Batch",
        )
        epoch_loss = []
        for _, x_data, action in pbar:
            optimizer.zero_grad()
            x_data.to(device)

            # x_data = x_data.reshape(-1, 16, 3).cuda()
            # x_data = x_data[:, 0].cuda()

            # x_data = x_data.reshape(-1, 16 * 3).cuda()
            loss = -flow(x_data)

            prior_loss = loss.mean()

            loss = prior_loss
            # loss = mse_mode_pose
            # loss = prior_loss.mean()
            loss.backward()

            epoch_loss.append(loss.item())

            nn.utils.clip_grad_norm_(flow.parameters(), max_norm=1.0)

            if use_wandb:
                wandb.log({"Prior Loss": prior_loss.item(), "Loss": loss.item()})

            optimizer.step()

            pbar.set_description(
                f"Epoch: {epoch + 1}/{epochs} | Loss {loss.item():.4f} | Prior Loss {prior_loss.item():.4f} | "
                f"Batch "
            )

        mean_loss = torch.mean(torch.Tensor(epoch_loss))
        pbar.set_description(
            f"Epoch: {epoch + 1}/{epochs} | Loss {torch.mean(mean_loss):.4f} | Prior Loss {prior_loss.item():.4f} | "
            f"Batch "
        )
        # lr_scheduler.step(mean_loss)

        if viz_batch is not None:
            post_eval_value = flow(viz_batch[0][0].cuda()).mean().detach().cpu().numpy()
            prior_eval_value = (
                flow(viz_batch[0][1].cuda()).mean().detach().cpu().numpy()
            )
            print(f"Posterior Evaluation: {post_eval_value.item():2f}")
            print(f"Prior Evaluation: {prior_eval_value.item():2f}")

            if use_wandb:
                df = pd.DataFrame(
                    flow(viz_batch[0][0].cuda()).detach().cpu().numpy().reshape(100, -1)
                )

                with sns.axes_style("whitegrid"):
                    fig = plt.figure(figsize=(5, 5))
                    sns.violinplot(data=df, palette="Set2", ax=plt.gca())

                wandb.log(
                    {
                        "Posterior Evaluation": post_eval_value.item(),
                        "Prior Evaluation": prior_eval_value.item(),
                        "Plot": wandb.Image(plt),
                    }
                )

                plt.close(fig)
