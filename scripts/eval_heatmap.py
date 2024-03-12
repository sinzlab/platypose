import sys
import time
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from tqdm import tqdm

sys.path.append("/src")

from chick.config import cfg_to_dict, get_experiment_config
from chick.dataset.h36m import H36MVideoDataset
from chick.energies import (
    fit_gaussian_2d_energy,
    gaussian_2d_energy,
    heatmap_energy,
    inpaint_2d_energy,
    learned_2d_energy,
    monocular_2d_energy,
)
from chick.heatmap import get_heatmaps, hrnet
from chick.pipeline import SkeletonPipeline
from chick.platform import platform
from chick.projection import Projection
from chick.utils.plot_utils import plot_2D, plot_3D
from chick.utils.reproducibility import set_random_seed
from chick.wehrbein.data.data_h36m import H36MDataset
from propose.propose.cameras.Camera import Camera, DummyCamera
from propose.propose.evaluation.calibration import calibration
from propose.propose.evaluation.mpjpe import mpjpe
from propose.propose.poses.human36m import Human36mPose

# Parameters
cfg = get_experiment_config()

Cam = {
    "dummy": DummyCamera,
    "camera": Camera,
}[cfg.experiment.projection]

if __name__ == "__main__":
    # print(cfg)
    # platform.init(project="chick", entity="sinzlab", name=f"eval_{time.time()}")
    # platform.config.update(cfg_to_dict(cfg))
    #
    set_random_seed(cfg.seed)

    actions = [
        "Directions",
        "Discussion",
        "Eating",
        "Greeting",
        "Phoning",
        "Photo",
        "Posing",
        "Purchases",
        "Sitting",
        "SittingDown",
        "Smoking",
        "Waiting",
        "WalkDog",
        "WalkTogether",
        "Walking",
    ]
    quick_eval_stride = 16
    test_file = "./dataset/testset_h36m.pickle"
    dataset = H36MDataset(
        test_file,
        quick_eval=True,
        quick_eval_stride=quick_eval_stride,
        actions=actions,
        train_set=False,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=False
    )

    pipe = SkeletonPipeline.from_pretrained(cfg.model.name)

    # metric storage
    mpjpes = []
    total = 0
    quantile_counts = torch.zeros((21, 17))

    pbar = tqdm(enumerate(dataloader), total=len(dataloader))
    for k, batch in pbar:
        if k < 1:
            continue

        if k > 10:
            exit()

        gt_3D = batch["p3d_gt"]
        gauss_fits = batch["gauss_fits"].reshape(-1, 6)
        mu = gauss_fits[:, [1, 2]]
        sigma = torch.eye(2).unsqueeze(0).repeat(
            gauss_fits.shape[0], 1, 1
        ).cuda() * gauss_fits[:, [3, 4]].unsqueeze(-1)

        dist = torch.distributions.MultivariateNormal(mu, covariance_matrix=sigma)
        x_coords = torch.dstack(
            torch.meshgrid(torch.arange(-2, 2, 0.1), torch.arange(-2, 2, 0.1))
        )
        x_coords = x_coords.reshape(-1, 1, 2).float().cuda()

        heatmap = dist.log_prob(x_coords).exp().sum(0).reshape(1, 1, 40, 40)

        plt.imshow(heatmap)
        plt.savefig(f"./frame_{k}.png")

        # cam_dict = dataset.dataset.cameras()[subject[0]][cam_ind.item()]

        exit()

        #
        # # plt.imshow(heatmap[0].sum(0))
        # # plt.imshow(image.permute(1, 2, 0))
        # print(heatmap.sum().sum().shape)
        # plt.imshow(heatmap.sum().sum().log())
        #
        # input_2D = input_2D[0]
        # input_2D = input_2D * 500 + 500# - input_2D[:, 0] - input_2D[:, 0]
        # pose = Human36mPose(input_2D[0].numpy())
        # pose.plot()
        # input_2D = input_2D[0]
        # input_2D = input_2D * 500 + 500  # - input_2D[:, 0] - input_2D[:, 0]
        # pose = Human36mPose(input_2D[0].numpy())
        # pose.plot()
        # plt.savefig(f"./frame_{k}.png")

        # plt.close()
        #
        # plt.imshow(frame)
        # input_2D = input_2D[0]
        # input_2D = input_2D * 500 + 500# - input_2D[:, 0] - input_2D[:, 0]
        # pose = Human36mPose(input_2D[0].numpy())
        # pose.plot()
        # plt.savefig(f"./frame_{k}_2.png")
        # plt.close()

        # continue

        camera = Cam(
            intrinsic_matrix=Cam.construct_intrinsic_matrix(*cam_dict["intrinsic"][:4]),
            rotation_matrix=torch.eye(3),
            translation_vector=torch.zeros((1, 3)),
            tangential_distortion=torch.from_numpy(
                np.reshape(cam_dict["tangential_distortion"], (1, 2))
            ),
            radial_distortion=torch.from_numpy(
                np.reshape(cam_dict["radial_distortion"], (1, 3))
            ),
        )

        gt_3D = gt_3D.to(cfg.device)

        center = gt_3D[:, :, 0].clone()
        # center = torch.zeros_like(center)

        gt_3D[:, :, 0] = 0
        gt_3D = gt_3D - center.unsqueeze(-2)

        gt_3D = gt_3D.permute(0, 2, 3, 1)
        gt_3D_projected = camera.proj2D(gt_3D.permute(0, 3, 1, 2))
        input_2D = gt_3D_projected.clone().permute(0, 2, 3, 1)

        # input_2D = input_2D.to(cfg.device).permute(0, 2, 3, 1)
        # center_2d = input_2D[:, [0]].clone()
        # input_2D = input_2D - center_2d
        # input_2D = -input_2D
        # input_2D = input_2D + center_2d

        gt_3D = gt_3D + center.unsqueeze(0).permute(0, 1, 3, 2)
        gt_3D[..., [1, 2], :] = -gt_3D[..., [2, 1], :]

        # samples = []
        # for frame_idx in range(29):
        stds = torch.ones(1, 1, 17, 2).cuda()
        stds[:, :, 3] = 1
        stds[:, :, 16] = 1

        samples = []
        print("sampling")
        for _ in range(cfg.experiment.num_repeats):
            out = pipe.sample(
                num_samples=cfg.experiment.num_samples,
                num_frames=cfg.model.num_frames,
                num_substeps=cfg.experiment.num_substeps,
                energy_fn=partial(
                    fit_gaussian_2d_energy,
                    heatmap=heatmap_full,
                    x_2d=input_2D,
                    x_2d_std=stds,
                    center=center,
                    camera=camera,
                ),
                energy_scale=cfg.experiment.energy_scale,
            )
            *_, _sample = out  # get the last element of the generator (the real pose)
            sample = _sample["sample"].detach()
            samples.append(sample)

        sample = torch.cat(samples, dim=0)

        gt_3D = gt_3D.cpu()
        sample = sample.cpu()
        center = center.cpu()
        camera.device = "cpu"

        sample = sample - center.unsqueeze(0).permute(0, 1, 3, 2)
        sample_2D_proj = camera.proj2D(sample.permute(0, 3, 1, 2))
        sample_2D_proj = sample_2D_proj.permute(0, 2, 3, 1)

        # samples_2D.append(sample_2D_proj)
        sample = sample + center.unsqueeze(0).permute(0, 1, 3, 2)

        sample[..., [1, 2], :] = -sample[..., [2, 1], :]
        # gt_3D[..., [1, 2], :] = -gt_3D[..., [2, 1], :]
        # samples_3D.append(sample)

        sample = sample.permute(0, 3, 1, 2)
        gt_3D = gt_3D.permute(0, 3, 1, 2)

        # m = mpjpe(gt_3D * 1000, sample * 1000, mean=False).mean(-1).mean(-1)
        # idx = m.argmin()
        # mpjpes.append(np.nanmin(m.cpu().numpy()))
        m = mpjpe(gt_3D * 1000, sample * 1000, mean=False).mean(-1).mean(-1)
        idx = m.argmin()
        # idx = m.argmin(0)
        # m = m[idx, torch.arange(m.shape[1])]
        m = m[idx]
        mpjpes.append(np.mean(m.cpu().numpy()))

        v, quantiles = calibration(sample, gt_3D)

        if not np.isnan(v).any():
            total += 1
            quantile_counts += v

        _quantile_freqs = quantile_counts / total
        _calibration_score = np.abs(
            np.median(_quantile_freqs, axis=1) - quantiles
        ).mean()

        pbar.set_description(
            f"{mpjpes[-1]:.2f} | {np.nanmean(mpjpes):.2f} | {_calibration_score:.2f}"
        )

        platform.log(
            {
                "running_avg_mpjpe": np.mean(mpjpes),
                "mpjpe": m.min().cpu(),
                "action": action[0],
            }
        )

        plot_2D(
            gt_3D_projected.permute(0, 2, 3, 1)[..., :1],
            input_2D[..., :1],
            sample_2D_proj[..., :1],
            f"2{k:02}: 2D {action[0]} {1} frames {1} samples energy scale",
            n_frames=1,
            # n_frames=cfg.model.num_frames,
            alpha=0.1,
        )
        # #
        # # for s in sample:
        # plot_3D(
        #     gt_3D.permute(0, 2, 3, 1),#[..., :1],
        #     sample.permute(
        #         0, 2, 3, 1
        #     ),#[..., :1],  # [idx, ..., torch.arange(29)].permute(1, 2, 0).unsqueeze(0),#[idx][None],
        #     f"2{k:02}: 3D {action[0]} {1} frames {1} samples energy scale",
        #     alpha=0.1,
        # )
        #
        # plot_3D(
        #     gt_3D.permute(0, 2, 3, 1),
        #     sample.permute(
        #         0, 2, 3, 1
        #     )[idx][None],#[idx, ..., torch.arange(sample.shape[1])].permute(1, 2, 0).unsqueeze(0),#[idx][None],
        #     f"2{k:02}: 3D {action[0]} {1} frames {1} samples energy scale",
        #     alpha=0.1,
        # )

        if k >= 0:
            exit(0)
