import sys
import time
from functools import partial

import numpy as np
import torch
from tqdm import tqdm
import gc

import subprocess

# subprocess.run(["pip", "install", "seaborn"])

from propose.propose.poses.human36m import Human36mPose

sys.path.append("/src")

from chick.config import cfg_to_dict, get_experiment_config
from chick.dataset.temporal import Human36mDataset
from chick.energies import energies
from chick.pipeline import SkeletonPipeline
from chick.platform import platform
from chick.utils.palettes import palettes
from chick.utils.plot_utils import plot_2D, plot_3D, plot_arrows
from chick.model.simplebaseline import LinearModel, init_weights
from chick.utils.reproducibility import set_random_seed
from propose.propose.cameras.Camera import Camera, DummyCamera
from propose.propose.poses.human36m import MPIIPose
from propose.propose.evaluation.calibration import calibration
from propose.propose.evaluation.mpjpe import mpjpe, pa_mpjpe, p_mpjpe, mpjve

def show_memory_stats():
    print("== Memory Stats ==")
    print("torch.cuda.memory_allocated: %fGB" % (torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024))
    print("torch.cuda.memory_reserved: %fGB" % (torch.cuda.memory_reserved(0) / 1024 / 1024 / 1024))
    print("torch.cuda.max_memory_reserved: %fGB" % (torch.cuda.max_memory_reserved(0) / 1024 / 1024 / 1024))
    print("====")

def get_confidence(heatmap):
    heatmap = heatmap / heatmap.sum(dim=(-1, -2), keepdim=True)
    heatmap = heatmap**2 * 30
    # get max value of each heatmap
    max_val = heatmap.amax(dim=-1).amax(dim=-1)
    print(max_val)
    return max_val


def sample_from_heatmap(heatmap, x_2d):
    # Normalize the heatmap along the last two dimensions
    heatmap = heatmap**4
    heatmap = heatmap / heatmap.sum(dim=(-1, -2), keepdim=True)

    # Create a categorical distribution
    dist = torch.distributions.Categorical(heatmap.view(heatmap.shape[-3], -1))

    # Sample from the distribution
    indices = dist.sample((20, )).repeat(10, 1)

    # Convert the indices to 2D coordinates
    coords = torch.stack((indices // 64, indices % 64), dim=-1)

    coords[..., [0, 1]] = coords[..., [1, 0]]
    coords = coords * 2

    # coords = torch.cat([coords, x_2d.squeeze().unsqueeze(0).repeat(5, 1, 1)], dim=0)

    return coords

# Parameters
cfg = get_experiment_config()

if __name__ == "__main__":
    # print file modification date
    import os
    print("Last modified: %s" % time.ctime(os.path.getmtime(__file__)))

    print(cfg)
    platform.init(project="chick", entity="sinzlab", name=f"eval_{time.time()}")
    platform.config.update(cfg_to_dict(cfg))

    set_random_seed(cfg.seed)

    dataset = Human36mDataset(path=cfg.dataset.full_path, augment=False, train=False, chunk_size=cfg.model.num_frames, stride=cfg.dataset.stride, subjects=cfg.dataset.subjects)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)

    pipe = SkeletonPipeline.from_pretrained(cfg.model.name)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    mpjpes = []
    vels = []
    pa_mpjpes = []
    # depths = []
    total = 0
    quantile_counts = torch.zeros((21, 17))
    # for gt_3D, x_2d, subject, cameras, heatmaps, crop_size, crop_bb in tqdm(dataloader):
    for gt_3D, x_2d, subject, cameras in tqdm(dataloader):
        # total += 1
        # if total < 2:
        #     continue
        # if total > 3:
        #     exit()

        camera_idx = np.random.choice(len(cameras), size=(cfg.experiment.num_cameras,))
        # print(camera_idx)

        # cameras = [Camera(**camera) for camera in cameras][: cfg.experiment.num_cameras]
        cameras = [Camera(**cameras[idx]) for idx in camera_idx]
        # randomly choose cameras
        # cameras = np.random.choice(cameras, cfg.experiment.num_cameras, replace=False)

        # slice the gt_3D to the number of frames defined in the config
        gt_3D = gt_3D[:, :cfg.model.num_frames]
        gt_3D = gt_3D.to(device)

        # x_2d = x_2d[:, :cfg.experiment.num_cameras, :cfg.model.num_frames, ..., :2].permute(1, 0, 2, 3, 4)
        x_2d = x_2d[:, camera_idx][:, :, :cfg.model.num_frames, ..., :2].permute(1, 0, 2, 3, 4)

        # x_2d = MPIIPose(x_2d.numpy()).to_human36m()
        # x_2d = torch.Tensor(x_2d.pose_matrix)
        x_2d = x_2d.to(device)
        # x_2d[..., 9, :] = (x_2d[..., 8, :] + x_2d[..., 10, :]) / 2


        # add in the trajectory
        gt_3D[:, :, 1:, :] = gt_3D[:, :, 1:, :] + gt_3D[:, :, :1, :]

        # get the hip location in the first frame
        center = gt_3D[:, :, [0], :].cuda()

        # project the gt_3D to 2D
        x_2d_proj = torch.stack([cam.proj2D(gt_3D) for cam in cameras], dim=0)
        # x_2d_proj = torch.stack([cam.proj2D(gt_3D) + torch.randn(size=(1, 1, 17, 2)).to('cuda') * 0.005 for cam in cameras], dim=0)

        x_2d = x_2d - x_2d[..., :1, :] + x_2d_proj[..., :1, :]

        error_per_joint = (x_2d_proj - x_2d).abs()
        error_per_joint[..., 0, :] = 1 / 80
        error_per_joint = 1 / error_per_joint / 80
        error_per_joint = error_per_joint.clip(0, 2)

        # x_2d = x_2d.permute(0, 2, 1, 3, 4)


        if cfg.experiment.keypoints == 'gt':
            error_per_joint = torch.ones_like(error_per_joint)
            x_2d = x_2d_proj

        # x_2d = x_2d - x_2d[..., [0], :]
        # x_2d = x_2d# + torch.randn(size=(200, 1, 17, 2)).to('cuda') * 0.001

        # intrinsic_matrix = cameras[0].intrinsic_matrix
        # R = cameras[0].rotation_matrix
        # t = cameras[0].translation_vector
        # inv_intrinsic_matrix = torch.inverse(intrinsic_matrix)
        #
        # R_inv = torch.inverse(R).cuda()
        # T_inv = -t.cuda()
        #
        # center_in_cam = torch.matmul(center + T_inv, R_inv)# + torch.randn(size=(cfg.experiment.num_samples, 17, 3)).to('cuda') * torch.Tensor([0.1, 0.1, 1]).to('cuda')

        # depth = torch.norm(center).cpu().numpy()
        # depths.append(depth)

        # print(depth)
        # energy_scale = (cfg.experiment.energy_scale - 1) / (np.log(5) - np.log(2)) * np.log(depth)

        r = cameras[0].proj3D(x_2d, center)
        r = r - r[..., :, [0], :]
        # r[..., 16:240, :, :] = torch.randn_like(r[..., 16:240, :, :]) * 1
        r = r.reshape(1, cfg.model.num_frames, -1).repeat(cfg.experiment.num_samples, 1, 1)#

        # r = r.reshape(cfg.experiment.num_samples, 1, 17, 3)
        # r = r + gt_3D[:, :, :1, :]
        # idx = torch.norm(r - gt_3D, dim=-1).mean(-1).argmin()
        # r_proj = torch.stack([cam.proj2D(r) for cam in cameras], dim=0)
        # print(r.shape)
        # print(r_proj.shape)
        # plot_3D(gt_3D.permute(0, 2, 3, 1), r[idx].unsqueeze(-1), alpha=1, name="3d")
        # plot_2D(x_2d_proj[0], x_2d[0], r_proj[:, idx].cpu().numpy(), name="2d", n_frames=cfg.model.num_frames, alpha=1)
        # exit()


        # run the sampling
        gradient_descent = False
        if gradient_descent:
            center_samples = center.repeat(200, 1, 1, 1)
            # sample = cameras[0].proj3D(x_2d, center_samples + torch.randn_like(center_samples) * 1)
            sample = cameras[0].proj3D(x_2d, center_samples + torch.randn_like(center_samples) * 0)
            sample = sample - sample[..., :, [0], :]
            # sample = torch.nn.Parameter(r[[0]])
            # optimizer = torch.optim.Adam([sample], lr=0.1)
            #
            # for _ in range(100):
            #     optimizer.zero_grad()
            #     loss = energies[cfg.experiment.energy_fn](
            #             sample,
            #             x_3d=gt_3D,
            #             x_2d=x_2d,
            #             center=center,
            #             camera=cameras,
            #             mpii=False,
            #             scale=error_per_joint)["train"]
            #     loss.backward()
            #     optimizer.step()

            sample = sample.data.cpu()
            sample = sample.reshape(200, cfg.model.num_frames, 17, 3)

        else:
            samples = []
            for _ in range(cfg.experiment.num_repeats):
                out = pipe.sample(
                    num_samples=cfg.experiment.num_samples,
                    num_frames=cfg.model.num_frames,
                    num_substeps=cfg.experiment.num_substeps,
                    energy_fn=partial(
                        energies[cfg.experiment.energy_fn],
                        x_3d=gt_3D,
                        x_2d=x_2d,
                        center=center,
                        camera=cameras,
                        mpii=False,
                        scale=error_per_joint
                    ),
                    energy_scale=cfg.experiment.energy_scale,
                    ddim=True,
                    init_pose=None,
                    skip_timesteps=2
                )
                # *_, _sample = out  # don't do this, it is memory inefficient

                # memory efficient sampling
                steps = 8
                for i in range(steps):
                    _sample = next(out)
                    if i == steps-1:
                        sample = _sample["sample"].detach().cpu()
                        _sample.clear()
                    else:
                        _sample.clear()

                samples.append(sample)

            sample = torch.cat(samples, dim=0)
            sample = sample.reshape(cfg.experiment.num_repeats * cfg.experiment.num_samples, cfg.model.num_frames, 17, 3)

        # Remove trajectory

        # gt_3D[..., [1, 2]] = gt_3D[..., [2, 1]]
        # gt_3D[..., 2] = -gt_3D[..., 2]

        # sample = sample - sample[:, :, 0:1, :]
        sample[:, :, 0, :] = 0

        # sample = sample[:, [32]]
        # gt_3D = gt_3D[:, [32]]

        # shuffled_sample = torch.stack(
        #     [sample[np.random.choice(np.arange(cfg.experiment.num_samples), size=cfg.experiment.num_samples, replace=False), i] for i in
        #      range(cfg.model.num_frames)], dim=1)
        # print(shuffled_sample.shape)

        sample_2d = [cam.proj2D(sample.cuda() + gt_3D[:, :, 0:1, :]).cuda() for cam in cameras][0]
        x_2d = [cam.proj2D(gt_3D.cuda()).cuda() for cam in cameras][0]

        gt_3D = gt_3D - gt_3D[:, :, 0:1, :]
        gt_3D = gt_3D.cpu()

        m = mpjpe(gt_3D * 1000, sample * 1000, mean=False)

        idx = m.mean(-1).argmin(0)
        # m = m.mean(-1).amin(0).mean(-1) # Seq = False
        # s = sample[idx, np.arange(cfg.model.num_frames)].unsqueeze(0)

        m = m.mean(-1).mean(-1) # Seq = True
        idx = m.argmin()
        idxs = m.argsort()[:5]
        m = m.min()
        s = sample[idx].unsqueeze(0)

        # s = shuffled_sample[idx].unsqueeze(0)

        mpjpes.append(m)

        vel = mpjve(gt_3D * 1000, s * 1000, mean=False).mean(-1).mean(-1)
        vels.append(vel.item())

        # m = pa_mpjpe(gt_3D[:, 0].permute(1, 0, 2) * 1000, sample[:, 0].permute(1, 0, 2) * 1000, mean=False)
        # m = pa_mpjpe(gt_3D[:, 0] * 1000, sample[:, 0] * 1000, mean=False)
        m = p_mpjpe(gt_3D * 1000, sample * 1000, mean=False)
        m = m.mean(-1).mean(-1).min()
        # m = m.mean(-1).amin(0).mean(-1)
        # .mean(-1).mean(-1)
        pa_mpjpes.append(m.min())

        #
        # break

        v, quantiles = calibration(sample, gt_3D)

        if not np.isnan(v).any():
            total += 1
            quantile_counts += v

        _quantile_freqs = quantile_counts / total
        _calibration_score = np.abs(
            np.median(_quantile_freqs, axis=1) - quantiles
        ).mean()

        print(f"mean MPJPE: {np.mean(mpjpes)} | pa-MPJPE: {np.mean(pa_mpjpes)} | MPJVE: {np.mean(vels)} | _calibration_score: {_calibration_score}")

        # del sample

        #
        # plot_2D(
        #     x_2d_proj[0],
        #     x_2d[0],
        #     sample_2d.cpu().numpy(),
        #     name="2d",
        #     n_frames=cfg.model.num_frames,
        #     alpha=0.1,
        # )
        # # exit()
        # #
        # # # print(m.argmin().data)
        # #
        # import matplotlib.pyplot as plt
        # import seaborn as sns
        #
        # sns.set_context('talk')
        # fig = plt.figure(figsize=(10, 5), dpi=300)
        # ax = fig.add_subplot(1, 1, 1)
        # s = sample
        # # s = s + center.cpu()
        # # s = s - s[:, 0:1, :, :]
        # # s = s.numpy()
        #
        # # g = gt_3D + center.cpu()
        # g = gt_3D
        # # g = g - g[:, 0:1, :, :]
        # # g = g.numpy()
        #
        # ax.plot(s[:, :, 3, 0].T, c='k', alpha=0.1)
        # ax.plot(g[:, :, 3, 0].T, c=palettes['chick']['red'])
        # sns.despine()
        #
        # plt.savefig(f"./poses/3d_trajectory.png", dpi=300, bbox_inches="tight", pad_inches=0)
        #
        #
        # fig = plt.figure(figsize=(12, 6), dpi=150)
        # ax = fig.add_subplot(1, 1, 1)
        # ax.set_xlim(-0.25, 1)
        # ax.set_ylim(-1, 1)
        # # plt.axis("off")
        #
        # # gizmo arrows
        # # arrowstyle = ArrowStyle.Fancy(head_width=5, head_length=7)
        # # ax.add_artist(
        # #     FancyArrowPatch(
        # #         (-0.5, -0.97),
        # #         (-0.5, -0.9 + 0.2),
        # #         fc="k",
        # #         ec="k",
        # #         arrowstyle=arrowstyle,
        # #         mutation_scale=1,
        # #     )
        # # )
        # # ax.add_artist(
        # #     FancyArrowPatch(
        # #         (-0.52, -0.95),
        # #         (-0.5 + 0.2, -0.95),
        # #         fc="k",
        # #         ec="k",
        # #         arrowstyle=arrowstyle,
        # #         mutation_scale=1,
        # #     )
        # # )
        #
        # frames = [64, 96, 128, 160, 192, 224, 255]
        # d2 = x_2d[:, frames].cpu().numpy()
        # d2[..., 1] = -d2[..., 1]
        # aux = Human36mPose(d2)
        # aux.plot(ax, plot_type="none", c="tab:red", alpha=1)
        #
        # plt.savefig(f"./poses/2d_example.png", dpi=150, bbox_inches="tight", pad_inches=0)
        # plt.close()
        #
        # fig = plt.figure(figsize=(18, 6), dpi=150)
        # ax = fig.add_subplot(1, 1, 1, projection="3d")
        # ax.view_init(elev=10, azim=0.0)
        # ax.set_xlim(-0.5, 1.75)
        # ax.set_ylim(-0.5, 3.5)
        # ax.set_zlim(-1, 1)
        # ax.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        # ax.yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        # ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        #
        # ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        # ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        # ax.zaxis.set_pane_color((0.95, 0.95, 0.95, 1.0))
        #
        # ax.set_xticks([])
        # ax.set_yticks([])
        # ax.set_zticks([])
        #
        # plt.gca().set_box_aspect(aspect=(1.5, 3.5, 1))
        #
        # # for i in [idx]:
        # s = sample[idxs]
        #
        # s = s + center.cpu()
        # s = s - s[:, 0:1, 0:1, :]
        # s = s.numpy()
        #
        # g = gt_3D + center.cpu()
        # g = g - g[:, 0:1, 0:1, :]
        # g = g.numpy()
        #
        # for sample_idx in range(5):
        #     for i in range(64, 255):
        #         ax.plot(s[sample_idx, i:i + 2, 3, 0], s[sample_idx, i:i + 2, 3, 1], s[sample_idx, i:i + 2, 3, 2], c=palettes["chick"]["black"],
        #                 lw=0.5, alpha=i / 255, zorder=10)
        #         ax.plot(s[sample_idx, i:i + 2, 6, 0], s[sample_idx, i:i + 2, 6, 1], s[sample_idx, i:i + 2, 6, 2], c=palettes["chick"]["black"],
        #                 lw=0.5, alpha=i / 255, zorder=10)
        #         ax.plot(s[sample_idx, i:i + 2, 13, 0], s[sample_idx, i:i + 2, 13, 1], s[sample_idx, i:i + 2, 13, 2], c=palettes["chick"]["black"],
        #                 lw=0.5, alpha=i / 255, zorder=10)
        #         ax.plot(s[sample_idx, i:i + 2, 16, 0], s[sample_idx, i:i + 2, 16, 1], s[sample_idx, i:i + 2, 16, 2], c=palettes["chick"]["black"],
        #                 lw=0.5, alpha=i / 255, zorder=10)
        #
        #
        #     aux = Human36mPose(s[sample_idx, frames])
        #     aux.plot(
        #         ax,
        #         plot_type="none",
        #         c=palettes["chick"]["yellow"],
        #         lw=1.5,
        #         zorder=10,
        #     )
        #
        #     for frame in frames:
        #         aux = Human36mPose(s[sample_idx, frame])
        #         aux.plot(
        #             ax,
        #             plot_type="none",
        #             c=palettes["chick"]["red"],
        #             lw=1.5,
        #             alpha=frame/255,
        #             zorder=10,
        #         )
        #
        # from matplotlib.patches import ArrowStyle
        #
        # arrowstyle = ArrowStyle.Fancy(head_width=5, head_length=7)
        # ax.arrow3D(
        #     -0.56,
        #     -0.57,
        #     -1.0,
        #     6 / 4,
        #     0,
        #     0,
        #     arrowstyle=arrowstyle,
        #     mutation_scale=1,
        #     color=palettes["chick"]["black"],
        # )
        # ax.arrow3D(
        #     -0.5,
        #     -0.59,
        #     -1.0,
        #     0,
        #     1 / 4,
        #     0,
        #     arrowstyle=arrowstyle,
        #     mutation_scale=1,
        #     color=palettes["chick"]["black"],
        # )
        # ax.arrow3D(
        #     -0.5,
        #     -0.57,
        #     -1.06,
        #     0,
        #     0,
        #     2 / 4,
        #     arrowstyle=arrowstyle,
        #     mutation_scale=1,
        #     color=palettes["chick"]["black"],
        # )
        #
        # plt.savefig(
        #     f"./poses/3d_example.png",
        #     dpi=600,
        #     bbox_inches="tight",
        #     pad_inches=0,
        # )

        #
        # plot_3D(
        #     gt_3D.permute(0, 2, 3, 1),
        #     s.permute(
        #         0, 2, 3, 1
        #     ),#[[m.argmin().item()]],
        #     alpha=1,
        #     name="3d",
        # )
        # exit()

        gc.collect()