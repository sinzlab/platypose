from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import torch

from chick.config import get_experiment_config
from chick.energies import full_gaussian_2d_energy, w_monocular_2d_energy
from chick.pipeline import SkeletonPipeline
from chick.utils.reproducibility import set_random_seed
from chick.wehrbein.data.data_h36m import H36MDataset
from propose.propose.cameras.Camera import Camera
from propose.propose.evaluation.mpjpe import mpjpe
from propose.propose.poses.human36m import MPII_2_H36M, Human36mPose, MPIIPose

cfg = get_experiment_config()

if __name__ == "__main__":
    set_random_seed(cfg.seed)

    dataset = H36MDataset(
        "./dataset/testset_h36m.pickle",
        quick_eval=True,
        quick_eval_stride=16,
        train_set=False,
        # actions=['Smoking']
        # actions=['SittingDown']
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

    # plt.figure(figsize=(5, 5))
    # # Human36mPose(x_gt_projected.squeeze().detach().cpu().numpy()).plot(plot_type='none', c='r')
    # MPIIPose(-dataset.mean[:200].squeeze().detach().cpu().numpy() / 64 + 0.5).plot(plot_type='none', c='g', alpha=0.1)
    # plt.axis('equal')
    # plt.savefig('test.png')
    # plt.close()

    # exit()

    pipe = SkeletonPipeline.from_pretrained(cfg.model.name)

    ms = []
    for idx, batch in enumerate(dataloader):
        # if idx < 17:
        #     continue

        poses_3d = batch["poses_3d"]
        center = batch["center"]

        poses_2d = torch.distributions.MultivariateNormal(**batch["poses_2d"])
        camera = Camera(**batch["camera"])

        loc = poses_2d.loc[0]
        covariance_matrix = poses_2d.covariance_matrix[0]
        dist = torch.distributions.MultivariateNormal(
            loc=loc,
            covariance_matrix=covariance_matrix,
        )

        xx = (
            torch.stack(
                torch.meshgrid(torch.arange(0, 64, 0.1), torch.arange(0, 64, 0.1)),
                dim=-1,
            )
            .reshape(-1, 1, 2)
            .float()
        )
        log_prob = dist.log_prob(xx.cuda()).reshape(640, 640, 16).exp().sum(-1)

        # plt.imshow(log_prob.detach().cpu().numpy().T, extent=[0, -64, -64, 0])
        # MPIIPose(-loc.detach().cpu().numpy()).plot()
        # plt.savefig('dist.png')
        # plt.close()

        # exit()

        # print(center)
        # # center[..., -1] = 5
        # x_gt_projected = -camera.proj2D(poses_3d + center)
        # x_gt_projected = x_gt_projected - x_gt_projected[:, [0]]
        # # x_gt_projected = x_gt_projected / 250 * 32
        # x_gt_projected = x_gt_projected / x_gt_projected.std() * 32 / 3
        #
        # poses_2d = -poses_2d.loc.squeeze().detach().cpu().numpy()
        # poses_2d = poses_2d - poses_2d[[6]]
        #
        # plt.figure(figsize=(5, 5))
        # Human36mPose(x_gt_projected.squeeze().detach().cpu().numpy()).plot(plot_type='none', c='r')
        # MPIIPose(poses_2d).plot(plot_type='none', c='g', alpha=1)
        # plt.axis('equal')
        # plt.savefig('test.png')
        # plt.close()
        #
        # exit()

        og_pose = poses_2d.loc.clone().cpu()
        og_pose = og_pose - og_pose[:, 6]
        og_pose = og_pose * 4

        poses_2d.loc = poses_2d.loc[0, MPII_2_H36M]
        poses_2d.covariance_matrix = poses_2d.covariance_matrix[0, MPII_2_H36M]
        poses_2d.loc = poses_2d.loc - poses_2d.loc[0]
        # poses_2d.loc *= -1
        # poses_2d.loc = poses_2d.loc[:, 1:]
        # poses_2d.covariance_matrix = poses_2d.covariance_matrix[:, 1:]

        # pose2d = poses_2d.loc.clone().cpu()
        # insert 0 at index 9
        # pose2d = torch.cat((pose2d[:, :9], torch.zeros((1, 1, 2)), pose2d[:, 9:]), dim=1).to(cfg.device)
        # Human36mPose(pose2d.squeeze().detach().cpu().numpy()).plot()
        # plt.savefig('test_2d.png')
        # plt.close()

        # MPIIPose(poses_2d.loc.squeeze().detach().cpu().numpy()).plot(c='r', plot_type='none')
        # plt.savefig('test_2d.png')
        # plt.close()

        # x_gt_projected = -camera.proj2D(poses_3d + center)
        # x_gt_projected = x_gt_projected - x_gt_projected[:, [0]]
        #
        # poses_2d.loc = x_gt_projected
        # poses_2d.loc = torch.cat((dist.loc[:, :9], dist.loc[:, 10:]), dim=1)
        # # poses_2d.loc = poses_2d.loc[MPII_2_H36M]
        #
        # print(poses_2d.loc.shape)

        x_gt_projected = -camera.proj2D(poses_3d + center)
        x_gt_projected = x_gt_projected - x_gt_projected[:, [0]]
        scale = x_gt_projected.std()

        poses_2d = torch.distributions.MultivariateNormal(
            loc=poses_2d.loc,
            covariance_matrix=poses_2d.covariance_matrix,
        )

        out = pipe.sample(
            num_samples=50,
            num_frames=cfg.model.num_frames,
            num_substeps=cfg.experiment.num_substeps,
            energy_fn=partial(
                full_gaussian_2d_energy,
                dist=poses_2d,
                center=center,
                camera=camera
                # w_monocular_2d_energy, x_2d=poses_2d.loc, dist=poses_2d, center=center, camera=camera
            ),
            energy_scale=cfg.experiment.energy_scale,
            # energy_scale=0,
        )

        *_, _sample = out  # get the last element of the generator (the real pose)
        sample = _sample["sample"].detach()  # * 1000
        # sample = sample[:1]
        # print(sample.shape)

        x_sample = sample + center.T

        x_2d_projected = -camera.proj2D(x_sample[..., 0])
        x_2d_projected = x_2d_projected - x_2d_projected[:, [0]]

        # print(x_2d_projected.mean(), x_2d_projected.std())
        # exit()

        x_gt_projected = x_gt_projected / x_gt_projected.std()
        x_2d_projected = x_2d_projected / x_2d_projected.std()
        og_pose = og_pose / og_pose.std()

        # energy = full_gaussian_2d_energy(poses_3d.unsqueeze(-1), dist = poses_2d, center = center, camera = camera, plot=True)
        # print(energy)

        # scale = og_pose.std() / x_2d_projected.std()
        # x_2d_projected = x_2d_projected * scale
        # x_gt_projected = x_gt_projected * scale

        poses_3d[:, :, [1, 2]] = -poses_3d[:, :, [2, 1]]
        sample[:, :, [1, 2]] = -sample[:, :, [2, 1]]

        #
        # # print(sample.mean(), sample.std(), poses_3d.mean(), poses_3d.std())
        #

        # print(sample.shape)
        # sample = sample / sample.std((1, 2, 3), keepdim=True) * poses_3d.std()

        m = mpjpe(
            poses_3d.squeeze() * 1000, sample.squeeze() * 1000, mean=False
        )  # .min()
        m_idx = m.mean(-1).argmin()
        ms.append(m.mean(-1)[m_idx].detach().cpu().numpy())
        # print(m[m_idx].detach().cpu().numpy())
        print(
            f"{idx}: {np.mean(ms):<7.2f} | {ms[-1]:<7.2f} | {batch['metadata']['action'][0]}"
        )

        # 3d plot of the sample
        # ax = plt.axes(projection='3d')
        # ax.view_init(0, 0)
        # Human36mPose(sample.squeeze().detach().cpu().numpy()).plot(ax=ax, plot_type='none', c='k', alpha=.1)
        # Human36mPose(poses_3d.squeeze().detach().cpu().numpy()).plot(ax=ax, plot_type='none', c='r')
        # ax.set_xlim(-0.5, 0.5)
        # ax.set_ylim(-0.5, 0.5)
        # ax.set_zlim(-1, 1)
        # ax.set_box_aspect(aspect=(0.5, 0.5, 1))
        # plt.savefig('test_3d.png')
        # plt.close()
        #
        # ax = plt.axes(projection='3d')
        # ax.view_init(0, 90)
        # Human36mPose(sample.squeeze().detach().cpu().numpy()).plot(ax=ax, plot_type='none', c='k', alpha=.1)
        # Human36mPose(poses_3d.squeeze().detach().cpu().numpy()).plot(ax=ax, plot_type='none', c='r')
        # ax.set_xlim(-0.5, 0.5)
        # ax.set_ylim(-0.5, 0.5)
        # ax.set_zlim(-1, 1)
        # ax.set_box_aspect(aspect=(0.5, 0.5, 1))
        # plt.savefig('test_3d_rot.png')
        # plt.close()
        #
        # plt.figure(figsize=(5, 5))
        # Human36mPose(x_2d_projected.squeeze().detach().cpu().numpy()).plot(plot_type='none', c='k', alpha=.1)
        # Human36mPose(x_gt_projected.squeeze().detach().cpu().numpy()).plot(plot_type='none', c='r')
        # MPIIPose(-og_pose.squeeze().detach().cpu().numpy()).plot(plot_type='none', c='g')
        # plt.axis('equal')
        # plt.savefig('test.png')
        # plt.close()

        # exit()
        if idx >= 100:
            break
