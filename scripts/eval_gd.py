import sys
import time
from functools import partial

import numpy as np
import torch
from tqdm import tqdm

from propose.propose.poses.human36m import Human36mPose

sys.path.append("/src")

from platypose.config import cfg_to_dict, get_experiment_config
from platypose.dataset.temporal import Human36mDataset
from platypose.energies import energies
from platypose.pipeline import SkeletonPipeline
from platypose.platform import platform
from platypose.utils.palettes import palettes
from platypose.utils.plot_utils import plot_2D, plot_3D
from platypose.utils.reproducibility import set_random_seed
from propose.propose.cameras.Camera import Camera, DummyCamera
from propose.propose.evaluation.calibration import calibration
from propose.propose.evaluation.mpjpe import mpjpe, pa_mpjpe

# Parameters
cfg = get_experiment_config()

Cam = {
    "dummy": DummyCamera,
    "camera": Camera,
}[cfg.experiment.projection]

# camera = Cam()

minMPJPES = []

if __name__ == "__main__":
    print(cfg)
    platform.init(project="platypose", entity="sinzlab", name=f"eval_{time.time()}")
    platform.config.update(cfg_to_dict(cfg))

    set_random_seed(cfg.seed)

    dataset = Human36mDataset(
        path=cfg.dataset.full_path, augment=False, train=False, stride=256 + 64
    )

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=False
    )

    # dataloader = iter(dataloader)
    # gt_3D, subject, cameras = next(dataloader)
    # gt_3D, subject, cameras = next(dataloader)

    pipe = SkeletonPipeline.from_pretrained(cfg.model.name)

    idx = 0
    for gt_3D, subject, cameras in tqdm(dataloader):
        idx += 1

        if idx == 2:
            break

        # print('load cameras')
        cameras = [Camera(**camera) for camera in cameras][: cfg.experiment.num_cameras]

        # print('load gt_3D')
        gt_3D = gt_3D[:, : cfg.model.num_frames]
        x_3d = gt_3D.reshape(1, cfg.model.num_frames, 17, 3)
        # x_3d[..., [1, 2]] = x_3d[..., [2, 1]]
        # center = x_3d[:, :, :1, :].clone()
        # # x_3d[..., 1] = -x_3d[..., 1]
        # x_3d[:, :, 0, :] = 0
        # x_3d = x_3d + center

        # rotate x_3d
        # x_3d = torch.matmul(x_3d + T_inv, R_inv)

        # x_2d = x_3d[..., :2] #cameras[0].proj2D(x_3d.cuda())
        # x_2d = cameras[0].proj2D(x_3d.cuda()).cuda()
        # x_2d_2 = cameras[1].proj2D(x_3d.cuda()).cuda()
        # print('project 3D to 2D')
        x_2d = [cam.proj2D(x_3d.cuda()).cuda() for cam in cameras]
        # x_2d_1 = cameras[1].proj2D(x_3d.cuda()).cuda()
        # x_2d_2 = cameras[2].proj2D(x_3d.cuda()).cuda()
        # x_2d_3 = cameras[3].proj2D(x_3d.cuda()).cuda()
        #
        # import matplotlib.pyplot as plt
        # plt.figure()
        # ax = plt.subplot(1, 4, 1)
        # Human36mPose(x_2d_0[0, 0].cpu().numpy()).plot(ax=ax)
        # plt.axis('equal')
        #
        # ax = plt.subplot(1, 4, 2)
        # Human36mPose(x_2d_1[0, 0].cpu().numpy()).plot(ax=ax)
        # plt.axis('equal')
        #
        # ax = plt.subplot(1, 4, 3)
        # Human36mPose(x_2d_2[0, 0].cpu().numpy()).plot(ax=ax)
        # plt.axis('equal')
        #
        # ax = plt.subplot(1, 4, 4)
        # Human36mPose(x_2d_3[0, 0].cpu().numpy()).plot(ax=ax)
        # plt.axis('equal')
        #
        # plt.savefig('x_2d.png')
        # plt.close()
        #
        # exit()

        # print('load center')
        center = gt_3D.reshape(1, cfg.model.num_frames, 17, 3)[:, 0, 0, :].cuda()

        # print('reshape gt_3D')
        gt_3D = gt_3D.reshape(1, cfg.model.num_frames, 17, 3).cuda()

        # metric storage
        # mpjpes = []
        # pa_mpjpes = []
        # total = 0
        # quantile_counts = torch.zeros((21, 17))

        # pbar = tqdm(enumerate(dataloader), total=len(dataloader))
        # sampling_count = 0
        # for k, (
        #     gt_3D, subject
        # ) in pbar:
        #     if k < 3:
        #         continue
        #     if k > 3:
        #         exit()

        # print('beginning to sample')

        sample = torch.nn.Parameter(torch.zeros(1, cfg.model.num_frames, 17, 3).cuda())
        optimizer = torch.optim.Adam([sample], lr=0.1)

        for _ in range(100):
            optimizer.zero_grad()
            loss = energies[cfg.experiment.energy_fn](sample, x_2d, cameras)["train"]
            loss.backward()
            optimizer.step()
            # print(loss)

        sample = sample.detach().cpu().numpy()

        # for _ in range(cfg.experiment.num_repeats):
        #     out = pipe.sample(
        #         num_samples=cfg.experiment.num_samples,
        #         num_frames=cfg.model.num_frames,
        #         num_substeps=cfg.experiment.num_substeps,
        #         energy_fn=partial(
        #             energies[cfg.experiment.energy_fn],
        #             x_3d=x_3d,
        #             x_2d=x_2d, center=center, camera=cameras
        #         ),
        #         energy_scale=cfg.experiment.energy_scale,
        #     )
        #     *_, _sample = out  # get the last element of the generator (the real pose)
        #     sample = _sample["sample"].detach().cpu()
        #     samples.append(sample)

        # print('concatenating samples')
        # sample = torch.cat(samples, dim=0)
        #
        # sample = sample[..., :17 * 3].reshape(cfg.experiment.num_samples * cfg.experiment.num_repeats, cfg.model.num_frames, 17, 3)
        #
        # print(sample[0, :, 0])

        import matplotlib.pyplot as plt
        from lipstick import GifMaker
        from tqdm import tqdm

        # 3D scatter plot
        # sample = next(iter(dataset))[0]
        def full_to_position(sample):
            return sample[..., : 17 * 3].reshape(
                sample.shape[0], sample.shape[1], 17, 3
            )[..., :3]

        # sample = full_to_position(sample).cpu().numpy()

        # reapply the center
        center = sample[:, :, :1, :].copy()
        sample[:, :, 0, :] = 0
        # center = center.cpu().numpy()
        sample = sample + center

        # x_3d = x_3d.cpu().numpy()
        # center = x_3d[:, :, :1, :].copy()
        # x_3d[:, :, 0, :] = 0
        # x_3d = x_3d + center

        x_3d = x_3d.cpu().numpy()
        center = x_3d[:, :, :1, :].copy()
        x_3d[:, :, 0, :] = 0
        x_3d = x_3d + center

        # Remove trajectory
        # x_3d = x_3d - x_3d[:, :, 0:1, :]
        # sample = sample - sample[:, :, 0:1, :]

        # Align the first frame
        x_3d = x_3d - x_3d[:, 0:1, 0:1, :]
        sample = sample - sample[:, 0:1, 0:1, :]

        m = mpjpe(x_3d * 1000, sample * 1000, mean=False).mean(-1).mean(-1)
        minMPJPES.append(m.min())

        print(f"MPJPE: {m.min()} | {np.mean(minMPJPES)}")

        # # print(sample.shape)
        with GifMaker("./poses.gif") as g:
            for i in tqdm(range(cfg.model.num_frames)):
                fig = plt.figure(figsize=(10, 10), dpi=150)
                ax = fig.add_subplot(111, projection="3d")

                # draw a 2d plane at z=0
                xx, yy = np.meshgrid(np.linspace(-5, 5, 10), np.linspace(-5, 5, 10))
                z = np.zeros(xx.shape)
                ax.plot_surface(xx, yy, z, alpha=0.025, color="#221114")

                Human36mPose(sample[:, i]).plot(ax=ax, plot_type="none", c="#f96113")
                Human36mPose(x_3d[:, i]).plot(ax=ax, plot_type="none", c="#648FFF")
                # ax.scatter(sample[:, i, :, 0], sample[:, i, :, 1], sample[:, i, :, 2], c='#f96113', marker='o', s=20)
                # ax.scatter(x_3d[:, i, :, 0], x_3d[:, i, :, 1], x_3d[:, i, :, 2], c='b', marker='o', s=20)

                # plot a line following the first joint
                for j in range(len(sample)):
                    ax.plot(
                        sample[j, :i, 0, 0],
                        sample[j, :i, 0, 1],
                        sample[j, :i, 0, 2],
                        c="#221114",
                    )
                    ax.plot(
                        sample[j, :i, 3, 0],
                        sample[j, :i, 3, 1],
                        sample[j, :i, 3, 2],
                        c="#221114",
                    )

                # range -1, 1
                ax.set_xlim(-2, 2)
                ax.set_ylim(-2, 2)
                ax.set_zlim(0, 2.5)

                plt.gca().set_box_aspect(aspect=(1, 1, 2.5 / 4))

                # grid off
                ax.grid(False)
                ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
                ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

                g.add(fig)

            g.save()

            # plot_2D(
            #     gt_3D_projected.permute(0, 2, 3, 1),
            #     input_2D,
            #     sample_2D_proj,
            #     name="2d",
            #     n_frames=cfg.model.num_frames,
            #     alpha=0.1,
            # )
            #
            # plot_3D(
            #     gt_3D.permute(0, 2, 3, 1),
            #     sample.permute(
            #         0, 2, 3, 1
            #     )[idx][None],
            #     alpha=1,
            #     name = "3d",
            # )


# 4 cams: 3.8
# 3 cams: 21.7
# 2 cams: 24.5
# 1 cams: 971.1

# 4 cams | 01 samples: 00.94
# 2 cams | 01 samples: 06.55
# 1 cams | 40 samples: 35.83
