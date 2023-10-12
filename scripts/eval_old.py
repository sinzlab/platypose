import sys
import time
from functools import partial

import numpy as np
import torch
from tqdm import tqdm

sys.path.append("/src")

from chick.config import cfg_to_dict, get_experiment_config
from chick.dataset.h36m import H36MVideoDataset
from chick.energies import energies
from chick.pipeline import SkeletonPipeline
from chick.platform import platform
from chick.utils.palettes import palettes
from chick.utils.plot_utils import plot_2D, plot_3D
from chick.utils.reproducibility import set_random_seed
from propose.propose.cameras.Camera import Camera, DummyCamera
from propose.propose.evaluation.calibration import calibration
from propose.propose.evaluation.mpjpe import mpjpe, pa_mpjpe
from propose.propose.poses.human36m import Human36mPose, MPIIPose

# Parameters
cfg = get_experiment_config()

Cam = {
    "dummy": DummyCamera,
    "camera": Camera,
}[cfg.experiment.projection]

if __name__ == "__main__":
    print(cfg)
    platform.init(project="chick", entity="sinzlab", name=f"eval_{time.time()}")
    platform.config.update(cfg_to_dict(cfg))

    set_random_seed(cfg.seed)

    dataset = H36MVideoDataset(
        path=cfg.dataset.full_path,
        root_path=cfg.dataset.root,
        frames=cfg.model.num_frames,
        keypoints=cfg.experiment.keypoints,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=False
    )

    pipe = SkeletonPipeline.from_pretrained(cfg.model.name)
    pipe.eval()

    # metric storage
    mpjpes = []
    pa_mpjpes = []
    total = 0
    quantile_counts = torch.zeros((21, 17))

    pbar = tqdm(enumerate(dataloader), total=len(dataloader))
    sampling_count = 0
    for k, (
        batch_cam,
        gt_3D,
        input_2D,
        action,
        subject,
        scale,
        bb_box,
        cam_ind,
        start_3d,
    ) in pbar:
        if k < 3:
            continue
        if k > 3:
            exit()

        cam_dict = dataset.dataset.cameras()[subject[0]][cam_ind.item()]

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

        print(
            dict(
                intrinsic_matrix=cam_dict["intrinsic"][:4],
                rotation_matrix=torch.eye(3),
                translation_vector=torch.zeros((1, 3)),
                tangential_distortion=torch.from_numpy(
                    np.reshape(cam_dict["tangential_distortion"], (1, 2))
                ),
                radial_distortion=torch.from_numpy(
                    np.reshape(cam_dict["radial_distortion"], (1, 3))
                ),
            )
        )

        gt_3D = gt_3D.to(cfg.device)

        center = gt_3D[:, :, 0].clone()

        gt_3D[:, :, 0] = 0
        gt_3D = gt_3D - center.unsqueeze(-2)

        # gt_3D = gt_3D.permute(0, 2, 3, 1)
        # gt_3D_projected = camera.proj2D(gt_3D.permute(0, 3, 1, 2))
        gt_3D_projected = camera.proj2D(gt_3D)
        gt_3D_simple = gt_3D[..., :2]

        import matplotlib.pyplot as plt

        plt.figure()
        ax = plt.subplot(111)
        Human36mPose(gt_3D_projected[0, 0].cpu().numpy()).plot(ax=ax)
        plt.axis("equal")
        plt.savefig("x_2d.png")
        plt.close()

        plt.figure()
        ax = plt.subplot(111)
        Human36mPose(gt_3D_simple[0, 0].cpu().numpy()).plot(ax=ax)
        plt.axis("equal")
        plt.savefig("x_2d_simple.png")
        plt.close()

        exit()

        gt_3D = gt_3D + center.unsqueeze(0).permute(0, 1, 3, 2)

        if cfg.experiment.keypoints == "gt":
            input_2D = gt_3D_projected.clone().permute(0, 2, 3, 1)
        else:
            input_2D = input_2D.to("cuda").permute(0, 2, 3, 1)
            c = input_2D[:, :1]
            input_2D = input_2D - c
            input_2D = -input_2D
            input_2D = input_2D + c

        samples = []
        for _ in range(cfg.experiment.num_repeats):
            out = pipe.sample(
                num_samples=cfg.experiment.num_samples,
                num_frames=cfg.model.num_frames,
                num_substeps=cfg.experiment.num_substeps,
                energy_fn=partial(
                    energies[cfg.experiment.energy_fn],
                    x_2d=input_2D,
                    x_3d=gt_3D,
                    center=center,
                    camera=camera,
                ),
                energy_scale=cfg.experiment.energy_scale,
            )
            *_, _sample = out  # get the last element of the generator (the real pose)
            sample = _sample["sample"].detach()
            samples.append(sample)

        sample = torch.cat(samples, dim=0)

        gt_3D[..., [1, 2], :] = -gt_3D[..., [2, 1], :]

        # moving to cpu to save memory
        gt_3D = gt_3D.cpu()
        sample = sample.cpu()
        center = center.cpu()
        camera.device = "cpu"

        sample = sample - center.unsqueeze(0).permute(0, 1, 3, 2)
        sample_2D_proj = camera.proj2D(sample.permute(0, 3, 1, 2))
        sample_2D_proj = sample_2D_proj.permute(0, 2, 3, 1)

        sample = sample + center.unsqueeze(0).permute(0, 1, 3, 2)
        sample[..., [1, 2], :] = -sample[..., [2, 1], :]

        sample = sample.permute(0, 3, 1, 2)
        gt_3D = gt_3D.permute(0, 3, 1, 2)

        m = mpjpe(gt_3D * 1000, sample * 1000, mean=False).mean(-1).mean(-1)
        idx = m.argmin()
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
            f"minMPJPE: {mpjpes[-1]:.2f} | {np.nanmean(mpjpes):.2f} | ECE: {_calibration_score:.2f} | sampling_count: {sampling_count}"
        )

        platform.log(
            {
                "running_avg_mpjpe": np.mean(mpjpes),
                "mpjpe": m.min().cpu(),
                "action": action[0],
            }
        )

        wrist = sample[..., 13, :]
        wrist = wrist.std(0).mean(-1)

        ankle = sample[..., 6, :]
        ankle = ankle.std(0).mean(-1)

        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 3))
        plt.plot(wrist, c=palettes["chick"]["orange"], marker="o", lw=1, label="Wrist")
        plt.plot(ankle, c=palettes["chick"]["red"], marker="o", lw=1, label="Ankle")

        plt.fill_between(
            np.arange(0, 64),
            0,
            wrist.max(),
            color=palettes["chick"]["black"],
            alpha=0.1,
        )
        plt.fill_between(
            np.arange(192, 256),
            0,
            wrist.max(),
            color=palettes["chick"]["black"],
            alpha=0.1,
        )

        plt.legend(frameon=False)
        plt.xlim(0, 256)
        plt.xlabel("Frame")
        plt.ylabel("Std")

        plt.savefig("wrist_std.png", dpi=150, bbox_inches="tight")
        plt.close()

        # joint_a = sample[..., 1, :]
        # joint_b = sample[..., 2, :]
        #
        # angle = torch.atan2(joint_b[..., 1] - joint_a[..., 1], joint_b[..., 0] - joint_a[..., 0]).detach().cpu().numpy()
        # # unravel the angle so that values can be greater than pi
        # angle = np.unwrap(angle)
        #
        # joint_a = gt_3D[..., 1, :]
        # joint_b = gt_3D[..., 2, :]
        #
        # angle_gt = torch.atan2(joint_b[..., 1] - joint_a[..., 1], joint_b[..., 0] - joint_a[..., 0]).detach().cpu().numpy()
        # # unravel the angle so that values can be greater than pi
        # angle_gt = np.unwrap(angle_gt)
        #
        # import matplotlib.pyplot as plt
        #
        # plt.figure(figsize=(10, 3))
        # plt.plot(angle_gt[0], c=palettes['chick']['orange'], lw=3, label='Ground Truth')
        # plt.plot(angle[0], c=palettes['chick']['black'], lw=1, alpha=0.25, label='Hypothesis')
        # for i in range(1, cfg.experiment.num_samples * cfg.experiment.num_repeats):
        #     plt.plot(angle[i], c=palettes['chick']['black'], lw=1, alpha=0.25)
        #
        # # shade in the observed region 0:64 and 192:256
        # plt.fill_between(np.arange(0, 64), -5, 5, color=palettes['chick']['black'], alpha=0.1)
        # plt.fill_between(np.arange(192, 256), -5, 5, color=palettes['chick']['black'], alpha=0.1)
        #
        # plt.ylim(-5, 5)
        #
        # plt.legend(frameon=False)
        # plt.xlabel('Frame')
        # plt.ylabel('Hip angle')
        # plt.savefig('angle.png', dpi=150, bbox_inches='tight')
        # plt.close()
        #
        # joint_a = sample[..., 2, :]
        # joint_b = sample[..., 3, :]
        #
        # angle = torch.atan2(joint_b[..., 1] - joint_a[..., 1], joint_b[..., 0] - joint_a[..., 0])
        #
        # joint_a = gt_3D[..., 2, :]
        # joint_b = gt_3D[..., 3, :]
        #
        # angle_gt = torch.atan2(joint_b[..., 1] - joint_a[..., 1], joint_b[..., 0] - joint_a[..., 0])
        #
        # plt.figure(figsize=(10, 3))
        # plt.plot(angle[idx], c=palettes['chick']['black'], lw=3, label='Hypothesis')
        # plt.plot(angle_gt[0], c=palettes['chick']['orange'], lw=3, label='Ground Truth')
        # plt.legend(frameon=False)
        # plt.xlabel('Frame')
        # plt.ylabel('Knee angle')
        # plt.savefig('angle_knee.png', dpi=150, bbox_inches='tight')
        # plt.close()

        # plot_2D(
        #     gt_3D_projected.permute(0, 2, 3, 1),
        #     input_2D,
        #     sample_2D_proj,
        #     name="2d",
        #     n_frames=cfg.model.num_frames,
        #     alpha=0.1,
        # )

        # plot_3D(
        #     gt_3D.permute(0, 2, 3, 1),
        #     sample.permute(
        #         0, 2, 3, 1
        #     )[idx][None],
        #     alpha=1,
        #     name = "3d",
        # )
