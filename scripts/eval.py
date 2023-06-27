import sys
import time
from functools import partial

import numpy as np
import torch
from tqdm import tqdm

sys.path.append("/src")

from chick.config import cfg_to_dict, get_experiment_config
from chick.dataset.h36m import H36MVideoDataset
from chick.energies import inpaint_2d_energy
from chick.pipeline import SkeletonPipeline
from chick.platform import platform
from chick.utils.plot_utils import plot_2D, plot_3D
from chick.utils.reproducibility import set_random_seed
from propose.propose.cameras.Camera import Camera, DummyCamera
from propose.propose.evaluation.calibration import calibration
from propose.propose.evaluation.mpjpe import mpjpe

# Parameters
cfg = get_experiment_config()

Cam = {
    "dummy": DummyCamera,
    "camera": Camera,
}[cfg.experiment.projection]

if __name__ == "__main__":
    platform.init(project="chick", entity="sinzlab", name=f"eval_{time.time()}")
    platform.config.update(cfg_to_dict(cfg))

    set_random_seed(cfg.seed)

    dataset = H36MVideoDataset(
        path=cfg.dataset.full_path,
        root_path=cfg.dataset.root,
        frames=cfg.model.num_frames,
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
    for k, (
        batch_cam,
        gt_3D,
        input_2D,
        action,
        subject,
        scale,
        bb_box,
        cam_ind,
    ) in pbar:
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

        gt_3D = gt_3D.to(cfg.device)

        center = gt_3D[:, :, 0].clone()
        center = torch.zeros_like(center)

        gt_3D[:, :, 0] = 0
        gt_3D = gt_3D - center.unsqueeze(-2)

        gt_3D = gt_3D.permute(0, 2, 3, 1)
        gt_3D_projected = camera.proj2D(gt_3D.permute(0, 3, 1, 2))
        input_2D = gt_3D_projected.clone().permute(0, 2, 3, 1)

        gt_3D = gt_3D + center.unsqueeze(0).permute(0, 1, 3, 2)
        gt_3D[..., [1, 2], :] = -gt_3D[..., [2, 1], :]

        # samples = []
        # for frame_idx in range(29):
        stds = torch.ones(1, 1, 17, 2).cuda()
        stds[:, :, 3] = 1
        stds[:, :, 16] = 1

        samples = []
        for _ in range(cfg.experiment.num_repeats):
            out = pipe.sample(
                num_samples=cfg.experiment.num_samples,
                num_frames=cfg.model.num_frames,
                num_substeps=cfg.experiment.num_substeps,
                energy_fn=partial(
                    monocular_2d_energy, x_2d=input_2D, x_2d_std=stds, center=center, camera=camera
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
        # samples_3D.append(sample)

        sample = sample.permute(0, 3, 1, 2)
        gt_3D = gt_3D.permute(0, 3, 1, 2)

        # m = mpjpe(gt_3D * 1000, sample * 1000, mean=False).mean(-1).mean(-1)
        # idx = m.argmin()
        # mpjpes.append(np.nanmin(m.cpu().numpy()))
        m = mpjpe(gt_3D * 1000, sample * 1000, mean=False).mean(-1)
        idx = m.argmin(0)
        m = m[idx, torch.arange(m.shape[1])]
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

        # plot_2D(
        #     gt_3D_projected.permute(0, 2, 3, 1),
        #     input_2D,
        #     sample_2D_proj,
        #     f"2{k:02}: 2D {action[0]} {1} frames {n_samples} samples energy scale",
        #     1,
        #     alpha=0.1,
        # )
        #
        # for s in sample:
        #     plot_3D(
        #         gt_3D.permute(0, 2, 3, 1),
        #         s.unsqueeze(0).permute(
        #             0, 2, 3, 1
        #         ),  # [idx, ..., torch.arange(29)].permute(1, 2, 0).unsqueeze(0),#[idx][None],
        #         f"2{k:02}: 3D {action[0]} {1} frames {n_samples} samples energy scale",
        #         frames,
        #         alpha=0.1,
        #     )