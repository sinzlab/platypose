"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import os
import pickle
import random
import re
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import wandb

sys.path.append("/src")

from lipstick import GifMaker
from tqdm import tqdm

from chick.data_loaders.get_data import get_dataset_loader
from chick.model.cfg_sampler import ClassifierFreeSampleModel
from chick.train.train_platforms import (  # required for the eval operation
    ClearmlPlatform, NoPlatform, TensorboardPlatform)
from chick.utils import dist_util
from chick.utils.fixseed import fixseed
from chick.utils.model_util import (create_model_and_diffusion,
                                    load_model_wo_clip)
from chick.utils.parser_util import evaluation_parser, train_args
from common.h36m_dataset import Human36mDataset
from common.load_data_hm36 import Fusion
from common.opt import opts
from common.utils import *
from propose.propose.cameras.Camera import Camera
from propose.propose.evaluation.mpjpe import mpjpe, pa_mpjpe
from propose.propose.poses.human36m import Human36mPose


opt = opts().parse()
args = train_args()
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu


# 49.16 with skip on 1000 samples
# 48.42 with skip on 1000 samples

def calibration(n_sample):
    quantiles = torch.arange(0, 1.05, 0.05)

    if occluded:
        v = np.nanmean((q_vals > true_error.squeeze()).astype(int), axis=1)[
            :, np.newaxis
        ]
    else:
        v = (q_vals > true_error.squeeze()).astype(int)

    if not torch.isnan(v).any():
        total += 1
        quantile_counts += v

    quantile_freqs = quantile_counts / n_sample
    calibration_score = torch.abs(
        torch.median(quantile_freqs, axis=1) - quantiles
    ).mean()
    return calibration_score


def mpjpe(pred, gt, dim=None, mean=True):
    """
    `mpjpe` is the mean per joint position error, which is the mean of the Euclidean distance between the predicted 3D
    joint positions and the ground truth 3D joint positions

    Used in Protocol-I for Human3.6M dataset evaluation.

    :param pred: the predicted 3D pose
    :param gt: ground truth
    :param dim: the dimension to average over. If None, the average is taken over all dimensions
    :param mean: If True, returns the mean of the MPJPE across all frames. If False, returns the MPJPE for each frame,
    defaults to True (optional)
    :return: The mean of the pjpe
    """
    pjpe = ((pred - gt) ** 2).sum(-1) ** 0.5

    if not mean:
        return pjpe

    # if pjpe is torch.Tensor use dim if numpy.array use axis
    if isinstance(pjpe, torch.Tensor):
        if dim is None:
            return pjpe.mean()
        return pjpe.mean(dim=dim)

    if dim is None:
        return np.mean(pjpe)

    return np.mean(pjpe, axis=dim)


def pa_mpjpe(
    p_gt: torch.TensorType, p_pred: torch.TensorType, dim: int = None, mean: bool = True
):
    """
    PA-MPJPE is the Procrustes mean per joint position error, which is the mean of the Euclidean distance between the
    predicted 3D joint positions and the ground truth 3D joint positions, after projecting the ground truth onto the
    predicted 3D skeleton.

    Used in Protocol-II for Human3.6M dataset evaluation.

    Code adapted from:
    https://github.com/twehrbein/Probabilistic-Monocular-3D-Human-Pose-Estimation-with-Normalizing-Flows/

    :param p_gt: the ground truth 3D pose
    :type p_gt: torch.TensorType
    :param p_pred: predicted 3D pose
    :type p_pred: torch.TensorType
    :param dim: the dimension to average over. If None, the average is taken over all dimensions
    :type dim: int
    :param mean: If True, returns the mean of the MPJPE across all frames. If False, returns the MPJPE for each frame,
    defaults to True (optional)
    :return: The transformed coordinates.
    """
    if not isinstance(p_pred, torch.Tensor):
        p_pred = torch.Tensor(p_pred)

    if not isinstance(p_gt, torch.Tensor):
        p_gt = torch.Tensor(p_gt)

    og_gt = p_gt.clone()

    # p_gt = p_gt.repeat(1, p_pred.shape[1], 1)

    p_gt = p_gt.permute(1, 2, 0).contiguous()
    p_pred = p_pred.permute(1, 2, 0).contiguous()

    # Moving the tensors to the CPU as the following code is more efficient on the CPU
    p_pred = p_pred.cpu()
    p_gt = p_gt.cpu()

    mu_gt = p_gt.mean(dim=2)
    mu_pred = p_pred.mean(dim=2)

    p_gt = p_gt - mu_gt[:, :, None]
    p_pred = p_pred - mu_pred[:, :, None]

    ss_gt = (p_gt**2.0).sum(dim=(1, 2))
    ss_pred = (p_pred**2.0).sum(dim=(1, 2))

    # centred Frobenius norm
    norm_gt = torch.sqrt(ss_gt)
    norm_pred = torch.sqrt(ss_pred)

    # scale to equal (unit) norm
    p_gt /= norm_gt[:, None, None]
    p_pred /= norm_pred[:, None, None]

    # optimum rotation matrix of Y
    A = torch.bmm(p_gt, p_pred.transpose(1, 2))

    U, s, V = torch.svd(A, some=True)

    # Computing the rotation matrix.
    T = torch.bmm(V, U.transpose(1, 2))

    detT = torch.det(T)
    sign = torch.sign(detT)
    V[:, :, -1] *= sign[:, None]
    s[:, -1] *= sign
    T = torch.bmm(V, U.transpose(1, 2))

    # Computing the trace of the matrix A.
    traceTA = s.sum(dim=1)

    # transformed coords
    scale = norm_gt * traceTA

    p_pred_projected = (
        scale[:, None, None] * torch.bmm(p_pred.transpose(1, 2), T) + mu_gt[:, None, :]
    )

    return mpjpe(og_gt, p_pred_projected.permute(1, 0, 2), dim=0)


def plot_2D(projected_gt_3D, input_2D, samples, name, n_frames, alpha=0.1):
    projected_gt_3D = projected_gt_3D.cpu().detach().numpy().squeeze()
    input_2D = input_2D.squeeze().cpu().detach().numpy()
    samples = [sample.cpu().detach().numpy().squeeze() for sample in samples]

    if n_frames == 1:
        fig = plt.figure(figsize=(4, 4), dpi=150)
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        aux = Human36mPose(projected_gt_3D)
        aux.plot(ax, plot_type="none", c="tab:green")
        aux = Human36mPose(input_2D)
        aux.plot(ax, plot_type="none", c="tab:red")
        for sample in samples:
            aux = Human36mPose(sample)
            aux.plot(ax, plot_type="none", c="tab:blue", alpha=alpha)
        plt.savefig(f"./poses/{name}.png")
        plt.close()
    else:
        with GifMaker(f"./poses/{name}.gif", fps=10) as g:
            for i in range(n_frames):
                fig = plt.figure(figsize=(4, 4), dpi=150)
                ax = fig.add_subplot(1, 1, 1)
                ax.set_xlim(-1, 1)
                ax.set_ylim(-1, 1)
                aux = Human36mPose(projected_gt_3D[..., i])
                aux.plot(ax, plot_type="none", c="tab:green")
                aux = Human36mPose(input_2D[..., i])
                aux.plot(ax, plot_type="none", c="tab:red")
                for sample in samples:  # don't know if this makes sense, tbh
                    aux = Human36mPose(sample[..., i])
                    aux.plot(ax, plot_type="none", c="tab:blue")
                g.add(fig)


def plot_3D(gt_3D, samples, name, n_frames, alpha=0.1):
    gt_3D = gt_3D.cpu().detach().numpy().squeeze()
    samples = [sample.cpu().detach().numpy().squeeze() for sample in samples]

    if n_frames == 1:
        fig = plt.figure(figsize=(4, 4), dpi=150)
        ax = fig.add_subplot(1, 1, 1, projection="3d")
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        ax.view_init(elev=0.0, azim=0.0)
        aux = Human36mPose(gt_3D)
        aux.plot(ax, plot_type="none", c="tab:red")
        for sample in samples:
            aux = Human36mPose(sample)
            aux.plot(ax, plot_type="none", c="tab:blue", alpha=alpha)
        plt.savefig(f"./poses/{name}.png")
        plt.close()
    else:
        with GifMaker(f"./poses/{name}.gif", fps=10) as g:
            for i in range(n_frames):
                fig = plt.figure(figsize=(4, 4), dpi=150)
                ax = fig.add_subplot(1, 1, 1, projection="3d")
                ax.set_xlim(-1, 1)
                ax.set_ylim(-1, 1)
                ax.set_zlim(-1, 1)
                ax.view_init(elev=0.0, azim=0.0)
                aux = Human36mPose(gt_3D[..., i])
                aux.plot(ax, plot_type="none", c="tab:red")
                for sample in samples:  # don't know if this makes sense, tbh
                    aux = Human36mPose(sample[..., i])
                    aux.plot(ax, plot_type="none", c="tab:blue")
                g.add(fig)


if __name__ == "__main__":
    wandb.init(project="chick", entity="sinzlab", name=f"eval_{time.time()}")

    print("Loading options...")
    opt.manualSeed = 1
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    # Check 2D keypoint type
    print(f"Using {opt.keypoints} 2D keypoints")

    root_path = opt.root_path
    dataset_path = root_path + "data_3d_" + opt.dataset + ".npz"

    dataset = Human36mDataset(dataset_path, opt)
    actions = define_actions(opt.actions)

    test_data = Fusion(opt=opt, train=False, dataset=dataset, root_path=root_path)
    test_dataloader = torch.utils.data.DataLoader(
        test_data, batch_size=1, shuffle=True, num_workers=0, pin_memory=False
    )

    args.seed = opt.manualSeed
    fixseed(args.seed)
    train_platform_type = eval(args.train_platform_type)
    train_platform = train_platform_type(args.save_dir)
    train_platform.report_args(args, name="Args")

    model, diffusion = create_model_and_diffusion(args)
    state_dict = torch.load("./old_model.pt", map_location="cpu")
    # state_dict = torch.load('./output/model000633659_30.pt', map_location='cpu')
    # wandb.init(project="chick", entity="sinzlab", name=f"eval_{time.time()}")
    # artifact = wandb.Artifact(
    #     name="MDM_H36m_30_frames_50_steps",
    #     type="model",
    # )
    # artifact.add_file("./output/model000633659_30.pt", name="model.pt")
    # wandb.run.log_artifact(artifact)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    model.to(device)

    # 48.42

    sampling_model = ClassifierFreeSampleModel(
        model
    )  # wrapping model with the classifier-free sampler
    scale = {
        "skeleton": torch.ones(args.batch_size),
    }
    sampling_model.to(device)
    sampling_model.eval()  # disable random masking

    sample_fn = diffusion.p_sample_loop_progressive
    n_samples = 200
    energy_scale = 30
    n_frames = 1
    mms = []
    dataloader = test_dataloader
    pbar = tqdm(enumerate(dataloader), total=len(dataloader))
    total = 0
    quantiles = np.arange(0, 1.05, 0.05)
    quantile_counts = torch.zeros((len(quantiles), 17))
    q_val = []

    wandb.config.update({
        "n_samples": n_samples,
        "energy_scale": energy_scale,
        "n_frames": n_frames,
        "keypoints": opt.keypoints,
        "seed": opt.manualSeed,
    })

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
        def energy_fn(x):
            x[:, 0, :, :] = 0.0
            x[..., 0] = x[..., 0] - center[:, 0, ...]
            x_2D_projected = camera.proj2D(x[..., 0])
            # for i in range(n_frames):
            # x[0, ..., i] = x[0, ..., i] - center[0, i, ...]
            # x_2D_projected = torch.stack(
            #     [camera.proj2D(x[..., i]) for i in range(n_frames)],
            #     dim=-1
            # )

            energy = ((x_2D_projected - input_2D[..., 0]) ** 2).mean(-1).mean(-1)
            return {"train": energy}

        cam_dict = dataset.cameras()[subject[0]][cam_ind.item()]
        camera = Camera(
            intrinsic_matrix=Camera.construct_intrinsic_matrix(
                *cam_dict["intrinsic"][:4]
            ),
            rotation_matrix=torch.eye(3),
            translation_vector=torch.zeros((1, 3)),
            tangential_distortion=torch.from_numpy(
                np.reshape(cam_dict["tangential_distortion"], (1, 2))
            ),
            radial_distortion=torch.from_numpy(
                np.reshape(cam_dict["radial_distortion"], (1, 3))
            ),
        )

        input_2D = input_2D.to(device)
        gt_3D = gt_3D.to(device)
        input_2D = input_2D - 2 * input_2D[:, :, 0, :].unsqueeze(2)  # centralize
        input_2D *= -1.0
        center = gt_3D[:, :, 0].clone()  # [0, 1]
        gt_3D[:, :, 0] = 0  # "centralizes" the hip (first three coordinates) in 0
        gt_3D[0, 0, ...] = gt_3D[0, 0, ...] - center[0, 0, ...]
        # for i in range(n_frames):
        #     gt_3D[0, i, ...] = gt_3D[0, i, ...] - center[0, i, ...]
        gt_3D = gt_3D.permute(0, 2, 3, 1)  # does not make difference for sampling
        input_2D = input_2D.permute(0, 2, 3, 1)
        gt_3D_projected = camera.proj2D(gt_3D[..., 0])
        gt_3D[0, ..., 0] = gt_3D[0, ..., 0] + center[0, 0, ...]
        # gt_3D_projected = torch.stack(
        #     [camera.proj2D(gt_3D[..., i]) for i in range(n_frames)],
        #     dim=-1
        # )
        # for i in range(n_frames):
        #     gt_3D[0, ..., i] = gt_3D[0, ..., i] + center[0, i, ...]
        gt_3D[..., [1, 2], :] = -gt_3D[..., [2, 1], :]

        samples_2D = []
        samples_3D = []
        ms = []

        out = sample_fn(
            model,
            (n_samples, 17, 3, 1),
            clip_denoised=False,
            model_kwargs={"y": {"uncond": True}},
            skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
            init_image=None,
            progress=False,
            energy_fn=energy_fn,
            energy_scale=energy_scale,
            # when experimenting guidance_scale we want to nutrileze the effect of noise on generation
        )
        *_, sample = out  # get the last element of the generator (the real pose)

        sample["sample"][..., 0] = sample["sample"][..., 0] - center[:, 0, ...]
        sample_2D_proj = camera.proj2D(sample["sample"][..., 0])
        # for i in range(n_frames):
        #     sample["sample"][0, ..., i] = sample["sample"][0, ..., i] - center[0, i, ...]
        # sample_2d_proj = torch.stack(
        #     [camera.proj2D(sample["sample"][..., i]) for i in range(n_frames)],
        #     dim=-1
        # )
        samples_2D.append(sample_2D_proj)
        sample["sample"][..., 0] = sample["sample"][..., 0] + center[:, 0, ...]
        # for i in range(sample["sample"].shape[-1]):
        #     sample["sample"][0, ..., i] = sample["sample"][0, ..., i] + center[0, i, ...]
        sample["sample"][..., [1, 2], :] = -sample["sample"][..., [2, 1], :]
        samples_3D.append(sample["sample"])

        ##### mpjpe
        sample["sample"] = sample["sample"].permute(0, 3, 1, 2)
        gt_3D = gt_3D.permute(0, 3, 1, 2)
        m = mpjpe(gt_3D * 1000, sample["sample"] * 1000, dim=2)
        mms.append(m.min().cpu())
        gt_3D = gt_3D.permute(0, 2, 1, 3)
        sample["sample"] = sample["sample"].permute(1, 2, 0, 3)

        ##### calibration
        sample_mean = (
            sample["sample"].median(-2).values[..., None, :]
        )
        errors = ((sample_mean / 0.0036 - sample["sample"] / 0.0036) ** 2).sum(-1) ** 0.5
        true_error = (((sample_mean / 0.0036 - gt_3D / 0.0036) ** 2).sum(-1) ** 0.5).cpu().numpy()
        q_vals = np.quantile(errors.cpu().numpy(), quantiles, 2).squeeze(1)
        q_val.append(q_vals)

        v = (q_vals > true_error.squeeze()).astype(int)

        if not np.isnan(v).any():
            total += 1
            quantile_counts += v

        _quantile_freqs = quantile_counts / total

        # _calibration_score = np.median(_quantile_freqs, axis=1).sum() * 0.05
        _calibration_score = np.abs(
            np.median(_quantile_freqs, axis=1) - quantiles
        ).mean()

        pbar.set_description(
            f"{mms[-1]:.2f} | {np.mean(mms):.2f} | {_calibration_score:.2f}"
        )

        wandb.log(
            {
                "running_avg_mpjpe": np.mean(mms),
                "mpjpe": m.min().cpu(),
                "action": action[0],
            }
        )
        # pm = pa_mpjpe(gt_3D[0, ...] * 1000,
        #               sample["sample"][0, ...] * 1000,
        #               dim=0)
        #     pms.append(pm.min().cpu())
        # plot_2D(
        #     gt_3D_projected,
        #     input_2D,
        #     samples_2D,
        #     f"2{k:02}: 2D {action[0]} {n_frames} frames {n_samples} samples energy scale",
        #     n_frames,
        #     alpha=0.05
        # )
        # plot_3D(
        #     gt_3D,
        #     samples_3D,
        #     f"2{k:02}: 3D {action[0]} {n_frames} frames {n_samples} samples energy scale",
        #     n_frames,
        #     alpha=0.05
        # )

    quantile_freqs = quantile_counts / total
    calibration_score = np.abs(np.median(quantile_freqs, axis=1) - quantiles).mean()
    print(calibration_score)
