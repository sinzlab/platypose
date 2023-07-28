import sys
import time
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchvision import transforms

sys.path.append("/src")

from chick.config import cfg_to_dict, get_experiment_config
from chick.dataset.h36m import H36MVideoDataset
from chick.energies import inpaint_2d_energy, monocular_2d_energy, gaussian_2d_energy, learned_2d_energy, heatmap_energy, fit_gaussian_2d_energy
from chick.pipeline import SkeletonPipeline
from chick.platform import platform
from chick.projection import Projection
from chick.utils.plot_utils import plot_2D, plot_3D
from chick.utils.reproducibility import set_random_seed
from chick.heatmap import hrnet, get_heatmaps
from propose.propose.cameras.Camera import Camera, DummyCamera
from propose.propose.evaluation.calibration import calibration
from propose.propose.evaluation.mpjpe import mpjpe
from propose.propose.poses.human36m import Human36mPose, MPIIPose, MPII_2_H36M


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

    print(cfg.experiment.keypoints)

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

    def load_video_frame(path, frame_idx):
        import cv2
        cap = cv2.VideoCapture(path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        # convert to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cap.release()
        return frame

    # metric storage
    mpjpes = []
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
        start_3d
    ) in pbar:
        if k < 1:
            continue

        if k > 99:
            exit()


        cam_dict = dataset.dataset.cameras()[subject[0]][cam_ind.item()]

        # video_path = f"./dataset/{subject[0]}/Videos/{action[0]}.{cam_dict['id']}.mp4"
        # bbox_path = f"./dataset/{subject[0]}/MySegmentsMat/ground_truth_bb/{action[0]}.{cam_dict['id']}.mat"
        # frame = start_3d.item()
        # # print("video_path:", video_path, "frame:", frame)
        #
        # import h5py
        #
        # try:
        #     bbox_h5py = h5py.File(bbox_path)
        #     mask = np.array(bbox_h5py[bbox_h5py['Masks'][frame, 0]]).T
        # except Exception as e:
        #     # print(e)
        #     continue
        #
        # try:
        #     frame = load_video_frame(video_path, frame)
        # except Exception as e:
        #     # print(e)
        #     continue
        #
        # import matplotlib.pyplot as plt
        #
        # frame = frame[:1000, :1000]
        #
        # ys, xs = np.where(mask == 1)
        # bbox = np.array([np.min(xs), np.min(ys), np.max(xs) + 1, np.max(ys) + 1])
        # center = [(bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2]
        #
        # # make bbox square
        # size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
        # bbox[0] = center[0] - size / 2
        # bbox[1] = center[1] - size / 2
        # bbox[2] = center[0] + size / 2
        # bbox[3] = center[1] + size / 2
        #
        # image = transforms.ToTensor()(frame)
        # image = transforms.functional.crop(image, bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0])
        # # print(image.shape)
        # image = transforms.Resize((256, 256))(image)
        # image = image.unsqueeze(0)
        #
        # heatmap = get_heatmaps(image, hrnet)
        #
        #
        # def fit_multivariate_normal(heatmap, num_components=1, num_epochs=1000, learning_rate=0.001, upscale=4):
        #     # argmax = (heatmap == torch.max(heatmap, dim=0))
        #     reshaped_tensor = heatmap.view(heatmap.shape[0], -1)
        #     max_indices = torch.argmax(reshaped_tensor, dim=1)
        #     row_indices = max_indices // heatmap.shape[-1]
        #     col_indices = max_indices % heatmap.shape[-2]
        #     argmax = torch.stack((row_indices, col_indices), dim=1).float()
        #     mean = nn.Parameter(argmax.cuda())
        #     confidence = nn.Parameter(reshaped_tensor[torch.arange(reshaped_tensor.shape[0]), max_indices].unsqueeze(-1).cuda())
        #     log_sigma = nn.Parameter(torch.ones(heatmap.shape[0], 2).cuda() * np.log(4.5))
        #     corr = nn.Parameter(torch.zeros(heatmap.shape[0], 1).cuda())
        #
        #     # Define the negative log-likelihood loss function
        #     loss_function = nn.KLDivLoss(reduction='batchmean')
        #     x_coords = torch.dstack(torch.meshgrid(torch.arange(heatmap.shape[-2]), torch.arange(heatmap.shape[-1])))
        #     x_coords = x_coords.reshape(-1, 1, 2).float().cuda().repeat(1, heatmap.shape[0], 1)
        #
        #     # Create an optimizer to minimize the loss
        #     optimizer = optim.Adam([log_sigma], lr=0.1)
        #
        #     # Number of optimization steps
        #     num_steps = 10000
        #
        #     minimum = torch.amin(heatmap, dim=(1, 2)).unsqueeze(-1).unsqueeze(-1)
        #     maximum = torch.amax(heatmap, dim=(1, 2)).unsqueeze(-1).unsqueeze(-1)
        #     heatmap = heatmap - minimum#) / (maximum - minimum)
        #
        #     # print('min/max', torch.amin(heatmap), torch.amax(heatmap))
        #     #
        #     # old_loss = 1e6
        #     # # Optimization loop
        #     # for step in range(num_steps):
        #     #     optimizer.zero_grad()
        #     #
        #     #     # Generate the multivariate normal distribution
        #     #     mvn = torch.distributions.MultivariateNormal(loc=mean, covariance_matrix=torch.eye(2).cuda().unsqueeze(0).repeat(heatmap.shape[0], 1, 1) * log_sigma.exp().unsqueeze(-1))
        #     #
        #     #     # Calculate the negative log-likelihood loss
        #     #     nll = mvn.log_prob(x_coords).exp()
        #     #     nll = nll.reshape(heatmap.shape)
        #     #     # nll = nll / torch.amax(nll, dim=(1, 2)).unsqueeze(-1).unsqueeze(-1)
        #     #
        #     #     # print('nll.flatten(), heatmap.cuda().flatten()', nll.flatten().shape, heatmap.cuda().flatten().shape)
        #     #     loss = loss_function(nll.flatten(), heatmap.flatten().cuda())
        #     #
        #     #     # Perform backpropagation and optimization step
        #     #     loss.backward()
        #     #     optimizer.step()
        #     #
        #     #     threshold = 1e-6
        #     #     if abs(old_loss - loss.item()) < threshold:
        #     #         break
        #     #
        #     #     if step % 100 == 0:
        #     #         print(f"Loss: {loss.item()} | Steps: {step}")
        #     #
        #     #     old_loss = loss.item()
        #
        #     # print(f"Loss: {loss.item()} | Steps: {step}")
        #
        #     # Return the fitted mean and covariance matrix
        #     return torch.distributions.MultivariateNormal(loc=mean * upscale,
        #                                            covariance_matrix=torch.eye(2).cuda().unsqueeze(0).repeat(heatmap.shape[0], 1,
        #                                                                                                      1) * log_sigma.exp().unsqueeze(
        #                                                -1) * (upscale**2)), confidence
        #     # return torch.distributions.MultivariateNormal(loc=mean * upscale, covariance_matrix=torch.eye(2).cuda() * sigma * (upscale**2) + corr)
        #
        #
        #
        # #
        # # print(bbox)
        # #
        # width = min([bbox[2], 1000]) - bbox[0]
        # height = min([bbox[3], 1000]) - bbox[1]
        #
        # upscale = width / 64
        #
        # # print(width, height)
        #
        # h = heatmap[:, :]
        # dist, confidence = fit_multivariate_normal(h[0], upscale=upscale)
        # scale = 1 / confidence**2
        # scale = torch.ones_like(confidence) * 1
        # scale = scale[MPII_2_H36M]

        # insert a value of 10 at index 9
        # scale = torch.cat((scale[:9], torch.tensor([[10]]).cuda(), scale[9:]))
        # scale = scale.unsqueeze(0).unsqueeze(0)
        #
        # print(dist)
        #
        # heatmap = transforms.Resize((height, width))(h)
        # heatmap = heatmap # (256, 256)


        #
        #
        #
        # # # position the heatmap in the original image in place of the bbox
        # heatmap_full = torch.zeros((1, heatmap.shape[1], *frame.shape[:-1]))
        # # print(heatmap_full.shape, heatmap.shape)
        # heatmap_full[:, :, int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])] = heatmap
        #
        # # heatmap_full = heatmap_full.cuda()
        #
        #
        # # heatmap #= heatmap / heatmap.sum(0).sum(0).max() * 255
        # frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2]), 1] = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2]), 1] * (1 - heatmap.sum(0).sum(0).numpy()) + heatmap.sum(0).sum(0).numpy() * 255
        #
        # # plt.imshow(heatmap_full.sum(0).sum(0).squeeze())
        # # plt.savefig(f"heatmap_{k}.png")
        # # plt.clf()
        #
        # x_coords = torch.dstack(torch.meshgrid(torch.arange(1000), torch.arange(1000)))
        # x_coords = x_coords.reshape(-1, 1, 2).float().cuda().repeat(1, h.shape[0], 1)
        #
        # mean = dist.mean + torch.tensor([bbox[1], bbox[0]]).cuda()
        #
        #
        # # # print(mean)
        # # print(dist.covariance_matrix.cpu())
        # dist = torch.distributions.MultivariateNormal(loc=mean.cpu(), covariance_matrix=dist.covariance_matrix.cpu())
        # x_coords = x_coords.cpu()
        # nll = dist.log_prob(x_coords)
        # nll = nll.exp()
        # nll = nll / torch.amax(nll, dim=0).unsqueeze(0)
        # nll = nll.sum(-1)
        # nll = nll.reshape(1000, 1000)

        # plt.imshow(nll.cpu().detach().numpy())
        #
        # pose = Human36mPose(input_2D[0].numpy() * 500 + 500)
        # pose.plot()
        #
        # plt.savefig(f"nll_{k}.png")
        # plt.clf()


        # sampling_count += 1

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

        if cfg.experiment.keypoints == 'gt':
            input_2D = gt_3D_projected.clone().permute(0, 2, 3, 1)
        else:
            input_2D = input_2D.to('cuda').permute(0, 2, 3, 1)
            c = input_2D[:, :1]
            input_2D = input_2D - c
            input_2D = -input_2D
            input_2D = input_2D + c

        # mean = (mean - 500) / 500
        #
        # # add hip joint at input_2D[:, :, 0]
        # mean = torch.Tensor(MPIIPose(mean.unsqueeze(0).cpu().detach().numpy()).to_human36m().pose_matrix).cuda()
        # mean = mean[..., [1, 0]]
        # mean = mean - mean[:, 0:1, :]
        # mean = mean + input_2D[0, :, 0:1].cuda()

        # print('mse(mean, input_2D)', torch.nn.MSELoss()(mean.cuda(), input_2D[0].cuda()).item())

        # input_2D = mean.unsqueeze(-1)
        # input_2D = input_2D.to('cuda')
        # c = input_2D[:, :1]
        # input_2D = input_2D - c
        # input_2D = -input_2D
        # input_2D = input_2D + c

        # print(input_2D.shape)


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
        scale = torch.ones(1, 1, 17, 1).cuda() * 1

        scale = ((input_2D.permute(0, 3, 1, 2) - gt_3D_projected)**2).mean(-1, keepdim=True) * 1e3
        scale = scale.clamp(1, 5)

        for _ in range(cfg.experiment.num_repeats):
            out = pipe.sample(
                num_samples=cfg.experiment.num_samples,
                num_frames=cfg.model.num_frames,
                num_substeps=cfg.experiment.num_substeps,
                energy_fn=partial(
                    monocular_2d_energy, x_2d=input_2D, x_2d_std=scale, center=center, camera=camera
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
            f"{mpjpes[-1]:.2f} | {np.nanmean(mpjpes):.2f} | {_calibration_score:.2f} | sampling_count: {sampling_count}"
        )

        platform.log(
            {
                "running_avg_mpjpe": np.mean(mpjpes),
                "mpjpe": m.min().cpu(),
                "action": action[0],
            }
        )

        exit()

        # plot_2D(
        #     gt_3D_projected.permute(0, 2, 3, 1)[..., :1],
        #     input_2D[..., :1],
        #     sample_2D_proj[..., :1],
        #     f"2{k:02}: 2D {action[0]} {1} frames {1} samples energy scale",
        #     n_frames=1,
        #     # n_frames=cfg.model.num_frames,
        #     alpha=0.1,
        # )
        # # #
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

        # if k >= 0:
        #     exit(0)

