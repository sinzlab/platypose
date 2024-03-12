import copy
import functools
import os
import time
from types import SimpleNamespace

import blobfile as bf
import numpy as np
import torch
from torch.optim import AdamW
from tqdm import tqdm

from platypose.diffusion import logger
from platypose.diffusion.fp16_util import MixedPrecisionTrainer
from platypose.diffusion.resample import (
    LossAwareSampler,
    UniformSampler,
    create_named_schedule_sampler,
)
from platypose.model.cfg_sampler import ClassifierFreeSampleModel
from platypose.utils import dist_util

# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0


class TrainLoop:
    def __init__(self, args, train_platform, model, diffusion, data):
        self.args = args
        self.dataset = args.dataset
        self.train_platform = train_platform
        self.model = model
        self.diffusion = diffusion
        self.cond_mode = model.cond_mode
        self.data = data
        self.batch_size = args.batch_size
        self.microbatch = args.batch_size  # deprecating this option
        self.lr = args.lr
        self.log_interval = args.log_interval
        self.save_interval = args.save_interval
        self.resume_checkpoint = args.resume_checkpoint
        self.use_fp16 = False  # deprecating this option
        self.fp16_scale_growth = 1e-3  # deprecating this option
        self.weight_decay = args.weight_decay
        self.lr_anneal_steps = args.lr_anneal_steps

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size  # * dist.get_world_size()
        self.num_steps = args.num_steps
        self.num_epochs = self.num_steps // len(self.data) + 1
        print(f"epochs: {self.num_epochs}")
        self.sync_cuda = torch.cuda.is_available()

        self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=self.fp16_scale_growth,
        )

        self.save_dir = args.save_dir
        self.overwrite = args.overwrite

        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )

        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.opt, mode="min", factor=0.5, patience=5, verbose=True
        )

        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.

        self.device = torch.device("cpu")
        if torch.cuda.is_available() and dist_util.dev() != "cpu":
            self.device = torch.device(dist_util.dev())

        self.schedule_sampler_type = "uniform"
        self.schedule_sampler = create_named_schedule_sampler(
            self.schedule_sampler_type, diffusion
        )
        self.eval_wrapper, self.eval_data, self.eval_gt_data = None, None, None

        self.use_ddp = False
        self.ddp_model = self.model

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
            self.model.load_state_dict(
                dist_util.load_state_dict(
                    resume_checkpoint, map_location=dist_util.dev()
                )
            )

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:09}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)

    def run_loop(self):

        for epoch in range(self.num_epochs):
            print(f"Starting epoch {epoch}")
            for (
                batch_cam,
                gt_3D,
                input_2D,
                action,
                subject,
                scale,
                bb_box,
                cam_ind,
            ) in tqdm(self.data):
                gt_3D[:, :, 0] = 0  # set the root to 0

                # Delete the root joint
                # gt_3D = gt_3D[:, :, 1:, :]

                motion = gt_3D.permute(0, 2, 3, 1)

                input_2D = input_2D - input_2D[:, :, 0, :].unsqueeze(2)

                # add noise to the input with random variance and use the variance as an additional input
                var = (
                    torch.rand(
                        input_2D.shape[0], input_2D.shape[1], input_2D.shape[2], 1
                    ).to(input_2D.device)
                    * 0
                )
                input_2D = input_2D + torch.randn_like(input_2D) * var
                input_2D = torch.cat([input_2D, var], dim=-1)

                cond = {
                    "y": {
                        "mask": torch.ones((motion.shape[0], 1)),
                        "uncond": True,
                        # 'skeleton': input_2D.permute(0, 2, 3, 1),
                    },
                }
                if not (
                    not self.lr_anneal_steps
                    or self.step + self.resume_step < self.lr_anneal_steps
                ):
                    break

                motion = motion.to(self.device)
                cond["y"] = {
                    key: val.to(self.device) if torch.is_tensor(val) else val
                    for key, val in cond["y"].items()
                }

                # cond = {} # no conditioning training
                self.run_step(motion, cond)

                if self.step % self.log_interval == 0 and self.step > 0:
                    # self.model.eval()
                    # self.evaluate()
                    # self.model.train()
                    for k, v in logger.get_current().name2val.items():
                        if k == "loss":
                            print(
                                "step[{}]: loss[{:0.5f}]".format(
                                    self.step + self.resume_step, v
                                )
                            )

                        if k in ["step", "samples"] or "_q" in k:
                            continue
                        else:
                            self.train_platform.report_scalar(
                                name=k, value=v, iteration=self.step, group_name="Loss"
                            )

                if self.step % self.save_interval == 0:
                    self.save()
                    # self.model.eval()
                    # self.evaluate()
                    # self.model.train()

                    # Run for a finite amount of time in integration tests.
                    if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                        return
                self.step += 1
            if not (
                not self.lr_anneal_steps
                or self.step + self.resume_step < self.lr_anneal_steps
            ):
                break
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()
            # self.evaluate()

    def evaluate(self):
        sampling_model = ClassifierFreeSampleModel(
            self.model
        )  # wrapping model with the classifier-free sampler
        sampling_model.to(dist_util.dev())
        sampling_model.eval()  # disable random masking

        sample_fn = self.diffusion.p_sample_loop

        for (
            batch_cam,
            gt_3D,
            input_2D,
            action,
            subject,
            scale,
            bb_box,
            cam_ind,
        ) in self.data:
            motion = gt_3D.permute(0, 2, 3, 1).to(dist_util.dev())
            cond = {
                "y": {
                    "mask": torch.ones((motion.shape[0], 1)).to(dist_util.dev()),
                    "skeleton": input_2D.permute(0, 2, 3, 1).to(dist_util.dev()),
                    "scale": torch.ones(self.global_batch).to(dist_util.dev()),
                },
            }

            sample = sample_fn(
                self.model,
                motion.shape,
                clip_denoised=False,
                model_kwargs=cond,
                skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
                init_image=None,
                progress=False,
                dump_steps=None,
                noise=None,
                const_noise=False,
                # when experimenting guidance_scale we want to nutrileze the effect of noise on generation
            )

            sample = sample[..., -1]  # take the last frame for evaluation
            gt = motion[..., -1]

            mpjpe = torch.mean(torch.norm(sample * 1000 - gt * 1000, dim=-1))
            print("MPJPE", mpjpe)

            break

    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
        self.mp_trainer.optimize(self.opt)
        self._anneal_lr()
        self.log_step()

    def forward_backward(self, batch, cond):
        self.mp_trainer.zero_grad()
        for i in range(0, batch.shape[0], self.microbatch):
            # Eliminates the microbatch feature
            assert i == 0
            assert self.microbatch == self.batch_size
            micro = batch
            micro_cond = cond
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,  # [bs, ch, image_size, image_size]
                t,  # [bs](int) sampled timesteps
                model_kwargs=micro_cond,
                dataset=self.data.dataset,
            )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()

            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            self.mp_trainer.backward(loss)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def ckpt_file_name(self):
        return f"model{(self.step+self.resume_step):09d}_30_frames.pt"

    def save(self):
        def save_checkpoint(params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            # if dist.get_rank() == 0:
            logger.log(f"saving model...")
            filename = self.ckpt_file_name()
            with bf.BlobFile(bf.join(self.save_dir, filename), "wb") as f:
                torch.save(state_dict, f)

        save_checkpoint(self.mp_trainer.master_params)

        with bf.BlobFile(
            bf.join(self.save_dir, f"opt{(self.step+self.resume_step):09d}.pt"),
            "wb",
        ) as f:
            torch.save(self.opt.state_dict(), f)


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
