import os

import torch
import torch.nn as nn
import tqdm

from chick.diffusion.fp16_util import MixedPrecisionTrainer
from chick.diffusion.resample import UniformSampler
from chick.utils.model_util import create_model_and_diffusion
from chick.utils.wandb import download_wandb_artefact


class Chick(nn.Module):
    def __init__(
        self,
        config=None,
    ):
        super().__init__()
        # Model settings
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model_config = {
            "modeltype": "",
            "njoints": 17,
            "nfeats": 3,
            "num_actions": 1,
            "translation": True,
            "pose_rep": "rot6d",
            "glob": True,
            "glob_rot": True,
            "latent_dim": 1024,
            "ff_size": 1024,
            "num_layers": 2,
            "num_heads": 4,
            "dropout": 0.1,
            "activation": "gelu",
            "data_rep": "h36m",
            "cond_mode": "skeleton",
            "cond_mask_prob": 0.1,
            "action_emb": "tensor",
            "arch": "trans_enc",
            "emb_trans_dec": False,
            "clip_version": "ViT-B/32",
            "dataset": "humanml",
            "noise_schedule": "cosine",
            "sigma_small": True,
            "lambda_rcxyz": 0.0,
            "lambda_vel": 0.0,
            "lambda_fc": 0.0,
        }

        if config is not None:
            self.model_config.update(config)

        self.model, self.diffusion = create_model_and_diffusion(self.model_config)

    def sample(self, energy_fn, energy_scale=1, num_samples=1, num_frames=1):
        """
        This function samples from a diffusion model using a given energy function and other optional parameters.

        :param energy_fn: The energy function used to calculate the energy of the samples generated by the model. It takes
        in the generated samples and returns a dict of energies of the samples
        :param energy_scale: The energy scale is a parameter that scales the energy function used in the sampling process.
        It can be used to adjust the importance of the energy function relative to other factors in the sampling process. A
        higher energy scale will result in a stronger influence of the energy function on the samples generated, defaults to
        1 (optional)
        :param num_samples: The number of samples to generate, defaults to 1 (optional)
        :return: a set of samples generated using the progressive sampling loop of the diffusion model. The samples are
        generated based on the given energy function and energy scale, and the number of samples is determined by the
        `num_samples` parameter. The samples are returned as a tensor of shape `(num_samples, 3, image_size, image_size)`.
        The function also has optional parameters for using alpha
        """
        return self.diffusion.p_sample_loop_progressive(
            self.model,
            (num_samples, 17, 3, num_frames),
            clip_denoised=False,
            model_kwargs={"y": {"uncond": True}},
            skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
            init_image=None,
            progress=False,
            energy_fn=energy_fn,
            energy_scale=energy_scale,
        )

    @staticmethod
    def _get_state_dict(path_or_artefact: str, use_cache: bool = True):
        """
        Checks if path_or_artefact is a path if not tries to download the model from wandb
        :param path_or_artefact: path to model or wandb artifact
        :param use_cache: if true uses the cached model
        :return: state dict of model
        """

        cache_path = "./models/" + path_or_artefact
        # check if ends with .pt
        if not path_or_artefact.endswith(".pt"):
            cache_path = cache_path + ".pt"

        if os.path.exists(cache_path) and use_cache:
            print("Using cached model")
            return torch.load(cache_path, map_location="cpu")
        else:
            return download_wandb_artefact(path_or_artefact)

    @classmethod
    def from_pretrained(
        cls, diffusion_artefact="sinzlab/chick/MDM_H36m_30_frames_50_steps:latest"
    ):
        """
        Loads a pretrained model from wandb
        :param diffusion_artefact: wandb artifact name
        :return: None
        """
        chick = cls()

        state_dict = chick._get_state_dict(diffusion_artefact)

        chick.model.load_state_dict(state_dict)
        chick.requires_grad_(True).eval()
        chick.to(chick.device)

        return chick

    @classmethod
    def train(cls, dataloader, cfg):
        self = cls()
        self.to(cfg.device)

        mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=False,
            fp16_scale_growth=1e-3,
        )
        schedule_sampler = UniformSampler(self.diffusion)
        opt = torch.optim.AdamW(
            mp_trainer.master_params,
            lr=cfg.train.lr,
            weight_decay=cfg.train.weight_decay,
        )

        num_epochs = cfg.num_steps // len(dataloader) + 1
        for epoch in range(num_epochs):
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
            ) in tqdm(dataloader):
                gt_3D[:, :, 0] = 0  # set the root to 0

                batch = gt_3D.permute(0, 2, 3, 1)
                batch = batch.to(cfg.device)

                cond = {}  # no conditioning training

                mp_trainer.zero_grad()
                for i in range(0, batch.shape[0], cfg.batch_size):
                    # Eliminates the microbatch feature
                    assert i == 0
                    micro = batch
                    micro_cond = cond
                    t, weights = schedule_sampler.sample(micro.shape[0], cfg.device)

                    losses = self.diffusion.training_losses(
                        self.model,
                        micro,  # [bs, ch, image_size, image_size]
                        t,  # [bs](int) sampled timesteps
                        model_kwargs=micro_cond,
                        dataset=dataloader.dataset,
                    )

                    loss = (losses["loss"] * weights).mean()

                    mp_trainer.backward(loss)

                mp_trainer.optimize(opt)

        return self
