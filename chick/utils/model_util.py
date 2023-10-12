from chick.diffusion import gaussian_diffusion as gd
from chick.diffusion.respace import SpacedDiffusion, space_timesteps
from chick.model.overcomplete import MDM


def load_model_wo_clip(model, state_dict):
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    assert len(unexpected_keys) == 0
    assert all([k.startswith("clip_model.") for k in missing_keys])


def create_model_and_diffusion(args):
    model = MDM(**args)
    diffusion = create_gaussian_diffusion(**args)
    return model, diffusion


def diffusion_defaults():
    """
    Defaults for image and classifier training.
    """
    return dict(
        learn_sigma=True,
        diffusion_steps=1000,
        noise_schedule="linear",
        timestep_respacing="250",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
    )


def model_and_diffusion_defaults():
    """
    Defaults for image training.
    """
    res = dict(
        image_size=128,
        num_channels=256,
        num_res_blocks=2,
        num_heads=4,
        num_heads_upsample=-1,
        num_head_channels=-1,
        attention_resolutions="32,16,8",
        channel_mult="",
        dropout=0.0,
        class_cond=False,
        use_checkpoint=False,
        use_scale_shift_norm=True,
        resblock_updown=True,
        use_fp16=False,
        use_new_attention_order=False,
    )
    res.update(diffusion_defaults())
    return res


def get_model_args(args):
    # default args
    clip_version = "ViT-B/32"
    action_emb = "tensor"
    if args.unconstrained:
        cond_mode = "no_cond"
    elif args.dataset in ["kit", "humanml"]:
        cond_mode = "text"
    else:
        cond_mode = "action"

    cond_mode = "skeleton"
    num_actions = 1

    data_rep = "h36m"
    njoints = 17
    nfeats = 3

    # SMPL defaults
    # data_rep = 'rot6d'
    # njoints = 25
    # nfeats = 6
    #
    # if args.dataset == 'humanml':
    #     data_rep = 'hml_vec'
    #     njoints = 263
    #     nfeats = 1
    # elif args.dataset == 'kit':
    #     data_rep = 'hml_vec'
    #     njoints = 251
    #     nfeats = 1

    return {
        "modeltype": "",
        "njoints": njoints,
        "nfeats": nfeats,
        "num_actions": num_actions,
        "translation": True,
        "pose_rep": "rot6d",
        "glob": True,
        "glob_rot": True,
        "latent_dim": args.latent_dim,
        "ff_size": 1024,
        "num_layers": args.layers,
        "num_heads": 4,
        "dropout": 0.1,
        "activation": "gelu",
        "data_rep": data_rep,
        "cond_mode": cond_mode,
        "cond_mask_prob": args.cond_mask_prob,
        "action_emb": action_emb,
        "arch": args.arch,
        "emb_trans_dec": args.emb_trans_dec,
        "clip_version": clip_version,
        "dataset": args.dataset,
    }


def create_gaussian_diffusion(
    noise_schedule, sigma_small, lambda_vel, lambda_rcxyz, lambda_fc, **kwargs
):
    # default params
    predict_xstart = True  # we always predict x_start (a.k.a. x0), that's our deal!
    steps = 50
    scale_beta = 1.0  # no scaling
    timestep_respacing = ""  # can be used for ddim sampling, we don't use it.
    learn_sigma = False
    rescale_timesteps = False

    betas = gd.get_named_beta_schedule(noise_schedule, steps, scale_beta)
    loss_type = gd.LossType.MSE

    if not timestep_respacing:
        timestep_respacing = [steps]

    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
        lambda_vel=lambda_vel,
        lambda_rcxyz=lambda_rcxyz,
        lambda_fc=lambda_fc,
    )
