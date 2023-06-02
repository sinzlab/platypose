import numpy as np


def calibration(sample, gt_3D):
    quantiles = np.arange(0, 1.05, 0.05)

    sample_mean = sample.median(0).values[None]
    errors = ((sample_mean / 0.0036 - sample / 0.0036) ** 2).sum(-1).mean(-1) ** 0.5
    true_error = (
        (((sample_mean / 0.0036 - gt_3D / 0.0036) ** 2).sum(-1).mean(-1) ** 0.5)
        .cpu()
        .numpy()
    )
    q_vals = np.quantile(errors.cpu().numpy(), quantiles, 0)  # .squeeze(1)
    v = (q_vals > true_error.squeeze()).astype(int)

    return v, quantiles