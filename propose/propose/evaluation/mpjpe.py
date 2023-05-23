import numpy as np
import torch


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

    p_gt = p_gt.repeat(1, p_pred.shape[1], 1)

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
