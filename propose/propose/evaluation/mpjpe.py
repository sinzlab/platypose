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


# def pa_mpjpe(
#     p_gt: torch.TensorType, p_pred: torch.TensorType, dim: int = None, mean: bool = True
# ):
#     """
#     PA-MPJPE is the Procrustes mean per joint position error, which is the mean of the Euclidean distance between the
#     predicted 3D joint positions and the ground truth 3D joint positions, after projecting the ground truth onto the
#     predicted 3D skeleton.
#
#     Used in Protocol-II for Human3.6M dataset evaluation.
#
#     Code adapted from:
#     https://github.com/twehrbein/Probabilistic-Monocular-3D-Human-Pose-Estimation-with-Normalizing-Flows/
#
#     :param p_gt: the ground truth 3D pose
#     :type p_gt: torch.TensorType
#     :param p_pred: predicted 3D pose
#     :type p_pred: torch.TensorType
#     :param dim: the dimension to average over. If None, the average is taken over all dimensions
#     :type dim: int
#     :param mean: If True, returns the mean of the MPJPE across all frames. If False, returns the MPJPE for each frame,
#     defaults to True (optional)
#     :return: The transformed coordinates.
#     """
#     if not isinstance(p_pred, torch.Tensor):
#         p_pred = torch.Tensor(p_pred)
#
#     if not isinstance(p_gt, torch.Tensor):
#         p_gt = torch.Tensor(p_gt)
#
#     og_gt = p_gt.clone()
#
#     p_gt = p_gt.expand_as(p_pred)
#
#     # p_gt = p_gt.permute(1, 2, 0).contiguous()
#     # p_pred = p_pred.permute(1, 2, 0).contiguous()
#     # 200, 3, 17 <- old
#     # 200, 64, 17, 3 <- new
#
#     # Moving the tensors to the CPU as the following code is more efficient on the CPU
#     p_pred = p_pred.cpu()
#     p_gt = p_gt.cpu()
#
#     mu_gt = p_gt.mean(dim=-2, keepdim=True )
#     mu_pred = p_pred.mean(dim=-2, keepdim=True)
#
#     p_gt = p_gt - mu_gt
#     p_pred = p_pred - mu_pred
#
#     # frobenius norm
#
#     # ss_gt = (p_gt**2.0).sum(dim=(-1, -2), keepdims=True)
#     # ss_pred = (p_pred**2.0).sum(dim=(-1, -2), keepdims=True)
#
#     # centred Frobenius norm
#     norm_gt = torch.norm(p_gt, dim=(-1, -2), keepdim=True)
#     norm_pred = torch.norm(p_pred, dim=(-1, -2), keepdim=True)
#
#     # scale to equal (unit) norm
#     p_gt /= norm_gt#[:, None, None]
#     p_pred /= norm_pred#[:, None, None]
#
#     # optimum rotation matrix of Y
#
#     # batch matrix multiplication (200, 64, 17, 3) x (200, 64, 3, 17) -> (200, 64, 17, 17)
#     A = p_gt.transpose(-1, -2) @ p_pred
#
#     # A = torch.bmm(p_gt, p_pred.transpose(-1, -2))
#
#     U, s, V = torch.svd(A, some=True)
#
#     # Computing the rotation matrix.
#     T = V @ U.transpose(-1, -2)
#     # T = torch.bmm(V, U.transpose(-1, -2))
#
#     detT = torch.det(T)
#     sign = torch.sign(detT)
#     V[:, :, -1] *= sign[:, None]
#     # V[..., -1] *= sign[..., None]
#     s[:, -1] *= sign
#
#     T = V @ U.transpose(-1, -2)
#     # T = torch.bmm(V, U.transpose(-1, -2))
#
#     # Computing the trace of the matrix A.
#     traceTA = s.sum(dim=-1)
#
#     # transformed coords
#     scale = norm_gt * traceTA.view(*norm_gt.shape)
#
#     p_pred = p_pred @ T.transpose(-1, -2)
#
#     p_pred_projected = (
#         scale * p_pred + mu_gt
#     )
#
#     return mpjpe(og_gt, p_pred_projected, dim=0, mean=mean)


def p_mpjpe(target, predicted, mean=True):
    """
    Pose error: MPJPE after rigid alignment (scale, rotation, and translation),
    often referred to as "Protocol #2" in many papers.
    """
    target = target.expand_as(predicted)

    # predicted = predicted.clone().numpy()
    # target = target.clone().numpy()

    assert predicted.shape == target.shape

    muX = torch.mean(target, dim=-2, keepdims=True) # (200, 1, 3)
    muY = torch.mean(predicted, dim=-2, keepdims=True) # (200, 1, 3)

    X0 = target - muX # (200, 17, 3)
    Y0 = predicted - muY # (200, 17, 3)

    normX = torch.sqrt(torch.sum(X0 ** 2, dim=(-1, -2), keepdims=True)) # (200, 1, 1)
    normY = torch.sqrt(torch.sum(Y0 ** 2, dim=(-1, -2), keepdims=True)) # (200, 1, 1)

    X0 /= normX
    Y0 /= normY

    H = torch.matmul(X0.transpose(-2, -1), Y0) # (200, 3, 3)
    U, s, Vt = np.linalg.svd(H) # (200, 3, 3), (200, 3), (200, 3, 3)
    U = torch.from_numpy(U)
    Vt = torch.from_numpy(Vt)
    s = torch.from_numpy(s)
    V = Vt.transpose(-2, -1) # (200, 3, 3)
    R = torch.matmul(V, U.transpose(-2, -1)) # (200, 3, 3)

    # Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
    sign_detR = torch.sign(torch.linalg.det(R).unsqueeze(-1)) # (200, 1)
    V[..., -1] *= sign_detR # (200, 3, 3)
    s[..., -1] *= sign_detR[..., 0] # (200, 3)


    R = torch.matmul(V, U.transpose(-2, -1))  # Rotation # (200, 3, 3)
    tr = torch.sum(s, dim=-1, keepdims=True).unsqueeze(-1) # (200, 1, 1)
    a = tr * normX / normY  # Scale # (200, 1, 1)
    t = muX - a * np.matmul(muY, R)  # Translation (200, 1, 3)
    # Perform rigid transformation on the input
    predicted_aligned = a * np.matmul(predicted, R) + t
    # Return MPJPE
    return mpjpe(target, predicted_aligned, dim=0, mean=mean)


def pa_mpjpe(target, predicted, mean=True):
    """
    Pose error: MPJPE after rigid alignment (scale, rotation, and translation),
    often referred to as "Protocol #2" in many papers.
    """
    target = target.expand_as(predicted)

    predicted = predicted.clone().numpy()
    target = target.clone().numpy()

    muX = np.mean(target, axis=1, keepdims=True)
    muY = np.mean(predicted, axis=1, keepdims=True)

    X0 = target - muX
    Y0 = predicted - muY

    normX = np.sqrt(np.sum(X0 ** 2, axis=(1, 2), keepdims=True))
    normY = np.sqrt(np.sum(Y0 ** 2, axis=(1, 2), keepdims=True))

    X0 /= normX
    Y0 /= normY

    H = np.matmul(X0.transpose(0, 2, 1), Y0)
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))

    # Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1))  # Rotation
    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)
    a = tr * normX / normY  # Scale
    t = muX - a * np.matmul(muY, R)  # Translation
    # Perform rigid transformation on the input
    predicted_aligned = a * np.matmul(predicted, R) + t
    # Return MPJPE
    return mpjpe(target, predicted_aligned, dim=0, mean=mean)

def mpjve(pred, gt, dim=None, mean=True):
    """
    `mpjve` is the mean per joint velociy error, which is the mean of the Euclidean distance between the predicted 3D
    joint velocity and the ground truth 3D joint velocity
    :param pred:
    :param gt:
    :param dim:
    :param mean:
    :return:
    """
    pred_vel = pred[:, 1:] - pred[:, :-1]
    gt_vel = gt[:, 1:] - gt[:, :-1]

    pjve = ((pred_vel - gt_vel) ** 2).sum(-1) ** 0.5

    if not mean:
        return pjve

    # if pjve is torch.Tensor use dim if numpy.array use axis
    if isinstance(pjve, torch.Tensor):
        if dim is None:
            return pjve.mean()
        return pjve.mean(dim=dim)

    if dim is None:
        return np.mean(pjve)

    return np.mean(pjve, axis=dim)