import torch

human36m_joints_to_use = [1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13, 14, 15, 16]


def pck(
    poses_gt: torch.Tensor,
    poses_pred: torch.Tensor,
    threshold: float = 150,
    return_distances: bool = False,
) -> torch.BoolTensor:
    """
    It computes the percentage of frames in which the predicted pose is within a threshold distance of the ground truth
    pose

    :param poses_gt: the ground truth poses with only the joints of interest (frames x joints x 3)
    :type poses_gt: torch.Tensor
    :param poses_pred: the predicted poses with only the joints of interest (frames x joints x 3)
    :type poses_pred: torch.Tensor
    :param threshold: The threshold for the distance between the predicted and ground truth pose, defaults to 180
    :type threshold: float (optional)
    :param return_distances: If True, returns the distances between the predicted and ground truth pose, defaults to False
    :type return_distances: bool (optional)
    """
    if not isinstance(poses_pred, torch.Tensor):
        poses_pred = torch.Tensor(poses_pred)

    if not isinstance(poses_gt, torch.Tensor):
        poses_gt = torch.Tensor(poses_gt)

    distances = torch.sqrt(torch.sum((poses_gt - poses_pred) ** 2, dim=-1))

    if return_distances:
        return distances

    n_correct_joints = torch.count_nonzero(distances < threshold, dim=1)
    correct_poses = n_correct_joints / poses_gt.shape[1]

    return correct_poses
