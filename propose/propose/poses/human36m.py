import os

import numpy as np
import torch

from .base import YamlPose

MPII_2_H36M = [
    6,
    2,
    1,
    0,
    3,
    4,
    5,
    7,
    8,
    9,
    13,
    14,
    15,
    12,
    11,
    10,
]


class Human36mPose(YamlPose):
    """
    Pose Class for the Human3.6M dataset.
    """

    def __init__(self, pose_matrix=None, **kwargs):
        dirname = os.path.dirname(__file__)
        path = os.path.join(dirname, "metadata/human36m.yaml")

        if pose_matrix is None:
            pose_matrix = np.zeros((1, 17, 3))

        super().__init__(pose_matrix, path)

    @classmethod
    def remove_root_edges(cls, edges, context_edges, num_context_samples):
        """
        We remove the root edges from the full edges, and then we subtract 1 from the full edges and context edges to
        make them zero-indexed

        :param cls: the class of the object
        :param edges: the edges of the full graph
        :param context_edges: the edges that are in the context graph
        :param num_context_samples: The number of samples in the context
        :return: The edges are being returned with the root edges removed.
        """
        full_edges = edges[:, torch.where(edges[0] != 0)[0]]
        context_edges = context_edges[:, torch.where(context_edges[1] != 0)[0]]
        root_edges = edges[:, torch.where(edges[0] == 0)[0]]

        full_edges -= 1
        context_edges[0] -= num_context_samples
        context_edges[1] -= 1
        root_edges[1] -= 1

        return full_edges, root_edges, context_edges


class MPIIPose(YamlPose):
    """
    Pose Class for the Human3.6M dataset.
    """

    def __init__(self, pose_matrix, **kwargs):
        dirname = os.path.dirname(__file__)
        path = os.path.join(dirname, "metadata/mpii.yaml")

        super().__init__(pose_matrix, path)

    def to_human36m(self):
        """
        Convert the pose to the Human3.6M format.
        :return: A Human3.6M pose.
        """
        pose_matrix = self.pose_matrix.copy()
        pose_matrix = pose_matrix[:, MPII_2_H36M]
        pose_matrix = np.insert(pose_matrix, 9, 0, axis=1)

        pose = Human36mPose(pose_matrix)
        pose.occluded_markers = self.occluded_markers[0, MPII_2_H36M, 0]
        pose.occluded_markers = np.insert(pose.occluded_markers, 9, True, axis=0)

        return pose