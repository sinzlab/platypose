import numpy as np

from .base import BasePose


class Rat7mPose(BasePose):
    """
    Pose Class for the Rat7M dataset.
    """

    marker_names = [
        "HeadF",
        "HeadB",
        "HeadL",
        "SpineF",
        "SpineM",
        "SpineL",
        "Offset1",
        "Offset2",
        "HipL",
        "HipR",
        "ElbowL",
        "ArmL",
        "ShoulderL",
        "ShoulderR",
        "ElbowR",
        "ArmR",
        "KneeR",
        "KneeL",
        "ShinL",
        "ShinR",
    ]

    def __init__(self, pose_matrix):
        super().__init__(pose_matrix)

    def set_adjacency_matrix(self):
        self.adjacency_matrix = np.eye(len(self.marker_names))

        edges = [
            self.get_edge("HeadF", "HeadB"),
            self.get_edge("HeadF", "HeadL"),
            self.get_edge("HeadF", "SpineF"),
            self.get_edge("HeadL", "SpineF"),
            self.get_edge("HeadL", "HeadB"),
            self.get_edge("HeadB", "SpineF"),
            self.get_edge("SpineF", "SpineM"),
            self.get_edge("SpineM", "SpineL"),
            self.get_edge("SpineF", "Offset1"),
            self.get_edge("SpineM", "Offset1"),
            self.get_edge("SpineM", "Offset2"),
            self.get_edge("SpineL", "Offset2"),
            self.get_edge("SpineL", "HipL"),
            self.get_edge("HipL", "KneeL"),
            self.get_edge("KneeL", "ShinL"),
            self.get_edge("SpineL", "HipR"),
            self.get_edge("HipR", "KneeR"),
            self.get_edge("KneeR", "ShinR"),
            self.get_edge("SpineF", "ShoulderL"),
            self.get_edge("ShoulderL", "ElbowL"),
            self.get_edge("ElbowL", "ArmL"),
            self.get_edge("SpineF", "ShoulderR"),
            self.get_edge("ShoulderR", "ElbowR"),
            self.get_edge("ElbowR", "ArmR"),
        ]

        for edge in edges:
            self.adjacency_matrix[edge] = 1
            self.adjacency_matrix[edge[::-1]] = 1

    @property
    def edge_groups(self):
        """
        Edge groups for plotting.
        :return: dict of edge groups
        """
        head_edges = np.array(
            [
                self._edge("HeadF", "HeadB"),
                self._edge("HeadF", "HeadL"),
                self._edge("HeadF", "SpineF"),
                self._edge("HeadL", "SpineF"),
                self._edge("HeadL", "HeadB"),
                self._edge("HeadB", "SpineF"),
            ]
        )

        spine_edges = np.array(
            [
                self._edge("SpineF", "SpineM"),
                self._edge("SpineM", "SpineL"),
            ]
        )

        leg_l_edges = np.array(
            [
                self._edge("SpineL", "HipL"),
                self._edge("HipL", "KneeL"),
                self._edge("KneeL", "ShinL"),
            ]
        )

        leg_r_edges = np.array(
            [
                self._edge("SpineL", "HipR"),
                self._edge("HipR", "KneeR"),
                self._edge("KneeR", "ShinR"),
            ]
        )

        arm_l_edges = np.array(
            [
                self._edge("SpineF", "ShoulderL"),
                self._edge("ShoulderL", "ElbowL"),
                self._edge("ElbowL", "ArmL"),
            ]
        )

        arm_r_edges = np.array(
            [
                self._edge("SpineF", "ShoulderR"),
                self._edge("ShoulderR", "ElbowR"),
                self._edge("ElbowR", "ArmR"),
            ]
        )

        return {
            "head": head_edges,
            "spine": spine_edges,
            "leg_l": leg_l_edges,
            "leg_r": leg_r_edges,
            "arm_l": arm_l_edges,
            "arm_r": arm_r_edges,
        }
