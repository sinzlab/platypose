from collections import namedtuple

import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData

import propose.preprocessing.rat7m as pp
from propose.poses.rat7m import Rat7mPose


class ScalePose(object):
    def __init__(self, scale):
        self.scale = scale

    def __call__(self, x):
        key_vals = {k: v for k, v in zip(x._fields, x)}

        key_vals["poses"] = pp.scale_pose(pose=x.poses, scale=self.scale)
        if "poses2d" in key_vals:
            key_vals["poses2d"] = pp.scale_pose(pose=x.poses2d, scale=self.scale)

        return x.__class__(**key_vals)


class CenterPose(object):
    def __init__(self, center_marker_name="SpineM"):
        self.center_marker_name = center_marker_name

    def __call__(self, x):
        key_vals = {k: v for k, v in zip(x._fields, x)}

        key_vals["poses"] = pp.center_pose(
            pose=key_vals["poses"], center_marker_name=self.center_marker_name
        )

        return x.__class__(**key_vals)


class CropImageToPose(object):
    def __init__(self, width=350):
        self.width = width

    def __call__(self, x):
        key_vals = {k: v for k, v in zip(x._fields, x)}

        pose = key_vals["poses"]
        image = key_vals["images"]
        camera = key_vals["cameras"]

        pose2D = Rat7mPose(camera.proj2D(pose))

        key_vals["images"] = pp.square_crop_to_pose(
            image=image, pose2D=pose2D, width=self.width
        )

        return x.__class__(**key_vals)


class RotatePoseToCamera(object):
    def __call__(self, x):
        key_vals = {k: v for k, v in zip(x._fields, x)}

        pose = key_vals["poses"]
        camera = key_vals["cameras"]

        key_vals["poses"] = pp.rotate_to_camera(pose=pose, camera=camera)

        return x.__class__(**key_vals)


class ToGraph(object):
    def __init__(self):
        self.graph_data_point = namedtuple(
            "GraphDataPoint", ("pose_matrix", "adjacency_matrix", "image")
        )

    def __call__(self, x):
        return self.graph_data_point(
            pose_matrix=x.poses.pose_matrix,
            adjacency_matrix=x.poses.adjacency_matrix,
            image=x.images,
        )


class SwitchArmsElbows(object):
    def __call__(self, x):
        key_vals = {k: v for k, v in zip(x._fields, x)}
        pose = key_vals["poses"]

        key_vals["poses"] = pp.switch_arms_elbows(pose=pose)

        return x.__class__(**key_vals)


class ScalePixelRange(object):
    def __call__(self, x):
        key_vals = {k: v for k, v in zip(x._fields, x)}
        key_vals["images"] = pp.scale_pixel_range(image=key_vals["images"])

        return x.__class__(**key_vals)


class Project2D(object):
    def __init__(self, idx):
        self.idx = idx

    def __call__(self, x):
        key_vals = {k: v for k, v in zip(x._fields, x)}
        poses = key_vals["poses"]
        key_vals["poses2d"] = poses[..., self.idx]

        return namedtuple("With2DPose", key_vals.keys())(**key_vals)


class ToHeteroData(object):
    def __init__(self, encode_joints: bool = False):
        """
        Converts a dataset to a HeteroData object.
        :param encode_joints: Whether to encode the joints as a one-hot vector
        """
        self.encode_joints = encode_joints

    def __call__(self, x):
        key_vals = {k: v for k, v in zip(x._fields, x)}

        pose3d = x.poses
        # pose2d = x.poses2d

        # one_hot_encoding = F.one_hot(torch.arange(len(pose3d)), len(pose3d)).float()

        # c = torch.Tensor(pose2d.pose_matrix)
        # if self.encode_joints:
        #     c = torch.cat([c, one_hot_encoding], dim=1)

        data = HeteroData()
        data["x"].x = torch.Tensor(pose3d.pose_matrix)
        data["x", "->", "x"].edge_index = torch.LongTensor(pose3d.edges).T
        data["x", "<-", "x"].edge_index = torch.LongTensor(pose3d.edges).T

        if "poses2d" in key_vals:
            pose2d = x.poses2d

            data["c"].x = torch.Tensor(pose2d.pose_matrix)
            context_edges = (
                torch.arange(0, pose3d.pose_matrix.shape[0])
                .repeat(2)
                .reshape(2, pose3d.pose_matrix.shape[0])
                .long()
            )
            data["c", "->", "x"].edge_index = context_edges

            del key_vals["poses2d"]

        # pose = HeteroData(
        #     {
        #         "x": {"x": torch.Tensor(pose3d.pose_matrix)},
        #         # "c": {"x": c},
        #         "edge_index": {
        #             ("x", "->", "x"): torch.LongTensor(pose3d.edges),
        #             ("x", "<-", "x"): torch.LongTensor(pose3d.edges),
        #             # ("c", "->", "x"): torch.arange(0, len(pose3d.edges)).repeat(2).reshape(2, len(pose3d.edges)).T.long(),
        #         },
        #     }
        # )

        key_vals["poses"] = data

        graph_data_point = namedtuple("GraphDataPoint", ("poses"))

        # return graph_data_point(**key_vals)
        return graph_data_point(**key_vals)

        # return data
