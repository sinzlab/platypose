import numpy as np
import torch
import torch.distributions as D
from torch_geometric.data import HeteroData
from torch_geometric.loader.dataloader import Collater


class ScaleGraphPose(object):
    def __init__(self, scale):
        self.scale = scale

    def __call__(self, x):
        key_vals = {k: v for k, v in zip(x._fields, x)}

        pose_graph = x.poses
        pose_graph["x"]["x"] = pose_graph["x"]["x"] * self.scale

        if "x" in pose_graph["c"]:
            pose_graph["c"]["x"] = pose_graph["c"]["x"] * self.scale

        key_vals["poses"] = pose_graph

        return x.__class__(**key_vals)


class AppendOneHot(object):
    def __call__(self, x):
        key_vals = {k: v for k, v in zip(x._fields, x)}

        pose_graph = x.poses
        num_joints = pose_graph["x"]["x"].shape[0]

        one_hot_vector = torch.eye(num_joints)

        if "c" not in pose_graph or "x" not in pose_graph["c"]:
            pose_graph["c"]["x"] = one_hot_vector
            pose_graph["edge_index"][("c", "->", "x")] = (
                torch.arange(0, num_joints).repeat(2).reshape(2, num_joints).T.long()
            )

        key_vals["poses"] = pose_graph

        return x.__class__(**key_vals)


class BinomialMask(object):
    def __init__(self, p, n):
        self.p = p
        self.n = n

    def __call__(self, x):
        key_vals = {k: v for k, v in zip(x._fields, x)}

        pose_graph = x.poses
        num_joints = pose_graph["x"]["x"].shape[0]

        # pose_graph['c']['x'] = pose_graph['c']['x'][mask]
        pose_graph_dict = pose_graph.to_dict()
        pose_graphs = [pose_graph]
        for _ in range(self.n):
            mask = D.Bernoulli(probs=self.p).sample(sample_shape=(num_joints,)).bool()

            if ("c", "->", "x") in pose_graph_dict.keys():
                new_pose_graph = {
                    **pose_graph_dict,
                    ("c", "->", "x"): {
                        "edge_index": pose_graph_dict[("c", "->", "x")]["edge_index"][
                            :, mask
                        ],
                    },
                }
            # new_pose_graph[('c', '->', 'x')]['edge_index'] = new_pose_graph[('c', '->', 'x')]['edge_index'][:, mask]

            pose_graphs.append(HeteroData(new_pose_graph))

        pose_graphs = Collater(follow_batch=None, exclude_keys=None).__call__(
            pose_graphs
        )

        key_vals["poses"] = pose_graphs

        return x.__class__(**key_vals)


class BinaryMask(object):
    def __init__(self, p: list[float]):
        self.p = p

    def __call__(self, x):
        key_vals = {k: v for k, v in zip(x._fields, x)}

        pose_graph = x.poses
        num_joints = pose_graph["x"]["x"].shape[0]

        # pose_graph['c']['x'] = pose_graph['c']['x'][mask]
        pose_graph_dict = pose_graph.to_dict()
        pose_graphs = [pose_graph]
        for p in self.p:
            mask = np.ones(num_joints)
            mask[: int(p * num_joints)] = 0
            # shuffle mask
            np.random.shuffle(mask)
            mask = mask.astype(bool)

            new_pose_graph = {
                **pose_graph_dict,
            }
            if ("c", "->", "x") in pose_graph_dict.keys():
                new_pose_graph = {
                    **pose_graph_dict,
                    ("c", "->", "x"): {
                        "edge_index": pose_graph_dict[("c", "->", "x")]["edge_index"][
                            :, mask
                        ],
                    },
                }
            # new_pose_graph[('c', '->', 'x')]['edge_index'] = new_pose_graph[('c', '->', 'x')]['edge_index'][:, mask]

            pose_graphs.append(HeteroData(new_pose_graph))

        pose_graphs = Collater(follow_batch=None, exclude_keys=None).__call__(
            pose_graphs
        )

        key_vals["poses"] = pose_graphs

        return x.__class__(**key_vals)
