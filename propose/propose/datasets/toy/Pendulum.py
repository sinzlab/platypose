from collections import namedtuple
from itertools import combinations

import brax
import torch
from torch.utils.data import Dataset
from torch_geometric.data import HeteroData
from tqdm import tqdm

from propose.training.utils import get_x_graph


class PendulumDataset(Dataset):
    def __init__(self, double=True, N=10, frames=100):
        data_list = []

        pendulum = self._setup_pendulum(double)

        for i in tqdm(range(N)):
            sys = brax.System(pendulum)
            qp = sys.default_qp()
            qp.pos[:, 2] -= 2.5
            qp.vel[-1, 0] = torch.randn(1) * 10

            for i in range(frames):
                points = torch.Tensor(qp.pos)
                points[:, [1, 2]] = points[:, [2, 1]].clone()

                data = HeteroData()
                data["x"].x = points

                n_points = 3 if double else 2
                point_indexes = torch.arange(n_points)
                c = data["x"].x[..., :2]
                data["c"].x = c

                if double:
                    data["c", "->", "x"].edge_index = torch.LongTensor(
                        [[0, 0], [1, 1], [2, 2]]
                    ).T
                    data["x", "->", "x"].edge_index = torch.LongTensor(
                        [[0, 1], [1, 2]]
                    ).T
                    data["x", "<-", "x"].edge_index = torch.LongTensor(
                        [[0, 1], [1, 2]]
                    ).T
                else:
                    data["c", "->", "x"].edge_index = torch.LongTensor(
                        [[0, 0], [1, 1]]
                    ).T
                    data["x", "->", "x"].edge_index = torch.LongTensor([[0, 1]]).T
                    data["x", "<-", "x"].edge_index = torch.LongTensor([[0, 1]]).T

                data_list.append(data)

                if double:
                    for r in [1, 2]:
                        for combs in combinations([(0, 0), (1, 1), (2, 2)], r=r):
                            data = HeteroData()
                            data["x"].x = points

                            n_points = 3
                            point_indexes = torch.arange(n_points)
                            c = data["x"].x[..., :2]
                            data["c"].x = c

                            data["c", "->", "x"].edge_index = torch.LongTensor(
                                [*combs]
                            ).T
                            data["x", "->", "x"].edge_index = torch.LongTensor(
                                [[0, 1], [1, 2]]
                            ).T
                            data["x", "<-", "x"].edge_index = torch.LongTensor(
                                [[0, 1], [1, 2]]
                            ).T

                            data_list.append(data)

                else:
                    for r in [1, 2]:
                        for combs in combinations([(0, 0), (1, 1)], r=r):
                            data = HeteroData()
                            data["x"].x = points

                            n_points = 3
                            point_indexes = torch.arange(n_points)
                            c = data["x"].x[..., :2]
                            data["c"].x = c

                            data["c", "->", "x"].edge_index = torch.LongTensor(
                                [*combs]
                            ).T
                            data["x", "->", "x"].edge_index = torch.LongTensor(
                                [[0, 1]]
                            ).T
                            data["x", "<-", "x"].edge_index = torch.LongTensor(
                                [[0, 1]]
                            ).T

                            data_list.append(data)

                qp, _ = sys.step(qp, [])

        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        prior_data = get_x_graph(data)
        return data, prior_data

    def metadata(self):
        return self.data[0].metadata()

    def _setup_pendulum(self, double=True):
        bouncy_ball = brax.Config(dt=0.05, substeps=4)

        # ground is a frozen (immovable) infinite plane
        ground = bouncy_ball.bodies.add(name="ground")
        ground.frozen.all = True
        plane = ground.colliders.add().plane
        plane.SetInParent()  # for setting an empty oneof

        # ball weighs 1kg, has equal rotational inertia along all axes, is 1m long, and
        # has an initial rotation of identity (w=1,x=0,y=0,z=0) quaternion
        ball = bouncy_ball.bodies.add(name="ball", mass=1)
        cap = ball.colliders.add().capsule
        cap.radius, cap.length = 0.05, 1

        # gravity is -9.8 m/s^2 in z dimension
        bouncy_ball.gravity.z = -9.8

        # @title A pendulum config for Brax
        pendulum = brax.Config(dt=0.01, substeps=4)

        # start with a frozen anchor at the root of the pendulum
        anchor = pendulum.bodies.add(name="anchor", mass=1.0)
        anchor.frozen.all = True

        # now add a middle and bottom ball to the pendulum
        pendulum.bodies.append(ball)
        pendulum.bodies[1].name = "middle"

        # connect anchor to middle
        joint = pendulum.joints.add(
            name="joint1",
            parent="anchor",
            child="middle",
            stiffness=10000,
            angular_damping=5,
        )
        joint.angle_limit.add(min=-180, max=180)
        joint.child_offset.z = 1
        joint.rotation.z = 90

        # connect middle to bottom
        if double:
            pendulum.bodies.append(ball)
            pendulum.bodies[2].name = "bottom"

            pendulum.joints.append(joint)
            pendulum.joints[1].name = "joint2"
            pendulum.joints[1].parent = "middle"
            pendulum.joints[1].child = "bottom"

        # gravity is -9.8 m/s^2 in z dimension
        pendulum.gravity.z = -9.8

        return pendulum
