from itertools import combinations

import torch
import torch.distributions as D
from torch.utils.data.dataset import Dataset
from torch_geometric.data import HeteroData
from torch_geometric.loader.dataloader import Collater


class PointDataset(Dataset):
    def __init__(self, prior=None):
        if prior is None:
            self.prior = D.MultivariateNormal(torch.zeros(3), torch.eye(3))

        if prior == "bimodal":
            self.prior = D.mixture_same_family.MixtureSameFamily(
                D.Categorical(torch.ones(2)),
                D.MultivariateNormal(
                    torch.Tensor([[0, 0, 10], [0, 0, -10]]),
                    covariance_matrix=torch.stack((torch.eye(3), torch.eye(3))),
                ),
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.data[idx], {"name": "Single"}

    def metadata(self):
        return self.data[0].metadata()


class SinglePointPriorDataset(PointDataset):
    def __init__(self, samples=100, prior=None):
        super().__init__(prior=prior)

        data_list = []

        for i in range(samples):
            data = HeteroData()
            data["x"].x = self.prior.sample((1,))

            data_list.append(data)

        self.data = data_list


class SinglePointDataset(PointDataset):
    def __init__(self, samples=100, prior=None):
        super().__init__(prior=prior)

        data_list = []

        for i in range(samples):
            data = HeteroData()
            data["x"].x = self.prior.sample((1,))
            data["c"].x = data["x"].x[..., :2]

            data["c", "->", "x"].edge_index = torch.LongTensor([[0, 0]]).T

            data_list.append(data)

        self.data = data_list


class TwoPointDataset(PointDataset):
    def __init__(self, samples=100, prior=None):
        super().__init__(prior=prior)

        data_list = []
        # semi_data_list = []
        prior_data_list = []

        collater = Collater(follow_batch=None, exclude_keys=None)

        for i in range(samples):
            M1 = self.prior.sample((1,))

            direction = torch.randn(3)
            direction[..., 2] = torch.randn(1, 1) / 5
            direction = direction / torch.norm(direction, dim=-1)
            M2 = M1 + direction

            data = HeteroData()

            data["x"].x = torch.stack([M1, M2]).squeeze()
            data["c"].x = data["x"].x[..., :2]

            data["x", "->", "x"].edge_index = torch.LongTensor([[0, 1]]).T
            data["x", "<-", "x"].edge_index = torch.LongTensor([[0, 1]]).T
            prior_data_list.append(data)

            data = HeteroData()

            data["x"].x = torch.stack([M1, M2]).squeeze()
            data["c"].x = data["x"].x[..., :2]

            data["c", "->", "x"].edge_index = torch.LongTensor([[0, 0], [1, 1]]).T
            data["x", "->", "x"].edge_index = torch.LongTensor([[0, 1]]).T
            data["x", "<-", "x"].edge_index = torch.LongTensor([[0, 1]]).T
            data_list.append(data)

            # for combs in combinations([(0, 0), (1, 1)], r=1):
            #     data = HeteroData()
            #     data["x"].x = torch.stack([M1, M2]).squeeze()
            #     data["c"].x = data["x"].x[..., :2]
            #
            #     data["c", "->", "x"].edge_index = torch.LongTensor([*combs]).T
            #     data["x", "->", "x"].edge_index = torch.LongTensor(
            #         [[0, 1]]
            #     ).T
            #     data["x", "<-", "x"].edge_index = torch.LongTensor(
            #         [[0, 1]]
            #     ).T
            #
            #     data_list.append(data)

        self.data = data_list
        # self.data = collater(data_list)
        self.prior_data = prior_data_list
        # self.semi_data = collater(semi_data_list)

    def __getitem__(self, idx):
        return (self.data[idx], self.prior_data[idx]), "None"  # , self.semi_data[idx]


class ThreePointDataset(PointDataset):
    def __init__(self, samples=100, prior=None):
        super().__init__(prior=prior)

        data_list = []

        for i in range(samples):
            M1 = self.prior.sample((1,))

            direction = torch.randn(3)
            direction[..., 2] = torch.randn(1, 1) / 5
            direction = direction / torch.norm(direction, dim=-1)
            M2 = M1 + direction

            direction = torch.randn(3)
            direction[..., 2] = torch.randn(1, 1) / 5
            direction = direction / torch.norm(direction, dim=-1)
            M3 = M2 + direction

            data = HeteroData()

            data["x"].x = torch.stack([M1, M2, M3]).squeeze()
            data["c"].x = data["x"].x[..., :2]

            data["c", "->", "x"].edge_index = torch.LongTensor(
                [[0, 0], [1, 1], [2, 2]]
            ).T
            data["x", "->", "x"].edge_index = torch.LongTensor([[0, 1], [1, 2]]).T
            data["x", "<-", "x"].edge_index = torch.LongTensor([[0, 1], [1, 2]]).T
            data_list.append(data)

            for r in [1, 2]:
                for combs in combinations([(0, 0), (1, 1), (2, 2)], r=r):
                    data = HeteroData()
                    data["x"].x = torch.stack([M1, M2, M3]).squeeze()
                    data["c"].x = data["x"].x[..., :2]

                    data["c", "->", "x"].edge_index = torch.LongTensor([*combs]).T
                    data["x", "->", "x"].edge_index = torch.LongTensor(
                        [[0, 1], [1, 2]]
                    ).T
                    data["x", "<-", "x"].edge_index = torch.LongTensor(
                        [[0, 1], [1, 2]]
                    ).T

                    data_list.append(data)

            data = HeteroData()
            data["x"].x = torch.stack([M1, M2, M3]).squeeze()
            data["c"].x = data["x"].x[..., :2]

            data["c", "->", "x"].edge_index = torch.LongTensor([[0, 0], [2, 2]]).T
            data["x", "->", "x"].edge_index = torch.LongTensor([[0, 1], [1, 2]]).T
            data["x", "<-", "x"].edge_index = torch.LongTensor([[0, 1], [1, 2]]).T

            data_list.append(data)

        self.data = data_list
