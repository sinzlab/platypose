from typing import Optional, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn.dense import DenseSAGEConv


class Embedding(nn.Module):
    """
    Base class for the embedding layer.
    """

    def __init__(self):
        super().__init__()
        self.embedding = nn.Identity()

    def forward(
        self, x: Union[tuple[Tensor, Tensor], Tensor]
    ) -> Union[tuple[Tensor, Tensor], Tensor]:
        if isinstance(x, tuple):
            return self.embedding(x[0]), x[1]
        else:
            return self.embedding(x)


class LinearEmbedding(Embedding):
    """
    Linear embedding layer.
    """

    def __init__(self, input_size: int, output_size: int):
        """
        :param input_size
        :param output_size
        """
        super().__init__()
        self.embedding = nn.Linear(input_size, output_size)


class MLPEmbedding(Embedding):
    """
    MLP embedding layer.
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        """
        :param input_size
        :param output_size
        """
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )


class FlatMLPEmbedding(nn.Module):
    """
    MLP embedding layer.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        """
        :param input_dim
        :param output_dim
        """
        super().__init__()
        self.features = [
            input_dim,
            hidden_dim,
            output_dim,
        ]

        self.embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, data):
        x = data["c"].x
        batch_size = x.shape[0]
        # batch_size = data['c'].x.max() + 1

        # print(data['c', '->', 'x'].edge_index[0])
        # x = x[data['c', '->', 'x'].edge_index[0]]
        x = x.reshape(batch_size, -1, self.features[0])
        batch_size = x.shape[0]
        # x = x.reshape(batch_size, 16, 2)
        # x = x[:, data['c', '->', 'x'].edge_index[0]]
        x = x.flatten(1)
        x = self.embedding(x)

        x = x.reshape(-1, self.features[0])

        return x


class SageEmbedding(nn.Module):
    """
    Sage Convolutional Embedding Layer
    """

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SageEmbedding, self).__init__()

        self.features = [input_dim, hidden_dim, output_dim]
        self.layers = nn.ModuleList(
            [
                DenseSAGEConv(input_dim, hidden_dim),
                DenseSAGEConv(hidden_dim, output_dim),
            ]
        )

        self.relu = nn.ReLU()

        self.adj = nn.Parameter(torch.ones((16, 16)))

    def forward(self, data):
        x = data["c"].x

        # batch_size = data['c'].x.max() + 1

        x = x.reshape(-1, 16, self.features[0])
        batch_size = x.shape[0]

        index = x.new_ones(batch_size, 16).bool().flatten()
        index[data["c", "->", "x"].edge_index[0]] = False
        index = index.reshape(batch_size, 16)

        adj = self.adj.repeat(batch_size, 1, 1)

        adj[index.unsqueeze(-1).repeat(1, 1, 16)] = 0
        adj[index.unsqueeze(1).repeat(1, 16, 1)] = 0

        for layer in self.layers[:-1]:
            x = layer(x, adj)
            x = self.relu(x)

        x = self.layers[-1](x, adj)

        x = x.reshape(batch_size * 16, -1)

        return x


class SplitEmbedding(nn.Module):
    """
    Split embedding layer.
    """

    def __init__(self, split_size: int):
        """
        :param split_size: size of the first split
        """
        super().__init__()
        self.split_size = split_size

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        return x[:, : self.split_size], x[:, self.split_size :]


class JoinEmbedding(nn.Module):
    """
    Join embedding layer.
    """

    def forward(self, x: tuple[Tensor]) -> Tensor:
        return torch.cat(x, dim=1)


class SplitLinearEmbedding(Embedding):
    """
    Split linear embedding layer.
    """

    def __init__(self, split_size, input_size, output_size):
        super().__init__()

        self.embedding = nn.Sequential(
            SplitEmbedding(split_size),
            LinearEmbedding(input_size, output_size),
            JoinEmbedding(),
        )


embeddings = {
    "linear": LinearEmbedding,
    "mlp": MLPEmbedding,
    "flat_mlp": FlatMLPEmbedding,
    "sage": SageEmbedding,
    "split_linear": SplitLinearEmbedding,
}
