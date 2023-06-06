from typing import Union

import torch
import torch.nn as nn
from propose.models.layers.CondGCN import CondGCN
from torch_geometric.data import HeteroData


class CondGNN(nn.Module):
    """
    Conditional Graph neural network.
    """

    def __init__(
        self,
        in_features: int = 3,
        context_features: int = 2,
        out_features: int = 3,
        hidden_features: int = 10,
        root_features: int = 3,
        relations: list[str] = None,
        use_attention: bool = False,
    ):
        super().__init__()

        self.gcn = CondGCN

        self.layers = nn.ModuleList(
            [
                self.gcn(
                    in_features=in_features,
                    hidden_features=hidden_features,
                    out_features=hidden_features,
                    context_features=context_features,
                    root_features=root_features,
                    relations=relations,
                    use_attention=use_attention,
                ),
                self.gcn(
                    in_features=hidden_features,
                    hidden_features=hidden_features,
                    out_features=out_features,
                    context_features=hidden_features,
                    root_features=hidden_features,
                    relations=relations,
                    use_attention=use_attention,
                ),
            ]
        )

    def forward(self, data: Union[HeteroData, dict]) -> torch.Tensor:
        if isinstance(data, dict):
            data = HeteroData(data)

        x_dict = self._get_x_dict(data)
        edge_index_dict = self._get_edge_index(data)

        for layer in self.layers:
            x_dict, edge_index_dict = layer(x_dict, edge_index_dict)

        return x_dict["x"]

    @staticmethod
    def _get_x_dict(data: HeteroData) -> dict:
        """
        Ensure that the 'c' key is present in the data.
        :param data: HeteroData instance
        :return: x_dict: dict
        """
        x_dict = data.x_dict
        if "c" not in x_dict:
            x_dict["c"] = None

        if "r" not in x_dict:
            x_dict["r"] = None

        return x_dict

    @staticmethod
    def _get_edge_index(data: HeteroData) -> dict:
        """
        Ensure that there is an edge_index_dict in the data.
        :param data: HeteroData instance
        :return: edge_index_dict: dict
        """
        try:
            edge_index_dict = data.edge_index_dict
        except AttributeError:
            edge_index_dict = {}

        return edge_index_dict
