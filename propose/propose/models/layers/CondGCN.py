import itertools
from typing import Literal

import torch
import torch.nn as nn
import torch_sparse as ts


class CondGCN(nn.Module):
    """
    Conditional GCN layer.
    """

    def __init__(
        self,
        in_features: int = 3,
        context_features: int = 2,
        out_features: int = 3,
        hidden_features: int = 10,
        root_features: int = 3,
        aggr: Literal["add", "mean", "max"] = "add",
        relations: list[str] = None,
        use_attention: bool = False,
    ) -> None:
        super().__init__()

        default_relations: list[str] = [
            "x",
            "c",
            "r",  # self loop
            "x->x",
            "x<-x",  # symmetric
            "c->x",
            "r->x",
        ]  # context

        self.relations = relations if relations else default_relations

        self.features = {
            "x": in_features,
            "c": context_features,
            "r": root_features,
            "hidden": hidden_features,
            "out": out_features,
        }

        self.layers = self._build_layers()

        self.pool = nn.Linear(hidden_features, out_features)
        self.act = nn.ReLU()

        self.attention = nn.Linear(in_features * 2, 1)
        self.use_attention = use_attention

        self.aggr = aggr

    def forward(self, x_dict: dict, edge_index_dict: dict) -> tuple[dict, dict]:
        """
        The function takes in a dictionary of node features and a dictionary of edge features, and returns a dictionary of
        node features and a dictionary of edge features.

        :param x_dict: a dictionary of node features
        :type x_dict: dict
        :param edge_index_dict: a dictionary of edge indices for each edge type
        :type edge_index_dict: dict
        :return: The output of the pooling layer.
        """
        x = x_dict["x"]

        self_x = self.act(self.layers["x"](x))  # self loop values

        message = self.aggregate(self.message(x_dict, edge_index_dict), self_x)

        if "c" in x_dict and x_dict["c"] is not None:
            x_dict["c"] = self.act(self.layers["c"](x_dict["c"]))

        if "r" in x_dict and x_dict["r"] is not None:
            x_dict["r"] = self.act(self.layers["r"](x_dict["r"]))

        x_dict["x"] = self.pool(message)

        return x_dict, edge_index_dict

    def message(
        self, x_dict: dict, edge_index_dict: dict, target: Literal["x", "c", "r"] = "x"
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the message for each edge.
        :param x_dict: x_dict['x'] is the node features.
        :param edge_index_dict: edge_index_dict['x<->x'] is the edge indices.
        :return: message and destination index
        """
        for key in edge_index_dict.keys():
            src_name, direction, dst_name = key  # e.g. "c", "->", "x"
            layer_name = "".join(key)  # e.g. "c->x"
            src, dst = edge_index_dict[key]

            # If the edge is an inverse edge, swap the source and destination
            if direction == "<-":
                src_name, dst_name = dst_name, src_name
                src, dst = dst, src
                layer_name = "x->x"  # .join(key[::-1])

            if dst_name != target:
                continue

            # attention mechanism
            if self.use_attention and src_name == "x":
                yield self.attention_mechanism(x_dict, src_name, dst_name, layer_name)
                continue

            message = self.act(self.layers[layer_name](x_dict[src_name][src]))

            yield message, dst

    def attention_mechanism(
        self, x_dict: dict, src_name: str, dst_name: str, layer_name: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        We create a fully connected graph, then we use the attention mechanism to compute the attention
        between each pair of nodes, which controls the computed message from each node to each other node

        :param x_dict: a dictionary of node features
        :type x_dict: dict
        :param src_name: the name of the source node
        :type src_name: str
        :param dst_name: the name of the node that the message is being sent to
        :type dst_name: str
        :param layer_name: The name of the layer to use for the message
        :type layer_name: str
        :return: The message and the destination node.
        """
        n_nodes = x_dict[src_name].shape[0]

        indexs = (
            torch.Tensor(
                list(itertools.product(list(range(n_nodes)), list(range(n_nodes))))
            )
            .long()
            .t()
            .to(self.device)
        )

        src = indexs[1]
        dst = indexs[0]

        src_x = x_dict[src_name][src]
        dst_x = x_dict[dst_name][dst]

        attention = self.attention(
            torch.cat(
                [
                    src_x,
                    dst_x,
                ],
                dim=-1,
            )
        )

        attention = torch.softmax(attention, dim=0)

        i = torch.arange(n_nodes).long().to(self.device)

        message = self.act(self.layers[layer_name](x_dict[src_name][i]))
        message = message.repeat(n_nodes, *[1] * (message.dim() - 1))

        message = torch.multiply(message, attention)

        message = message.reshape(
            -1, n_nodes, message.shape[-2], message.shape[-1]
        ).sum(0)

        dst = dst.reshape(-1, n_nodes)[:, 0]

        return message, dst

    def aggregate(
        self, message: tuple[torch.Tensor, torch.Tensor], self_x: torch.Tensor
    ) -> torch.Tensor:
        """
        Aggregates the messages according to the aggregation method.
        :param message: message tensor
        :param self_x: self loop features
        :return: aggregated message
        """
        message = list(message)

        values, indexes = [], []
        if len(message):
            values, indexes = list(zip(*message))

        index = torch.cat(
            [
                *indexes,  # concatenate all message indices
                torch.arange(
                    self_x.shape[0],
                    dtype=torch.long,
                    device=self.device,  # if torch.cuda.is_available() else 'cpu'
                ),  # add self loop indices
            ]
        )
        index = torch.stack(
            [index, torch.zeros_like(index)]  # Only one column index, thus all zeros
        )

        # Concatenate all messages with self loop features

        # when sampling the messages from context need to be repeated to match the number of samples

        if len(values) == 4 and values[0].dim() == 3:
            values = list(values)
            samples = values[0].shape[1]
            if values[2].shape[1] == 1:
                values[2] = values[2].repeat(1, samples, 1)
            if values[3].shape[1] == 1:
                values[3] = values[3].repeat(1, samples, 1)

        if len(values) == 3 and values[0].dim() == 3:
            values = list(values)
            samples = self_x.shape[1]
            if values[0].shape[1] == 1:
                values[0] = values[0].repeat(1, samples, 1)
            if values[2].shape[1] == 1:
                values[2] = values[2].repeat(1, samples, 1)

        if len(values) == 1 and values[0].dim() == 3:
            values = list(values)
            # samples = values[0].shape[1]
            # self_x = self_x.repeat(1, samples, 1)
            samples = self_x.shape[1]
            values[0] = values[0].repeat(1, samples, 1)

        value = torch.cat([*values, self_x])  # concatenate all the message values

        # Aggregates the messages according to the aggregation method where the same index is used
        _, aggr_message = ts.coalesce(
            index, value, m=index[0].max() + 1, n=index[1].max() + 1, op=self.aggr
        )

        return aggr_message

    @property
    def device(self) -> Literal["cpu", "cuda"]:
        """
        It returns the device of the module
        :return: The device of the first parameter of the model.
        """
        return next(self.parameters()).device

    def _build_layers(self) -> nn.ModuleDict:
        """
        For each relation in the relations list, create a linear layer with the number of features of the first node in
        the relation as the input size and the number of features of the hidden layer as the output size
        :return: A dictionary of linear layers.
        """
        layers_dict = {}
        for relation in self.relations:
            n_features: int = self.features[relation[0]]
            layers_dict[relation] = nn.Linear(n_features, self.features["hidden"])

        return nn.ModuleDict(layers_dict)
