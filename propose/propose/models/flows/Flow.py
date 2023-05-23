import numpy as np
import torch
from nflows.distributions import StandardNormal
from nflows.flows import Flow
from nflows.transforms import AffineCouplingTransform, BatchNorm, CompositeTransform
from torch import nn
from torch.nn import functional as F


class NF(Flow):
    def __init__(
        self,
        features=20 * 3,
        num_layers=5,
        hidden_features=100,
        context_features=0,
    ):
        """
        :param features: Number of features in the input.
        :param num_layers: Number of flow layers.
        :param hidden_features: Number of features in the hidden layers.
        """

        def create_net(in_features, out_features):
            return MLP(
                in_shape=(in_features + context_features,),
                out_shape=(out_features,),
                hidden_sizes=(hidden_features,)
                if isinstance(hidden_features, int)
                else hidden_features,
            )

        coupling_constructor = AffineCouplingTransform

        mask = torch.ones(features)
        mask[::2] = -1

        layers = []
        for _ in range(num_layers):
            layers.append(BatchNorm(features=features))
            layers.append(
                coupling_constructor(mask=mask, transform_net_create_fn=create_net)
            )
            mask *= -1

        super().__init__(
            transform=CompositeTransform(layers),
            distribution=StandardNormal([features]),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.log_prob(x)


class MLP(nn.Module):
    """A standard multi-layer perceptron."""

    def __init__(
        self,
        in_shape,
        out_shape,
        hidden_sizes,
        activation=F.relu,
        activate_output=False,
    ):
        """
        Args:
            in_shape: tuple, list or torch.Size, the shape of the input.
            out_shape: tuple, list or torch.Size, the shape of the output.
            hidden_sizes: iterable of ints, the hidden-layer sizes.
            activation: callable, the activation function.
            activate_output: bool, whether to apply the activation to the output.
        """
        super().__init__()
        self._in_shape = torch.Size(in_shape)
        self._out_shape = torch.Size(out_shape)
        self._hidden_sizes = hidden_sizes
        self._activation = activation
        self._activate_output = activate_output

        if len(hidden_sizes) == 0:
            raise ValueError("List of hidden sizes can't be empty.")

        self._input_layer = nn.Linear(np.prod(in_shape), hidden_sizes[0])
        self._hidden_layers = nn.ModuleList(
            [
                nn.Linear(in_size, out_size)
                for in_size, out_size in zip(hidden_sizes[:-1], hidden_sizes[1:])
            ]
        )
        self._output_layer = nn.Linear(hidden_sizes[-1], np.prod(out_shape))

    def forward(self, inputs, context=None):
        if context is not None:
            inputs = torch.cat([inputs, context], dim=1)

        inputs = inputs.reshape(-1, np.prod(self._in_shape))

        outputs = self._input_layer(inputs)
        outputs = self._activation(outputs)

        for hidden_layer in self._hidden_layers:
            outputs = hidden_layer(outputs)
            outputs = self._activation(outputs)

        outputs = self._output_layer(outputs)
        if self._activate_output:
            outputs = self._activation(outputs)
        outputs = outputs.reshape(-1, *self._out_shape)

        return outputs
