import nflows.utils.typechecks as check
import torch
from nflows.transforms.base import CompositeTransform, Transform
from nflows.transforms.coupling import CouplingTransform
from nflows.utils import torchutils
from torch import nn
from torch_geometric.data import HeteroData


class GraphAffineCouplingTransform(CouplingTransform):
    def forward(self, inputs):
        inputs, identity_split, transform_split = self._split_inputs(inputs)

        transform_params = self.transform_net(identity_split)
        transform_split, logabsdet = self._coupling_transform_forward(
            inputs=transform_split, transform_params=transform_params
        )

        if self.unconditional_transform is not None:
            identity_split, logabsdet_identity = self.unconditional_transform(
                identity_split
            )
            logabsdet += logabsdet_identity

        outputs = self._join_splits(inputs, identity_split, transform_split)

        return outputs, logabsdet

    def inverse(self, inputs):
        inputs, identity_split, transform_split = self._split_inputs(inputs)

        logabsdet = 0.0
        if self.unconditional_transform is not None:
            identity_split, logabsdet = self.unconditional_transform.inverse(
                identity_split
            )

        transform_params = self.transform_net(identity_split)
        transform_split, logabsdet_split = self._coupling_transform_inverse(
            inputs=transform_split, transform_params=transform_params
        )
        logabsdet += logabsdet_split

        outputs = self._join_splits(inputs, identity_split, transform_split)

        return outputs, logabsdet

    def _join_splits(self, inputs, identity_split, transform_split):
        inputs["x"]["x"][..., self.identity_features] = identity_split["x"]["x"]
        inputs["x"]["x"][..., self.transform_features] = transform_split["x"]["x"]

        outputs = HeteroData(inputs)

        return outputs

    def _split_inputs(self, inputs):
        if not isinstance(inputs, dict):
            inputs_dict = inputs.to_dict()
        else:
            inputs_dict = inputs

        identity_split = {
            **inputs_dict,
            "x": dict(x=inputs["x"]["x"][..., self.identity_features]),
        }
        transform_split = {
            **inputs_dict,
            "x": dict(x=inputs["x"]["x"][..., self.transform_features]),
        }

        return inputs_dict, identity_split, transform_split

    def _transform_dim_multiplier(self):
        return 2

    def _scale_and_shift(self, transform_params):
        unconstrained_scale = transform_params[..., self.num_transform_features :]
        shift = transform_params[..., : self.num_transform_features]
        scale = torch.sigmoid(unconstrained_scale + 2) + 1e-3
        return scale, shift

    def _coupling_transform_forward(self, inputs, transform_params):
        scale, shift = self._scale_and_shift(transform_params)
        # _, shift = self._scale_and_shift(transform_params)
        log_scale = torch.log(scale)
        #
        outputs = {**inputs, "x": dict(x=inputs["x"]["x"] * scale + shift)}
        # outputs = {**inputs, "x": dict(x=inputs["x"]["x"] + shift)}
        #
        if log_scale.dim() == 3:
            log_scale = log_scale.mean(dim=1)
        #
        logabsdet = torchutils.sum_except_batch(log_scale, num_batch_dims=1)
        # logabsdet = torch.zeros(log_scale.shape[0], 3).to(log_scale.device)
        # if log_scale.shape[1] == 1:
        #     logabsdet[:, -1:] = log_scale
        # else:
        #     logabsdet[:, :-1] = log_scale

        return outputs, logabsdet

    def _coupling_transform_inverse(self, inputs, transform_params):
        scale, shift = self._scale_and_shift(transform_params)
        # _, shift = self._scale_and_shift(transform_params)
        log_scale = torch.log(scale)

        outputs = {**inputs, "x": dict(x=(inputs["x"]["x"] - shift) / scale)}
        # outputs = {**inputs, "x": dict(x=(inputs["x"]["x"] - shift))}

        if log_scale.dim() == 3:
            log_scale = log_scale.mean(dim=1)

        logabsdet = -torchutils.sum_except_batch(log_scale, num_batch_dims=1)
        return outputs, logabsdet


class GraphActNorm(Transform):
    def __init__(self, features):
        """
        Transform that performs activation normalization. Works for 2D and 4D inputs. For 4D
        inputs (images) normalization is performed per-channel, assuming BxCxHxW input shape.

        Reference:
        > D. Kingma et. al., Glow: Generative flow with invertible 1x1 convolutions, NeurIPS 2018.
        """
        if not check.is_positive_int(features):
            raise TypeError("Number of features must be a positive integer.")
        super().__init__()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.register_buffer(
            "initialized",
            torch.tensor(False, dtype=torch.bool, device=torch.device(self.device)),
        )
        self.log_scale = nn.Parameter(torch.zeros(features))
        self.shift = nn.Parameter(torch.zeros(features))

    @property
    def scale(self):
        return torch.exp(self.log_scale)

    def _initialize(self, inputs):
        """Data-dependent initialization, s.t. post-actnorm activations have zero mean and unit
        variance."""
        dims = list(range(inputs.dim()))[:-1]
        with torch.no_grad():
            std = inputs.std(dim=dims)
            mu = (inputs / std).mean(dim=dims)
            self.log_scale.data = -torch.log(std)
            self.shift.data = -mu
            self.initialized.data = torch.tensor(
                True, dtype=torch.bool, device=torch.device(self.device)
            )

    def forward(self, inputs, context=None):
        x = inputs["x"]["x"]

        if self.training and not self.initialized:
            self._initialize(x)

        scale, shift = self.scale.view(1, -1), self.shift.view(1, -1)

        scaled_x = scale * x + shift

        batch_size = x.shape[0]
        logabsdet = torch.sum(self.log_scale) * scaled_x.new_ones(batch_size)
        # logabsdet = self.log_scale * scaled_x.new_ones(batch_size, 3)

        inputs_dict = inputs.to_dict()
        outputs = {**inputs_dict, "x": {**inputs_dict["x"], "x": scaled_x}}
        outputs = HeteroData(outputs)

        return outputs, logabsdet

    def inverse(self, inputs, context=None):
        x = inputs["x"]["x"]

        scale, shift = self.scale.view(1, -1), self.shift.view(1, -1)
        scaled_x = (x - shift) / scale

        batch_size = x.shape[0]
        logabsdet = -torch.sum(self.log_scale) * scaled_x.new_ones(batch_size)

        inputs_dict = inputs.to_dict()
        outputs = {**inputs_dict, "x": {**inputs_dict["x"], "x": scaled_x}}
        outputs = HeteroData(outputs)

        return outputs, logabsdet


class GraphCompositeTransform(CompositeTransform):
    @staticmethod
    def _cascade(inputs, funcs):
        batch_size = inputs["x"].x.shape[0]
        outputs = inputs
        total_logabsdet = torch.zeros(batch_size, device=inputs["x"]["x"].device)
        # total_logabsdet = torch.zeros(batch_size, 3, device=inputs["x"]["x"].device)
        for func in funcs:
            outputs, logabsdet = func(outputs)
            total_logabsdet += logabsdet

        return outputs, total_logabsdet

    def forward(self, inputs):
        funcs = self._transforms
        return self._cascade(inputs, funcs)

    def inverse(self, inputs):
        funcs = (transform.inverse for transform in self._transforms[::-1])
        return self._cascade(inputs, funcs)
