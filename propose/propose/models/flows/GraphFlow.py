"""Basic definitions for the flows module."""

import nflows.utils.typechecks as check
import torch
from nflows.flows.base import Flow
from torch_geometric.data import HeteroData


class GraphFlow(Flow):
    """
    Adaptation of Flow class for Hetero data.
    """

    def embed_inputs(self, inputs):
        """Embed the context inputs into the flow.

        Args:
            inputs: Tensor, input variables.

        Returns:
            A Tensor containing the embedded inputs, with shape [num_nodes, num_samples, ...].
        """
        inputs_dict = inputs.to_dict()

        if "c" not in inputs_dict or "x" not in inputs_dict["c"]:
            return inputs

        inputs_dict["c"]["x"] = self._embedding_net(inputs)

        if isinstance(inputs_dict["c"]["x"], HeteroData):
            inputs_dict["c"]["x"] = inputs_dict["c"]["x"]["c"]["x"]

        return HeteroData(inputs_dict)

    def log_prob(self, inputs):
        """Calculate log probability under the distribution.

        Args:
            inputs: Tensor, input variables.
            context: Tensor or None, conditioning variables. If a Tensor, it must have the same
                number or rows as the inputs. If None, the context is ignored.

        Returns:
            A Tensor of shape [input_size], the log probability of the inputs given the context.
        """
        inputs = self.embed_inputs(inputs)

        return self._log_prob(inputs)

    def _log_prob(self, inputs):
        noise, logabsdet = self._transform(inputs)

        log_prob = self._distribution.log_prob(noise["x"]["x"])

        return log_prob + logabsdet

    def sample(self, num_samples, context, temperature=1.0):
        """Generates samples from the distribution. Samples can be generated in batches.

        Args:
            num_samples: int, number of samples to generate.
            context: Tensor or None, conditioning variables. If None, the context is ignored.
            temperature: float, temperature to use for sampling.

        Returns:
            A Tensor containing the samples, with shape [num_samples, ...] if context is None, or
            [context_size, num_samples, ...] if context is given.
        """
        context = self.embed_inputs(context)

        if not check.is_positive_int(num_samples):
            raise TypeError("Number of samples must be a positive integer.")

        return self._sample(num_samples, context, temperature)

    def _sample(self, num_samples, context, temperature=1.0):
        num_nodes = context["x"]["x"].shape[0] if context is not None else 1
        num_features = context["x"]["x"].shape[-1] if context is not None else 3
        noise = self._distribution.sample(
            (num_samples * num_nodes), temperature=temperature
        ).reshape((num_nodes, num_samples, num_features))

        noise = self._noise_to_hetero(noise, context)

        samples, _ = self._transform.inverse(noise)

        return samples

    def mode_sample(self, context):
        context = self.embed_inputs(context)

        num_nodes = context["x"]["x"].shape[0] if context is not None else 1
        num_features = context["x"]["x"].shape[-1] if context is not None else 3
        noise = torch.zeros((num_nodes, 1, num_features)).to(self.device)

        noise = self._noise_to_hetero(noise, context)

        samples, _ = self._transform.inverse(noise)

        return samples

    def sample_and_log_prob(self, num_samples, context=None):
        """Generates samples from the flow, together with their log probabilities.

        For flows, this is more efficient that calling `sample` and `log_prob` separately.
        """
        context = self.embed_inputs(context)

        num_nodes = context["x"]["x"].shape[0] if context is not None else 1
        num_features = context["x"]["x"].shape[-1] if context is not None else 3

        noise, log_prob = self._distribution.sample_and_log_prob(
            (num_samples * num_nodes)
        )

        noise = noise.reshape((num_nodes, num_samples, num_features))
        log_prob = log_prob.reshape((num_nodes, num_samples)).mean(-1)

        noise = self._noise_to_hetero(noise, context)

        samples, logabsdet = self._transform.inverse(noise)

        return samples, log_prob - logabsdet

    def transform_to_noise(self, inputs):
        """Transforms given data into noise. Useful for goodness-of-fit checking.

        Args:
            inputs: A `Tensor` of shape [batch_size, ...], the data to be transformed.

        Returns:
            A `Tensor` of shape [batch_size, ...], the noise.
        """
        inputs = self.embed_inputs(inputs)
        noise, _ = self._transform(inputs)
        return noise

    @staticmethod
    def _noise_to_hetero(noise, context):
        context_dict = context.to_dict()
        noise = {**context_dict, "x": {**context_dict["x"], "x": noise}}
        if "c" in noise and "x" in noise["c"]:
            noise["c"]["x"] = noise["c"]["x"].unsqueeze(1)

        if "r" in noise and "x" in noise["r"]:
            noise["r"]["x"] = noise["r"]["x"].unsqueeze(1)

        noise = HeteroData(noise)

        return noise

    @property
    def device(self):
        return next(self.parameters()).device
