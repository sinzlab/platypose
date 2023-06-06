import nflows.utils.typechecks as check
import numpy as np
import torch
from nflows.distributions.base import Distribution
from nflows.utils import torchutils


class StandardNormal(Distribution):
    """
    A multivariate Normal with zero mean and unit covariance.
    Adapted from nflows.distributions.StandardNormal such that it works with graph data
    """

    def __init__(self, shape):
        super().__init__()
        self._shape = torch.Size(shape)

        self.register_buffer(
            "_log_z",
            torch.tensor(0.5 * np.prod(shape) * np.log(2 * np.pi), dtype=torch.float64),
            persistent=False,
        )

    def _log_prob(self, inputs, context):
        # Note: the context is ignored.
        if inputs.dim() == 3:
            inputs = inputs.mean(1)

        neg_energy = -0.5 * torchutils.sum_except_batch(inputs**2, num_batch_dims=1)
        return neg_energy - self._log_z

    def _sample(self, num_samples, context, temperature=1.0):
        if context is None:
            return (
                torch.randn(num_samples, *self._shape, device=self._log_z.device)
                * temperature
            )
        else:
            # The value of the context is ignored, only its size and device are taken into account.
            context_size = context.shape[0]
            samples = (
                torch.randn(
                    context_size * num_samples, *self._shape, device=context.device
                )
                * temperature
            )
            return torchutils.split_leading_dim(samples, [context_size, num_samples])

    def _mean(self, context):
        if context is None:
            return self._log_z.new_zeros(self._shape)
        else:
            # The value of the context is ignored, only its size is taken into account.
            return context.new_zeros(context.shape[0], *self._shape)

    def sample(self, num_samples, context=None, batch_size=None, temperature=1.0):
        """Generates samples from the distribution. Samples can be generated in batches.
        Args:
            num_samples: int, number of samples to generate.
            context: Tensor or None, conditioning variables. If None, the context is ignored.
            batch_size: int or None, number of samples per batch. If None, all samples are generated
                in one batch.
            temperature: float, temperature to use for sampling.
        Returns:
            A Tensor containing the samples, with shape [num_samples, ...] if context is None, or
            [context_size, num_samples, ...] if context is given.
        """
        if not check.is_positive_int(num_samples):
            raise TypeError("Number of samples must be a positive integer.")

        if context is not None:
            context = torch.as_tensor(context)

        if batch_size is None:
            return self._sample(num_samples, context, temperature=temperature)

        else:
            if not check.is_positive_int(batch_size):
                raise TypeError("Batch size must be a positive integer.")

            num_batches = num_samples // batch_size
            num_leftover = num_samples % batch_size
            samples = [self._sample(batch_size, context) for _ in range(num_batches)]
            if num_leftover > 0:
                samples.append(self._sample(num_leftover, context))
            return torch.cat(samples, dim=0)
