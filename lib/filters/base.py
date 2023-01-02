import jax
import jax.numpy as jnp

from ..base import module


class Filter(module):
    """
    GLM coupling filter base class.
    """

    cross_coupling: bool
    filter_time: jnp.ndarray

    def __init__(self, cross_coupling, filter_length, array_type):
        """
        Filter length includes instantaneous part
        """
        if filter_length <= 0:
            raise ValueError("Filter length must be bigger than zero")
        super().__init__(array_type)
        self.cross_coupling = cross_coupling
        self.filter_time = jnp.arange(filter_length, dtype=array_type)

    def compute_filter(self, prng_state):
        """
        Potentially stochastic filters
        """
        raise NotImplementedError

    def forward(self, prng_state, inputs):
        """
        Introduces the spike coupling by convolution with the spike train, no padding and left removal
        for causal convolutions.

        :param torch.Tensor input: input spiketrain or covariates with shape (trials, neurons, filter_length)
                                   or (samples, neurons, filter_length)
        :returns: filtered input of shape (trials, neurons, filter_length)
        """
        h, KL = self.compute_filter(prng_state)
        return F.conv1d(input, h_, groups=self.conv_groups), KL
