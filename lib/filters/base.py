import jax
from jax import lax, vmap
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

    def compute_filter(self, prng_state, compute_KL):
        """
        Potentially stochastic filters
        """
        raise NotImplementedError

    def apply_filter(self, prng_state, inputs, compute_KL):
        """
        Introduces the spike coupling by convolution with the spike train, no padding and left removal
        for causal convolutions.

        :param jnp.ndarray input: input spiketrain or covariates with shape (trials, neurons, filter_length)
        :returns: filtered input of shape (trials, neurons, filter_length)
        """
        # num_samps = inputs.shape[0]
        h, KL = self.compute_filter(prng_state, compute_KL)  # sample one trajectory
        
        if self.cross_coupling:
            dn = lax.conv_dimension_numbers(
                inputs.shape, h.shape, ('NCW', 'WIO', 'NCW'))

            out = lax.conv_general_dilated(inputs,   # lhs = image tensor
                                           h,      # rhs = conv kernel tensor
                                           (1,),   # window strides
                                           'VALID', # padding mode
                                           (1,),   # lhs/image dilation
                                           (1,),   # rhs/kernel dilation
                                           dn)     # dimension_numbers = lhs, rhs, out dimension permutation
            
        else:
            dn = lax.conv_dimension_numbers(
                (inputs.shape[0], 1, 1), (h.shape[0], 1, 1), ('NCW', 'WIO', 'NCW'))
            inputs = inputs[..., None, :]
            h = h[..., None]
            print(inputs.shape)
            print(h.shape)
            out = vmap(lax.conv_general_dilated, (1, 1, None, None, None, None, None), 1)(
                inputs,   # lhs = image tensor
                h,      # rhs = conv kernel tensor
                (1,),   # window strides
                'VALID', # padding mode
                (1,),   # lhs/image dilation
                (1,),   # rhs/kernel dilation
                dn,     # dimension_numbers = lhs, rhs, out dimension permutation
            )[..., 0, :]  # vmap over out_dims
            
        return out, KL
