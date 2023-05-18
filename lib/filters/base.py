import jax
import jax.numpy as jnp
from jax import lax, vmap

from ..base import module


class Filter(module):
    """
    GLM coupling filter base class.
    """

    cross_coupling: bool
    filter_time: jnp.ndarray
    filter_length: int
    obs_dims: int

    def __init__(self, obs_dims, cross_coupling, filter_length, array_type):
        """
        Filter length includes instantaneous part
        """
        if filter_length <= 0:
            raise ValueError("Filter length must be bigger than zero")
        super().__init__(array_type)
        self.cross_coupling = cross_coupling
        self.filter_time = jnp.arange(filter_length, dtype=self.array_dtype())
        self.filter_length = filter_length
        self.obs_dims = obs_dims

    def compute_posterior(self, mean_only, sel_outdims, jitter):
        """
        Compute filter
        """
        raise NotImplementedError

    def sample_prior(self, prng_state, num_samps, sel_outdims, jitter):
        """
        Valid for deterministic filters
        """
        h = self.compute_posterior(True, sel_outdims, jitter)[0]
        return h[None].repeat(num_samps, axis=0)

    def sample_posterior(self, prng_state, num_samps, compute_KL, sel_outdims, jitter):
        """
        Valid for deterministic filters
        """
        h = self.compute_posterior(True, sel_outdims, jitter)[0]
        return h[None].repeat(num_samps, axis=0), 0.0

    def apply_filter(
        self, prng_state, inputs, joint_samples, compute_KL, prior, sel_outdims, jitter
    ):
        """
        Introduces the spike coupling by convolution with the spike train, no padding and left removal
        for causal convolutions.

        :param jnp.ndarray inputs: input spiketrain or covariates with shape (trials, neurons, ts)
        :param bool joint_samples: apply joint samples from the filter (relevant when probabilistic),
                                   otherwise applies the mean filter
        :returns:
            filtered input of shape (trials, neurons, filter_length)
        """
        if sel_outdims is None:
            sel_outdims = jnp.arange(self.obs_dims)
        num_samps = inputs.shape[0]

        if joint_samples:
            if prior:
                KL = 0.0
                h = self.sample_prior(
                    prng_state, num_samps, sel_outdims, jitter
                )  # (num_samps, filter_len, post, pre)

            else:
                h, KL = self.sample_posterior(
                    prng_state, num_samps, compute_KL, sel_outdims, jitter
                )  # (num_samps, filter_len, post, pre)

        else:
            KL = 0.0
            h, _ = self.compute_posterior(
                False, sel_outdims, jitter
            )  # (filter_len, post, pre)
            h = h[None].repeat(num_samps, axis=0)

        def conv_func(inputs, h, dn):
            return lax.conv_general_dilated(
                inputs,  # lhs = image tensor
                h,  # rhs = conv kernel tensor
                (1,),  # window strides
                "VALID",  # padding mode
                (1,),  # lhs/image dilation
                (1,),  # rhs/kernel dilation
                dn,
            )  # dimension_numbers = lhs, rhs, out dimension permutation

        if self.cross_coupling:
            dn = lax.conv_dimension_numbers(
                inputs.shape[1:], h.shape[1:], ("NCW", "WIO", "NCW")
            )
            out = vmap(conv_func, (0, 0, None))(inputs, h, dn)  # (num_samps, post, ts)

        else:
            filter_len = h.shape[1]
            inputs = inputs[:, sel_outdims]  # convolution per channel

            dn = lax.conv_dimension_numbers(
                (inputs.shape[1], 1, 1), (filter_len, 1, 1), ("NCW", "WIO", "NCW")
            )
            inputs = inputs.reshape(
                -1, 1, 1, inputs.shape[-1]
            )  # (num_samps * N, 1, 1, ts)
            h = h.transpose(0, 2, 1, 3).reshape(
                -1, filter_len, 1, 1
            )  # (num_samps * N, filter_len, 1, 1)

            out = vmap(conv_func, (0, 0, None))(inputs, h, dn).reshape(
                num_samps, len(sel_outdims), -1
            )  # (num_samps, post, ts)

        return out, KL
