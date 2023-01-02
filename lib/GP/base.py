import math
from functools import partial
from typing import Union

import jax.numpy as jnp
import jax.random as jr

from jax import lax, vmap
from jax.numpy.linalg import cholesky
from jax.scipy.linalg import solve_triangular

from ..base import module

from .kernels import Kernel, MarkovianKernel
from .linalg import mvn_conditional


class GP(module):
    """
    GP with function and RFF kernel
    """

    kernel: Kernel
    RFF_num_feats: int

    mean: jnp.ndarray

    def __init__(self, kernel, mean, RFF_num_feats):
        super().__init__(kernel.array_type)
        self.kernel = kernel
        self.mean = self._to_jax(mean)  # (out_dims,)
        self.RFF_num_feats = RFF_num_feats  # use random Fourier features

    def apply_constraints(self):
        """
        PSD constraint
        """
        kernel = self.kernel.apply_constraints(self.kernel)

        model = jax.tree_map(lambda p: p, self)  # copy
        model = eqx.tree_at(
            lambda tree: tree.kernel,
            model,
            replace_fn=lambda _: kernel,
        )

        return model

    def evaluate_conditional(self, x, x_obs, f_obs, mean_only, diag_cov, jitter):
        """
        Compute the conditional distribution

        :param jnp.array x: shape (num_samps, out_dims, time, in_dims)
        :param jnp.array x_obs: shape (num_samps, out_dims, obs_pts, in_dims)
        :param jnp.array f_obs: shape (num_samps, out_dims, obs_pts, 1)
        :return:
            conditional mean of shape (num_samps, out_dims, ts, 1)
            conditional covariance of shape (num_samps, out_dims, ts, 1)
        """
        cond_out = vmap(
            mvn_conditional,
            (0, 0, 0, None, None, None, None),
            0 if mean_only else (0, 0),
        )(x, x_obs, f_obs, self.kernel.K, mean_only, diag_cov, jitter)

        if mean_only:
            return cond_out + self.mean[None, :, None, None]
        else:
            return cond_out[0] + self.mean[None, :, None, None], cond_out[1]

    def sample_prior(self, prng_state, x, jitter):
        """
        Prior distribution p(f(x)) = N(0, K_xx)
        Can use approx_points as number of points

        :param jnp.array x: shape (num_samps, out_dims, time, in_dims)
        :return:
            sample of shape (num_samps, out_dims, time)
        """
        in_dims = self.kernel.in_dims
        out_dims = self.kernel.out_dims

        # if x.ndim == 3:
        #    x = x[..., None, :]  # (time, num_samps, out_dims, in_dims)
        num_samps, ts = x.shape[0], x.shape[2]

        if self.RFF_num_feats > 0:  # random Fourier features
            prng_keys = jr.split(prng_state, 2)
            ks, amplitude = self.kernel.sample_spectrum(
                prng_keys[0], num_samps, self.RFF_num_feats
            )  # (num_samps, out_dims, feats, in_dims), (out_dims,)
            phi = (
                2
                * jnp.pi
                * jr.uniform(
                    prng_keys[1], shape=(num_samps, out_dims, self.RFF_num_feats)
                )
            )
            cos_terms = jnp.cos(
                (ks[..., None, :, :] * x[..., None, :]).sum(-1) + phi[..., None, :]
            )  # (num_samps, out_dims, time, feats)
            amps = amplitude[None, :, None] * jnp.sqrt(
                2.0 / self.RFF_num_feats
            )  # (num_samps, out_dims, feats)
            samples = self.mean[None, :, None] + amps * cos_terms.sum(
                -1
            )  # (num_samps, out_dims, time)

        else:
            Kxx = vmap(self.kernel.K, (0, None, None), 0)(
                x, None, False
            )  # (num_samps, out_dims, time, time)
            eps_I = jitter * jnp.eye(ts)
            Lcov = cholesky(Kxx + eps_I)
            mean = self.mean[None, :, None, None]  # match (num_samps, out_dims, ts, 1)
            samples = mean + Lcov @ jr.normal(
                prng_state, shape=(num_samps, out_dims, ts, 1)
            )
            samples = samples[..., 0]

        return samples


class SSM(module):
    """
    Gaussian Linear Time-Invariant System

    Temporal multi-output kernels have separate latent processes that can be coupled.
    Spatiotemporal kernel modifies the process noise across latent processes, but dynamics uncoupled.
    Multi-output GPs generally mix latent processes via dynamics as well.
    """

    site_locs: Union[jnp.ndarray, None]
    site_obs: jnp.ndarray
    site_Lcov: jnp.ndarray

    def __init__(self, site_locs, site_obs, site_Lcov, array_type):
        """
        :param module markov_kernel: (hyper)parameters of the state space model
        :param jnp.ndarray site_locs: means of shape (time, 1)
        :param jnp.ndarray site_obs: means of shape (time, x_dims, 1)
        :param jnp.ndarray site_Lcov: covariances of shape (time, x_dims, x_dims)
        """
        super().__init__(array_type)
        self.site_locs = self._to_jax(site_locs) if site_locs is not None else None
        self.site_obs = self._to_jax(site_obs)
        self.site_Lcov = self._to_jax(site_Lcov)
