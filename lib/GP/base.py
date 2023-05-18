from typing import Union

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import lax, vmap
from jax.numpy.linalg import cholesky
from jax.scipy.linalg import solve_triangular

from ..base import ArrayTypes_, module

from .kernels import Kernel
from .linalg import mvn_conditional


class GP(module):
    """
    GP with function and RFF kernel, zero mean
    """

    kernel: Kernel
    RFF_num_feats: int

    def __init__(self, kernel, RFF_num_feats):
        super().__init__(ArrayTypes_[kernel.array_type])
        self.kernel = kernel
        self.RFF_num_feats = RFF_num_feats  # use random Fourier features

    def apply_constraints(self):
        """
        PSD constraint
        """
        model = jax.tree_map(lambda p: p, self)  # copy
        model = eqx.tree_at(
            lambda tree: tree.kernel,
            model,
            replace_fn=lambda obj: obj.apply_constraints(),
        )

        return model

    def evaluate_conditional(
        self, x, x_obs, f_obs, mean_only, diag_cov, jitter, sel_outdims
    ):
        """
        Compute the conditional distribution

        :param jnp.array x: shape (num_samps, out_dims, time, in_dims)
        :param jnp.array x_obs: shape (num_samps, out_dims, obs_pts, in_dims)
        :param jnp.array f_obs: shape (num_samps, out_dims, obs_pts, 1)
        :return:
            conditional mean of shape (num_samps, out_dims, ts, 1)
            conditional covariance of shape (num_samps, out_dims, ts, 1)
        """
        if sel_outdims is None:
            sel_outdims = jnp.arange(self.kernel.out_dims)

        Koo = vmap(self.kernel.K, (0, None, None), 0)(x_obs, None, False, sel_outdims)
        Kox = vmap(self.kernel.K, (0, 0, None), 0)(
            x_obs, x, False, sel_outdims
        )  # (num_samps, out_dims, obs_pts, time)

        if mean_only is False:
            Kxx = vmap(self.kernel.K, (0, None, None, None), 0)(
                x, None, diag_cov, sel_outdims
            )  # (num_samps, out_dims, time, time or 1)
        else:
            Kxx = None

        cond_out = vmap(
            mvn_conditional,
            (None if mean_only else 0, 0, 0, None, None, None),
            0 if mean_only else (0, 0),
        )(Kxx, Kox, Koo, f_obs, mean_only, diag_cov, jitter)

        return cond_out

    def sample_prior(self, prng_state, x, jitter, sel_outdims=None):
        """
        Prior distribution p(f(x)) = N(0, K_xx)
        Can use approx_points as number of points

        :param jnp.array x: shape (num_samps, out_dims, time, in_dims)
        :return:
            sample of shape (num_samps, out_dims, time)
        """
        if sel_outdims is None:
            out_dims = self.kernel.out_dims
            sel_outdims = jnp.arange(out_dims)

        num_samps, ts = x.shape[0], x.shape[2]

        if self.RFF_num_feats > 0:  # random Fourier features
            prng_keys = jr.split(prng_state, 3)
            ks, amplitude = self.kernel.sample_spectrum(
                prng_keys[0], num_samps, self.RFF_num_feats, sel_outdims
            )  # (num_samps, out_dims, feats, in_dims), (num_samps, out_dims, feats)

            phi = (
                2
                * jnp.pi
                * jr.uniform(
                    prng_keys[1],
                    shape=(num_samps, len(sel_outdims), self.RFF_num_feats),
                )
            )
            cos_terms = jnp.cos(
                (ks[..., None, :, :] * x[..., None, :]).sum(-1) + phi[..., None, :]
            )  # (num_samps, out_dims, time, feats)
            amps = amplitude * jnp.sqrt(
                2.0 / self.RFF_num_feats
            )  # (num_samps, out_dims, feats)

            weights = jr.normal(
                prng_keys[2], shape=(num_samps, len(sel_outdims), 1, self.RFF_num_feats)
            )
            samples = (amps[..., None, :] * weights * cos_terms).sum(
                -1
            )  # (num_samps, out_dims, time)

        else:
            Kxx = vmap(self.kernel.K, (0, None, None, None), 0)(
                x, None, False, sel_outdims
            )  # (num_samps, out_dims, time, time)
            eps_I = jitter * jnp.eye(ts)
            Lcov = cholesky(Kxx + eps_I)
            samples = Lcov @ jr.normal(
                prng_state, shape=(num_samps, len(sel_outdims), ts, 1)
            )
            samples = samples[..., 0]

        return samples
