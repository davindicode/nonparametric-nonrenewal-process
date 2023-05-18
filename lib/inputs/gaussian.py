from typing import List, Union

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr

import numpy as np
from jax import vmap
from jax.numpy.linalg import cholesky

from ..base import ArrayTypes, module

from ..GP.sparse import SparseGP

from ..utils.jax import safe_sqrt


class GaussianLatentObservedSeries(module):
    """
    Time series with latent GP and observed covariates
    """

    gp: Union[SparseGP, None]
    lat_dims: List[int]
    obs_dims: List[int]
    x_dims: int
    diagonal_cov: bool

    def __init__(
        self, gp, lat_dims, obs_dims, diagonal_cov=False, array_type="float32"
    ):
        if gp is not None:  # checks
            assert gp.array_type == ArrayTypes[array_type]
            assert len(lat_dims) == gp.kernel.out_dims
            assert gp.kernel.in_dims == 1  # temporal GP

        super().__init__(array_type)
        self.gp = gp
        self.lat_dims = lat_dims
        self.obs_dims = obs_dims
        self.x_dims = len(self.lat_dims) + len(self.obs_dims)
        self.diagonal_cov = diagonal_cov

    def apply_constraints(self):
        """
        Constrain parameters in optimization
        """
        model = super().apply_constraints()
        if model.gp is not None:
            model = eqx.tree_at(
                lambda tree: tree.gp,
                model,
                replace_fn=lambda obj: obj.apply_constraints(),
            )

        return model

    def sample_prior(self, prng_state, num_samps, timestamps, x_eval, jitter):
        """
        Combines observed inputs with latent trajectories

        :param jnp.ndarray timestamps: time stamps of inputs (ts,)
        :param jnp.ndarray x_eval: inputs for evaluation (ts, x_dims)
        """
        ts = len(timestamps)
        if len(self.obs_dims) > 0:
            x_eval = jnp.broadcast_to(x_eval, (num_samps, ts, len(self.obs_dims)))

        if len(self.lat_dims) == 0:
            x = x_eval

        else:
            xx = timestamps[None, None, :, None].repeat(num_samps, axis=0)
            x_samples = self.gp.sample_prior(prng_state, xx, jitter)

            if len(self.obs_dims) == 0:
                x = x_samples

            else:
                x = jnp.empty((num_samps, ts, self.x_dims))
                x = x.at[..., self.obs_dims].set(x_eval)
                x = x.at[..., self.lat_dims].set(x_samples)

        return x  # (num_samps, ts, x_dims)

    def sample_posterior(
        self, prng_state, num_samps, timestamps, x_eval, jitter, compute_KL
    ):
        """
        Combines observed inputs with latent trajectories

        :param jnp.ndarray timestamps:
        """
        ts = len(timestamps)
        if len(self.obs_dims) > 0:
            x_eval = jnp.broadcast_to(x_eval, (num_samps, ts, len(self.obs_dims)))

        if len(self.lat_dims) == 0:
            x, KL = x_eval, 0.0

        else:
            xx = timestamps[None, None, :, None].repeat(num_samps, axis=0)
            x_samples, KL = self.gp.sample_posterior(
                prng_state, xx, compute_KL, jitter
            )  # (tr, time, x_dims)
            x_samples = x_samples[..., 0]

            if len(self.obs_dims) == 0:
                x = x_samples

            else:
                x = jnp.empty((num_samps, len(timestamps), self.x_dims))
                x = x.at[..., self.obs_dims].set(x_eval)
                x = x.at[..., self.lat_dims].set(x_samples)

        return x, KL

    def marginal_posterior(self, num_samps, timestamps, x_eval, jitter, compute_KL):
        """
        Combines observed inputs with latent marginal samples

        :param jnp.ndarray x_eval: observed covariates (ts, obs_dims)
        :return:
            x_mean (tr, time, x_dims)
            x_cov (tr, time, x_dims, 1 or x_dims) depending on diagonal_cov
            KL divergence (scalar)
        """
        ts = len(timestamps)
        if len(self.obs_dims) > 0:
            x_eval = jnp.broadcast_to(x_eval, (num_samps, ts, len(self.obs_dims)))

        if len(self.lat_dims) == 0:
            x_mean, KL = x_eval, 0.0
            x_cov = jnp.zeros_like(x_eval)[..., None]

        else:
            xx = timestamps[None, None, :, None].repeat(num_samps, axis=0)
            post_mean, post_cov, KL, _ = self.gp.evaluate_posterior(
                xx, False, self.diagonal_cov, compute_KL, False, jitter
            )

            if len(self.obs_dims) == 0:
                x_mean, x_cov = post_mean, post_cov

            else:
                x_mean, KL = (
                    jnp.empty(
                        (num_samps, len(timestamps), self.x_dims),
                        dtype=self.array_dtype(),
                    ),
                    0.0,
                )
                x_mean = x_mean.at[..., self.obs_dims].set(x_eval)
                x_mean = x_mean.at[..., self.lat_dims].set(post_mean)

                if self.diagonal_cov:
                    x_cov = jnp.empty(
                        (num_samps, len(timestamps), self.x_dims, 1),
                        dtype=self.array_dtype(),
                    )
                    x_cov = x_cov.at[..., self.obs_dims, 0].set(jnp.zeros_like(x_eval))
                    x_cov = x_cov.at[..., self.lat_dims, 0].set(
                        vmap(vmap(jnp.diag))(post_cov)
                    )

                else:
                    x_cov = jnp.zeros(
                        (num_samps, len(timestamps), self.x_dims, self.x_dims),
                        dtype=self.array_dtype(),
                    )
                    acc = jnp.array(self.lat_dims)[:, None].repeat(
                        len(self.lat_dims), axis=1
                    )
                    x_cov = x_cov.at[..., acc, acc.T].set(post_cov)

        return x_mean, x_cov, KL

    def sample_marginal_posterior(
        self, prng_state, num_samps, timestamps, x_eval, jitter, compute_KL
    ):
        x_mean, x_cov, KL = self.marginal_posterior(
            num_samps, timestamps, x_eval, jitter, compute_KL
        )

        # conditionally independent sampling
        if self.diagonal_cov:
            x_std = safe_sqrt(x_cov)
            x = x_mean + x_std[..., 0] * jr.normal(
                prng_state, shape=x_mean.shape
            )  # (num_samps, ts, x_dims)
        else:
            eps = jitter * jnp.eye(self.x_dims)[None, None]
            Lcov = cholesky(x_cov + eps)
            x = (
                x_mean
                + (Lcov @ jr.normal(prng_state, shape=x_mean.shape)[..., None])[..., 0]
            )  # (num_samps, ts, x_dims)
        return x, KL
