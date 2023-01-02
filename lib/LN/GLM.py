import math

import jax.numpy as jnp
import jax.random as jr

# from functools import partial

from ..base import module


class Mapping(module):
    """
    The observation model class, E_q[ p(yₙ|fₙ) p(fₙ|xₙ) ], defines a mapping combined with a likelihood

    The default functions here use cubature/MC approximation methods, exact integration is specific
    to certain observation classes.
    """

    in_dims: int
    out_dims: int

    def __init__(self, in_dims, out_dims):
        """
        :param hyp: (hyper)parameters of the likelihood model
        """
        self.in_dims = in_dims
        self.out_dims = out_dims

    ### distributions ###
    def evaluate_posterior(self, x, params, var_params, mean_only, compute_KL, jitter):
        """ """
        raise NotImplementedError("KL term for this mapping is not implemented")

    def sample_prior(self, prng_state, x, jitter):
        raise NotImplementedError(
            "Variational posterior for this mapping is not implemented"
        )

    def sample_posterior(self, prng_state, x, jitter, compute_KL):
        raise NotImplementedError("Prior for this mapping is not implemented")


class Constant(module):
    """
    Constant value
    """

    value: jnp.ndarray  # (out_dims,)

    def __init__(self, in_dims, out_dims):
        """
        :param variance: The observation noise variance, σ²
        """
        super().__init__(in_dims, out_dims)

    # @partial(jit, static_argnums=(0, 4, 5))
    def evaluate_posterior(self, x, mean_only, compute_KL, jitter):
        """
        :param jnp.array x: input of shape (time, num_samps, in_dims, 1)
        :returns:
            means of shape (time, num_samps, out_dims)
            covariances of shape (time, time, num_samps, out_dims)
        """
        ts, num_samps = x.shape[:2]
        post_means = jnp.ones((ts, num_samps, 1)) * self.value[None, None, :]
        post_covs = None if mean_only else jnp.zeros((ts, ts, num_samps, self.out_dims))
        return post_means, post_covs, 0.0

    # @partial(jit, static_argnums=(0,))
    def sample_prior(self, prng_state, x, jitter):
        """
        Prior distribution p(f(x)) = N(0, K_xx)
        Can use approx_points as number of points

        :param jnp.ndarray x: shape (time, num_samps, in_dims)
        :return:
            sample of shape (time, num_samps, out_dims)
        """
        return self.evaluate_posterior(x, None, True, False, jitter)[0]

    # @partial(jit, static_argnums=(0, 6))
    def sample_posterior(self, prng_state, x, jitter, compute_KL):
        """
        Sample from posterior q(f|x)

        :param jnp.array x: input of shape (time, num_samps, in_dims)
        :return:
            sample of shape (time, num_samps, out_dims)
        """
        samples, _, KL = self.evaluate_posterior(x, True, compute_KL, jitter)
        return samples, KL


class Identity(module):
    """
    Direct regression to likelihood
    """

    def __init__(self, in_dims):
        """
        :param variance: The observation noise variance, σ²
        """
        super().__init__(in_dims, in_dims)

    # @partial(jit, static_argnums=(0, 4, 5))
    def evaluate_posterior(self, x, mean_only, compute_KL, jitter):
        """
        :param jnp.array x: input of shape (time, num_samps, in_dims, 1)
        :returns:
            means of shape (time, num_samps, out_dims)
            covariances of shape (time, time, num_samps, out_dims)
        """
        ts, num_samps = x.shape[:2]

        post_means = x[..., 0]
        post_covs = None if mean_only else jnp.zeros((ts, ts, num_samps, self.out_dims))
        return post_means, post_covs, 0.0

    # @partial(jit, static_argnums=(0,))
    def sample_prior(self, prng_state, x, jitter):
        """
        Prior distribution p(f(x)) = N(0, K_xx)
        Can use approx_points as number of points

        :param jnp.ndarray x: shape (time, num_samps, in_dims)
        :return:
            sample of shape (time, num_samps, out_dims)
        """
        return self.evaluate_posterior(x, params, None, True, False, jitter)[0]

    # @partial(jit, static_argnums=(0, 6))
    def sample_posterior(self, prng_state, x, jitter, compute_KL):
        """
        Sample from posterior q(f|x)

        :param jnp.array x: input of shape (time, num_samps, in_dims)
        :return:
            sample of shape (time, num_samps, out_dims)
        """
        samples, _, KL = self.evaluate_posterior(x, True, compute_KL, jitter)
        return samples, KL


class Linear(module):
    """
    Factor analysis
    """

    C: jnp.ndarray
    b: jnp.ndarray

    def __init__(self, C, b):
        """
        :param variance: The observation noise variance, σ²
        """
        in_dims, out_dims = C.shape
        super().__init__(in_dims, out_dims)
        self.C = C
        self.b = b

    # @partial(jit, static_argnums=(0, 4, 5))
    def evaluate_posterior(self, x, mean_only, compute_KL, jitter):
        """
        :param jnp.array x: input of shape (time, num_samps, in_dims, 1)
        :returns:
            means of shape (time, num_samps, out_dims)
            covariances of shape (time, time, num_samps, out_dims)
        """
        params = self.params if params is None else params
        C = params["C"][None, None, ...]
        b = params["b"][None, None, :, None]
        post_means = (C @ x + b)[..., 0]  # (time, num_samps, in_dims)

        ts, num_samps = x.shape[:2]
        post_covs = None if mean_only else jnp.zeros((ts, ts, num_samps, self.out_dims))
        return post_means, post_covs, 0.0

    # @partial(jit, static_argnums=(0,))
    def sample_prior(self, prng_state, x, jitter):
        """
        Prior distribution p(f(x)) = N(0, K_xx)
        Can use approx_points as number of points

        :param jnp.ndarray x: shape (time, num_samps, in_dims)
        :return:
            sample of shape (time, num_samps, out_dims)
        """
        return self.evaluate_posterior(x, None, True, False, jitter)[0]

    # @partial(jit, static_argnums=(0, 6))
    def sample_posterior(self, prng_state, x, jitter, compute_KL):
        """
        Sample from posterior q(f|x)

        :param jnp.array x: input of shape (time, num_samps, in_dims)
        :return:
            sample of shape (time, num_samps, out_dims)
        """
        samples, _, KL = self.evaluate_posterior(x, True, compute_KL, jitter)
        return samples, KL
