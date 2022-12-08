import math
from functools import partial

from .base import module

import jax.numpy as jnp
import jax.random as jr
from jax import grad, jacrev, jit, random, tree_map, value_and_grad, vjp, vmap
from jax.numpy.linalg import cholesky

from jax.scipy.linalg import cho_factor, cho_solve, solve_triangular

from .utils.jax import (
    expsum,
    mc_sample,
    sample_gaussian_noise,
    sigmoid,
    softplus,
    softplus_inv,
)
from .utils.linalg import enforce_positive_diagonal, gauss_hermite, inv

vdiag = vmap(jnp.diag)


class Mapping(module):
    """
    The observation model class, E_q[ p(yₙ|fₙ) p(fₙ|xₙ) ], defines a mapping combined with a likelihood

    The default functions here use cubature/MC approximation methods, exact integration is specific
    to certain observation classes.
    """

    def __init__(self, x_dims, f_dims, params, var_params):
        """
        :param hyp: (hyper)parameters of the likelihood model
        """
        self.x_dims = x_dims
        self.f_dims = f_dims

        self.params = params
        self.var_params = var_params

    ### distributions ###
    def evaluate_posterior(self, x, params, var_params, mean_only, compute_KL, jitter):
        """ """
        raise NotImplementedError("KL term for this mapping is not implemented")

    def sample_prior(self, params, prng_state, x, jitter):
        """ """
        raise NotImplementedError(
            "Variational posterior for this mapping is not implemented"
        )

    def sample_posterior(self, params, var_params, prng_state, x, jitter, compute_KL):
        """ """
        raise NotImplementedError("Prior for this mapping is not implemented")


class Constant(module):
    """
    Constant value
    """

    def __init__(self, x_dims, f_dims, params):
        """
        :param variance: The observation noise variance, σ²
        """
        super().__init__(x_dims, f_dims, params, {})

    @partial(jit, static_argnums=(0, 4, 5))
    def evaluate_posterior(self, x, params, var_params, mean_only, compute_KL, jitter):
        """
        :param jnp.array x: input of shape (time, num_samps, x_dims, 1)
        :returns:
            means of shape (time, num_samps, f_dims)
            covariances of shape (time, time, num_samps, f_dims)
        """
        params = self.params if params is None else params
        value = params["value"]  # (f_dims,)

        ts, num_samps = x.shape[:2]
        post_means = jnp.ones((ts, num_samps, 1)) * value[None, None, :]
        post_covs = None if mean_only else jnp.zeros((ts, ts, num_samps, self.f_dims))
        return post_means, post_covs, 0.0

    @partial(jit, static_argnums=(0,))
    def sample_prior(self, params, prng_state, x, jitter):
        """
        Prior distribution p(f(x)) = N(0, K_xx)
        Can use approx_points as number of points

        :param jnp.ndarray x: shape (time, num_samps, x_dims)
        :return:
            sample of shape (time, num_samps, f_dims)
        """
        return self.evaluate_posterior(x, params, None, True, False, jitter)[0]

    @partial(jit, static_argnums=(0, 6))
    def sample_posterior(self, params, var_params, prng_state, x, jitter, compute_KL):
        """
        Sample from posterior q(f|x)

        :param jnp.array x: input of shape (time, num_samps, x_dims)
        :return:
            sample of shape (time, num_samps, f_dims)
        """
        samples, _, KL = self.evaluate_posterior(
            x, params, var_params, True, compute_KL, jitter
        )
        return samples, KL


class Identity(module):
    """
    Direct regression to likelihood
    """

    def __init__(self, x_dims):
        """
        :param variance: The observation noise variance, σ²
        """
        super().__init__(x_dims, x_dims, {}, {})

    @partial(jit, static_argnums=(0, 4, 5))
    def evaluate_posterior(self, x, params, var_params, mean_only, compute_KL, jitter):
        """
        :param jnp.array x: input of shape (time, num_samps, x_dims, 1)
        :returns:
            means of shape (time, num_samps, f_dims)
            covariances of shape (time, time, num_samps, f_dims)
        """
        ts, num_samps = x.shape[:2]

        post_means = x[..., 0]
        post_covs = None if mean_only else jnp.zeros((ts, ts, num_samps, self.f_dims))
        return post_means, post_covs, 0.0

    @partial(jit, static_argnums=(0,))
    def sample_prior(self, params, prng_state, x, jitter):
        """
        Prior distribution p(f(x)) = N(0, K_xx)
        Can use approx_points as number of points

        :param jnp.ndarray x: shape (time, num_samps, x_dims)
        :return:
            sample of shape (time, num_samps, f_dims)
        """
        return self.evaluate_posterior(x, params, None, True, False, jitter)[0]

    @partial(jit, static_argnums=(0, 6))
    def sample_posterior(self, params, var_params, prng_state, x, jitter, compute_KL):
        """
        Sample from posterior q(f|x)

        :param jnp.array x: input of shape (time, num_samps, x_dims)
        :return:
            sample of shape (time, num_samps, f_dims)
        """
        samples, _, KL = self.evaluate_posterior(
            x, params, var_params, True, compute_KL, jitter
        )
        return samples, KL


class Linear(module):
    """
    Factor analysis
    """

    def __init__(self, C, b):
        """
        :param variance: The observation noise variance, σ²
        """
        x_dims, f_dims = C.shape
        params = {"C": C, "b": b}
        super().__init__(x_dims, f_dims, params, {})

    @partial(jit, static_argnums=(0, 4, 5))
    def evaluate_posterior(self, x, params, var_params, mean_only, compute_KL, jitter):
        """
        :param jnp.array x: input of shape (time, num_samps, x_dims, 1)
        :returns:
            means of shape (time, num_samps, f_dims)
            covariances of shape (time, time, num_samps, f_dims)
        """
        params = self.params if params is None else params
        C = params["C"][None, None, ...]
        b = params["b"][None, None, :, None]
        post_means = (C @ x + b)[..., 0]  # (time, num_samps, x_dims)

        ts, num_samps = x.shape[:2]
        post_covs = None if mean_only else jnp.zeros((ts, ts, num_samps, self.f_dims))
        return post_means, post_covs, 0.0

    @partial(jit, static_argnums=(0,))
    def sample_prior(self, params, prng_state, x, jitter):
        """
        Prior distribution p(f(x)) = N(0, K_xx)
        Can use approx_points as number of points

        :param jnp.ndarray x: shape (time, num_samps, x_dims)
        :return:
            sample of shape (time, num_samps, f_dims)
        """
        return self.evaluate_posterior(x, params, None, True, False, jitter)[0]

    @partial(jit, static_argnums=(0, 6))
    def sample_posterior(self, params, var_params, prng_state, x, jitter, compute_KL):
        """
        Sample from posterior q(f|x)

        :param jnp.array x: input of shape (time, num_samps, x_dims)
        :return:
            sample of shape (time, num_samps, f_dims)
        """
        samples, _, KL = self.evaluate_posterior(
            x, params, var_params, True, compute_KL, jitter
        )
        return samples, KL


class tSVGP(module):
    """
    Sparse variational Gaussian process

    Variational parameters are (u_mean, u_cov)
    site_mean has shape (f_dims, num_induc, 1)
    site_mean has shape (f_dims, num_induc, num_induc)

    Note q(f) factorizes along f_dims, not optimal for heteroscedastic likelihoods

    References:
        [1] site-based Sparse Variational Gaussian Processes
    """

    def __init__(self, kernel, mean_f, induc_locs, var_params, RFF_num_feats=0):
        """
        :param induc_locs: inducing point locations z, array of shape (f_dims, num_induc, in_dims)
        :param variance: The observation noise variance, σ²
        """
        params = {"induc_locs": induc_locs, "mean": mean_f, "kernel": kernel.hyp}
        if (not "lambda_1" in var_params.keys()) or (
            not "chol_Lambda_2" in var_params.keys()
        ):
            raise ValueError("Missing keys in site parameter dictionary")

        if induc_locs.shape[2] != kernel.in_dims:
            raise ValueError(
                "Dimensions of inducing locations do not match kernel input dimensions"
            )
        if induc_locs.shape[0] != kernel.out_dims:
            raise ValueError(
                "Dimensions of inducing locations do not match kernel output dimensions"
            )

        super().__init__(kernel.in_dims, kernel.out_dims, params, var_params)
        self.num_induc = induc_locs.shape[1]
        self.kernel = kernel

        self.RFF_num_feats = RFF_num_feats  # use random Fourier features

    def apply_constraints(self):
        """
        PSD constraint
        """
        chol = var_params["chol_Lambda_2"]
        chol = vmap(enforce_positive_diagonal)(np.tril(chol))
        var_params["chol_Lambda_2"] = chol
        return params, var_params

    @partial(jit, static_argnums=(0, 4, 5, 6))
    def evaluate_posterior(
        self, x, params, var_params, mean_only, compute_KL, compute_aux, jitter
    ):
        """
        :param jnp.array x: input of shape (time, num_samps, x_dims, 1)
        :returns:
            means of shape (f_dims, num_samps, time, 1)
            covariances of shape (f_dims, num_samps, time, time)
        """
        params = self.params if params is None else params
        var_params = self.var_params if var_params is None else var_params

        eps_I = jitter * jnp.eye(self.num_induc)[None, ...]

        z = params["induc_locs"]  # (f_dims, num_induc, in_dims)
        kern_hyp = params["kernel"]
        mean_f = params["mean"]  # (f_dims,)

        lambda_1, chol_Lambda_2 = var_params["lambda_1"], var_params["chol_Lambda_2"]

        Kzz = self.kernel.K(z, z, kern_hyp)
        Kzz = jnp.broadcast_to(Kzz, (self.f_dims, self.num_induc, self.num_induc))
        induc_cov = chol_Lambda_2 @ chol_Lambda_2.transpose(0, 2, 1)
        chol_R = cholesky(Kzz + induc_cov)  # (f_dims, num_induc, num_induc)

        ts, num_samps = x.shape[:2]
        K = lambda x, y: self.kernel.K(x, y, kern_hyp)

        Kxx = vmap(K, (2, 2), 1)(
            x[None, ...], x[None, ...]
        )  # (f_dims, num_samps, time, time)
        Kxx = jnp.broadcast_to(Kxx, (self.f_dims, num_samps, ts, ts))

        Kzx = vmap(K, (None, 2), 1)(
            z, x[None, ...]
        )  # (f_dims, num_samps, num_induc, time)
        Kzx = jnp.broadcast_to(Kzx, (self.f_dims, num_samps, self.num_induc, ts))
        Kxz = Kzx.transpose(0, 1, 3, 2)  # (f_dims, num_samps, time, num_induc)

        if mean_only is False or compute_KL:
            chol_Kzz = cholesky(Kzz + eps_I)  # (f_dims, num_induc, num_induc)
            Kxz_invKzz = cho_solve(
                (chol_Kzz[:, None, ...].repeat(num_samps, axis=1), True), Kzx
            ).transpose(0, 1, 3, 2)

        Rinv_lambda_1 = cho_solve((chol_R, True), lambda_1)  # (f_dims, num_induc, 1)
        post_means = (
            Kxz @ Rinv_lambda_1[:, None, ...].repeat(num_samps, axis=1)
            + mean_f[:, None, None, None]
        )  # (f_dims, num_samps, time, 1)

        if mean_only is False:
            invR_Kzx = cho_solve(
                (chol_R[:, None, ...].repeat(num_samps, axis=1), True), Kzx
            )  # (f_dims, num_samps, num_induc, time)
            post_covs = (
                Kxx - Kxz_invKzz @ Kzx + Kxz @ invR_Kzx
            )  # (f_dims, num_samps, time, time)
        else:
            post_covs = None

        if compute_KL:
            trace_term = jnp.trace(cho_solve((chol_R, True), Kzz)).sum()
            quadratic_form = (
                Rinv_lambda_1.transpose(0, 2, 1) @ Kzz @ Rinv_lambda_1
            ).sum()
            log_determinants = (jnp.log(vdiag(chol_R)) - jnp.log(vdiag(chol_Kzz))).sum()
            KL = 0.5 * (trace_term + quadratic_form - self.num_induc) + log_determinants
        else:
            KL = 0.0

        if compute_aux:
            aux = (Kxz, Kxz_invKzz, chol_R)  # squeeze shape
        else:
            aux = None

        return post_means, post_covs, KL, aux

    @partial(jit, static_argnums=(0,))
    def sample_prior(self, params, prng_state, x, jitter):
        """
        Prior distribution p(f(x)) = N(0, K_xx)
        Can use approx_points as number of points

        :param jnp.array x: shape (time, num_samps, f_dims, x_dims)
        :return:
            sample of shape (time, num_samps, f_dims)
        """
        params = self.params if params is None else params
        mean_f = params["mean"]  # (f_dims,)
        kern_hyp = params["kernel"]

        if x.ndim == 3:
            x = x[..., None, :]  # (time, num_samps, f_dims, x_dims)
        ts, num_samps = x.shape[:2]

        if self.RFF_num_feats > 0:  # random Fourier features
            prng_keys = jr.split(prng_state, 2)
            variance = softplus(kern_hyp["variance"])  # (f_dims,)
            ks = self.kernel.sample_spectrum(
                kern_hyp, prng_keys[0], num_samps, self.RFF_num_feats
            )  # (num_samps, f_dims, feats, x_dims)
            phi = (
                2
                * jnp.pi
                * jr.uniform(
                    prng_keys[1], shape=(num_samps, self.f_dims, self.RFF_num_feats)
                )
            )

            samples = mean_f[None, None, :] + jnp.sqrt(
                2.0 * variance / self.RFF_num_feats
            ) * (jnp.cos((ks[None, ...] * x[..., None, :]).sum(-1) + phi)).sum(
                -1
            )  # (time, num_samps, f_dims)

        else:
            K = lambda x: self.kernel.K(x, x, kern_hyp)
            Kxx = vmap(K, 1, 1)(
                x.transpose(2, 1, 0, 3)
            )  # (f_dims, num_samps, time, time)
            eps_I = jnp.broadcast_to(jitter * jnp.eye(ts), (self.f_dims, 1, ts, ts))
            cov = Kxx + eps_I
            mean = jnp.broadcast_to(
                mean_f[:, None, None, None], (self.f_dims, num_samps, ts, 1)
            )
            samples = sample_gaussian_noise(prng_state, mean, cov)[..., 0].transpose(
                2, 1, 0
            )

        return samples

    @partial(jit, static_argnums=(0, 6))
    def sample_posterior(self, params, var_params, prng_state, x, jitter, compute_KL):
        """
        Sample from posterior q(f|x)

        :param jnp.array x: input of shape (time, num_samps, x_dims)
        :return:
            sample of shape (time, num_samps, f_dims)
        """
        if self.RFF_num_feats > 0:
            post_means, _, KL, aux = self.evaluate_posterior(
                x, params, var_params, True, compute_KL, True, jitter
            )
            Kxz, Kxz_invKzz, chol_R = aux
            z = params["induc_locs"].transpose(1, 0, 2)  # (f_dims, num_induc, in_dims)

            prng_keys = jr.split(prng_state, 2)
            ts, num_samps = x.shape[:2]

            x_aug = jnp.concatenate(
                (
                    x[..., None, :].repeat(self.f_dims, axis=2),
                    z[:, None, ...].repeat(num_samps, axis=1),
                ),
                axis=0,
            )  # (nums, num_samps, f_dims, x_dims)
            print(x_aug.shape)
            prior_samps = self.sample_prior(params, prng_keys[0], x_aug, jitter)
            prior_samps_x, prior_samps_z = prior_samps[:ts, ...], prior_samps[ts:, ...]

            noise = solve_triangular(
                chol_R[:, None, ...].repeat(num_samps, axis=1),
                jr.normal(
                    prng_keys[1], shape=(self.f_dims, num_samps, self.num_induc, 1)
                ),
                lower=True,
            )

            prior_samps_z = prior_samps_z.transpose(2, 1, 0)[..., None]
            smoothed_samps = Kxz_invKzz @ prior_samps_z + Kxz @ noise
            smoothed_samps = smoothed_samps[..., 0].transpose(2, 1, 0)
            post_means = post_means[..., 0].transpose(2, 1, 0)

            # Matheron's rule pathwise samplig
            samples = prior_samps_x + post_means - smoothed_samps

        else:
            post_means, post_covs, KL, _ = self.evaluate_posterior(
                x, params, var_params, False, compute_KL, False, jitter
            )
            samples = sample_gaussian_noise(prng_state, post_means, post_covs)
            samples = samples[..., 0].transpose(2, 1, 0)

        return samples, KL
