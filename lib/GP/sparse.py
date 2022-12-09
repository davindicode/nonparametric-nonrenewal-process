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



class tSVGP(module):
    """
    Sparse variational Gaussian process

    Variational parameters are (u_mean, u_cov)
    site_mean has shape (out_dims, num_induc, 1)
    site_mean has shape (out_dims, num_induc, num_induc)

    Note q(f) factorizes along out_dims, not optimal for heteroscedastic likelihoods

    References:
        [1] site-based Sparse Variational Gaussian Processes
    """
    # hyperparameters
    induc_locs: jnp.ndarray
    mean: jnp.ndarray
        
    # variational parameters
    lambda_1: jnp.ndarray
    chol_Lambda_2: jnp.ndarray
        
    kernel: module
    RFF_num_feats: int

    def __init__(self, kernel, mean_f, induc_locs, lambda_1, chol_Lambda_2, RFF_num_feats=0):
        """
        :param induc_locs: inducing point locations z, array of shape (out_dims, num_induc, in_dims)
        :param variance: The observation noise variance, σ²
        """
        if induc_locs.shape[2] != kernel.in_dims:
            raise ValueError(
                "Dimensions of inducing locations do not match kernel input dimensions"
            )
        if induc_locs.shape[0] != kernel.out_dims:
            raise ValueError(
                "Dimensions of inducing locations do not match kernel output dimensions"
            )

        super().__init__(kernel.in_dims, kernel.out_dims)
        self.induc_locs = induc_locs  # (num_induc, out_dims, in_dims)
        self.mean = mean  # (out_dims,)

        self.lambda_1 = lambda_1
        self.chol_Lambda_2 = chol_Lambda_2
        
        self.kernel = kernel
        self.RFF_num_feats = RFF_num_feats  # use random Fourier features

    def apply_constraints(self):
        """
        PSD constraint
        """
        model = jax.tree_map(lambda p: p, self)  # copy
        
        def update(Lcov):
            epdfunc = lambda x: enforce_positive_diagonal(x, lower_lim=1e-2)
            Lcov = vmap(epdfunc)(jnp.tril(Lcov))
            Lcov = jnp.tril(Lcov)
            return Lcov

        model = eqx.tree_at(
            lambda tree: tree.chol_Lambda_2,
            model,
            replace_fn=update,
        )
        
        kernel = self.kernel.apply_constraints(self.kernel)
        model = eqx.tree_at(
            lambda tree: tree.kernel,
            model,
            replace_fn=lambda _: kernel,
        )

        return model

    def evaluate_posterior(
        self, x, mean_only, compute_KL, compute_aux, jitter
    ):
        """
        :param jnp.array x: input of shape (time, num_samps, in_dims, 1)
        :returns:
            means of shape (out_dims, num_samps, time, 1)
            covariances of shape (out_dims, num_samps, time, time)
        """
        num_induc = induc_locs.shape[1]
        eps_I = jitter * jnp.eye(num_induc)[None, ...]

        Kzz = self.kernel.K(z, z)
        Kzz = jnp.broadcast_to(Kzz, (self.out_dims, num_induc, num_induc))
        induc_cov = chol_Lambda_2 @ chol_Lambda_2.transpose(0, 2, 1)
        chol_R = cholesky(Kzz + induc_cov)  # (out_dims, num_induc, num_induc)

        ts, num_samps = x.shape[:2]
        K = lambda x, y: self.kernel.K(x, y)

        Kxx = vmap(K, (2, 2), 1)(
            x[None, ...], x[None, ...]
        )  # (out_dims, num_samps, time, time)
        Kxx = jnp.broadcast_to(Kxx, (self.out_dims, num_samps, ts, ts))

        Kzx = vmap(K, (None, 2), 1)(
            z, x[None, ...]
        )  # (out_dims, num_samps, num_induc, time)
        Kzx = jnp.broadcast_to(Kzx, (self.out_dims, num_samps, num_induc, ts))
        Kxz = Kzx.transpose(0, 1, 3, 2)  # (out_dims, num_samps, time, num_induc)

        if mean_only is False or compute_KL:
            chol_Kzz = cholesky(Kzz + eps_I)  # (out_dims, num_induc, num_induc)
            Kxz_invKzz = cho_solve(
                (chol_Kzz[:, None, ...].repeat(num_samps, axis=1), True), Kzx
            ).transpose(0, 1, 3, 2)

        Rinv_lambda_1 = cho_solve((chol_R, True), lambda_1)  # (out_dims, num_induc, 1)
        post_means = (
            Kxz @ Rinv_lambda_1[:, None, ...].repeat(num_samps, axis=1)
            + mean_f[:, None, None, None]
        )  # (out_dims, num_samps, time, 1)

        if mean_only is False:
            invR_Kzx = cho_solve(
                (chol_R[:, None, ...].repeat(num_samps, axis=1), True), Kzx
            )  # (out_dims, num_samps, num_induc, time)
            post_covs = (
                Kxx - Kxz_invKzz @ Kzx + Kxz @ invR_Kzx
            )  # (out_dims, num_samps, time, time)
        else:
            post_covs = None

        if compute_KL:
            trace_term = jnp.trace(cho_solve((chol_R, True), Kzz)).sum()
            quadratic_form = (
                Rinv_lambda_1.transpose(0, 2, 1) @ Kzz @ Rinv_lambda_1
            ).sum()
            log_determinants = (jnp.log(vdiag(chol_R)) - jnp.log(vdiag(chol_Kzz))).sum()
            KL = 0.5 * (trace_term + quadratic_form - num_induc) + log_determinants
        else:
            KL = 0.0

        if compute_aux:
            aux = (Kxz, Kxz_invKzz, chol_R)  # squeeze shape
        else:
            aux = None

        return post_means, post_covs, KL, aux

    def sample_prior(self, prng_state, x, jitter):
        """
        Prior distribution p(f(x)) = N(0, K_xx)
        Can use approx_points as number of points

        :param jnp.array x: shape (time, num_samps, out_dims, in_dims)
        :return:
            sample of shape (time, num_samps, out_dims)
        """
        if x.ndim == 3:
            x = x[..., None, :]  # (time, num_samps, out_dims, in_dims)
        ts, num_samps = x.shape[:2]

        if self.RFF_num_feats > 0:  # random Fourier features
            prng_keys = jr.split(prng_state, 2)
            variance = softplus(kern_hyp["variance"])  # (out_dims,)
            ks = self.kernel.sample_spectrum(
                kern_hyp, prng_keys[0], num_samps, self.RFF_num_feats
            )  # (num_samps, out_dims, feats, in_dims)
            phi = (
                2
                * jnp.pi
                * jr.uniform(
                    prng_keys[1], shape=(num_samps, self.out_dims, self.RFF_num_feats)
                )
            )

            samples = mean_f[None, None, :] + jnp.sqrt(
                2.0 * variance / self.RFF_num_feats
            ) * (jnp.cos((ks[None, ...] * x[..., None, :]).sum(-1) + phi)).sum(
                -1
            )  # (time, num_samps, out_dims)

        else:
            K = lambda x: self.kernel.K(x, x, kern_hyp)
            Kxx = vmap(K, 1, 1)(
                x.transpose(2, 1, 0, 3)
            )  # (out_dims, num_samps, time, time)
            eps_I = jnp.broadcast_to(jitter * jnp.eye(ts), (self.out_dims, 1, ts, ts))
            cov = Kxx + eps_I
            mean = jnp.broadcast_to(
                mean_f[:, None, None, None], (self.out_dims, num_samps, ts, 1)
            )
            samples = sample_gaussian_noise(prng_state, mean, cov)[..., 0].transpose(
                2, 1, 0
            )

        return samples

    def sample_posterior(self, prng_state, x, jitter, compute_KL):
        """
        Sample from posterior q(f|x)

        :param jnp.array x: input of shape (time, num_samps, in_dims)
        :return:
            sample of shape (time, num_samps, out_dims)
        """
        if self.RFF_num_feats > 0:
            post_means, _, KL, aux = self.evaluate_posterior(
                x, params, var_params, True, compute_KL, True, jitter
            )
            Kxz, Kxz_invKzz, chol_R = aux
            z = params["induc_locs"].transpose(1, 0, 2)  # (out_dims, num_induc, in_dims)

            prng_keys = jr.split(prng_state, 2)
            ts, num_samps = x.shape[:2]

            x_aug = jnp.concatenate(
                (
                    x[..., None, :].repeat(self.out_dims, axis=2),
                    z[:, None, ...].repeat(num_samps, axis=1),
                ),
                axis=0,
            )  # (nums, num_samps, out_dims, in_dims)
            print(x_aug.shape)
            prior_samps = self.sample_prior(params, prng_keys[0], x_aug, jitter)
            prior_samps_x, prior_samps_z = prior_samps[:ts, ...], prior_samps[ts:, ...]

            noise = solve_triangular(
                chol_R[:, None, ...].repeat(num_samps, axis=1),
                jr.normal(
                    prng_keys[1], shape=(self.out_dims, num_samps, self.num_induc, 1)
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
