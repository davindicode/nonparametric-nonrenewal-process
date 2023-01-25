import math
from functools import partial

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from jax import vmap
from jax.numpy.linalg import cholesky
from jax.scipy.linalg import cho_solve, solve_triangular

from ..utils.jax import constrain_diagonal
from .base import GP

from .linalg import evaluate_qsparse_posterior, evaluate_tsparse_posterior


def t_to_q_svgp_moments(Kzz, lambda_1, chol_Lambda_2, jitter):
    """
    Get the posterior q(u) moments
    """
    induc_cov = chol_Lambda_2 @ chol_Lambda_2.transpose(0, 2, 1)
    chol_R = cholesky(Kzz + induc_cov)  # (out_dims, num_induc, num_induc)
    Rinv_Kzz = cho_solve((chol_R, True), Kzz)

    u_mu = Rinv_Kzz @ lambda_1
    u_cov = Kzz @ Rinv_Kzz

    eps_I = jitter * jnp.eye(Kzz.shape[-1])
    u_Lcov = cholesky(u_cov + eps_I)  # (out_dims, num_induc, num_induc)
    return u_mu, u_Lcov


def white_to_q_svgp_moments(Kzz, v_mu, v_Lcov, jitter):
    """
    Get the posterior q(u) moments
    """
    eps_I = jitter * jnp.eye(Kzz.shape[-1])
    chol_Kzz = cholesky(Kzz + eps_I)  # (out_dims, num_induc, num_induc)

    u_mu = chol_Kzz @ v_mu
    u_Lcov = chol_Kzz @ v_Lcov
    return u_mu, u_Lcov


def t_from_q_svgp_moments(Kzz, u_mu, u_Lcov):
    """
    Get tSVGP variational parameters from the posterior q(u) moments

    :param jnp.ndarray u_Lcov: cholesky factor of covariance (out_dims, num_induc, num_induc)
    """
    lambda_1 = Kzz @ cho_solve((u_Lcov, True), u_mu)
    chol_Lambda_2 = Kzz @ cho_solve((u_Lcov, True), Kzz) - Kzz
    return lambda_1, chol_Lambda_2


class SparseGP(GP):
    """
    GP with sparse approximation to the posterior
    """

    induc_locs: jnp.ndarray

    def __init__(self, kernel, RFF_num_feats, induc_locs):
        if induc_locs.shape[2] != kernel.in_dims:
            raise ValueError(
                "Dimensions of inducing locations do not match kernel input dimensions"
            )
        if induc_locs.shape[0] != kernel.out_dims:
            raise ValueError(
                "Dimensions of inducing locations do not match kernel output dimensions"
            )
        super().__init__(kernel, RFF_num_feats)
        self.induc_locs = self._to_jax(induc_locs)  # (num_induc, out_dims, in_dims)

    def sample_posterior(self, prng_state, x, jitter, compute_KL):
        """
        Sample from posterior q(f|x)

        :param jnp.array x: input of shape (num_samps, out_dims or 1, time, in_dims)
        :return:
            sample of shape (num_samps, out_dims, time)
        """
        num_samps, ts = x.shape[0], x.shape[2]
        in_dims = self.kernel.in_dims
        out_dims = self.kernel.out_dims
        num_induc = self.induc_locs.shape[1]

        if self.RFF_num_feats > 0:
            post_means, _, KL, aux = self.evaluate_posterior(
                x, True, False, compute_KL, True, jitter
            )
            chol_Kzz, Kxz_invLzz = aux

            prng_keys = jr.split(prng_state, 2)

            x_aug = jnp.concatenate(
                (
                    jnp.broadcast_to(x, (num_samps, out_dims, ts, in_dims)),
                    self.induc_locs[None, ...].repeat(num_samps, axis=0),
                ),
                axis=2,
            )  # (num_samps, out_dims, num_locs, in_dims)

            prior_samps = self.sample_prior(
                prng_keys[0], x_aug, jitter
            )  # (num_samps, out_dims, num_locs)
            prior_samps_x, prior_samps_z = prior_samps[..., :ts], prior_samps[..., ts:]

            prior_samps_z = prior_samps_z[..., None]
            smoothed_samps = Kxz_invLzz @ solve_triangular(
                chol_Kzz[None, ...].repeat(num_samps, axis=0),
                prior_samps_z,
                lower=True,
            )
            smoothed_samps = smoothed_samps[..., 0]
            post_means = post_means[..., 0]

            # Matheron's rule pathwise samplig
            samples = prior_samps_x + post_means - smoothed_samps

        else:
            post_means, post_covs, KL, _ = self.evaluate_posterior(
                x, False, False, compute_KL, False, jitter
            )
            eps_I = jitter * jnp.eye(post_means.shape[-2])
            post_Lcovs = cholesky(post_covs + eps_I)

            samples = post_means + post_Lcovs @ jr.normal(
                prng_state, shape=post_means.shape
            )
            samples = samples[..., 0]

        return samples, KL


class qSVGP(SparseGP):
    """
    References:
    1. Gaussian Processes for Big Data, James Hensman, Nicolo Fusi, Neil D. Lawrence
    """

    # variational parameters
    u_mu: jnp.ndarray
    u_Lcov: jnp.ndarray
    whitened: bool

    def __init__(
        self,
        kernel,
        induc_locs,
        u_mu,
        u_Lcov,
        RFF_num_feats=0,
        whitened=False,
    ):
        """
        :param induc_locs: inducing point locations z, array of shape (out_dims, num_induc, in_dims)
        :param variance: The observation noise variance, σ²
        """
        super().__init__(kernel, RFF_num_feats, induc_locs)
        self.u_mu = self._to_jax(u_mu)
        self.u_Lcov = self._to_jax(u_Lcov)
        self.whitened = whitened

    def apply_constraints(self):
        """
        PSD constraint
        """
        model = super().apply_constraints()

        def update(Lcov):
            Lcov = constrain_diagonal(jnp.tril(Lcov), lower_lim=1e-8)
            # Lcov = jnp.tril(Lcov) if self.spatial_MF else Lcov
            return Lcov

        model = eqx.tree_at(
            lambda tree: tree.u_Lcov,
            model,
            replace_fn=vmap(update),
        )

        return model

    def evaluate_posterior(
        self, x, mean_only, diag_cov, compute_KL, compute_aux, jitter
    ):
        """
        :param jnp.array x: input of shape (num_samps, out_dims or 1, time, in_dims)
        :returns:
            means of shape (num_samps, out_dims, time, 1)
            covariances of shape (num_samps, out_dims, time, time)
        """
        post_means, post_covs, KL, aux = evaluate_qsparse_posterior(
            self.kernel,
            self.induc_locs,
            x,
            self.u_mu,
            self.u_Lcov,
            self.whitened,
            mean_only,
            diag_cov,
            compute_KL,
            compute_aux,
            jitter,
        )

        return post_means, post_covs, KL, aux


class tSVGP(SparseGP):
    """
    Sparse variational Gaussian process

    Variational parameters are (u_mean, u_cov)
    site_mean has shape (out_dims, num_induc, 1)
    site_mean has shape (out_dims, num_induc, num_induc)

    Note q(f) factorizes along out_dims, not optimal for heteroscedastic likelihoods

    References:
        [1] site-based Sparse Variational Gaussian Processes
    """

    # variational parameters
    lambda_1: jnp.ndarray
    chol_Lambda_2: jnp.ndarray

    def __init__(self, kernel, induc_locs, lambda_1, chol_Lambda_2, RFF_num_feats=0):
        """
        :param induc_locs: inducing point locations z, array of shape (out_dims, num_induc, in_dims)
        :param variance: The observation noise variance, σ²
        """
        super().__init__(kernel, RFF_num_feats, induc_locs)
        self.lambda_1 = self._to_jax(lambda_1)
        self.chol_Lambda_2 = self._to_jax(chol_Lambda_2)

    def apply_constraints(self):
        """
        PSD constraint
        """
        model = super().apply_constraints()

        def update(Lcov):
            Lcov = constrain_diagonal(jnp.tril(Lcov), lower_lim=1e-2)
            return Lcov

        model = eqx.tree_at(
            lambda tree: tree.chol_Lambda_2,
            model,
            replace_fn=vmap(update),
        )

        return model

    def evaluate_posterior(
        self, x, mean_only, diag_cov, compute_KL, compute_aux, jitter
    ):
        """
        :param jnp.array x: input of shape (num_samps, out_dims or 1, time, in_dims)
        :returns:
            means of shape (num_samps, out_dims, time, 1)
            covariances of shape (num_samps, out_dims, time, time)
        """
        post_means, post_covs, KL, aux = evaluate_tsparse_posterior(
            self.kernel,
            self.induc_locs,
            x,
            self.lambda_1,
            self.chol_Lambda_2,
            mean_only,
            diag_cov,
            compute_KL,
            compute_aux,
            jitter,
        )

        return post_means, post_covs, KL, aux


#     def sample_posterior(self, prng_state, x, jitter, compute_KL):
#         """
#         Sample from posterior q(f|x)

#         :param jnp.array x: input of shape (num_samps, out_dims or 1, time, in_dims)
#         :return:
#             sample of shape (time, num_samps, out_dims)
#         """
#         num_samps, ts = x.shape[0], x.shape[2]
#         in_dims = self.kernel.in_dims
#         out_dims = self.kernel.out_dims
#         num_induc = self.induc_locs.shape[1]

#         if self.RFF_num_feats > 0:
#             post_means, _, KL, aux = self.evaluate_posterior(
#                 x, True, False, compute_KL, True, jitter
#             )
#             Kxz_Rinv, = aux

#             prng_keys = jr.split(prng_state, 2)

#             x_aug = jnp.concatenate(
#                 (
#                     jnp.broadcast_to(x, (num_samps, out_dims, ts, in_dims)),
#                     self.induc_locs[None, ...].repeat(num_samps, axis=0),
#                 ),
#                 axis=2,
#             )  # (num_samps, out_dims, num_locs, in_dims)

#             prior_samps = self.sample_prior(prng_keys[0], x_aug, jitter)  # (num_samps, out_dims, num_locs)
#             prior_samps_x, prior_samps_z = prior_samps[..., :ts], prior_samps[..., ts:]

#             Ln = self.chol_Lambda_2[None, ...].repeat(num_samps, axis=0)
#             noise = solve_triangular(
#                 Ln, jr.normal(prng_keys[1], shape=(num_samps, out_dims, num_induc, 1)), lower=True)

#             prior_samps_z = prior_samps_z[..., None]
#             smoothed_samps = Kxz_Rinv @ (prior_samps_z + noise)
#             smoothed_samps = smoothed_samps[..., 0]
#             post_means = post_means[..., 0]

#             # Matheron's rule pathwise samplig
#             samples = prior_samps_x + post_means - smoothed_samps

#         else:
#             post_means, post_covs, KL, _ = self.evaluate_posterior(
#                 x, False, False, compute_KL, False, jitter
#             )
#             eps_I = jitter * jnp.eye(post_means.shape[-2])
#             post_Lcovs = cholesky(post_covs + eps_I)

#             samples = post_means + post_Lcovs @ jr.normal(prng_state, shape=post_means.shape)
#             samples = samples[..., 0]

#         return samples, KL
