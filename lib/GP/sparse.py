import math
from functools import partial

from jax import vmap
import jax.numpy as jnp
import jax.random as jr
from jax.numpy.linalg import cholesky
from jax.scipy.linalg import cho_solve, solve_triangular

from .linalg import evaluate_qsparse_posterior, evaluate_tsparse_posterior
from ..utils.jax import sample_gaussian_noise
from .base import GP



class qSVGP(GP):
    """
    References:
    1. Gaussian Processes for Big Data, James Hensman, Nicolo Fusi, Neil D. Lawrence
    """
    induc_locs: jnp.ndarray
        
    # variational parameters
    u_mu: jnp.ndarray
    u_Lcov: jnp.ndarray
    whitened: bool

    def __init__(self, kernel, mean, induc_locs, u_mu, u_Lcov, RFF_num_feats=0, whitened=False):
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

        super().__init__(kernel, mean, RFF_num_feats)
        self.induc_locs = induc_locs  # (num_induc, out_dims, in_dims)
        
        self.u_mu = u_mu
        self.u_Lcov = u_Lcov
        self.whitened = whitened
        
    def apply_constraints(self):
        """
        PSD constraint
        """
        model = super().apply_constraints()
        
        def update(Lcov):
            epdfunc = lambda x: enforce_positive_diagonal(x, lower_lim=1e-2)
            Lcov = vmap(epdfunc)(jnp.tril(Lcov))
            Lcov = jnp.tril(Lcov)
            return Lcov

        model = eqx.tree_at(
            lambda tree: tree.u_Lcov,
            model,
            replace_fn=update,
        )

        return model
    
    def evaluate_posterior(
        self, x, mean_only, diag_cov, compute_KL, compute_aux, jitter
    ):
        """
        :param jnp.array x: input of shape (time, num_samps, in_dims)
        :returns:
            means of shape (out_dims, num_samps, time, 1)
            covariances of shape (out_dims, num_samps, time, time)
        """
        post_means, post_covs, KL, aux = evaluate_qsparse_posterior(
            self.kernel, self.induc_locs, self.mean, x, 
            self.u_mu, self.u_Lcov, self.whitened, 
            mean_only, diag_cov, compute_KL, compute_aux, jitter
        )

        return post_means, post_covs, KL, aux
    
    def sample_posterior(self, prng_state, x, jitter, compute_KL):
        """
        Sample from posterior q(f|x)

        :param jnp.array x: input of shape (time, num_samps, in_dims)
        :return:
            sample of shape (time, num_samps, out_dims)
        """
        in_dims = self.kernel.in_dims
        out_dims = self.kernel.out_dims
        num_induc = self.induc_locs.shape[1]
        
        if self.RFF_num_feats > 0:
            post_means, _, KL, aux = self.evaluate_posterior(
                x, True, False, compute_KL, True, jitter
            )
            chol_Kzz, Kxz_invLzz = aux
            z = self.induc_locs.transpose(1, 0, 2)  # (out_dims, num_induc, in_dims)

            prng_keys = jr.split(prng_state, 2)
            ts, num_samps = x.shape[:2]

            x_aug = jnp.concatenate(
                (
                    x[..., None, :].repeat(out_dims, axis=2),
                    z[:, None, ...].repeat(num_samps, axis=1),
                ),
                axis=0,
            )  # (nums, num_samps, out_dims, in_dims)
            
            prior_samps = self.sample_prior(prng_keys[0], x_aug, jitter)
            prior_samps_x, prior_samps_z = prior_samps[:ts, ...], prior_samps[ts:, ...]

            prior_samps_z = prior_samps_z.transpose(2, 1, 0)[..., None]
            smoothed_samps = Kxz_invLzz @ solve_triangular(
                chol_Kzz[:, None, ...].repeat(num_samps, axis=1),
                prior_samps_z, 
                lower=True,
            )
            smoothed_samps = smoothed_samps[..., 0].transpose(2, 1, 0)
            post_means = post_means[..., 0].transpose(2, 1, 0)

            # Matheron's rule pathwise samplig
            samples = prior_samps_x + post_means - smoothed_samps

        else:
            post_means, post_covs, KL, _ = self.evaluate_posterior(
                x, False, False, compute_KL, False, jitter
            )
            eps_I = jitter * jnp.eye(post_means.shape[-2])[None, None, ...]
            post_Lcovs = cholesky(post_covs + eps_I)
            
            samples = sample_gaussian_noise(prng_state, post_means, post_Lcovs)
            samples = samples[..., 0].transpose(2, 1, 0)

        return samples, KL
        



class tSVGP(GP):
    """
    Sparse variational Gaussian process

    Variational parameters are (u_mean, u_cov)
    site_mean has shape (out_dims, num_induc, 1)
    site_mean has shape (out_dims, num_induc, num_induc)

    Note q(f) factorizes along out_dims, not optimal for heteroscedastic likelihoods

    References:
        [1] site-based Sparse Variational Gaussian Processes
    """
    induc_locs: jnp.ndarray
        
    # variational parameters
    lambda_1: jnp.ndarray
    chol_Lambda_2: jnp.ndarray

    def __init__(self, kernel, mean, induc_locs, lambda_1, chol_Lambda_2, RFF_num_feats=0):
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

        super().__init__(kernel, mean, RFF_num_feats)
        self.induc_locs = induc_locs  # (num_induc, out_dims, in_dims)

        self.lambda_1 = lambda_1
        self.chol_Lambda_2 = chol_Lambda_2

    def apply_constraints(self):
        """
        PSD constraint
        """
        model = super().apply_constraints()
        
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

        return model
    
    def q_u_moments(self):
        """
        Get the posterior q(u) moments
        """
        Kzz = self.kernel.K(self.induc_locs, None, False)
        induc_cov = self.chol_Lambda_2 @ self.chol_Lambda_2.transpose(0, 2, 1)
        chol_R = cholesky(Kzz + induc_cov)  # (out_dims, num_induc, num_induc)
        Rinv_Kzz = cho_solve((chol_R, True), Kzz)
    
        induc_mu = Rinv_Kzz @ self.lambda_1
        induc_cov = Kzz @ Rinv_Kzz
        return induc_mu, induc_cov
    
    def evaluate_posterior(
        self, x, mean_only, diag_cov, compute_KL, compute_aux, jitter
    ):
        """
        :param jnp.array x: input of shape (time, num_samps, in_dims)
        :returns:
            means of shape (out_dims, num_samps, time, 1)
            covariances of shape (out_dims, num_samps, time, time)
        """
        post_means, post_covs, KL, aux = evaluate_tsparse_posterior(
            self.kernel, self.induc_locs, self.mean, x, 
            self.lambda_1, self.chol_Lambda_2, 
            mean_only, diag_cov, compute_KL, compute_aux, jitter
        )

        return post_means, post_covs, KL, aux

    def sample_posterior(self, prng_state, x, jitter, compute_KL):
        """
        Sample from posterior q(f|x)

        :param jnp.array x: input of shape (time, num_samps, in_dims)
        :return:
            sample of shape (time, num_samps, out_dims)
        """
        in_dims = self.kernel.in_dims
        out_dims = self.kernel.out_dims
        num_induc = self.induc_locs.shape[1]
        
        if self.RFF_num_feats > 0:
            post_means, _, KL, aux = self.evaluate_posterior(
                x, True, False, compute_KL, True, jitter
            )
            Kxz_Rinv, = aux
            z = self.induc_locs.transpose(1, 0, 2)  # (out_dims, num_induc, in_dims)

            prng_keys = jr.split(prng_state, 2)
            ts, num_samps = x.shape[:2]

            x_aug = jnp.concatenate(
                (
                    x[..., None, :].repeat(out_dims, axis=2),
                    z[:, None, ...].repeat(num_samps, axis=1),
                ),
                axis=0,
            )  # (nums, num_samps, out_dims, in_dims)
            
            prior_samps = self.sample_prior(prng_keys[0], x_aug, jitter)
            prior_samps_x, prior_samps_z = prior_samps[:ts, ...], prior_samps[ts:, ...]

            Ln = self.chol_Lambda_2[:, None, ...].repeat(num_samps, axis=1)
            noise = jr.normal(prng_keys[1], shape=(out_dims, num_samps, num_induc, 1))

            prior_samps_z = prior_samps_z.transpose(2, 1, 0)[..., None]
            smoothed_samps = Kxz_Rinv @ (prior_samps_z + Ln @ noise)
            smoothed_samps = smoothed_samps[..., 0].transpose(2, 1, 0)
            post_means = post_means[..., 0].transpose(2, 1, 0)

            # Matheron's rule pathwise samplig
            samples = prior_samps_x + post_means - smoothed_samps

        else:
            post_means, post_covs, KL, _ = self.evaluate_posterior(
                x, False, False, compute_KL, False, jitter
            )
            eps_I = jitter * jnp.eye(post_means.shape[-2])[None, None, ...]
            post_Lcovs = cholesky(post_covs + eps_I)
            
            samples = sample_gaussian_noise(prng_state, post_means, post_Lcovs)
            samples = samples[..., 0].transpose(2, 1, 0)

        return samples, KL
