import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from jax import vmap
from jax.numpy.linalg import cholesky
from jax.scipy.linalg import cho_solve, solve_triangular

from ..utils.jax import constrain_diagonal
from .base import GP

from .linalg import evaluate_qsparse_posterior, evaluate_tsparse_posterior


def t_q_svgp_moments(Kzz, mean, Lcov, jitter, t_to_q=True):
    """
    Get the posterior q(u) moments
    Get tSVGP variational parameters from the posterior q(u) moments

    :param jnp.ndarray u_Lcov: cholesky factor of covariance (out_dims, num_induc, num_induc)
    """
    if t_to_q:
        lambda_1, chol_Lambda_2 = mean, Lcov

        induc_cov = chol_Lambda_2 @ chol_Lambda_2.transpose(0, 2, 1)
        chol_R = cholesky(Kzz + induc_cov)  # (out_dims, num_induc, num_induc)
        Rinv_Kzz = cho_solve((chol_R, True), Kzz)
        Rinv_Kzz = cho_solve((chol_R, True), Kzz)

        u_mu = Rinv_Kzz @ lambda_1
        u_cov = Kzz @ Rinv_Kzz

        eps_I = jitter * jnp.eye(Kzz.shape[-1])
        u_Lcov = cholesky(u_cov + eps_I)  # (out_dims, num_induc, num_induc)
        return u_mu, u_Lcov

    else:
        u_mu, u_Lcov = mean, Lcov
        lambda_1 = Kzz @ cho_solve((u_Lcov, True), u_mu)

        eps_I = jitter * jnp.eye(Kzz.shape[-1])
        chol_Lambda_2 = cholesky(Kzz @ cho_solve((u_Lcov, True), Kzz) - Kzz + eps_I)
        return lambda_1, chol_Lambda_2


def w_q_svgp_moments(Kzz, mean, Lcov, jitter, w_to_q=True):
    """
    Get the posterior q(u) moments

    :param jnp.ndarray Lcov: (out_dims, num_induc, num_induc)
    """
    eps_I = jitter * jnp.eye(Kzz.shape[-1])
    chol_Kzz = cholesky(Kzz + eps_I)

    if w_to_q:
        v_mu, v_Lcov = mean, Lcov
        u_mu = chol_Kzz @ v_mu
        u_Lcov = chol_Kzz @ v_Lcov
        return u_mu, u_Lcov

    else:
        u_mu, u_Lcov = mean, Lcov
        v_mu = solve_triangular(chol_Kzz, u_mu, lower=True)
        v_Lcov = solve_triangular(chol_Kzz, u_Lcov, lower=True)
        return v_mu, v_Lcov


class SparseGP(GP):
    """
    GP with sparse approximation to the posterior
    """

    induc_locs: jnp.ndarray
    penalize_induc_proximity: bool

    def __init__(self, kernel, RFF_num_feats, induc_locs, penalize_induc_proximity):
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
        self.penalize_induc_proximity = penalize_induc_proximity

    def sample_posterior(self, prng_state, x, compute_KL, jitter, sel_outdims):
        """
        Sample from posterior q(f|x)

        :param jnp.array x: input of shape (num_samps, out_dims or 1, time, in_dims)
        :return:
            sample of shape (num_samps, out_dims, time)
        """
        post_means, post_covs, KL, _ = self.evaluate_posterior(
            x, False, False, compute_KL, False, jitter, sel_outdims
        )
        eps_I = jitter * jnp.eye(post_means.shape[-2])
        post_Lcovs = cholesky(post_covs + eps_I)

        samples = post_means + post_Lcovs @ jr.normal(
            prng_state, shape=post_means.shape
        )
        samples = samples[..., 0]

        return samples, KL

    def induc_proximity_cost(self):
        """
        Proximity cost for inducing point locations, to improve numerical stability
        """
        induc_locs = self.induc_locs
        num_induc = induc_locs.shape[1]

        repulsion_func = lambda x: jnp.maximum(1e-3 - jnp.abs(x).sum(), 0.0)
        dist_uu = vmap(vmap(vmap(repulsion_func)))(
            induc_locs[..., None, :] - induc_locs[:, None, ...]
        )
        repulsion = (
            dist_uu.at[:, jnp.arange(num_induc), jnp.arange(num_induc)].set(0.0).sum()
        )
        return repulsion


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
        penalize_induc_proximity=False,
    ):
        """
        :param np.ndarray induc_locs: inducing point locations z, array of shape (out_dims, num_induc, in_dims)
        :param np.ndarray u_mu: inducing means (out_dims, num_induc, 1)
        :param np.ndarray u_Lcov: inducing means (out_dims, num_induc, num_induc)
        """
        super().__init__(kernel, RFF_num_feats, induc_locs, penalize_induc_proximity)
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
        self, x, mean_only, diag_cov, compute_KL, compute_aux, jitter, sel_outdims=None
    ):
        """
        :param jnp.array x: input of shape (num_samps, out_dims or 1, time, in_dims)
        :returns:
            means of shape (num_samps, out_dims, time, 1)
            covariances of shape (num_samps, out_dims, time, time)
        """
        if sel_outdims is None:
            sel_outdims = jnp.arange(self.kernel.out_dims)

        Kzz = self.kernel.K(self.induc_locs, None, False, sel_outdims)
        Kzx = vmap(self.kernel.K, (None, 0, None, None), 2)(
            self.induc_locs, x, False, sel_outdims
        )  # (out_dims, num_induc, num_samps, time)

        if mean_only is False:
            Kxx = vmap(self.kernel.K, (0, None, None, None), 0)(
                x, None, diag_cov, sel_outdims
            )  # (num_samps, out_dims, time, time or 1)
        else:
            Kxx = None

        post_means, post_covs, KL, aux = evaluate_qsparse_posterior(
            Kxx,
            Kzx,
            Kzz,
            self.u_mu[sel_outdims],
            self.u_Lcov[sel_outdims],
            self.whitened,
            mean_only,
            diag_cov,
            compute_KL,
            compute_aux,
            jitter,
        )

        if self.penalize_induc_proximity:  # regularizer
            KL += self.induc_proximity_cost()

        return post_means, post_covs, KL, aux

    def sample_posterior(self, prng_state, x, compute_KL, jitter, sel_outdims=None):
        """
        Sample from posterior q(f|x)

        :param jnp.array x: input of shape (num_samps, out_dims or 1, time, in_dims)
        :return:
            sample of shape (num_samps, out_dims, time)
        """
        out_dims = self.kernel.out_dims
        if sel_outdims is None:
            sel_outdims = jnp.arange(out_dims)

        if self.RFF_num_feats > 0:
            num_samps, ts = x.shape[0], x.shape[2]
            in_dims = self.kernel.in_dims
            num_induc = self.induc_locs.shape[1]

            post_means, _, KL, aux = self.evaluate_posterior(
                x, True, False, compute_KL, True, jitter, sel_outdims
            )
            chol_Kzz, Kxz_invLzz = aux

            x_aug = jnp.concatenate(
                (
                    jnp.broadcast_to(x, (num_samps, out_dims, ts, in_dims)),
                    self.induc_locs[None, ...].repeat(num_samps, axis=0),
                ),
                axis=2,
            )[
                :, sel_outdims
            ]  # (num_samps, out_dims, num_locs, in_dims)

            prng_keys = jr.split(prng_state)

            prior_samps = self.sample_prior(
                prng_keys[0], x_aug, jitter, sel_outdims
            )  # (num_samps, out_dims, num_locs)
            prior_samps_x, prior_samps_z = prior_samps[..., :ts], prior_samps[..., ts:]

            prior_samps_z = prior_samps_z[..., None]
            if self.whitened:
                smoothed_samps = Kxz_invLzz @ (
                    solve_triangular(
                        chol_Kzz[None, ...].repeat(num_samps, axis=0),
                        prior_samps_z,
                        lower=True,
                    )
                    + self.u_Lcov[sel_outdims]
                    @ jr.normal(prng_keys[1], shape=prior_samps_z.shape)
                )

            else:
                smoothed_samps = Kxz_invLzz @ solve_triangular(
                    chol_Kzz[None, ...].repeat(num_samps, axis=0),
                    prior_samps_z
                    + self.u_Lcov[sel_outdims]
                    @ jr.normal(prng_keys[1], shape=prior_samps_z.shape),
                    lower=True,
                )

            smoothed_samps = smoothed_samps[..., 0]
            post_means = post_means[..., 0]

            # Matheron's rule pathwise samplig
            samples = prior_samps_x + post_means - smoothed_samps
            return samples, KL

        else:
            return super().sample_posterior(
                prng_state, x, compute_KL, jitter, sel_outdims
            )


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

    def __init__(
        self,
        kernel,
        induc_locs,
        lambda_1,
        chol_Lambda_2,
        RFF_num_feats=0,
        penalize_induc_proximity=False,
    ):
        """
        :param np.ndarray induc_locs: inducing point locations z, array of shape (out_dims, num_induc, in_dims)
        :param np.ndarray u_mu: inducing means (out_dims, num_induc, 1)
        :param np.ndarray u_Lcov: inducing means (out_dims, num_induc, num_induc)

        """
        super().__init__(kernel, RFF_num_feats, induc_locs, penalize_induc_proximity)
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
        self, x, mean_only, diag_cov, compute_KL, compute_aux, jitter, sel_outdims=None
    ):
        """
        :param jnp.array x: input of shape (num_samps, out_dims or 1, time, in_dims)
        :returns:
            means of shape (num_samps, out_dims, time, 1)
            covariances of shape (num_samps, out_dims, time, time)
        """
        if sel_outdims is None:
            sel_outdims = jnp.arange(self.kernel.out_dims)

        Kzz = self.kernel.K(self.induc_locs, None, False, sel_outdims)
        Kzx = vmap(self.kernel.K, (None, 0, None, None), 0)(
            self.induc_locs, x, False, sel_outdims
        )  # (num_samps, out_dims, num_induc, time)

        if mean_only is False:
            Kxx = vmap(self.kernel.K, (0, None, None, None), 0)(
                x, None, diag_cov, sel_outdims
            )  # (num_samps, out_dims, time, time or 1)
        else:
            Kxx = None

        post_means, post_covs, KL, aux = evaluate_tsparse_posterior(
            Kxx,
            Kzx,
            Kzz,
            self.lambda_1[sel_outdims],
            self.chol_Lambda_2[sel_outdims],
            mean_only,
            diag_cov,
            compute_KL,
            compute_aux,
            jitter,
        )

        if self.penalize_induc_proximity:  # regularizer
            KL += self.induc_proximity_cost()

        return post_means, post_covs, KL, aux

    def sample_posterior(self, prng_state, x, compute_KL, jitter, sel_outdims=None):
        """
        Sample from posterior q(f|x)

        :param jnp.array x: input of shape (num_samps, out_dims or 1, time, in_dims)
        :return:
            sample of shape (num_samps, out_dims, time)
        """
        out_dims = self.kernel.out_dims
        if sel_outdims is None:
            sel_outdims = jnp.arange(out_dims)

        if self.RFF_num_feats > 0:
            num_samps, ts = x.shape[0], x.shape[2]
            in_dims = self.kernel.in_dims
            out_dims = self.kernel.out_dims
            num_induc = self.induc_locs.shape[1]

            post_means, _, KL, aux = self.evaluate_posterior(
                x, True, False, compute_KL, True, jitter, sel_outdims
            )
            chol_Kzz, Kxz_invLzz, Kxz_invLR = aux

            x_aug = jnp.concatenate(
                (
                    jnp.broadcast_to(x, (num_samps, out_dims, ts, in_dims)),
                    self.induc_locs[None, ...].repeat(num_samps, axis=0),
                ),
                axis=2,
            )[
                :, sel_outdims
            ]  # (num_samps, out_dims, num_locs, in_dims)

            prng_keys = jr.split(prng_state)

            prior_samps = self.sample_prior(
                prng_keys[0], x_aug, jitter, sel_outdims
            )  # (num_samps, out_dims, num_locs)
            prior_samps_x, prior_samps_z = prior_samps[..., :ts], prior_samps[..., ts:]

            prior_samps_z = prior_samps_z[..., None]
            eps_I = jitter * jnp.eye(num_induc)
            smoothed_samps = Kxz_invLzz @ solve_triangular(
                chol_Kzz[None, ...].repeat(num_samps, axis=0),
                prior_samps_z,
                lower=True,
            ) + Kxz_invLR @ jr.normal(prng_keys[1], shape=prior_samps_z.shape)

            smoothed_samps = smoothed_samps[..., 0]
            post_means = post_means[..., 0]

            # Matheron's rule pathwise samplig
            samples = prior_samps_x + post_means - smoothed_samps
            return samples, KL

        else:
            return super().sample_posterior(
                prng_state, x, compute_KL, jitter, sel_outdims
            )
