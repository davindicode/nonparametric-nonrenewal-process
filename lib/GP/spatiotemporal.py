import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from jax import lax, vmap
from jax.numpy.linalg import cholesky

from ..utils.jax import constrain_diagonal
from .base import SSM
from .kernels import MarkovSparseKronecker

from .linalg import evaluate_LTI_posterior
from .markovian import interpolation_times, order_times, vmap_outdims


@eqx.filter_vmap(
    args=(None, None, None, None, -3, -3, None, None, None, None, None, None),
    out=(0, 0, 0),
)
def vmap_spatial(
    H,
    Pinf,
    As,
    Qs,
    site_obs,
    site_Lcov,
    ind_eval,
    A_fwd,
    A_bwd,
    mean_only,
    compute_KL,
    jitter,
):
    return vmap_outdims(
        H,
        Pinf,
        As,
        Qs,
        site_obs,
        site_Lcov,
        ind_eval,
        A_fwd,
        A_bwd,
        mean_only,
        compute_KL,
        jitter,
    )


class KroneckerLTI(SSM):
    """
    Factorized spatial and temporal GP kernel with temporal markov kernel

    We use the Kronecker kernel, with out_dims of the temporal kernel
    equal to out_dims of process
    """

    markov_sparse_kernel: MarkovSparseKronecker

    spatial_MF: bool
    fixed_grid_locs: bool

    def __init__(
        self,
        spatiotemporal_kernel,
        site_locs,
        site_obs,
        site_Lcov,
        spatial_MF=True,
        fixed_grid_locs=False,
    ):
        """
        :param module temporal_kernel: markov kernel module
        :param module spatial_kernel: spatial kernel
        :param jnp.ndarray site_locs: site locations (out_dims, timesteps)
        :param jnp.ndarray site_obs: site observations with shape (out_dims, timesteps, spatial_locs, 1)
        :param jnp.ndarray site_Lcov: site observations with shape (out_dims, timesteps, spatial_locs, spatial_locs or 1)
        """
        if spatial_MF:
            assert site_Lcov.shape[-1] == 1
        assert site_obs.shape[-2] == site_Lcov.shape[-2]  # spatial_locs
        super().__init__(
            site_locs, site_obs, site_Lcov, spatiotemporal_kernel.array_type
        )
        self.markov_sparse_kernel = spatiotemporal_kernel
        self.spatial_MF = spatial_MF
        self.fixed_grid_locs = fixed_grid_locs

    def apply_constraints(self):
        """
        PSD constraint
        """
        model = jax.tree_map(lambda p: p, self)  # copy

        def update(Lcov):
            epdfunc = lambda x: constrain_diagonal(x, lower_lim=1e-2)
            Lcov = vmap(epdfunc)(jnp.tril(Lcov))
            Lcov = jnp.triu(Lcov) if self.spatial_MF else Lcov
            return Lcov

        model = eqx.tree_at(
            lambda tree: tree.site_Lcov,
            model,
            replace_fn=update,
        )

        kernel = self.markov_sparse_kernel.apply_constraints(self.markov_sparse_kernel)
        model = eqx.tree_at(
            lambda tree: tree.markov_sparse_kernel,
            model,
            replace_fn=lambda _: kernel,
        )

        return model

    def get_site_locs(self):
        if self.fixed_grid_locs:
            locs = lax.stop_gradient(self.site_locs)  # (out_dims, ts)
            return locs, locs[:, 1:2] - locs[:, 0:1]
        else:
            return self.site_locs, jnp.diff(self.site_locs, axis=1)

    ### posterior ###
    def evaluate_posterior(
        self,
        t_eval,
        x_eval,
        mean_only,
        compute_KL,
        jitter,
    ):
        """
        predict at test locations X, which may includes training points
        (which are essentially fixed inducing points)

        :param jnp.ndarray t_eval: evaluation times of shape (out_dims or 1, locs)
        :param jnp.ndarray x_eval: evaluation times of shape (out_dims or 1, locs, x_dims)
        :return:
            means of shape (out_dims, time, 1)
            covariances of shape (out_dims, time, time)
        """
        num_evals, x_dims = x_eval.shape[1:]

        site_locs, site_dlocs = self.get_site_locs()
        t_eval = jnp.broadcast_to(t_eval, (site_locs.shape[0], num_evals))

        # evaluation locations
        ind_eval, dt_fwd, dt_bwd = vmap(interpolation_times, (0, 0), (1, 1, 1))(
            t_eval, site_locs
        )  # vmap over out_dims
        stack_dt = jnp.concatenate([dt_fwd, dt_bwd], axis=0)  # (2*num_evals, out_dims)

        if self.spatial_MF:  # vmap over temporal kernel out_dims = spatial_locs
            stack_A = vmap(self.markov_sparse_kernel.markov_factor._state_transition)(
                stack_dt
            )  # vmap over num_evals, (eval_inds, out_dims, sd, sd)

            # compute LDS matrices
            H, Pinf, As, Qs = self.markov_sparse_kernel._get_LDS(
                site_dlocs, site_locs.shape[1]
            )
            # (ts, out_dims, sd, sd)

            # vmap over spatial points
            post_means, post_covs, KL = vmap_spatial(
                H,
                Pinf,
                As,
                Qs,
                self.site_obs[..., None],
                self.site_Lcov[..., None],
                interp_sites,
                mean_only,
                compute_KL,
                jitter,
            )  # (spatial_locs, out_dims, timesteps, 1)

            post_means = post_means.transpose(1, 2, 0, 3)
            post_covs = post_covs.transpose(1, 2, 0, 3)

        else:
            stack_A = vmap(self.markov_sparse_kernel.state_transition)(
                stack_dt
            )  # vmap over num_evals
            A_fwd, A_bwd = stack_A[:num_evals], stack_A[-num_evals:]
            # (eval_inds, out_dims, spatial_pts*sd, spatial_pts*sd)

            H, Pinf, As, Qs = self.markov_sparse_kernel.get_LDS(
                site_dlocs, site_locs.shape[1]
            )
            # (ts, out_dims, spatial_pts*sd, spatial_pts*sd)

            post_means_, post_covs_, KL = vmap_outdims(
                H,
                Pinf,
                As,
                Qs,
                self.site_obs,
                self.site_Lcov,
                ind_eval,
                A_fwd,
                A_bwd,
                mean_only,
                compute_KL,
                jitter,
            )  # (out_dims, timesteps, spatial_locs, 1 or spatial_locs)

        C_krr, C_nystrom = self.markov_sparse_kernel.sparse_conditional(x_eval, jitter)
        Kmarkov = self.markov_sparse_kernel.markov_factor.K(
            t_eval, None, True
        )  # (out_dims, ts, 1)

        post_means = (C_krr * post_means_[..., 0]).sum(
            -1, keepdims=True
        )  # (out_dims, timesteps, 1)
        W = C_krr[..., None, :]  # (out_dims, timesteps, 1, spatial_locs)
        post_covs = (W @ post_covs_ @ W.transpose(0, 1, 3, 2))[
            ..., 0
        ] + Kmarkov * C_nystrom
        return post_means, post_covs, KL.sum()  # sum over out_dims

    ### sample ###
    def sample_prior(self, prng_state, t_eval, x_eval, jitter):
        """
        Sample from the model prior f~N(0,K) multiple times using a nested loop.
        :param num_samps: number of samples to draw
        :param jnp.ndarray tx_eval: input locations at which to sample (out_dims, locs)
        :return:
            f_sample: the prior samples (num_samps, out_dims, locs)
        """
        state_dims = self.markov_sparse_kernel.state_dims

        # evaluation locations
        t_all, site_ind, eval_ind = vmap(order_times, (None, 0), (0, 0, 0))(
            t_eval, site_locs
        )

        eps_I = jitter * jnp.eye(state_dims)

        # sample independent temporal processes

        # mix temporal trajectories

        # transition and noise process matrices
        H, Pinf, As, Qs = self.markov_sparse_kernel.get_LDS(dt, tsteps)

        prng_states = jr.split(prng_state, num_samps)  # (num_samps, 2)

        def step(carry, inputs):
            m = carry
            A, Q, prng_state = inputs
            L = cholesky(Q + eps_I)  # can be a bit unstable, lower=True

            q_samp = L @ jr.normal(prng_state, shape=(state_dims, 1))
            m = A @ m + q_samp
            f = H @ m
            return m, f

        def sample_i(prng_state):
            f0 = cholesky(Pinf) @ jr.normal(prng_state, shape=(state_dims, 1))
            prng_keys = jr.split(prng_state, tsteps)
            _, f_sample = lax.scan(step, init=f0, xs=(As[:-1], Qs[:-1], prng_keys))
            return f_sample

        f_samples = vmap(sample_i, 0, 1)(prng_states)
        return f_samples  # (time, tr, out_dims, 1)

    def sample_posterior(
        self,
        prng_state,
        num_samps,
        t_eval,
        x_eval,
        jitter,
        compute_KL,
    ):
        """
        :param jnp.ndarray tx_eval: input locations at which to sample (out_dims, locs)
        :return:
            f_sample: the prior samples (num_samps, out_dims, locs)
        """
        state_dims = self.markov_sparse_kernel.state_dims
        # sample independent temporal processes

        # RFF part

        # sparse kernel part
        site_locs, site_dlocs = self.get_site_locs()

        # evaluation locations
        t_all, site_ind, eval_ind = order_times(t_eval, site_locs)
        interp_sites = interpolation_transitions(
            t_eval, site_locs, self.markov_sparse_kernel.state_transition
        )

        # compute linear dynamical system
        H, Pinf, As, Qs = self.markov_sparse_kernel.get_LDS(site_locs, site_dlocs)

        # sample prior at obs and eval locs
        prng_keys = jr.split(prng_state, 2)
        prior_samps = self.sample_prior(
            prng_keys[0], num_samps, t_all, jitter
        )  # (time, num_samps, out_dims, 1)

        # posterior mean
        post_means, _, KL_ss = evaluate_LTI_posterior(
            H,
            Pinf,
            As,
            Qs,
            self.site_obs,
            self.site_Lcov,
            interp_sites,
            mean_only=True,
            compute_KL=compute_KL,
            jitter=jitter,
        )  # (time, out_dims, 1)

        # noisy prior samples at eval locs
        array_indexing = lambda ind, array: array[ind, ...]
        varray_indexing = vmap(array_indexing, (0, 2), 2)

        prior_samps_t = (varray_indexing(site_ind, prior_samps),)
        prior_samps_eval = (varray_indexing(eval_ind, prior_samps),)

        prior_samps_noisy = prior_samps_t + self.site_Lcov[:, None, ...] @ jr.normal(
            prng_keys[1], shape=prior_samps_t.shape
        )  # (time, tr, out_dims, 1)

        # smooth noisy samples
        def smooth_prior_sample(prior_samp_i):
            smoothed_sample, _, _ = evaluate_LTI_posterior(
                H,
                Pinf,
                As,
                Qs,
                prior_samp_i,
                self.site_Lcov,
                interp_sites,
                mean_only=True,
                compute_KL=False,
                jitter=jitter,
            )
            return smoothed_sample

        smoothed_samps = vmap(smooth_prior_sample, 1, 1)(prior_samps_noisy)

        # Matheron's rule pathwise samplig
        return prior_samps_eval - smoothed_samps + post_means[:, None, ...], KL_ss
