import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from jax import lax, vmap
from jax.numpy.linalg import cholesky

from .base import SSM
from .kernels import MarkovianKernel

from .linalg import evaluate_LGSSM_posterior, LTI_process_noise


def sample_LGSSM(H, m0, P0, As, Qs, prng_state, num_samps, jitter):
    """
    Sample sequentially from LGSSM, centered around 0
    """
    timesteps = As.shape[0] - 1
    state_dims = As.shape[-1]

    eps_I = jitter * jnp.eye(state_dims)

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
        f0 = m0 + cholesky(P0) @ jr.normal(prng_state, shape=(state_dims, 1))
        prng_keys = jr.split(prng_state, timesteps)
        _, f_sample = lax.scan(step, init=f0, xs=(As[:-1], Qs[:-1], prng_keys))
        return f_sample

    f_samples = vmap(sample_i, 0, 1)(prng_states)
    return f_samples  # (time, tr, out_dims, 1)


def order_times(t_eval, t_site):
    """
    Order all time points

    :param jnp.ndarray t_eval: evaluation points of shape (eval_locs,)
    :param jnp.ndarray t_site: evaluation points of shape (site_locs,)
    """
    num_sites = len(t_site)
    if t_eval is None:  # assume t_site is ordered and unique
        site_ind = jnp.arange(num_sites)
        return t_site, site_ind, site_ind
    
    num_eval = len(t_eval)
    t_all = jnp.concatenate([t_site, t_eval])
    #t_all, input_ind = jnp.unique(t_all, return_inverse=True)
    
    sort_ind = jnp.argsort(t_all)
    t_all = t_all[sort_ind]

    site_ind, eval_ind = (
        jnp.where(sort_ind < num_sites, size=num_sites)[0],#input_ind[:num_sites]],
        jnp.where(sort_ind >= num_sites, size=num_eval)[0],#input_ind[num_sites:]],
    )
    return t_all, site_ind, eval_ind


def interpolation_times(t_eval, t_site):
    """
    Transition matrices between observation points in a Markovian system

    :param jnp.ndarray t_site: site locations of shape (locs,)
    :return:
        evaluation indices of shape (eval_nums, out_dims)
        transition matrices of shape (eval_nums, out_dims, sd, sd)
    """
    num_evals = len(t_eval)
    out_dims = len(t_site)

    inf = 1e10 * jnp.ones(1)
    t_aug = jnp.concatenate([-inf, t_site, inf])

    ind_eval = jnp.searchsorted(t_aug, t_eval) - 1

    dt_fwd = t_eval - t_aug[ind_eval]  # (num_evals,)
    dt_bwd = t_aug[ind_eval + 1] - t_eval  # (num_evals,)
    return ind_eval, dt_fwd, dt_bwd


### classes ###
class GaussianLTI(SSM):
    """
    Gaussian Linear Time-Invariant System

    Temporal multi-output kernels have separate latent processes that can be coupled.
    Spatiotemporal kernel modifies the process noise across latent processes, but dynamics uncoupled.
    Multi-output GPs generally mix latent processes via dynamics as well.
    """

    kernel: MarkovianKernel

    fixed_grid_locs: bool
    diagonal_site: bool

    def __init__(
        self,
        kernel,
        site_locs,
        site_obs,
        site_Lcov,
        diagonal_site=True,
        fixed_grid_locs=False,
    ):
        """
        :param module kernel: Markovian kernel module
        :param jnp.ndarray site_locs: site locations with shape (timesteps,)
        :param jnp.ndarray site_obs: site observations with shape (timesteps, x_dims, 1)
        :param jnp.ndarray site_Lcov: site covariance cholesky with shape (timesteps, x_dims, x_dims)
        """
        super().__init__(site_locs, site_obs, site_Lcov, kernel.array_type)
        self.kernel = kernel

        self.fixed_grid_locs = fixed_grid_locs
        self.diagonal_site = diagonal_site

    def apply_constraints(self):
        """
        PSD constraint
        """
        model = jax.tree_map(lambda p: p, self)  # copy

        def update(Lcov):
            epdfunc = lambda x: constrain_diagonal(x, lower_lim=1e-2)
            Lcov = vmap(epdfunc)(jnp.tril(Lcov))
            Lcov = jnp.triu(Lcov) if self.diagonal_site else Lcov
            return Lcov

        model = eqx.tree_at(
            lambda tree: tree.site_Lcov,
            model,
            replace_fn=update,
        )

        kernel = self.kernel.apply_constraints(self.kernel)
        model = eqx.tree_at(
            lambda tree: tree.kernel,
            model,
            replace_fn=lambda _: kernel,
        )

        if fixed_grid_locs is False:  # enforce ordering in site locs

            def update(locs):  # ignore change in site params ordering
                return jnp.sort(locs)

            model = eqx.tree_at(
                lambda tree: tree.site_locs,
                model,
                replace_fn=update,
            )

        return model

    def get_site_locs(self):
        if self.fixed_grid_locs:
            locs = lax.stop_gradient(self.site_locs)  # (ts,)
            return locs, locs[1:2] - locs[0:1]
        else:
            return self.site_locs, jnp.diff(self.site_locs)

    ### posterior ###
    def evaluate_posterior(self, t_eval, mean_only, compute_KL, jitter):
        """
        predict at test locations X, which may includes training points
        (which are essentially fixed inducing points)

        :param jnp.ndarray t_eval: evaluation times of shape (time,)
        :return:
            means of shape (time, x_dims, 1)
            covariances of shape (time, x_dims, x_dims)
        """
        site_locs, site_dlocs = self.get_site_locs()
        
        # compute linear dynamical system
        H, Pinf, As, Qs = self.kernel.get_LDS(
            site_dlocs[None, :], site_locs.shape[0]
        )
        
        # interpolation
        ind_eval, dt_fwd, dt_bwd = interpolation_times(t_eval, site_locs)
        dt_fwd_bwd = jnp.concatenate([dt_fwd, dt_bwd])  # (2*num_evals,)
        A_fwd_bwd = vmap(self.kernel.state_transition)(
            dt_fwd_bwd
        )  # vmap over num_evals
        Q_fwd_bwd = vmap(LTI_process_noise, (0, None), 0)(A_fwd_bwd, Pinf)

        post_means, post_covs, KL = evaluate_LGSSM_posterior(
            H,
            Pinf,
            Pinf,
            As,
            Qs,
            self.site_obs,
            self.site_Lcov,
            ind_eval,
            A_fwd_bwd,
            Q_fwd_bwd,
            mean_only,
            compute_KL,
            jitter,
        )
        return post_means, post_covs, KL

    ### sample ###
    def sample_prior(self, prng_state, num_samps, t_eval, jitter):
        """
        Sample from the model prior f~N(0,K) via simulation of the LGSSM

        :param int num_samps: number of samples to draw
        :param jnp.ndarray t_eval: the input locations at which to sample (locs,)
        :return:
            f_sample: the prior samples (num_samps, time, x_dims, 1)
        """
        tsteps = len(t_eval)
        dt = jnp.diff(t_eval)  # (eval_locs-1,)
        if jnp.all(jnp.isclose(dt, dt[0])):  # grid
            dt = dt[:1]

        H, Pinf, As, Qs = self.kernel.get_LDS(dt[None, :], tsteps)
        minf = jnp.zeros((Pinf.shape[-1], 1))
        samples = sample_LGSSM(H, minf, Pinf, As, Qs, prng_state, num_samps, jitter)
        return samples.transpose(1, 0, 2, 3)

    def sample_posterior(
        self,
        prng_state,
        num_samps,
        t_eval,
        jitter,
        compute_KL,
    ):
        """
        Sample from the posterior at specified time locations.
        Posterior sampling works by smoothing samples from the prior using the approximate Gaussian likelihood
        model given by the pseudo-likelihood, 𝓝(f|μ*,σ²*), computed during training.
         - draw samples (f*) from the prior
         - add Gaussian noise to the prior samples using auxillary model p(y*|f*) = 𝓝(y*|f*,σ²*)
         - smooth the samples by computing the posterior p(f*|y*)
         - posterior samples = prior samples + smoothed samples + posterior mean
                             = f* + E[p(f*|y*)] + E[p(f|y)]
        See Arnaud Doucet's note "A Note on Efficient Conditional Simulation of Gaussian Distributions" for details.

        :param jnp.ndarray seed: JAX prng state
        :param int num_samps: the number of samples to draw
        :param jnp.ndarray t_eval: the sampling input locations (eval_nums,), if None, sample at site locs
        :return:
            the posterior samples (num_samps, eval_locs, x_dims, 1)
        """
        prng_keys = jr.split(prng_state, 2)
        site_locs, site_dlocs = self.get_site_locs()

        # compute linear dynamical system
        H, Pinf, As, Qs = self.kernel.get_LDS(
            site_dlocs[None, :], site_locs.shape[0]
        )
        
        # evaluation locations
        t_all, site_ind, eval_ind = order_times(t_eval, site_locs)
        
        if t_eval is not None:
            num_evals = len(t_eval)
            ind_eval, dt_fwd, dt_bwd = interpolation_times(t_eval, site_locs)

            dt_fwd_bwd = jnp.concatenate([dt_fwd, dt_bwd])  # (2*num_evals,)
            A_fwd_bwd = vmap(self.kernel.state_transition)(
                dt_fwd_bwd
            )  # vmap over num_evals
            Q_fwd_bwd = vmap(LTI_process_noise, (0, None), 0)(A_fwd_bwd, Pinf)
            
        else:
            ind_eval, A_fwd_bwd, Q_fwd_bwd = None, None, None
        
        # posterior mean
        post_means, _, KL_ss = evaluate_LGSSM_posterior(
            H,
            Pinf,
            Pinf,
            As,
            Qs,
            self.site_obs,
            self.site_Lcov,
            ind_eval,
            A_fwd_bwd,
            Q_fwd_bwd,
            mean_only=True,
            compute_KL=compute_KL,
            jitter=jitter,
        )  # (time, out_dims, 1)

        # sample prior at obs and eval locs
        prior_samps = self.sample_prior(
            prng_keys[0], num_samps, t_all, jitter
        )  # (num_samps, time, out_dims, 1)
        
        prior_samps_t = prior_samps[:, site_ind]
        prior_samps_eval = prior_samps[:, eval_ind]

        # noisy prior samples at eval locs
        prior_samps_noisy = prior_samps_t + self.site_Lcov[None, ...] @ jr.normal(
            prng_keys[1], shape=prior_samps_t.shape
        )  # (tr, time, out_dims, 1)

        # smooth noisy samples
        def smooth_prior_sample(prior_samp_i):
            smoothed_sample, _, _ = evaluate_LGSSM_posterior(
                H,
                Pinf,
                Pinf,
                As,
                Qs,
                prior_samp_i,
                self.site_Lcov,
                ind_eval,
                A_fwd_bwd,
                Q_fwd_bwd,
                mean_only=True,
                compute_KL=False,
                jitter=jitter,
            )
            return smoothed_sample

        smoothed_samps = vmap(smooth_prior_sample, 0, 0)(prior_samps_noisy)

        # Matheron's rule pathwise samplig
        return prior_samps_eval - smoothed_samps + post_means[None, ...], KL_ss


@eqx.filter_vmap(args=(0, 0, 0, 1, 1, 0, 0, 1, 1, 1, None, None, None), out=(0, 0, 0))
def vmap_outdims_LGSSM_posterior(
    H,
    P_init, 
    P_end,
    As,
    Qs,
    site_obs,
    site_Lcov,
    ind_eval,
    A_fwd_bwd,
    Q_fwd_bwd,
    mean_only,
    compute_KL,
    jitter,
):
    return evaluate_LGSSM_posterior(
        H,
        P_init, 
        P_end,
        As,
        Qs,
        site_obs,
        site_Lcov,
        ind_eval,
        A_fwd_bwd,
        Q_fwd_bwd,
        mean_only,
        compute_KL,
        jitter,
    )


class IndependentLTI(SSM):
    """
    Multi-output temporal GP with temporal markov kernel
    Independent outputs (posterior is factorized across output dimensions)

    We use the Kronecker kernel, with out_dims of the temporal kernel
    equal to out_dims of process
    """

    kernel: MarkovianKernel

    fixed_grid_locs: bool

    def __init__(
        self, kernel, site_locs, site_obs, site_Lcov, fixed_grid_locs=False
    ):
        """
        :param module temporal_kernel: markov kernel module
        :param module spatial_kernel: spatial kernel
        :param jnp.ndarray site_locs: site locations (out_dims, timesteps)
        :param jnp.ndarray site_obs: site observations with shape (out_dims, timesteps, 1)
        :param jnp.ndarray site_Lcov: site observations with shape (out_dims, timesteps, 1)
        """
        assert site_obs.shape[-2] == site_Lcov.shape[-2]  # spatial_locs
        super().__init__(site_locs, site_obs, site_Lcov, kernel.array_type)
        self.kernel = kernel
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
        num_evals = t_eval.shape[-1]

        site_locs, site_dlocs = self.get_site_locs()
        t_eval = jnp.broadcast_to(t_eval, (site_locs.shape[0], num_evals))

        # compute LDS matrices
        H, Pinf, As, Qs = self.kernel._get_LDS(
            site_dlocs, site_locs.shape[1]
        )  # (ts, out_dims, sd, sd)
        
        # evaluation locations
        ind_eval, dt_fwd, dt_bwd = vmap(interpolation_times, (0, 0), (1, 1, 1))(
            t_eval, site_locs
        )  # vmap over out_dims
        dt_fwd_bwd = jnp.concatenate([dt_fwd, dt_bwd], axis=0)  # (2*num_evals, out_dims)
        A_fwd_bwd = vmap(self.kernel._state_transition)(
            dt_fwd_bwd
        )  # vmap over num_evals, (eval_inds, out_dims, sd, sd)
        Q_fwd_bwd = vmap(vmap(LTI_process_noise, (0, 0), 0), (0, None), 0)(A_fwd_bwd, Pinf)

        # vmap over output dimensions
        post_means, post_covs, KL = vmap_outdims_LGSSM_posterior(
            H,
            Pinf,
            Pinf,
            As,
            Qs,
            self.site_obs[..., None],
            self.site_Lcov[..., None],
            ind_eval, 
            A_fwd_bwd, 
            Q_fwd_bwd, 
            mean_only,
            compute_KL,
            jitter,
        )  # (out_dims, timesteps, 1, 1)
        
        post_means = post_means[..., 0]  # (out_dims, timesteps, 1)
        post_covs = post_covs[..., 0]

        return post_means, post_covs, KL.sum()  # sum over out_dims

    ### sample ###
    def sample_prior(self, prng_state, num_samps, t_eval, jitter):
        """
        Sample from the model prior f~N(0,K)
        
        :param num_samps: number of samples to draw
        :param jnp.ndarray tx_eval: input locations at which to sample (out_dims, locs)
        :return:
            f_sample: the prior samples (num_samps, out_dims, locs, 1)
        """
        state_dims, out_dims = self.kernel.state_dims, self.kernel.out_dims
        eps_I = jitter * jnp.eye(state_dims)
        
        tsteps = t_eval.shape[-1]
        t_eval = jnp.broadcast_to(t_eval, (self.site_locs.shape[0], tsteps))
        
        dt = jnp.diff(t_eval, axis=-1)  # (out_dims, eval_locs-1)
        if jnp.all(jnp.isclose(dt, dt[0, 0])):  # grid
            dt = dt[:, :1]
            
        # sample independent temporal processes
        prng_state = jr.split(prng_state, out_dims)
        
        H, Pinf, As, Qs = self.kernel._get_LDS(dt, tsteps)  # (ts, out_dims, sd, sd)
        minf = jnp.zeros((*Pinf.shape[:2], 1))
        samples = vmap(sample_LGSSM, (0, 0, 0, 1, 1, 0, None, None), 0)(
            H, minf, Pinf, As, Qs, prng_state, num_samps, jitter)[..., 0]  # (out, ts, num_samps, 1)
        return samples.transpose(2, 0, 1, 3)

    def sample_posterior(
        self,
        prng_state,
        num_samps, 
        t_eval, 
        jitter, 
        compute_KL,
    ):
        """
        :param jnp.ndarray t_eval: input locations at which to sample (out_dims, locs)
        :return:
            f_sample: the prior samples (num_samps, out_dims, locs)
        """
        prng_keys = jr.split(prng_state, 2)
        state_dims = self.kernel.state_dims
        
        site_locs, site_dlocs = self.get_site_locs()
        t_eval = jnp.broadcast_to(t_eval, (site_locs.shape[0], t_eval.shape[-1]))

        # compute linear dynamical system
        H, Pinf, As, Qs = self.kernel._get_LDS(
            site_dlocs, site_locs.shape[1]
        )  # (ts, out_dims, sd, sd)
        
        # evaluation locations
        t_all, site_ind, eval_ind = vmap(order_times, (0, 0), (0, 0, 0))(
            t_eval, self.site_locs
        )  # vmap over out_dims
        
        ind_eval, dt_fwd, dt_bwd = vmap(interpolation_times, (0, 0), (1, 1, 1))(
            t_eval, site_locs
        )  # vmap over out_dims
        dt_fwd_bwd = jnp.concatenate([dt_fwd, dt_bwd], axis=0)  # (2*num_evals, out_dims)
        A_fwd_bwd = vmap(self.kernel._state_transition)(
            dt_fwd_bwd
        )  # vmap over num_evals, (eval_inds, out_dims, sd, sd)
        Q_fwd_bwd = vmap(vmap(LTI_process_noise, (0, 0), 0), (0, None), 0)(A_fwd_bwd, Pinf)
        
        # posterior mean
        post_means, _, KL = vmap_outdims_LGSSM_posterior(
            H,
            Pinf,
            Pinf,
            As,
            Qs,
            self.site_obs[..., None],
            self.site_Lcov[..., None],
            ind_eval, 
            A_fwd_bwd, 
            Q_fwd_bwd, 
            mean_only=True,
            compute_KL=compute_KL,
            jitter=jitter,
        )  # (time, out_dims, 1)

        # sample prior at obs and eval locs
        prior_samps = self.sample_prior(
            prng_keys[0], num_samps, t_all, jitter
        )  # (num_samps, out_dims, time, 1)
        
        # noisy prior samples at eval locs
        array_indexing = lambda ind, array: array[..., ind, :]
        varray_indexing = vmap(array_indexing, (0, 1), 1)  # vmap over out_dims

        prior_samps_t = varray_indexing(site_ind, prior_samps)
        prior_samps_eval = varray_indexing(eval_ind, prior_samps)
        
        prior_samps_noisy = prior_samps_t + self.site_Lcov[None, ...] * jr.normal(
            prng_keys[1], shape=prior_samps_t.shape
        )  # (tr, out_dims, time, 1)

        # smooth noisy samples
        def smooth_prior_sample(prior_samp_i):
            smoothed_sample, _, _ = vmap_outdims_LGSSM_posterior(
                H,
                Pinf,
                Pinf,
                As,
                Qs,
                prior_samp_i,
                self.site_Lcov[..., None],
                ind_eval, 
                A_fwd_bwd, 
                Q_fwd_bwd, 
                mean_only=True,
                compute_KL=False,
                jitter=jitter,
            )
            return smoothed_sample

        smoothed_samps = vmap(smooth_prior_sample)(prior_samps_noisy)[..., 0]  # vmap over MC

        # Matheron's rule pathwise samplig
        return prior_samps_eval - smoothed_samps + post_means[None, ..., 0], KL.sum()  # sum over out_dims
