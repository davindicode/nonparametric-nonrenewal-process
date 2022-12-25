from ..base import module

from jax import vmap, lax
import jax.numpy as jnp
import jax.random as jr
from jax.numpy.linalg import cholesky

import equinox as eqx

from .linalg import evaluate_LTI_posterior
from .base import SSM




def compute_dt(t):
    """
    :param jnp.ndarray t: time points of shape (locs,)
    """
    dt = jnp.diff(t)
    if jnp.all(jnp.isclose(dt, dt[0])):  # grid
        return dt[:1]
    else:
        return dt



def order_times(t_eval, t_site):
    """
    Order all time points
    
    :param jnp.ndarray t_eval: evaluation points of shape (eval_locs,)
    :param jnp.ndarray t_site: evaluation points of shape (site_locs,)
    """
    num_eval, num_sites = t_eval.shape[0], t_site.shape[0]
    
    if t_eval is None:  # assume t_site is ordered and unique
        site_ind = jnp.arange(num_sites)
        return t_site, site_ind, site_ind

    t_all = jnp.concatenate([t_site, t_eval])
    t_all, input_ind = jnp.unique(t_all, return_inverse=True)
    
    sort_ind = jnp.argsort(t_all)
    t_all = t_all[sort_ind]

    site_ind, eval_ind = sort_ind[input_ind[:num_sites]], sort_ind[input_ind[num_sites:]]
    return t_all, site_ind, eval_ind



def interpolation_transitions(t_eval, t_site, kernel_state_transition):
    """
    Transition matrices between observation points in a Markovian system
    
    :param jnp.ndarray t_site: site locations of shape (out_dims, locs)
    :return:
        evaluation indices of shape (eval_nums, out_dims)
        transition matrices of shape (eval_nums, out_dims, sd, sd)
    """
    if t_eval is None:
        return None, None, None
    
    num_evals = t_eval.shape[0]
    out_dims = t_site.shape[0]
    
    inf = 1e10 * jnp.ones((out_dims, 1))
    t_aug = jnp.concatenate([-inf, t_site, inf], axis=-1)
    
    def search_and_sort(t_aug, t_eval):
        ind_eval = jnp.searchsorted(t_aug, t_eval) - 1

        dt_fwd = t_eval - t_aug[ind_eval]  # (num_evals,)
        dt_bwd = t_aug[ind_eval + 1] - t_eval  # (num_evals,)
        stack_dt = jnp.concatenate([dt_fwd, dt_bwd])
        
        return ind_eval, stack_dt
    
    ind_eval, stack_dt = vmap(search_and_sort, (0, None), (1, 1))(t_aug, t_eval)
    stack_A = vmap(kernel_state_transition)(stack_dt)  # vmap over num_evals
    A_fwd, A_bwd = stack_A[:num_evals], stack_A[-num_evals:]
    
    return ind_eval, A_fwd, A_bwd
    


### classes ###
class GaussianLTI(SSM):
    """
    Gaussian Linear Time-Invariant System
    
    Temporal multi-output kernels have separate latent processes that can be coupled.
    Spatiotemporal kernel modifies the process noise across latent processes, but dynamics uncoupled.
    Multi-output GPs generally mix latent processes via dynamics as well.
    """
    
    markov_kernel: MarkovianKernel
        
    fixed_grid_locs: bool
    diagonal_site: bool

    def __init__(self, markov_kernel, site_locs, site_obs, site_Lcov, diagonal_site=True, fixed_grid_locs=False):
        """
        :param module markov_kernel: Markovian kernel module
        :param jnp.ndarray site_locs: site locations with shape (timesteps,)
        :param jnp.ndarray site_obs: site observations with shape (timesteps, x_dims, 1)
        :param jnp.ndarray site_Lcov: site covariance cholesky with shape (timesteps, x_dims, x_dims)
        """
        super().__init__(site_locs, site_obs, site_Lcov)
        self.markov_kernel = markov_kernel
        
        self.fixed_grid_locs = fixed_grid_locs
        self.diagonal_site = diagonal_site
        
    def apply_constraints(self):
        """
        PSD constraint
        """
        model = jax.tree_map(lambda p: p, self)  # copy
        
        def update(Lcov):
            epdfunc = lambda x: enforce_positive_diagonal(x, lower_lim=1e-2)
            Lcov = vmap(epdfunc)(jnp.tril(Lcov))
            Lcov = jnp.triu(Lcov) if self.spatial_MF else Lcov
            return Lcov

        model = eqx.tree_at(
            lambda tree: tree.site_Lcov,
            model,
            replace_fn=update,
        )
        
        kernel = self.markov_kernel.apply_constraints(self.markov_kernel)
        model = eqx.tree_at(
            lambda tree: tree.markov_kernel,
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
    def evaluate_posterior(
        self, t_eval, mean_only, compute_KL, jitter
    ):
        """
        predict at test locations X, which may includes training points
        (which are essentially fixed inducing points)

        :param jnp.ndarray t_eval: evaluation times of shape (locs,)
        :return:
            means of shape (time, out_dims, 1)
            covariances of shape (time, out_dims, out_dims)
        """
        site_locs, site_dlocs = self.get_site_locs()
        interp_sites = interpolation_transitions(
            t_eval, site_locs, self.markov_kernel.state_transition)
        
        # compute linear dynamical system
        H, minf, Pinf, As, Qs = self.markov_kernel.get_LDS(site_locs, site_dlocs)
        
        post_means, post_covs, KL = evaluate_LGSSM_posterior(
            H, minf, Pinf, As, Qs, self.site_obs, self.site_Lcov, 
            interp_sites, mean_only, compute_KL, jitter, 
        )
        return post_means, post_covs, KL

    ### sample ###
    def sample_prior(self, prng_state, num_samps, t_eval, jitter):
        """
        Sample from the model prior f~N(0,K) via simulation of the LGSSM
        
        :param int num_samps: the number of samples to draw [scalar]
        :param jnp.ndarray t_eval: the input locations at which to sample
        :return:
            f_sample: the prior samples (time, num_samps, out_dims, 1)
        """
        tsteps = t_eval.shape[0]
        dt = compute_dt(t_eval)  # (eval_locs,)
        t_eval, dt = t_eval[None, :], dt[None, :]
        
        eps_I = jitter * jnp.eye(self.markov_kernel.state_dims)
        
        # transition and noise process matrices
        H, minf, Pinf, As, Qs = self.markov_kernel.get_LDS(dt, tsteps)

        prng_states = jr.split(prng_state, num_samps)  # (num_samps, 2)

        def step(carry, inputs):
            m = carry
            A, Q, prng_state = inputs
            L = cholesky(Q + eps_I)  # can be a bit unstable, lower=True

            q_samp = L @ jr.normal(prng_state, shape=(self.markov_kernel.state_dims, 1))
            m = A @ m + q_samp
            f = H @ m
            return m, f

        def sample_i(prng_state):
            m0 = cholesky(Pinf) @ jr.normal(
                prng_state, shape=(self.markov_kernel.state_dims, 1)
            )
            prng_keys = jr.split(prng_state, tsteps)
            _, f_sample = lax.scan(step, init=m0, xs=(As[:-1], Qs[:-1], prng_keys))
            return f_sample

        f_samples = vmap(sample_i, 0, 1)(prng_states)
        return f_samples  # (time, tr, out_dims, 1)

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
            the posterior samples (eval_locs, num_samps, N, 1)
        """
        site_locs, site_dlocs = self.get_site_locs()
        
        # evaluation locations
        t_all, site_ind, eval_ind = order_times(t_eval, site_locs)
        interp_sites = interpolation_transitions(
            t_eval, site_locs, self.markov_kernel.state_transition)
        
        # compute linear dynamical system
        H, minf, Pinf, As, Qs = self.markov_kernel.get_LDS(site_locs, site_dlocs)

        # sample prior at obs and eval locs
        prng_keys = jr.split(prng_state, 2)
        prior_samps = self.sample_prior(
            prng_keys[0], num_samps, t_all, jitter
        )  # (time, num_samps, out_dims, 1)
        
        # posterior mean
        post_means, _, KL_ss = evaluate_LTI_posterior(
            H, minf, Pinf, As, Qs, self.site_obs, self.site_Lcov, 
            interp_sites, mean_only=True, compute_KL=compute_KL, jitter=jitter,
        )  # (time, out_dims, 1)

        # noisy prior samples at eval locs
        array_indexing = lambda ind, array: array[ind, ...]
        varray_indexing = vmap(array_indexing, (0, 2), 2)
            
        prior_samps_t = varray_indexing(site_ind, prior_samps),
        prior_samps_eval = varray_indexing(eval_ind, prior_samps),
        
        prior_samps_noisy = prior_samps_t + self.site_Lcov[:, None, ...] @ jr.normal(
            prng_keys[1], shape=prior_samps_t.shape
        )  # (time, tr, out_dims, 1)

        # smooth noisy samples
        def smooth_prior_sample(prior_samp_i):
            smoothed_sample, _, _ = evaluate_LTI_posterior(
                H, 
                minf, 
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
        
       
    
    
@eqx.filter_vmap(args=(0, 0, 0, 1, 1, 0, 0, (1, 1, 1), None, None, None), out=(0, 0, 0))
def vmap_outdims(H, minf, Pinf, As, Qs, site_obs, site_Lcov, interp_sites, mean_only, compute_KL, jitter):
    return evaluate_LTI_posterior(
        H, minf, Pinf, As, Qs, site_obs, site_Lcov, interp_sites, mean_only, compute_KL, jitter, 
    )


@eqx.filter_vmap(args=(None, None, None, None, None, -3, -3, (None, None, None), None, None, None), out=(0, 0, 0))
def vmap_spatial(H, minf, Pinf, As, Qs, site_obs, site_Lcov, interp_sites, mean_only, compute_KL, jitter):
    return vmap_outdims(
        H, minf, Pinf, As, Qs, site_obs, site_Lcov, interp_sites, mean_only, compute_KL, jitter, 
    )
    

class SpatioTemporalLTI(SSM):
    """
    Factorized spatial and temporal GP kernel with temporal markov kernel
    
    We use the Kronecker kernel, with out_dims of the temporal kernel 
    equal to out_dims of process
    """
    
    spatiotemporal_kernel: MarkovSparseKronecker
    
    spatial_MF: bool
    fixed_grid_locs: bool

    def __init__(self, spatiotemporal_kernel, site_locs, site_obs, site_Lcov, spatial_MF=True, fixed_grid_locs=False):
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
        super().__init__(site_locs, site_obs, site_Lcov)
        self.spatiotemporal_kernel = spatiotemporal_kernel
        self.spatial_MF = spatial_MF
        self.fixed_grid_locs = fixed_grid_locs
        
    def apply_constraints(self):
        """
        PSD constraint
        """
        model = jax.tree_map(lambda p: p, self)  # copy
        
        def update(Lcov):
            epdfunc = lambda x: enforce_positive_diagonal(x, lower_lim=1e-2)
            Lcov = vmap(epdfunc)(jnp.tril(Lcov))
            Lcov = jnp.triu(Lcov) if self.spatial_MF else Lcov
            return Lcov

        model = eqx.tree_at(
            lambda tree: tree.site_Lcov,
            model,
            replace_fn=update,
        )
        
        kernel = self.markov_kernel.apply_constraints(self.markov_kernel)
        model = eqx.tree_at(
            lambda tree: tree.markov_kernel,
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
        self, tx_eval, mean_only, compute_KL, jitter, 
    ):
        """
        predict at test locations X, which may includes training points
        (which are essentially fixed inducing points)

        :param jnp.ndarray t_eval: evaluation times of shape (locs,)
        :return:
            means of shape (time, out, 1)
            covariances of shape (time, out, out)
        """
        site_locs, site_dlocs = self.get_site_locs()
        t_eval, x_eval = tx_eval[..., 0], tx_eval[..., 1:]
        
        if self.spatial_MF:  # vmap over temporal kernel out_dims = spatial_locs
            # evaluation locations
            interp_sites = interpolation_transitions(
                t_eval, timedata[0], self.markov_kernel.markov_factor._state_transition)
            # (eval_inds, out_dims, sd, sd)
            
            # compute LDS matrices
            H, minf, Pinf, As, Qs = self.markov_kernel._get_LDS(site_locs, site_dlocs)
            # (ts, out_dims, sd, sd)

            # vmap over spatial points
            post_means, post_covs, KL = vmap_spatial(
                H, minf, Pinf, As, Qs, self.site_obs[..., None], self.site_Lcov[..., None], 
                interp_sites, mean_only, compute_KL, jitter, 
            )  # (spatial_locs, out_dims, timesteps, 1)
            
            post_means = post_means.transpose(1, 2, 0, 3)
            post_covs = post_covs.transpose(1, 2, 0, 3)
            
        else:
            # evaluation locations
            interp_sites = interpolation_transitions(
                t_eval, timedata[0], self.markov_kernel.state_transition)
            # (eval_inds, out_dims, spatial_pts*sd, spatial_pts*sd)
            
            H, minf, Pinf, As, Qs = self.markov_kernel.get_LDS(site_locs, site_dlocs)
            # (ts, out_dims, spatial_pts*sd, spatial_pts*sd)
            
            post_means_, post_covs_, KL = vmap_outdims(
                H, minf, Pinf, As, Qs, self.site_obs, self.site_Lcov, 
                interp_sites, mean_only, compute_KL, jitter, 
            )  # (out_dims, timesteps, spatial_locs, 1 or spatial_locs)
            
        C_krr, C_nystrom = self.markov_kernel.sparse_conditional(x_eval, jitter)
        Kmarkov = self.markov_kernel.markov_factor.K(t_eval, None, True)  # (out_dims, ts, 1)
        W = C_krr[..., None, :]  # (out_dims, timesteps, 1, spatial_locs)
        post_means = (W @ post_means_)[..., 0]  # (out_dims, timesteps, 1)
        post_covs = (W @ post_covs_ @ W.transpose(0, 1, 3, 2))[..., 0] + Kmarkov * C_nystrom
        
        return post_means, post_covs, KL.sum()  # sum over out_dims
        
    ### sample ###
    def sample_prior(self, prng_state, num_samps, tx_eval, jitter):
        """
        Sample from the model prior f~N(0,K) multiple times using a nested loop.
        :param num_samps: the number of samples to draw [scalar]
        :param t: the input locations at which to sample (defaults to train+test set) [N_samp, 1]
        :return:
            f_sample: the prior samples [S, N_samp]
        """
        vmap(order_times, (None, 0), (0, 0, 0))
    
        eps_I = jitter * jnp.eye(self.markov_kernel.state_dims)
        
        # sample independent temporal processes
        
        # mix temporal trajectories
        
        # transition and noise process matrices
        H, minf, Pinf, As, Qs = self.markov_kernel.get_LDS(dt, tsteps)

        prng_states = jr.split(prng_state, num_samps)  # (num_samps, 2)

        def step(carry, inputs):
            m = carry
            A, Q, prng_state = inputs
            L = cholesky(Q + eps_I)  # can be a bit unstable, lower=True

            q_samp = L @ jr.normal(prng_state, shape=(self.markov_kernel.state_dims, 1))
            m = A @ m + q_samp
            f = H @ m
            return m, f

        def sample_i(prng_state):
            m0 = cholesky(Pinf) @ jr.normal(
                prng_state, shape=(self.markov_kernel.state_dims, 1)
            )
            prng_keys = jr.split(prng_state, tsteps)
            _, f_sample = lax.scan(step, init=m0, xs=(As[:-1], Qs[:-1], prng_keys))
            return f_sample

        f_samples = vmap(sample_i, 0, 1)(prng_states)
        return f_samples  # (time, tr, out_dims, 1)

    def sample_posterior(
        self,
        prng_state,
        num_samps,
        tx_eval,
        jitter,
        compute_KL,
    ):
        # sample independent temporal processes
        
        # RFF part
        
        # sparse kernel part
        return
    
    

class LGSSM(SSM):
    """
    Linear Gaussian State Space Model
    
    Temporal multi-output kernels have separate latent processes that can be coupled.
    Spatiotemporal kernel modifies the process noise across latent processes, but dynamics uncoupled.
    Multi-output GPs generally mix latent processes via dynamics as well.
    """
    As: jnp.ndarray
    Qs: jnp.ndarray
    H: jnp.ndarray
    minf: jnp.ndarray
    P0: jnp.ndarray
        
    site_obs: jnp.ndarray
    site_Lcov: jnp.ndarray

    def __init__(self, As, Qs, H, minf, P0, site_obs, site_Lcov):
        """
        :param module markov_kernel: (hyper)parameters of the state space model
        :param jnp.ndarray As: transitions of shape (time, out, sd, sd)
        :param jnp.ndarray Qs: process noises of shape (time, out, sd, sd)
        :param jnp.ndarray site_locs: means of shape (time, out, 1)
        :param jnp.ndarray site_obs: means of shape (time, out, 1)
        :param jnp.ndarray site_Lcov: covariances of shape (time, out, out)
        """
        self.As = As
        self.Qs = Qs
        self.H = H
        self.minf = minf
        self.P0 = P0
        
        self.site_obs = site_obs
        self.site_Lcov = site_Lcov
    
    def get_LDS(self):
        Id = jnp.eye(self.As.shape[-1])
        Zs = jnp.zeros_like(Id)
        As = jnp.concatenate((Id[None, ...], self.As, Id[None, ...]), axis=0)
        Qs = jnp.concatenate((Zs[None, ...], self.Qs, Zs[None, ...]), axis=0)
        return self.H, self.minf, self.P0, As, Qs,
    
    ### posterior ###
    def entropy_posterior(self):
        """
        Precision of joint process is block-tridiagonal
        
        Compute the KL divergence and variational expectation of prior
        """
        return
    
    
    def evaluate_posterior(
        self, mean_only, compute_KL, compute_joint, jitter
    ):
        """
        predict at test locations X, which may includes training points
        (which are essentially fixed inducing points)

        :param jnp.ndarray t_eval: evaluation times of shape (locs,)
        :return:
            means of shape (time, out_dims, 1)
            covariances of shape (time, out_dims, out_dims)
        """
        H, minf, P0, As, Qs = self.get_LDS()
        
        smoother_means, smoother_covs, gains, logZ = fixed_interval_smoothing(
            H, minf, P0, As, Qs, self.site_obs, self.site_Lcov, compute_KL, compute_joint)

        post_means = H @ smoother_means
        if compute_KL:  # compute using pseudo likelihood
            post_covs = H @ smoother_covs @ H.T
        post_covs = None if mean_only else H @ smoother_covs @ H.T

        if compute_KL:
            site_log_lik = pseudo_log_likelihood(
                post_means, post_covs, site_obs, site_Lcov
            )
            KL = site_log_lik - logZ

        else:
            KL = 0.0
            
        if compute_joint:
            cross_cov = gain[ind] @ post_cov[ind + 1]
            post_joint_covs = jnp.block(
                [[post_cov[ind], cross_cov], [cross_cov.T, post_cov[ind + 1]]]
            )
            
            return post_means, post_covs, post_joint_covs, KL
        
        else:  # only marginal
            return post_means, post_covs, KL

    ### sample ###
    def sample_prior(self, prng_state, num_samps, jitter):
        """
        Sample from the model prior f~N(0,K) multiple times using a nested loop.
        :param num_samps: the number of samples to draw [scalar]
        :param t: the input locations at which to sample (defaults to train+test set) [N_samp, 1]
        :return:
            f_sample: the prior samples [S, N_samp]
        """
        H, minf, P0, As, Qs = self.get_LDS()
        
        eps_I = jitter * jnp.eye(self.markov_kernel.state_dims)
        tsteps = self.site_obs.shape[0]

        prng_states = jr.split(prng_state, num_samps)  # (num_samps, 2)

        def step(carry, inputs):
            m = carry
            A, Q, prng_state = inputs
            L = cholesky(Q + eps_I)  # can be a bit unstable, lower=True

            q_samp = L @ jr.normal(prng_state, shape=(self.markov_kernel.state_dims, 1))
            m = A @ m + q_samp
            f = H @ m
            return m, f

        def sample_i(prng_state):
            m0 = cholesky(Pinf) @ jr.normal(
                prng_state, shape=(self.markov_kernel.state_dims, 1)
            )
            prng_keys = jr.split(prng_state, tsteps)
            _, f_sample = lax.scan(step, init=m0, xs=(As[:-1], Qs[:-1], prng_keys))
            return f_sample

        f_samples = vmap(sample_i, 0, 1)(prng_states)
        return f_samples  # (time, tr, state_dims, 1)

    def sample_posterior(
        self,
        prng_state,
        num_samps,
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

        :param X: the sampling input locations [N, 1]
        :param num_samps: the number of samples to draw [scalar]
        :param seed: the random seed for sampling
        :return:
            the posterior samples (eval_locs, num_samps, N, 1)
        """
        H, minf, P0, As, Qs = self.get_LDS()
        
        # sample prior at obs and eval locs
        prng_keys = jr.split(prng_state, 2)
        prior_samps = self.sample_prior(
            prng_keys[0], num_samps, t_all, jitter
        )
        
        # posterior mean
        post_means, _, KL_ss = fixed_interval_smoothing(
            H, minf, Pinf, As, Qs, self.site_obs, self.site_Lcov, 
            interp_sites, mean_only=True, compute_KL=compute_KL, return_gains=False, 
        )  # (time, N, 1)

        # noisy prior samples at eval locs
        prior_samps_t, prior_samps_eval = (
            prior_samps[site_ind, ...],
            prior_samps[eval_ind, ...],
        )
        prior_samps_noisy = prior_samps_t + self.site_Lcov[:, None, ...] @ jr.normal(
            prng_keys[1], shape=prior_samps_t.shape
        )  # (time, tr, N, 1)

        # smooth noisy samples
        def smooth_prior_sample(prior_samp_i):
            smoothed_sample, _, _, _ = fixed_interval_smoothing(
                H, minf, P0, As, Qs, prior_samp_i, self.site_Lcov, 
                compute_KL=False, return_gains=False, 
            )
            
            return smoothed_sample

        smoothed_samps = vmap(smooth_prior_sample, 1, 1)(prior_samps_noisy)
        
        # Matheron's rule pathwise samplig
        return prior_samps_eval - smoothed_samps + post_means[:, None, ...], KL_ss