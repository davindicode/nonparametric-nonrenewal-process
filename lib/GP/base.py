import math
from functools import partial

from jax import vmap
import jax.numpy as jnp
import jax.random as jr
from jax.numpy.linalg import cholesky
from jax.scipy.linalg import solve_triangular

from ..base import module
from .kernels import Kernel, MarkovianKernel
from .linalg import mvn_conditional

from ..utils.jax import sample_gaussian_noise




class GP(module):
    """
    GP with function and RFF kernel
    """
    kernel: Kernel
    RFF_num_feats: int
        
    # hyperparameters
    mean: jnp.ndarray
        
    def __init__(self, kernel, mean, RFF_num_feats):
        super().__init__()
        self.kernel = kernel
        self.mean = mean  # (out_dims,)
        self.RFF_num_feats = RFF_num_feats  # use random Fourier features
        
    def apply_constraints(self):
        """
        PSD constraint
        """
        kernel = self.kernel.apply_constraints(self.kernel)
        
        model = jax.tree_map(lambda p: p, self)  # copy
        model = eqx.tree_at(
            lambda tree: tree.kernel,
            model,
            replace_fn=lambda _: kernel,
        )

        return model
        
    
    def evaluate_conditional(self, x, x_obs, f_obs, mean_only, diag_cov, jitter):
        """
        Compute the conditional distribution
        
        :param jnp.array x: shape (time, num_samps, in_dims)
        :param jnp.array x_obs: shape (out_dims, obs_pts, in_dims)
        :return:
            conditional mean of shape (out_dims, num_samps, ts, 1)
            conditional covariance of shape (out_dims, num_samps, ts, 1)
        """
        cond_out = vmap(
            mvn_conditional, 
            (2, None, None, None, None, None, None), 
            1 if mean_only else (1, 1),
        )(
            x[None, ...], x_obs, f_obs, self.kernel.K, mean_only, diag_cov, jitter
        )
        
        if mean_only:
            return cond_out + self.mean[:, None, None, None]
        else:
            return cond_out[0] + self.mean[:, None, None, None], cond_out[1]
        
        
    def sample_prior(self, prng_state, x, jitter):
        """
        Prior distribution p(f(x)) = N(0, K_xx)
        Can use approx_points as number of points

        :param jnp.array x: shape (time, num_samps, out_dims, in_dims)
        :return:
            sample of shape (time, num_samps, out_dims)
        """
        in_dims = self.kernel.in_dims
        out_dims = self.kernel.out_dims
        
        if x.ndim == 3:
            x = x[..., None, :]  # (time, num_samps, out_dims, in_dims)
        ts, num_samps = x.shape[:2]

        if self.RFF_num_feats > 0:  # random Fourier features
            prng_keys = jr.split(prng_state, 2)
            ks, amplitude = self.kernel.sample_spectrum(
                prng_keys[0], num_samps, self.RFF_num_feats
            )  # (num_samps, out_dims, feats, in_dims)
            phi = (
                2
                * jnp.pi
                * jr.uniform(
                    prng_keys[1], shape=(num_samps, out_dims, self.RFF_num_feats)
                )
            )

            samples = self.mean[None, None, :] + amplitude * jnp.sqrt(
                2.0 / self.RFF_num_feats
            ) * (jnp.cos((ks[None, ...] * x[..., None, :]).sum(-1) + phi)).sum(
                -1
            )  # (time, num_samps, out_dims)

        else:
            Kxx = vmap(self.kernel.K, (1, None, None), 1)(
                x.transpose(2, 1, 0, 3), None, False
            )  # (out_dims, num_samps, time, time)
            eps_I = jnp.broadcast_to(jitter * jnp.eye(ts), (out_dims, 1, ts, ts))
            cov = Kxx + eps_I
            mean = jnp.broadcast_to(
                self.mean[:, None, None, None], (out_dims, num_samps, ts, 1)
            )
            samples = sample_gaussian_noise(prng_state, mean, cov)[..., 0].transpose(
                2, 1, 0
            )

        return samples
    
    
    
    
class LTI(module):
    """
    Gaussian  Linear Time-Invariant System
    
    Temporal multi-output kernels have separate latent processes that can be coupled.
    Spatiotemporal kernel modifies the process noise across latent processes, but dynamics uncoupled.
    Multi-output GPs generally mix latent processes via dynamics as well.
    """
    
    markov_kernel: MarkovianKernel
        
    fixed_grid_locs: bool
    site_locs: jnp.ndarray
        
    site_obs: jnp.ndarray
    site_Lcov: jnp.ndarray

    def __init__(self, markov_kernel, site_locs, site_obs, site_Lcov, fixed_grid_locs):
        """
        :param module markov_kernel: (hyper)parameters of the state space model
        :param jnp.ndarray site_locs: means of shape (time, out, 1)
        :param jnp.ndarray site_obs: means of shape (time, out, 1)
        :param jnp.ndarray site_Lcov: covariances of shape (time, out, out)
        """
        self.markov_kernel = markov_kernel
        
        self.fixed_grid_locs = fixed_grid_locs
        self.site_locs = site_locs
        
        self.site_obs = site_obs
        self.site_Lcov = site_Lcov
        
        
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
    
        
    def get_timedata(self):
        if self.fixed_grid_locs:
            locs = lax.stop_gradient(self.site_locs)
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
        timedata = self.get_timedata()
        interp_sites = interpolation_transitions(
            t_eval, timedata[0], self.markov_kernel.state_transition)
        
        # compute linear dynamical system
        H, minf, Pinf, As, Qs = self.markov_kernel.get_LDS(timedata)
        
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
            f_sample: the prior samples (time, num_samps, state_dims, 1)
        """
        dt = jnp.diff(t_eval)
        if jnp.all(jnp.isclose(dt, dt[0])):  # grid
            timedata = (t_eval, dt[:1])
        else:
            timedata = (t_eval, dt)
        
        eps_I = jitter * jnp.eye(self.markov_kernel.state_dims)
        tsteps = timedata[0].shape[0]
        
        # transition and noise process matrices
        H, minf, Pinf, As, Qs = self.markov_kernel.get_LDS(timedata)

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
        timedata = self.get_timedata()
        
        # evaluation locations
        t_all, site_ind, eval_ind = get_evaldata(t_eval, timedata[0])
        t_site, t_eval = t_all[site_ind], t_all[eval_ind]
        interp_sites = interpolation_transitions(
            t_eval, t_site, self.markov_kernel.state_transition)
        
        # compute linear dynamical system
        H, minf, Pinf, As, Qs = self.markov_kernel.get_LDS(timedata)

        # sample prior at obs and eval locs
        prng_keys = jr.split(prng_state, 2)
        prior_samps = self.sample_prior(
            prng_keys[0], num_samps, t_all, jitter
        )
        
        # posterior mean
        post_means, _, KL_ss = evaluate_LGSSM_posterior(
            H, minf, Pinf, As, Qs, self.site_obs, self.site_Lcov, 
            interp_sites, mean_only=True, compute_KL=compute_KL, jitter=jitter,
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
            smoothed_sample, _, _ = evaluate_LGSSM_posterior(
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