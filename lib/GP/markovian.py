import math

from ..base import module

from jax import vmap, lax
import jax.numpy as jnp
import jax.random as jr
from jax.numpy.linalg import cholesky

import equinox as eqx

from .linalg import get_LGSSM_matrices, id_kronecker, evaluate_LGSSM_posterior

_log_twopi = math.log(2 * math.pi)



def get_evaldata(t_eval, timedata):
    """ """
    t = timedata[0]
    tsteps = t.shape[0]  # assume timedata is ordered and unique

    t_all = jnp.concatenate([t, t_eval], axis=0)
    t_all, input_ind = jnp.unique(t_all, return_inverse=True)

    sort_ind = jnp.argsort(t_all, axis=0)
    t_all = t_all[sort_ind]
    all_timedata = (t_all, jnp.diff(t_all))
    t_ind, eval_ind = sort_ind[input_ind[:tsteps]], sort_ind[input_ind[tsteps:]]
    return all_timedata, t_ind, eval_ind
    


### classes ###
class LGSSM(module):
    """
    Linear Gaussian State Space Model
    
    Temporal multi-output kernels have separate latent processes that can be coupled.
    Spatiotemporal kernel modifies the process noise across latent processes, but dynamics uncoupled.
    Multi-output GPs generally mix latent processes via dynamics as well.
    """
    
    markov_kernel: module
        
    site_obs: jnp.ndarray
    site_Lcov: jnp.ndarray

    def __init__(self, markov_kernel, site_obs, site_Lcov):
        """
        :param module markov_kernel: (hyper)parameters of the state space model
        :param jnp.ndarray site_obs: means of shape (time, out, 1)
        :param jnp.ndarray site_Lcov: covariances of shape (time, out, out)
        """
        self.markov_kernel = markov_kernel
        
        self.site_obs = site_obs
        self.site_Lcov = site_Lcov

    def get_LDS(self, timedata, Pinf):
        t, dt = timedata
        if dt.shape[0] == 1:
            A = self.markov_kernel.state_transition(dt[0])
        else:
            A = vmap(self.markov_kernel.state_transition)(dt)
        return get_LGSSM_matrices(A, Pinf, t.shape[0])
    
    ### posterior ###
    def evaluate_posterior(
        self, t_eval, timedata, mean_only, compute_KL, jitter
    ):
        """
        predict at test locations X, which may includes training points
        (which are essentially fixed inducing points)

        :param jnp.ndarray t_eval: evaluation times of shape (locs,)
        :return:
            means of shape (time, out, 1)
            covariances of shape (time, out, out)
        """
        # compute linear dynamical system
        H, minf, Pinf = self.markov_kernel.state_output()
        As, Qs = self.get_LDS(timedata, Pinf)

        post_means, post_covs, KL = evaluate_LGSSM_posterior(
            t_eval, As, Qs, H, minf, Pinf, self.markov_kernel.state_transition, 
            timedata[0], self.site_obs, self.site_Lcov, 
            mean_only, compute_KL, jitter, 
        )
        return post_means, post_covs, KL

    ### sample ###
    def sample_prior(self, prng_state, num_samps, timedata, jitter):
        """
        Sample from the model prior f~N(0,K) multiple times using a nested loop.
        :param num_samps: the number of samples to draw [scalar]
        :param t: the input locations at which to sample (defaults to train+test set) [N_samp, 1]
        :return:
            f_sample: the prior samples [S, N_samp]
        """
        eps_I = jitter * jnp.eye(self.markov_kernel.state_dims)
        H, minf, Pinf = self.markov_kernel.state_output()

        # transition and noise process matrices
        tsteps = timedata[0].shape[0]
        As, Qs = self.get_LDS(timedata, Pinf)

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
        timedata,
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

        :param X: the sampling input locations [N, 1]
        :param num_samps: the number of samples to draw [scalar]
        :param seed: the random seed for sampling
        :return:
            the posterior samples (eval_locs, num_samps, N, 1)
        """
        if t_eval is None:
            all_timedata = timedata
            t_ind = jnp.arange(timedata[0].shape[0])
            eval_ind = t_ind
        else:
            evaldata = get_evaldata(t_eval, timedata)
            all_timedata, t_ind, eval_ind = evaldata
        t_eval = all_timedata[0][eval_ind]
        
        # compute linear dynamical system
        H, minf, Pinf = self.markov_kernel.state_output()
        As, Qs = self.get_LDS(timedata, Pinf)

        # sample prior at obs and eval locs
        prng_keys = jr.split(prng_state, 2)
        prior_samps = self.sample_prior(
            prng_keys[0], num_samps, all_timedata, jitter
        )
        
        # posterior mean
        post_means, _, KL_ss = evaluate_LGSSM_posterior(
            t_eval, As, Qs, H, minf, Pinf, self.markov_kernel.state_transition, 
            timedata[0], self.site_obs, self.site_Lcov, 
            mean_only=True, compute_KL=compute_KL, jitter=jitter,
        )  # (time, N, 1)

        # noisy prior samples at eval locs
        prior_samps_t, prior_samps_eval = (
            prior_samps[t_ind, ...],
            prior_samps[eval_ind, ...],
        )
        prior_samps_noisy = prior_samps_t + self.site_Lcov[:, None, ...] @ jr.normal(
            prng_keys[1], shape=prior_samps_t.shape
        )  # (time, tr, N, 1)

        # smooth noisy samples
        def smooth_prior_sample(prior_samp_i):
            smoothed_sample, _, _ = evaluate_LGSSM_posterior(
                t_eval,
                As, 
                Qs, 
                H, 
                minf, 
                Pinf, 
                self.markov_kernel.state_transition, 
                timedata[0], 
                prior_samp_i, 
                self.site_Lcov, 
                mean_only=True,
                compute_KL=False,
                jitter=jitter,
            )
            return smoothed_sample

        smoothed_samps = vmap(smooth_prior_sample, 1, 1)(prior_samps_noisy)
        
        # Matheron's rule pathwise samplig
        return prior_samps_eval - smoothed_samps + post_means[:, None, ...], KL_ss

    
    
class FullLGSSM(LGSSM):
    """
    Full state space LDS with Gaussian noise
    
    Temporal multi-output kernels have separate latent processes that can be coupled.
    Spatiotemporal kernel modifies the process noise across latent processes, but dynamics uncoupled.
    Multi-output GPs generally mix latent processes via dynamics as well.
    """
    
    diagonal_site: bool

    def __init__(self, kernel, site_obs, site_Lcov, diagonal_site=True):
        """
        :param module kernel: kernel module
        :param jnp.ndarray site_obs: site observations with shape (timesteps, x_dims, 1)
        :param jnp.ndarray site_Lcov: site covariance cholesky with shape (timesteps, x_dims, x_dims)
        """
        super().__init__(kernel, site_obs, site_Lcov)
        self.diagonal_site = diagonal_site
        
    def apply_constraints(self):
        """
        PSD constraint
        """
        model = jax.tree_map(lambda p: p, self)  # copy
        
        def update(Lcov):
            epdfunc = lambda x: enforce_positive_diagonal(x, lower_lim=1e-2)
            Lcov = vmap(epdfunc)(jnp.tril(Lcov))
            Lcov = jnp.triu(Lcov) if self.diagonal_site else Lcov
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
    
    
@eqx.filter_vmap(args=(None, 0, 0, 0, 0, 0, 0, None, 0, 0, None, None, None), out=(0, 0, 0))
def vmap_outdims(t_eval, As, Qs, H, minf, Pinf, kernel_state_transition, 
                 t_obs, site_obs, site_Lcov, mean_only, compute_KL, jitter):
    return evaluate_LGSSM_posterior(
        t_eval, As, Qs, H, minf, Pinf, kernel_state_transition, 
        t_obs, site_obs, site_Lcov, mean_only, compute_KL, jitter, 
    )


@eqx.filter_vmap(args=(None, None, None, None, None, None, None, None, -3, -3, None, None, None), out=(0, 0, 0))
def vmap_spatial(t_eval, As, Qs, H, minf, Pinf, kernel_state_transition, 
                 t_obs, site_obs, site_Lcov, mean_only, compute_KL, jitter):
    return vmap_outdims(
        t_eval, As, Qs, H, minf, Pinf, kernel_state_transition, 
        t_obs, site_obs, site_Lcov, mean_only, compute_KL, jitter, 
    )
    


class SpatiotemporalLGSSM(LGSSM):
    """
    Factorized spatial and temporal GP kernel with temporal markov kernel
    
    Temporal multi-output kernels have separate latent processes that can be coupled.
    Spatiotemporal kernel modifies the process noise across latent processes, but dynamics uncoupled.
    Multi-output GPs generally mix latent processes via dynamics as well.
    """
    
    spatial_MF: bool
    
    spatial_locs: jnp.ndarray
    spatial_kernel: module

    def __init__(self, temporal_kernel, spatial_kernel, spatial_locs, site_obs, site_Lcov, spatial_MF=True):
        """
        :param module temporal_kernel: markov kernel module
        :param module spatial_kernel: spatial kernel
        :param jnp.ndarray site_obs: site observations with shape (out_dims, timesteps, spatial_locs, 1)
        :param jnp.ndarray site_Lcov: site observations with shape (out_dims, timesteps, spatial_locs, spatial_locs or 1)
        """
        if spatial_MF:
            assert site_Lcov.shape[-1] == 1
        assert site_obs.shape[-2] == site_Lcov.shape[-2]  # spatial_locs
        super().__init__(temporal_kernel, site_obs, site_Lcov)
        self.spatial_kernel = spatial_kernel
        self.spatial_locs = spatial_locs
        
        self.spatial_MF = spatial_MF
        
    def get_LDS(self, timedata, Pinf):
        """
        :param jnp.ndarray Pinf: matrix of shape (out_dims, state_dims, state_dims)
        """
        t, dt = timedata
        if dt.shape[0] == 1:
            A = self.markov_kernel._state_transition(dt[0])  # (out_dims, state_dims, 1)
        else:
            A = vmap(self.markov_kernel._state_transition, 0, 1)(dt)  # (out_dims, timesteps, state_dims, 1)
        return vmap(get_LGSSM_matrices, (0, 0, None), (0, 0))(A, Pinf, t.shape[0])
        
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
    
    ### posterior ###
    def evaluate_posterior(
        self, t_eval, x_eval, timedata, mean_only, compute_KL, jitter
    ):
        """
        predict at test locations X, which may includes training points
        (which are essentially fixed inducing points)

        :param jnp.ndarray t_eval: evaluation times of shape (locs,)
        :return:
            means of shape (time, out, 1)
            covariances of shape (time, out, out)
        """
        num_spatial = self.spatial_locs.shape[-2]
        
        # compute LDS matrices
        H, minf, Pinf = self.markov_kernel._state_output()
        As, Qs = self.get_LDS(timedata, Pinf)
        
        if self.spatial_MF:  # vmap over temporal kernel out_dims = spatial_locs
            # vmap over spatial points
            post_means, post_covs, KL = vmap_spatial(
                t_eval, As, Qs, H, minf, Pinf, self.markov_kernel.state_transition, 
                timedata[0], self.site_obs[..., None], self.site_Lcov[..., None], 
                mean_only, compute_KL, jitter, 
            )  # (spatial_locs, out_dims, timesteps, 1)
            
            post_means = post_means.transpose(1, 2, 0, 3)
            post_covs = post_covs.transpose(1, 2, 0, 3)
            
        else:
            # kronecker structure
            H, Pinf = id_kronecker(num_spatial, H), id_kronecker(num_spatial, Pinf)
            minf = jnp.tile(minf, (1, num_spatial))
            As, Qs = id_kronecker(num_spatial, As), id_kronecker(num_spatial, Qs)
            print(As.shape)
            print(Pinf.shape)
            print(Qs.shape)
            post_means, post_covs, KL = vmap_outdims(
                t_eval, As, Qs, H, minf, Pinf, self.markov_kernel.state_transition, 
                timedata[0], self.site_obs, self.site_Lcov, 
                mean_only, compute_KL, jitter, 
            )  # (out_dims, timesteps, spatial_locs)
            
            print(post_means.shape)
            
        # TODO: if R is fixed, only compute B, C once
        C_krr, C_nystrom = sparse_conditional(x_eval)  # (out_dims, timesteps, spatial_locs, spatial_locs)
        W = B @ H
        test_mean = post_means @ W
        test_var = W @ post_covs @ transpose(W) + self.temporal_kernel.K(X, X) * C
        
        return post_means, post_covs, KL   
        
    ### sample ###
    def sample_prior(self, prng_state, num_samps, timedata, jitter):
        """
        Sample from the model prior f~N(0,K) multiple times using a nested loop.
        :param num_samps: the number of samples to draw [scalar]
        :param t: the input locations at which to sample (defaults to train+test set) [N_samp, 1]
        :return:
            f_sample: the prior samples [S, N_samp]
        """
        f_samples = super().sample_prior()  # (time, tr, state_dims, 1)
        return Lzz @ f_samples

    def sample_posterior(
        self,
        prng_state,
        num_samps,
        timedata,
        t_eval,
        jitter,
        compute_KL,
    ):
        raise NotImplementedError