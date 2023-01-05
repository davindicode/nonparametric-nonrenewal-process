import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import lax, vmap
from jax.scipy.linalg import expm

from ..base import module
from ..utils.jax import safe_log

from .base import SSM
from .kernels import MarkovianKernel
from .linalg import bdiag
from .markovian import interpolation_times


### HMM ###
class DTMarkovProcess(module):
    """
    Discrete-time finite discrete state with Markovian transition dynamics
    
    Regular grid of time points
    """
    
    pre_T: jnp.ndarray  # pre-softmax
    site_ll: jnp.ndarray  # pseudo-observation log probabilities
    K: int  # state dimension

    def __init__(self, pre_T, site_ll, array_type=jnp.float32):
        """
        :param jnp.ndarray T: transition matrix of shape (K, K)
        :param jnp.ndarray site_ll: pseudo-observation log probabilities (ts, K)
        """
        super().__init__(array_type)
        self.pre_T = self._to_jax(pre_T)
        self.site_ll = self._to_jax(site_ll)
        self.K = pre_T.shape[-1]

    def compute_chain(self):
        """
        Solve overparameterized augmented linear equation via QR decomposition
        """
        log_T = jax.nn.log_softmax(self.pre_T, axis=0)

        P = jnp.exp(log_T) - jnp.eye(self.K)
        A = jnp.concatenate((P, jnp.ones(self.K)[None, :]), axis=0)
        b = jnp.zeros(self.K + 1).at[-1].set(1.0)

        # Q, R = jnp.linalg.qr(A)
        # jax.scipy.linalg.solve_triangular(R, jnp.Q.T @ b)
        pi = jnp.linalg.pinv(A) @ b  # via pinv
        return pi, log_T
    
    def filter_observations(self, log_p0, obs_ll, reverse, compute_marginal):
        """
        Filtering observations with discrete variable, message passing to marginalize efficiently.

        :param jnp.ndarray log_p0: initial probability of shape (K,)
        :param jnp.ndarray obs_ll: observation log likelihoods per latent state of shape (ts, K)
        """
        pi, log_T = self.compute_chain()
        log_pi = safe_log(pi)
        
        def step(carry, inputs):
            log_p_ahead, log_marg = carry  # log p(z_t | x_{<t})
            ll = inputs

            log_normalizer = jax.nn.logsumexp(log_p_ahead + ll)  # log p(x_t | x_{<t})
            if compute_marginal:
                log_marg += log_normalizer

            # log p(z_t | x_{<=t}) = log[ p(z_t | x_{<t}) * p(x_t|z_t) / p(x_t | x_{<t}) ]
            log_p_at = (log_p_ahead + ll) - log_normalizer  # (K,)

            # log[ \sum_{z_t} p(z_t|x_{<=t}) * p(z_{t+1}|z_t) ]
            log_p_ahead = jax.nn.logsumexp(log_p_at[None, :] + log_T, axis=-1)  # (K,)

            return (log_p_ahead, log_marg), log_p_at

        (_, log_marg), log_p_ats = lax.scan(
            step, init=(log_pi, 0.0), xs=obs_ll, reverse=reverse
        )
        return log_p_ats, log_marg

    def evaluate_posterior(self, obs_ll, compute_KL):
        """
        Forward-backward algorithm

        :param jnp.array obs_ll: observation log likelihoods of shape (ts, K)
        :returns:
            means of shape (out_dims, num_samps, time, 1)
            covariances of shape (out_dims, num_samps, time, time)
        """
        pi, log_T = self.compute_chain()
        log_pi = safe_log(pi)
        
        log_betas, log_marg = self.filter_observations(
            log_pi, obs_ll, True, compute_KL
        )  # backward

        # combine forward with backward sweep for marginals
        def step(carry, inputs):
            log_alpha, var_expect = carry  # log p(z_t | x_{<t})
            log_beta, ll = inputs

            log_post = log_alpha + log_beta  # log[ p(z_t | x_{<t}) * p(z_t | x_{>=t})]
            if compute_KL:
                var_expect += (jnp.exp(log_post) * ll).sum()

            # log p(z_t | x_{<=t}) = log[ p(z_t | x_{<t}) * p(x_t|z_t) / p(x_{<=t}) ]
            log_p_at = jax.nn.log_softmax(log_alpha + ll)  # (K,)
            # log[ \sum_{z_t} p(z_t|x_{<=t}) * p(z_{t+1}|z_t) ]
            log_alpha = jax.nn.logsumexp(log_p_at[None, :] + log_T, axis=-1)  # (K,)

            return (log_alpha, var_expect), log_post

        (_, var_expect), log_posts = lax.scan(
            step, init=(log_pi, 0.0), xs=(log_betas, obs_ll), reverse=False
        )  # (ts, K)

        KL = log_marg - var_expect
        aux = (log_marg, log_betas, log_pi, log_T)
        return log_posts, KL, aux

    def sample_prior(self, prng_state, num_samps, timesteps):
        """
        Sample from the HMM latent variables.

        :param jnp.ndarray p0: initial probabilities of shape (num_samps, state_dims)
        :returns: state tensor of shape (trials, time)
        :rtype: torch.tensor
        """
        pi, log_T = self.compute_chain()
        log_pi = safe_log(pi)

        prng_states = jr.split(prng_state, num_samps)  # (num_samps, 2)

        def step(carry, inputs):
            log_p_cond = carry  # log p(z_t | z_{<t})
            prng_state = inputs

            s = jr.categorical(prng_state, log_p_cond)  # (K,)
            log_p_cond = log_T[:, s] + log_p_cond[s]  # next step probabilities
            return log_p_cond, s

        def sample_i(prng_state):
            prng_keys = jr.split(prng_state, timesteps)
            _, states = lax.scan(step, init=log_pi, xs=prng_keys)
            return states

        states = vmap(sample_i, 0, 1)(prng_states)
        return states  # (time, tr, K)

    def sample_posterior(self, prng_state, num_samps, timesteps, compute_KL):
        """
        Forward-backward sampling algorithm

        :param jnp.array x: input of shape (time, num_samps, in_dims, 1)
        :returns:
            means of shape (out_dims, num_samps, time, 1)
            covariances of shape (out_dims, num_samps, time, time)
        """
        log_posts, KL, aux = self.evaluate_posterior(self.site_ll, compute_KL)
        log_marg, log_betas, log_pi, log_T = aux

        prng_states = jr.split(prng_state, num_samps)  # (num_samps, 2)

        # combine forward with backward sweep
        def step(carry, inputs):
            log_p_cond = carry  # log p(z_t | z_{<t}, x_{1:T})
            log_beta, prng_state = inputs

            log_p = log_beta + log_p_cond
            s = jr.categorical(prng_state, log_p_cond)  # (K,)
            log_p_cond = log_T[:, s] + log_p_cond  # next step probabilities

            return log_p_cond, s

        def sample_i(prng_state):
            prng_keys = jr.split(prng_state, timesteps)
            _, states = lax.scan(
                step, init=jnp.zeros(self.K), xs=(log_betas, prng_keys), reverse=False
            )  # (ts, K)
            return states

        states = vmap(sample_i, 0, 1)(prng_states)  # (time, tr, K)
        return states, KL


class CTMarkovProcess(module):
    """
    Continuous-time finite discrete state with Markovian transition dynamics
    """

    Q_f: jnp.ndarray  # Q is Q_f with right column to make Q.sum(-1) = 0

    fixed_grid_locs: bool
    site_locs: jnp.ndarray
    site_ll: jnp.ndarray  # pseudo-observation log probabilities
    K: int  # state dimension

    def __init__(self, Q_f, site_locs, site_ll, fixed_grid_locs=False, array_type=jnp.float32):
        """
        :param jnp.ndarray Q_f: transition matrix of shape (K, K-1)
        :param jnp.ndarray site_ll: pseudo-observation log probabilities (ts, K)
        """
        super().__init__(array_type)
        self.Q_f = self._to_jax(Q_f)

        self.fixed_grid_locs = fixed_grid_locs
        self.site_locs = self._to_jax(site_locs)
        self.site_ll = self._to_jax(site_ll)
        self.K = pre_T.shape[-1]
        
    def get_site_locs(self):
        if self.fixed_grid_locs:
            locs = lax.stop_gradient(self.site_locs)  # (ts,)
            return locs, locs[1:2] - locs[0:1]
        else:
            return self.site_locs, jnp.diff(self.site_locs)

    def compute_chain(self, dt):
        """
        Solve overparameterized augmented linear equation via QR decomposition
        """
        Q = jnp.concatenate((self.Q_f, -self.Q_f.sum(1, keepdims=True)), axis=1)
        T = expm(dt[:, None, None] * Q[None, ...])  # (ts, sd, sd)

        P = T - jnp.eye(self.K)
        A = jnp.concatenate((P, jnp.ones(self.K)[None, :]), axis=0)
        b = jnp.zeros(self.K + 1).at[-1].set(1.0)

        # Q, R = jnp.linalg.qr(A)
        # jax.scipy.linalg.solve_triangular(R, jnp.Q.T @ b)
        pi = jnp.linalg.pinv(A) @ b  # via pinv
        return pi


### switch transitions ###
class Transition(module):
    """
    Transition matrices
    """

    state_dims: int
    out_dims: int

    def __init__(self, out_dims, state_dims, array_type):
        """
        :parma jnp.ndarray A: transition matrices (num_switches, sd, sd)
        :parma jnp.ndarray Q: process noise matrices (num_switches, sd, sd)
        """
        super().__init__(array_type)
        self.state_dims = state_dims
        self.out_dims = out_dims
        
    def state_transition(self):
        """
        Block diagonal from (out_dims, x_dims, x_dims) to (state_dims, state_dims)

        :param jnp.ndarray dt: time intervals of shape broadcastable with (out_dims,)
        """
        A, Q = self._state_transition()  # vmap over output dims
        return vmap(bdiag, 1, 0)(A), vmap(bdiag, 1, 0)(Q)


class WhiteNoiseSwitch(Transition):
    """
    White noise processes at different orders
    Matches the diagonal stationary covariance of the Matern with corresponding order
    """
    
    noise_vars: jnp.ndarray

    def __init__(self, out_dims, state_dims, noise_vars, array_type=jnp.float32):
        """
        :param jnp.ndarray noise_vars: (out_dims, orders)
        """
        super().__init__(out_dims, state_dims, array_type)
        self.noise_vars = self._to_jax(noise_vars)
        
    @eqx.filter_vmap()
    def _state_transition(self):
        """
        :param dt: step size(s), Δtₙ = tₙ - tₙ₋₁ [scalar]
        :param hyperparams: the kernel hyperparameters, lengthscale is in index 1 [2]
        :return:
            state transition matrix A (orders, sd, sd)
        """
        O = len(self.noise_vars)
        A = jnp.zeros((O, self.state_dims, self.state_dims))
        Q = jnp.zeros((O, self.state_dims, self.state_dims))
        for o in range(O + 1):  # unroll, last order is identity transition
            inds = jnp.arange(o)
            A = A.at[o, inds, inds].set(1.)
            if o < O:
                Q = Q.at[o, o, o].set(self.noise_vars[o])
            
        return A, Q


### switching LGSSM ###
class SwitchingLTI(SSM):
    """
    In the switching formulation, every observation point is doubled to account
    for infinitesimal switches inbetween
    """

    switch_locs: jnp.ndarray
    state_HMM: DTMarkovProcess
    switch_HMM: DTMarkovProcess
        
    switch_transitions: Transition
    markov_kernel_group: MarkovianKernel
        
    fixed_grid_locs: bool
    
    def __init__(
        self,
        switch_locs, 
        state_HMM,
        switch_HMM,
        markov_kernel_group,
        switch_transitions,
        site_locs,
        site_obs,
        site_Lcov,
        fixed_switch_locs=False,
        fixed_grid_site_locs=False,
    ):
        """
        :param jnp.ndarray switch_locs: locations of switches (K, K, switches)
        """
        super().__init__(site_locs, site_obs, site_Lcov)
        self.switch_locs = self._to_jax(switch_locs)
        self.state_HMM = state_HMM
        self.switch_HMM = switch_HMM
        
        self.switch_transitions = switch_transitions
        self.markov_kernel_group = markov_kernel_group
        
        self.fixed_grid_locs = fixed_grid_locs

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
    
    ### site and switch locations ###
    def get_site_locs(self):
        """
        Both site obervation locs and switch locs contribute the site locations
        """
        locs = jnp.concatenate((self.site_locs, self.switch_locs), axis=0)
        sort_ind = jnp.argsort(locs)
        switch_inds = sort_ind[self.site_locs.shape[0]:]
        locs = locs[sort_ind]
        
        switch_inds += jnp.arange(len(switch_inds))  # increase as we insert
        obs_inds = jnp.delete(jnp.arange(locs.shape[0]), switch_inds)
        
        if self.fixed_grid_locs:
            locs = lax.stop_gradient(locs)  # (ts,)
            return locs, obs_inds, switch_inds, locs[1:2] - locs[0:1]
        else:
            return locs, obs_inds, switch_inds, jnp.diff(locs)
        
    def get_site_params(self, obs_inds):
        """
        Insert unobserved locations at switch edge locations
        """
        sd = self.site_obs.shape[-2]
        ts = len(obs_inds) + len(switch_inds)
        
        site_obs = jnp.zeros((ts, sd, 1))
        site_Lcov = 1e6 * jnp.eye(sd)[None, ...].repeat(ts, axis=0)  # empty observations
        
        site_obs = site_obs.at[obs_inds].set(self.site_obs)
        site_Lcov = site_Lcov.at[obs_inds].set(self.site_Lcov)
        return site_obs, site_Lcov

    def get_conditional_LDS(self, dt, tsteps, trans_inds, switch_inds, s_path, j_path):
        """
        Insert switches inbetween observation sites
        Each observation site has a state
        Each switch applies from A_{s}(t/2) * A_{ss'} * A_{s'}(t/2) with empty 
        observation at the * locations
        
        :param jnp.ndarray s_path: switch state trajectories (num_samps, s_dims)
        """
        switches, transitions, sd = len(switch_inds), len(trans_inds), As.shape[-1]
        As = jnp.empty((transitions + switches, sd, sd))
        Qs = jnp.empty((transitions + switches, sd, sd))
        
        # use out_dims of kernel as states
        H, minf, Pinf, As_, Qs_ = self.markov_kernel._get_LDS(dt, tsteps)  # (states, ts, sd, sd)
        As_, Qs_ = As_[s_path], Qs_[s_path]
        
        # jump matrices
        Ajs, Qjs = self.transitions._state_transition()
        Ajs, Qjs = Ajs[j_path], Qjs[j_path]
        
        # insert matrices
        As = As.at[trans_inds].set(As_).at[switch_inds].set(Ajs)
        Qs = Qs.at[trans_inds].set(Qs_).at[switch_inds].set(Qjs)
        
        # append boundary transitions
        Id = jnp.eye(As.shape[-1])
        Zs = jnp.zeros_like(Id)
        As = jnp.concatenate((Id[None, ...], As, Id[None, ...]), axis=0)
        Qs = jnp.concatenate((Zs[None, ...], Qs, Zs[None, ...]), axis=0)

        return H, minf, Pinf, As, Qs

    ### posterior ###
    def evaluate_conditional_posterior(self, t_eval, s_path):
        """
        Compute posterior conditioned on HMM path
        """
        site_locs, obs_inds, switch_inds, site_dlocs = self.get_site_locs()
        switches = len(switch_inds)
        
        # double the switch locs in site locs, left and right limit of switch point
        site_locs_ = jnp.empty((site_locs.shape[0] + switches,))
        site_locs_ = site_locs_.at[obs_inds]
        
        ind_eval, dt_fwd, dt_bwd = interpolation_times(t_eval, site_locs)
        
        # sample from HMMs
        s_path, KL_s = self.switch_HMM.sample_posterior()
        j_path, KL_j = self.switch_HMM.sample_posterior()
        
        site_obs, site_Lcovs = self.get_site_params(obs_inds)
        H, Pinf, As, Qs = self.get_conditional_LDS(
            dt, tsteps, trans_inds, switch_inds, s_path, j_path)
        
        post_means, post_covs, KL = evaluate_LTI_posterior(
            H,
            Pinf,
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
        return
    
    def evaluate_posterior(self, t_eval, mean_only, compute_KL, jitter):
        """
        predict at test locations X, which may includes training points
        (which are essentially fixed inducing points)

        :param jnp.ndarray t_eval: evaluation times of shape (time,)
        :return:
            means of shape (time, out_dims, 1)
            covariances of shape (time, out_dims, out_dims)
        """
        num_evals = t_eval.shape[0]

        site_locs, site_dlocs = self.get_site_locs()
        ind_eval, dt_fwd, dt_bwd = interpolation_times(t_eval, site_locs)

        stack_dt = jnp.concatenate([dt_fwd, dt_bwd])  # (2*num_evals,)
        stack_A = vmap(self.markov_kernel.state_transition)(
            stack_dt
        )  # vmap over num_evals
        A_fwd, A_bwd = stack_A[:num_evals], stack_A[-num_evals:]

        # compute linear dynamical system
        H, minf, Pinf, As, Qs = self.markov_kernel.get_LDS(
            site_dlocs[None, :], site_locs.shape[0]
        )

        
        return post_means, post_covs, KL
    
    ### sample ###
    def sample_prior(self, prng_state, num_samps, t_eval, jitter):
        """
        Sample from the model prior f~N(0,K) via simulation of the LGSSM

        :param int num_samps: number of samples to draw
        :param jnp.ndarray t_eval: the input locations at which to sample (out_dims, locs,)
        :return:
            f_sample: the prior samples (num_samps, time, out_dims, 1)
        """
        prng_state, num_samps, timesteps = self.state_HMM.sample_prior()
        self.switch_HMM.sample_prior()
        
        H, minf, Pinf, As, Qs = self.get_conditional_LDS()
        
        
        tsteps = t_eval.shape[0]
        dt = compute_dt(t_eval)  # (eval_locs,)

        H, minf, Pinf, As, Qs = self.get_conditional_LDS(dt[None, :], tsteps)
        samples = sample_LGSSM(H, minf, Pinf, As, Qs, prng_state, num_samps, jitter)
        x_samples = samples.transpose(1, 0, 2, 3)
        
        return x_samples, s_samples

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
        if t_eval is not None:
            num_evals = t_eval.shape[0]
            ind_eval, dt_fwd, dt_bwd = interpolation_times(t_eval, site_locs)

            stack_dt = jnp.concatenate([dt_fwd, dt_bwd])  # (2*num_evals,)
            stack_A = vmap(self.markov_kernel.state_transition)(
                stack_dt
            )  # vmap over num_evals
            A_fwd, A_bwd = stack_A[:num_evals], stack_A[-num_evals:]
            
        else:
            ind_eval, A_fwd, A_bwd = None, None, None

        # compute linear dynamical system
        H, minf, Pinf, As, Qs = self.markov_kernel.get_LDS(
            site_dlocs[None, :], site_locs.shape[0]
        )

        # sample prior at obs and eval locs
        prng_keys = jr.split(prng_state, 2)
        prior_samps = self.sample_prior(
            prng_keys[0], num_samps, t_all, jitter
        )  # (time, num_samps, out_dims, 1)

        # posterior mean
        post_means, _, KL_ss = evaluate_LTI_posterior(
            H,
            minf,
            Pinf,
            As,
            Qs,
            self.site_obs,
            self.site_Lcov,
            ind_eval,
            A_fwd,
            A_bwd,
            mean_only=True,
            compute_KL=compute_KL,
            jitter=jitter,
        )  # (time, out_dims, 1)

        # noisy prior samples at eval locs
        prior_samps_t = prior_samps[:, site_ind]
        prior_samps_eval = prior_samps[:, eval_ind]

        prior_samps_noisy = prior_samps_t + self.site_Lcov[None, ...] @ jr.normal(
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
                ind_eval,
                A_fwd,
                A_bwd,
                mean_only=True,
                compute_KL=False,
                jitter=jitter,
            )
            return smoothed_sample

        smoothed_samps = vmap(smooth_prior_sample, 0, 0)(prior_samps_noisy)

        # Matheron's rule pathwise samplig
        return prior_samps_eval - smoothed_samps + post_means[None, ...], KL_ss
