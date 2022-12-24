import jax
from jax import lax, vmap
import jax.random as jr
import jax.numpy as jnp


from ..base import module
from .base import LTI
from .markovian import LGSSM



# HMM
class CTHMM(module):
    """
    Continuous-time finite discrete state with Markovian transition dynamics
    """
    
    pre_T: jnp.ndarray  # pre-softmax
        
    fixed_grid_locs: bool
    site_locs: jnp.ndarray
    site_ll: jnp.ndarray  # pseudo-observation log probabilities
    K: int  # state dimension
    
    def __init__(self, pre_T, site_ll):
        """
        :param jnp.ndarray T: transition matrix of shape (K, K)
        :param jnp.ndarray site_ll: pseudo-observation log probabilities (ts, K)
        """
        self.pre_T = pre_T
        
        self.fixed_grid_locs = fixed_grid_locs
        self.site_locs = site_locs
        self.site_ll = site_ll
        self.K = pre_T.shape[-1]
    
    
    def compute_stationary_distribution(self):
        """
        Solve overparameterized augmented linear equation via QR decomposition
        """
        T = jax.nn.softmax(self.pre_T, axis=0)
        
        P = T - jnp.eye(self.K)
        A = jnp.concatenate((P, jnp.ones(self.K)[None, :]), axis=0)
        b = jnp.zeros(self.K+1).at[-1].set(1.)
        
        #Q, R = jnp.linalg.qr(A)
        #jax.scipy.linalg.solve_triangular(R, jnp.Q.T @ b)
        pi = jnp.linalg.pinv(A) @ b  # via pinv
        return pi
        
    
    def filter_observations(self, log_p0, obs_ll, reverse, compute_marginal):
        """
        Filtering observations with discrete variable, message passing to marginalize efficiently.
        
        :param jnp.ndarray log_p0: initial probability of shape (K,)
        :param jnp.ndarray obs_ll: observation log likelihoods per latent state of shape (ts, K)
        """
        log_T = jax.nn.log_softmax(self.pre_T, axis=0)
        
        def step(carry, inputs):
            log_p_ahead, log_marg = carry  # log p(z_t | x_{<t})
            ll = inputs
            
            log_normalizer = jax.nn.logsumexp(log_p_ahead + ll)  # log p(x_t | x_{<t})
            if compute_marginal:
                log_marg += log_normalizer
                
            # log p(z_t | x_{<=t}) = log[ p(z_t | x_{<t}) * p(x_t|z_t) / p(x_t | x_{<t}) ]
            log_p_at = (log_p_ahead + ll) - log_normalizer  # (K,)
            
            # log[ \sum_{z_t} p(z_t|x_{<=t}) * p(z_{t+1}|z_t) ]
            log_p_ahead = jax.nn.logsumexp(
                log_p_at[None, :] + log_T, axis=-1)  # (K,)
            
            return (log_p_ahead, log_marg), log_p_at
                
        (_, log_marg), log_p_ats = lax.scan(step, init=(log_p0, 0.), xs=obs_ll, reverse=reverse)
        return log_p_ats, log_marg
    
    
    def evaluate_posterior(
        self, obs_ll, compute_KL
    ):
        """
        Forward-backward algorithm
        
        :param jnp.array obs_ll: observation log likelihoods of shape (ts, K)
        :returns:
            means of shape (out_dims, num_samps, time, 1)
            covariances of shape (out_dims, num_samps, time, time)
        """
        pi = self.compute_stationary_distribution()
        log_pi = jnp.log(jnp.maximum(pi, 1e-12))
        log_T = jax.nn.log_softmax(self.pre_T, axis=0)  # transition matrix
        
        log_betas, log_marg = self.filter_observations(log_pi, obs_ll, True, compute_KL)  # backward
        
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
            log_alpha = jax.nn.logsumexp(
                log_p_at[None, :] + log_T, axis=-1)  # (K,)
            
            return (log_alpha, var_expect), log_post
                
        (_, var_expect), log_posts = lax.scan(
            step, init=(log_pi, 0.), xs=(log_betas, obs_ll), reverse=False)  # (ts, K)
        
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
        pi = self.compute_stationary_distribution()
        log_pi = jnp.log(jnp.maximum(pi, 1e-12))
        
        prng_states = jr.split(prng_state, num_samps)  # (num_samps, 2)
        log_T = jax.nn.log_softmax(self.pre_T, axis=0)  # transition matrix

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
                step, init=jnp.zeros(self.K), xs=(log_betas, prng_keys), reverse=False)  # (ts, K)
            return states
        
        states = vmap(sample_i, 0, 1)(prng_states)  # (time, tr, K)
        return states, KL
    
    
    
# switching LGSSM
class SwitchingLTI(LTI):
    """
    In the switching formulation, every observation point is doubled to account 
    for infinitesimal switches inbetween
    """
    
    CTHMM: module
    
    def __init__(self, CTHMM, markov_kernel_group, site_locs, site_obs, site_Lcov, fixed_grid_locs=False):
        super().__init__(markov_kernel_group, site_locs, site_obs, site_Lcov, fixed_grid_locs)
        self.CTHMM = CTHMM
    
    def sample_prior(self):
        """
        Sample from the joint prior
        """
        prng_state, num_samps, timesteps = self.CTHMM.sample_prior()
    
    
    def evaluate_conditional_posterior(self):
        return
    
    def sample_posterior(self):
        """
        Sample from joint posterior
        """
        return