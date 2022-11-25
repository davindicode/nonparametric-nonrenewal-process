from functools import partial
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import value_and_grad, lax, jit, random, vmap, tree_map
from jax import lax

from jax.scipy.linalg import cho_solve, solve_triangular
from jax.numpy.linalg import cholesky
    
import math
_log_twopi = math.log(2 * math.pi)




### inputs ###
def process_inputs(t, x_obs, y, dtype):
    """
    Order the inputs.
    :param t: training inputs (T,)
    :param x_obs: non-temporal coordinates (T, dims)
    :param y: observations at the training inputs (T, out)
    :return:
        dt_train: training step sizes, Δtₙ = tₙ - tₙ₋₁ (steps-1,)
        t_train: training inputs (steps,)
        x_obs_train: observation inputs
        y_train: training observations (steps, N)
    """
    assert t.shape[0] == y.shape[0] # matching number of time points
    ind = jnp.argsort(t, axis=0)
    
    t = jnp.array(t[ind, ...], dtype=dtype)
    y = jnp.array(y[ind, ...], dtype=dtype)
    dt = jnp.diff(t)
    
    if jnp.abs(jnp.diff(dt)).max() / dt.max() < 1e-2:  # uniform time grid
        dt = dt[:1]
        
    if x_obs is not None:
        if x_obs.ndim < 2:
            x_obs = x_obs.expand_dims(x_obs, 1)
        x_obs = jnp.array(x_obs[ind, ...], dtype=dtype)
    else:
        x_obs = None
    
    return t, dt, x_obs, y




### base ###
class SVI_batched(object):
    """
    """
    def __init__(self, dtype=jnp.float32):
        self.dtype = dtype

        
class SVI_state_space(object):
    """
    Class for state space priors, handles data sequentially
    """
    def __init__(self, state_space, dtype=jnp.float32):
        self.dtype = dtype
        self.x_dims = state_space.kernel.out_dims
        self.state_space = state_space
        
        self.y = None # no training data set
    
    def set_data(self, t, x_obs, y, mask=None):
        """
        :param t: training inputs
        :param y: training data / observations
        
        :param y: observed data [N, obs_dim]
        :param dt: step sizes Δtₙ = tₙ - tₙ₋₁ [N, 1]
        """
        self.t, self.dt, self.x_obs, self.y = process_inputs(t, x_obs, y, self.dtype)
        self.mask = mask
        
    def get_data(self):
        """
        Get data for Kalman filtering/smoothing
        """
        if self.y is None:
            raise ValueError('Training data not set with set_data()')
        return [self.t, self.dt, self.x_obs, self.y, self.mask]
    
    def get_x_posterior(self):
        raise NotImplementedError

        
        

### classes ###    
class IF_SSGP(SVI_state_space):
    """
    Integrate-and-fire state space model
    """
    def __init__(self, state_space, eps_mapping, mu_mapping, lsigma_mapping, IF_model, q_vh_ic, 
                 dtype=jnp.float32, jitter=1e-6):
        """
        The state-space conjugate-computation variational inference class.
        
        The stochastic differential equation (SDE) form of a Gaussian process (GP) model.
        Implements methods for inference and learning in models with GP priors of the form
            f(t) ~ GP(0,k(t,t'))
        using state space methods, i.e. Kalman filtering and smoothing.
        Constructs a linear time-invariant (LTI) stochastic differential equation (SDE) of the following form:
            dx(t)/dt = F x(t) + L w(t)
                  yₙ ~ p(yₙ | f(tₙ)=H x(tₙ))
        where w(t) is a white noise process and where the state x(t) is Gaussian distributed with initial
        state distribution x(t)~𝓝(0,Pinf).
        Combined likelihood and mapping to compute p(y|z;theta) becomes the observation model
        Refs:
            Chang, Wilkinson, Khan & Solin 2020 "Fast variational learning in state space Gaussian process models"
        
        :param prior: the model prior p(f)=GP(0,k(.,.)) object which constructs the required state space model matrices
        :param likelihood: the likelihood model object which performs parameter updates and evaluates p(y|f)
        """
        super().__init__(state_space)
        self.state_space = state_space   
        self.state_dims = state_space.kernel.state_dims
        
        self.eps_mapping = eps_mapping
        if self.eps_mapping.x_dims != self.x_dims:
            raise ValueError('Latent space dimensions do not match for state space and observation models')
        self.f_dims = self.eps_mapping.f_dims
        
        self.mu_mapping = mu_mapping
        self.lsigma_mapping = lsigma_mapping
        
        self.IF_model = IF_model
        self.q_vh_ic = q_vh_ic
        
        self.jitter = jitter
        
        
    ### getters and setters ###
    def set_params(self, params):
        """
        Hyperparmaeters to set, will be called in run functions
        """
        self.q_vh_ic = params['ic']
        self.state_space.params = params['state_space']
        self.eps_mapping.params = params['eps_mapping']
        self.mu_mapping.params = params['mu_mapping']
        self.lsigma_mapping.params = params['lsigma_mapping']
        self.IF_model.params = params['IF_model']
        
    def get_params(self):
        """
        Return a copy of parameters to be trained with SGD
        """
        return {'ic': self.q_vh_ic, 
                'state_space': self.state_space.params, 
                'eps_mapping': self.eps_mapping.params,
                'mu_mapping': self.mu_mapping.params,
                'lsigma_mapping': self.lsigma_mapping.params,
                'IF_model': self.IF_model.params}
    
    def set_var_params(self, var_params):
        """
        Site parameters are variational parameters trained with natural gradient CVI
        first Kalman pass will initialize site params
        """
        self.state_space.var_params = var_params['state_space']
        self.eps_mapping.var_params = var_params['eps_mapping']
        self.mu_mapping.var_params = var_params['mu_mapping']
        self.lsigma_mapping.var_params = var_params['lsigma_mapping']
        
    def get_var_params(self):
        """
        Return a copy of parameters to be trained with SGD
        """
        return {'state_space': self.state_space.var_params, 
                'eps_mapping': self.eps_mapping.var_params,
                'mu_mapping': self.mu_mapping.var_params,
                'lsigma_mapping': self.lsigma_mapping.var_params}
    
    def set_all_params(self, all_params):
        params, var_params = all_params['hyp'], all_params['sites']
        self.set_params(params)
        self.set_var_params(var_params)
    
    def get_all_params(self):
        params = self.get_params()
        var_params = self.get_var_params()
        return {'hyp': params, 'sites': var_params}
    
    
    ### inference ###
    def constraints(self, all_params):
        """
        PSD constraint
        """
        params, var_params = all_params['hyp'], all_params['sites']
        params['state_space'], var_params['state_space'] = self.state_space.constraints(
            params['state_space'], var_params['state_space'])
        params['eps_mapping'], var_params['eps_mapping'] = self.eps_mapping.constraints(
            params['eps_mapping'], var_params['eps_mapping'])
        params['mu_mapping'], var_params['mu_mapping'] = self.mu_mapping.constraints(
            params['mu_mapping'], var_params['mu_mapping'])
        params['lsigma_mapping'], var_params['lsigma_mapping'] = self.lsigma_mapping.constraints(
            params['lsigma_mapping'], var_params['lsigma_mapping'])
        params['IF_model'] = self.IF_model.constraints(params['IF_model'])
        return {'hyp': params, 'sites': var_params}
    
    
    @partial(jit, static_argnums=(0, 8))
    def simulate_spiketrains(self, params, var_params, prng_state, dt, x_samples, x_obs, jitter, prior):
        """
        :param jnp.array q_vh_ic: neuron state initial conditions of shape (tr, N, state_dims)
        :param jnp.array x_samples: noise process samples of shape (time, tr, x_dims)
        :param jnp.array x_obs: observation time series (time, tr, obs_dims)
        """
        q_vh_ic = params['ic']
        eps_mapping = params['eps_mapping']
        mu_params = params['mu_mapping']
        lsigma_params = params['lsigma_mapping']
        IF_params = params['IF_model']
        prng_keys = jr.split(prng_state, 4)
        
        if prior:
            eps_samples = self.eps_mapping.sample_prior(
                eps_mapping, prng_keys[0], x_samples, jitter)
            mu = self.mu_mapping.sample_prior(
                mu_params, prng_keys[1], x_obs, jitter)
            log_sigma = self.lsigma_mapping.sample_prior(
                lsigma_params, prng_keys[2], x_obs, jitter)
            
        else:
            eps_var_params = var_params['eps_mapping']
            mu_var_params = var_params['mu_mapping']
            lsigma_var_params = var_params['lsigma_mapping']
        
            eps_samples, _ = self.eps_mapping.sample_posterior(
                eps_mapping, eps_var_params, prng_keys[0], x_samples, jitter, False)  # (time, tr, N)
            mu, _ = self.mu_mapping.sample_posterior(
                mu_params, mu_var_params, prng_keys[1], x_obs, jitter, False)
            log_sigma, _ = self.lsigma_mapping.sample_posterior(
                lsigma_params, lsigma_var_params, prng_keys[2], x_obs, jitter, False)
        
        I = mu + jnp.exp(log_sigma) * eps_samples
        y, q_vh = self.IF_model.Euler_simulate(
            IF_params, prng_keys[3], dt, q_vh_ic, I)
        return y, q_vh, I, eps_samples
    
    
    @partial(jit, static_argnums=(0, 4))
    def filter_smoother(self, learned_all_params, fixed_all_params, prng_state, num_samps):
        """
        Run Kalman filtering and RTS smoothing to obtain the ELBO
        """
        all_params = tree_map(
            lambda x, y: x if x is not None else y, 
            learned_all_params, fixed_all_params, is_leaf=lambda x: x is None
        )
        
        params, var_params = all_params['hyp'], all_params['sites']
        jitter = self.jitter
        
        # parameters
        q_vh_ic = params['ic']
        IF_params = params['IF_model']
        
        ss_params, ss_var_params = params['state_space'], var_params['state_space']
        eps_mapping, eps_var_params = params['eps_mapping'], var_params['eps_mapping']
        mu_params, mu_var_params = params['mu_mapping'], var_params['mu_mapping']
        lsigma_params, lsigma_var_params = params['lsigma_mapping'], var_params['lsigma_mapping']
        
        # data
        t, dt, x_obs, y, mask = self.get_data()
        timedata = (t, dt)
        assert dt.shape[0] == 1  # assume uniform time intervals
        dt = dt[0]
        prng_keys = jr.split(prng_state, 4)
        
        ### E_q(x) [log p(x)/q(x)] and sample from posterior ###
        params, var_params = all_params['hyp'], all_params['sites']
        
        x_samples, KL_ss = self.state_space.sample_posterior(
            ss_params, ss_var_params, prng_keys[0], num_samps, 
            timedata, None, jitter, compute_KL=True)  # (time, tr, x_dims, 1)
        
        eps_samples, KL_eps = self.eps_mapping.sample_posterior(
            eps_mapping, eps_var_params, prng_keys[1], x_samples, jitter, True)  # (time, tr, N)
        mu, KL_mu = self.mu_mapping.sample_posterior(
            mu_params, mu_var_params, prng_keys[2], x_obs, jitter, True)
        log_sigma, KL_lsigma = self.lsigma_mapping.sample_posterior(
            lsigma_params, lsigma_var_params, prng_keys[3], x_obs, jitter, True)
        
        KL = KL_ss + KL_eps + KL_mu + KL_lsigma  # mean(0) for MC/trials
        
        ### E_q(x) [log p(y|x)] ###
        I = mu + jnp.exp(log_sigma) * eps_samples
        ll = self.IF_model.Euler_fit(IF_params, dt, q_vh_ic, I, y)
        
        neg_ELBO = -ll + KL  # final objective
        return neg_ELBO
        

    def train_ELBO(self, all_params, prng_state, num_samps, split_all_params_func=None, take_grads=True):
        """
        Compute the full ELBO for hyperparameter learning
        E step is done in this function by updating site parameters
        M step given the site parameters before this E step is done by gradient descent with grads
        
        :param tuple inference_state: state for inference hyperparameters PRNG state, damping, jitter
        :param lambda split_all_params_func: function to split into learnable and fixed parameters
        :param bool take_grads: if True, use autodiff to get hyperparameter gradients. Running pure 
                                E steps involves setting this to False
        :return:
            objective: the negative ELBO -E_q(x)[ ELBO(x) ]
            grads: the derivative of the objective w.r.t. the model hyperparameters
            inference_state: state list for inference settings
        """
        if split_all_params_func is None:
            learned_all_params = all_params
            fixed_all_params = tree_map(lambda prms: None, all_params)
        else:
            learned_all_params, fixed_all_params = split_all_params_func(all_params)
            
        # filtering-smoothing
        if take_grads: # compute ELBO and gradients via autodiff
            objective, grads = value_and_grad(self.filter_smoother, argnums=0)(
                learned_all_params, fixed_all_params, prng_state, num_samps)
            grads = tree_map(lambda grds, prms: jnp.zeros_like(prms) if grds is None else grds, 
                             grads, all_params, is_leaf=lambda x: x is None)
            
        else:
            objective, aux = self.filter_smoother(
                learned_all_params, fixed_all_params, prng_state, num_samps)
            grads = None
            
        return objective, grads
    
    
    ### sample ###
    def sample_prior_spikes(self, prng_state, num_samps, x_obs=None, timedata=None):
        """
        Sample from the generative model
        """
        params = self.get_all_params()['hyp']
        jitter = self.jitter
        
        if x_obs is None:
            x_obs = self.x_obs
        
        if timedata is None:  # default to training data
            t, dt = self.t, self.dt
        else:
            t, dt = timedata
        assert dt.shape[0] == 1  # assume uniform time intervals
        
        x_samples = self.state_space.sample_prior(
            params['state_space'], prng_state, num_samps, (t, dt), jitter)  # (time, tr, x_dims, 1)
        
        y, q_vh, I, eps_samples = self.simulate_spiketrains(
            params, None, prng_state, dt[0], x_samples, x_obs, jitter, prior=True)
        return y, q_vh, I, eps_samples
    
    
    def sample_posterior_spikes(self, prng_state, num_samps):
        """
        Sample from posterior predictive
        """
        all_params = self.get_all_params()
        params, var_params = all_params['hyp'], all_params['sites']
        jitter = self.jitter
        
        timedata = (self.t, self.dt)
        assert self.dt.shape[0] == 1  # assume uniform time intervals
        prng_keys = jr.split(prng_state, 2)
        
        x_samples = self.state_space.sample_posterior(
            params['state_space'], var_params['state_space'], prng_keys[0], num_samps, 
            timedata, None, jitter, compute_KL=False)[0]  # (time, tr, x_dims, 1)
        y, q_vh, I, eps_samples = self.simulate_spiketrains(
            params, var_params, prng_keys[1], self.dt[0], x_samples, self.x_obs, jitter, prior=False)
        return y, q_vh, I, eps_samples
        
    
    ### evaluation ###
    def evaluate_ELBO(self):
        return
        

    
    
class PP_SSGP(SVI_state_space):
    """
    Point process state space Gaussian process
    """
    def __init__(self, state_space, mapping, likelihood, dtype=jnp.float32):
        """
        The state-space conjugate-computation variational inference class.
        
        The stochastic differential equation (SDE) form of a Gaussian process (GP) model.
        Implements methods for inference and learning in models with GP priors of the form
            f(t) ~ GP(0,k(t,t'))
        using state space methods, i.e. Kalman filtering and smoothing.
        Constructs a linear time-invariant (LTI) stochastic differential equation (SDE) of the following form:
            dx(t)/dt = F x(t) + L w(t)
                  yₙ ~ p(yₙ | f(tₙ)=H x(tₙ))
        where w(t) is a white noise process and where the state x(t) is Gaussian distributed with initial
        state distribution x(t)~𝓝(0,Pinf).
        Combined likelihood and mapping to compute p(y|z;theta) becomes the observation model
        Refs:
            Chang, Wilkinson, Khan & Solin 2020 "Fast variational learning in state space Gaussian process models"
        
        :param prior: the model prior p(f)=GP(0,k(.,.)) object which constructs the required state space model matrices
        :param likelihood: the likelihood model object which performs parameter updates and evaluates p(y|f)
        """
        x_dims = state_space.kernel.out_dims
        super().__init__(x_dims, observation)
        self.state_space = state_space   
        self.state_dims = state_space.kernel.state_dims
        
        
    
    
    
class SRM_SSGP(SVI_state_space):
    """
    GLM or SRM model with noisy input
    """
    def __init__(self, state_space, mapping, likelihood, dtype=jnp.float32):
        """
        The state-space conjugate-computation variational inference class.
        """
        x_dims = state_space.kernel.out_dims
        super().__init__(x_dims, observation)
        self.state_space = state_space   
        self.state_dims = state_space.kernel.state_dims
        
        