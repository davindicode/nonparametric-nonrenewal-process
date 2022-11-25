import jax
import jax.numpy as jnp
import jax.random as jr
from jax import lax, vmap

from functools import partial




# neuron models
class neuron_model(object):
    """
    Individual neuron dynamics and transitions
    """
    def __init__(self, log_beta, log_gamma, tau_s, v_t, v_r, q_d):
        self.q_d = q_d
        self.adaptation = True if q_d > 1 else False
            
        self.params = {} # model specific parameters
        self.params['log_beta'] = log_beta
        self.params['log_gamma'] = log_gamma
        self.params['v_t'] = v_t
        self.params['v_r'] = v_r
        self.params['tau_s'] = tau_s
        
    def constraints(self, params):
        return params
        
    def f_vh(self, params, q_vh, I_e):
        raise NotImplementedError
        
    ### state transitions ###
    def r_vh(self, params, q_vh, N_pre):
        """
        Transition is applied to a single event at a time
        """
        raise NotImplementedError
        
    ### synaptic current transitions ###
    @partial(jax.jit, static_argnums=(0,))
    def r_s(self, params, q_s, N_pre):
        """
        Transition is applied to a single event at a time
        """
        tau_s = params['tau_s']
        tau_s = tau_s[0] if tau_s.shape[0] == 1 else tau_s[N_pre]
            
        return q_s+1./tau_s
    
    ### solver ###
    @partial(jax.jit, static_argnums=(0,))
    def Euler_simulate(self, params, prng_state, dt, q_vh_ic, I):
        """
        Euler integration, using Heaviside with surrogate gradients
        :param jnp.array y: observed spike train of shape (time, tr, N)
        :param jnp.array prng_state: random state of shape (time, 2)
        :param jnp.array mu: input mean of shape (time, tr, N)
        :param jnp.array sigma: input noise magnitude of shape (time, tr, N)
        :param jnp.array epsilon: input noise of shape (time, tr, N, dims)
        """
        v_r = params['v_r'][None, :]
        Tsteps = I.shape[0]
        prng_states = jr.split(prng_state, Tsteps)
        
        def step(carry, inputs):
            I_e, prng_state = inputs
            q_vh = carry
            
            f_vh = self.f_vh(params, q_vh, I_e)
            r_vh = self.r_vh(params, q_vh)
            p = jnp.exp(self.log_p_spike(params, q_vh[..., 0]))
            y = jr.bernoulli(prng_state, p)[..., None]  # (tr, N, 1)
            
            # next time step
            q_vh = q_vh + (1. - y) * dt * f_vh + y * r_vh
            return q_vh, (y[..., 0], q_vh)

        init = q_vh_ic
        xs = (I, prng_states)
        _, (y, q_vh) = lax.scan(step, init, xs)
        return y, q_vh
    
    
    ## fitting ###
    def log_p_spike(self, params, q_v):
        """
        :param jnp.array q_v: voltage of shape (tr, N)
        """
        beta = jnp.exp(params['log_beta'])[None, :]
        gamma = jnp.exp(params['log_gamma'])[None, :]
        v_t = params['v_t'][None, :]
        return -gamma * jax.nn.softplus(beta * (v_t - q_v)) / beta
            
        
    @partial(jax.jit, static_argnums=(0,))
    def Euler_fit(self, params, dt, q_vh_ic, I, y):
        """
        :param jnp.array y: spike train of shape (time, tr, N)
        """
        def step(carry, inputs):
            I_e, y = inputs
            q_vh = carry

            f_vh = self.f_vh(params, q_vh, I_e)  # (tr, N, dims)
            r_vh = self.r_vh(params, q_vh)
            
            # heavi = heaviside(q_vh[..., 0], surro_log_beta)
            q_vh = q_vh + (1. - y[..., None]) * dt * f_vh + y[..., None] * r_vh
            
            q_v = q_vh[..., 0]
            log_p = self.log_p_spike(params, q_v)  # (tr, N)
            cross_ent = y * log_p + \
                (1. - y) * jnp.log(jnp.maximum(1. - jnp.exp(log_p), 1e-10))
            
            return q_vh, cross_ent
        
        init = q_vh_ic
        xs = (I, y)
        _, cross_ent = lax.scan(step, init, xs)  # (time, tr, N)
        return cross_ent.mean(1).sum()  # mean over trials



class NIF(neuron_model):
    """
    Individual neuron dynamics and transitions
    
    Transition functions r() take state values at a trial and time step of shape (dims,)
    """
    def __init__(self, log_beta, log_gamma, v_t, v_r, tau_s):
        super().__init__(log_beta, log_gamma, tau_s, v_t, v_r, 1)
        
    @partial(jax.jit, static_argnums=(0,))
    def f_vh(self, params, q, I_e):
        """
        q has shape (tr, N, dims)
        """
        return I_e[..., None]
        
    @partial(jax.jit, static_argnums=(0,))
    def r_vh(self, params, q_vh):
        """
        q_vh of shape (tr, N, dims)
        """
        v_r = params['v_r'][None, :, None]
        return v_r - q_vh



class LIF(neuron_model):
    """
    Individual neuron dynamics and transitions
    
    Transition functions r() take state values at a trial and time step of shape (dims,)
    """
    def __init__(self, log_beta, log_gamma, v_t, v_r, tau_s, tau_m):
        super().__init__(log_beta, log_gamma, tau_s, v_t, v_r, 1)
        self.params['tau_m'] = tau_m
        
    @partial(jax.jit, static_argnums=(0,))
    def f_vh(self, params, q, I_e):
        """
        q has shape (tr, N, dims)
        """
        tau_m = params['tau_m'][None, :]
        return ((-q[..., 0] + I_e) / tau_m)[..., None]
        
    @partial(jax.jit, static_argnums=(0,))
    def r_vh(self, params, q_vh):
        """
        q_vh of shape (tr, N, dims)
        """
        v_r = params['v_r'][None, :, None]
        return v_r - q_vh
    
    
    
class AdLIF(neuron_model):
    """
    Individual neuron dynamics and transitions
    """
    def __init__(self, log_beta, log_gamma, v_t, v_r, tau_s, tau_m, tau_h, a, b):
        super().__init__(log_beta, log_gamma, tau_s, v_t, v_r, 2)
        self.params['tau_m'] = tau_m
        self.params['tau_h'] = tau_h
        self.params['a'] = a
        self.params['b'] = b
        
    @partial(jax.jit, static_argnums=(0,))
    def f_vh(self, params, q, I_e):
        tau_m = params['tau_m'][None, :]
        a = params['a'][None, :]
        tau_h = params['tau_h'][None, :]
        return jnp.stack(
            [-q[..., 0]/tau_m - q[..., 1] + I_e, 
             ((-q[..., 1] + a*q[..., 0])/tau_h)], 
            axis=-1
        )
        
    @partial(jax.jit, static_argnums=(0,))
    def r_vh(self, params, q_vh):
        v_r = params['v_r'][None, :]
        b = params['b'][None, :]
        v, h = q_vh[..., 0], q_vh[..., 1]
        return jnp.stack([v_r - v, h + b], axis=-1)
    
    
    
class AdExpIF(neuron_model):
    """
    Individual neuron dynamics and transitions
    """
    def __init__(self, v_t, v_r, tau_s, tau_m, tau_h, a, b):
        super().__init__(tau_s, v_t, v_r, 4)
        self.param_dict['tau_m'] = tau_m
        self.param_dict['tau_h'] = tau_h
        self.param_dict['delta_T'] = delta_T
        self.param_dict['v_T'] = v_T
        self.param_dict['a'] = a
        self.eb_param_dict['b'] = b
        
    @partial(jax.jit, static_argnums=(0,))
    def f_vh(self, params, q, I_e):
        tau_m = params['tau_m'][None, :]
        a = params['a'][None, :]
        delta_T = params['delta_T'][None, :]
        tau_h = params['tau_h'][None, :]
        return jnp.stack(
            ((-q[..., 0] + delta_T*jnp.exp( (q[..., 0] - v_T)/delta_T ) - \
              q[..., 1] + q[..., -1])/tau_m + I_e, 
             ((-q[..., 1] + a*q[..., 0])/tau_h)), 
            axis=-1
        )
        
    @partial(jax.jit, static_argnums=(0,))
    def r_vh(self, params, q_vh):
        v_r = params['v_r'][None, :]
        b = params['b'][None, :]
        v, h = q_vh[..., 0], q_vh[..., 1]
        return jnp.stack([v_r - v, h + b], axis=-1)
    
    
    
class thetaIF(neuron_model):
    """
    Individual neuron dynamics and transitions
    
    Transition functions r() take state values at a trial and time step of shape (dims,)
    """
    def __init__(self, v_t, v_r, tau_s, tau_m):
        super().__init__(tau_s, v_t, v_r, 3)
        self.param_dict['tau_m'] = tau_m
        
    @partial(jax.jit, static_argnums=(0,))
    def f_vh(self, params, q, I_e):
        """
        q has shape (tr, N, dims)
        """
        tau_m = params['tau_m'][None, :]
        return (
            (1 + jnp.cos(2*jnp.pi*q[..., 0]))/tau_m + \
            (1 - jnp.cos(2*jnp.pi*q[..., 0]))*(q[..., -1] + I_e)
        )[..., None]
        
    @partial(jax.jit, static_argnums=(0,))
    def r_vh(self, params, q_vh):
        """
        q_vh of shape (tr, N, dims)
        """
        v_r = params['v_r'][None, :, None]
        return v_r - q_vh
    
    
    
class QIF(neuron_model):
    """
    Individual neuron dynamics and transitions
    
    Transition functions r() take state values at a trial and time step of shape (dims,)
    """
    def __init__(self, v_t, v_r, tau_s, tau_m):
        super().__init__(tau_s, v_t, v_r, 3)
        self.param_dict['tau_m'] = tau_m
        self.param_dict['v_c'] = v_c
        
    @partial(jax.jit, static_argnums=(0,))
    def f_vh(self, params, q, I_e):
        """
        q has shape (tr, N, dims)
        """
        tau_m = params['tau_m'][None, :]
        v_c = params['v_c'][None, :]
        return (q[..., 0]*(q[..., 0] - v_c)/tau_m + I_e)[..., None]
        
    @partial(jax.jit, static_argnums=(0,))
    def r_vh(self, params, q_vh):
        """
        q_vh of shape (tr, N, dims)
        """
        v_r = params['v_r'][None, :, None]
        return v_r - q_vh
    
    
    
    
class Izhikevich(neuron_model):
    """
    Individual neuron dynamics and transitions
    Note that c is the reset v_r parameter
    """
    def __init__(self, a, b, c, d, v_t, tau_s):
        super().__init__(tau_s, v_t, c, 4)
        self.param_dict['a'] = a
        self.param_dict['b'] = b
        self.eb_param_dict['d'] = d
        
    @partial(jax.jit, static_argnums=(0,))
    def f_vh(self, params, q, I_e):
        a = params['a'][None, :]
        b = params['b'][None, :]
        return jnp.stack(
            (0.04*q[..., 0]**2 + 5.*q[..., 0] + 140. - q[..., 1] + q[..., -1] + I_e, 
             (a*(-q[..., 1] + b*q[..., 0]))), 
            axis=-1
        )
        
    @partial(jax.jit, static_argnums=(0,))
    def r_vh(self, params, q_vh):
        v_r = params['v_r'][None, :]
        d = params['d'][None, :]
        v, h = q_vh[..., 0], q_vh[..., 1]
        return jnp.array([v_r - v, h + d])