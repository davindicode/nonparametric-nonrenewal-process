from functools import partial

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import lax, vmap



# neuron models
class neuron_model(object):
    """
    Individual neuron dynamics and transitions
    """

    def __init__(self, log_beta, log_gamma, tau_s, v_t, v_r, q_d):
        self.q_d = q_d
        self.adaptation = True if q_d > 1 else False

        self.params = {}  # model specific parameters
        self.params["log_beta"] = log_beta
        self.params["log_gamma"] = log_gamma
        self.params["v_t"] = v_t
        self.params["v_r"] = v_r
        self.params["tau_s"] = tau_s

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
        tau_s = params["tau_s"]
        tau_s = tau_s[0] if tau_s.shape[0] == 1 else tau_s[N_pre]

        return q_s + 1.0 / tau_s

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
        v_r = params["v_r"][None, :]
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
            q_vh = q_vh + (1.0 - y) * dt * f_vh + y * r_vh
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
        beta = jnp.exp(params["log_beta"])[None, :]
        gamma = jnp.exp(params["log_gamma"])[None, :]
        v_t = params["v_t"][None, :]
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
            q_vh = q_vh + (1.0 - y[..., None]) * dt * f_vh + y[..., None] * r_vh

            q_v = q_vh[..., 0]
            log_p = self.log_p_spike(params, q_v)  # (tr, N)
            cross_ent = y * log_p + (1.0 - y) * jnp.log(
                jnp.maximum(1.0 - jnp.exp(log_p), 1e-10)
            )

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
        v_r = params["v_r"][None, :, None]
        return v_r - q_vh


class LIF(neuron_model):
    """
    Individual neuron dynamics and transitions

    Transition functions r() take state values at a trial and time step of shape (dims,)
    """

    def __init__(self, log_beta, log_gamma, v_t, v_r, tau_s, tau_m):
        super().__init__(log_beta, log_gamma, tau_s, v_t, v_r, 1)
        self.params["tau_m"] = tau_m

    @partial(jax.jit, static_argnums=(0,))
    def f_vh(self, params, q, I_e):
        """
        q has shape (tr, N, dims)
        """
        tau_m = params["tau_m"][None, :]
        return ((-q[..., 0] + I_e) / tau_m)[..., None]

    @partial(jax.jit, static_argnums=(0,))
    def r_vh(self, params, q_vh):
        """
        q_vh of shape (tr, N, dims)
        """
        v_r = params["v_r"][None, :, None]
        return v_r - q_vh


class AdLIF(neuron_model):
    """
    Individual neuron dynamics and transitions
    """

    def __init__(self, log_beta, log_gamma, v_t, v_r, tau_s, tau_m, tau_h, a, b):
        super().__init__(log_beta, log_gamma, tau_s, v_t, v_r, 2)
        self.params["tau_m"] = tau_m
        self.params["tau_h"] = tau_h
        self.params["a"] = a
        self.params["b"] = b

    @partial(jax.jit, static_argnums=(0,))
    def f_vh(self, params, q, I_e):
        tau_m = params["tau_m"][None, :]
        a = params["a"][None, :]
        tau_h = params["tau_h"][None, :]
        return jnp.stack(
            [
                -q[..., 0] / tau_m - q[..., 1] + I_e,
                ((-q[..., 1] + a * q[..., 0]) / tau_h),
            ],
            axis=-1,
        )

    @partial(jax.jit, static_argnums=(0,))
    def r_vh(self, params, q_vh):
        v_r = params["v_r"][None, :]
        b = params["b"][None, :]
        v, h = q_vh[..., 0], q_vh[..., 1]
        return jnp.stack([v_r - v, h + b], axis=-1)


class AdExpIF(neuron_model):
    """
    Individual neuron dynamics and transitions
    """

    def __init__(self, v_t, v_r, tau_s, tau_m, tau_h, a, b):
        super().__init__(tau_s, v_t, v_r, 4)
        self.param_dict["tau_m"] = tau_m
        self.param_dict["tau_h"] = tau_h
        self.param_dict["delta_T"] = delta_T
        self.param_dict["v_T"] = v_T
        self.param_dict["a"] = a
        self.eb_param_dict["b"] = b

    @partial(jax.jit, static_argnums=(0,))
    def f_vh(self, params, q, I_e):
        tau_m = params["tau_m"][None, :]
        a = params["a"][None, :]
        delta_T = params["delta_T"][None, :]
        tau_h = params["tau_h"][None, :]
        return jnp.stack(
            (
                (
                    -q[..., 0]
                    + delta_T * jnp.exp((q[..., 0] - v_T) / delta_T)
                    - q[..., 1]
                    + q[..., -1]
                )
                / tau_m
                + I_e,
                ((-q[..., 1] + a * q[..., 0]) / tau_h),
            ),
            axis=-1,
        )

    @partial(jax.jit, static_argnums=(0,))
    def r_vh(self, params, q_vh):
        v_r = params["v_r"][None, :]
        b = params["b"][None, :]
        v, h = q_vh[..., 0], q_vh[..., 1]
        return jnp.stack([v_r - v, h + b], axis=-1)


class thetaIF(neuron_model):
    """
    Individual neuron dynamics and transitions

    Transition functions r() take state values at a trial and time step of shape (dims,)
    """

    def __init__(self, v_t, v_r, tau_s, tau_m):
        super().__init__(tau_s, v_t, v_r, 3)
        self.param_dict["tau_m"] = tau_m

    @partial(jax.jit, static_argnums=(0,))
    def f_vh(self, params, q, I_e):
        """
        q has shape (tr, N, dims)
        """
        tau_m = params["tau_m"][None, :]
        return (
            (1 + jnp.cos(2 * jnp.pi * q[..., 0])) / tau_m
            + (1 - jnp.cos(2 * jnp.pi * q[..., 0])) * (q[..., -1] + I_e)
        )[..., None]

    @partial(jax.jit, static_argnums=(0,))
    def r_vh(self, params, q_vh):
        """
        q_vh of shape (tr, N, dims)
        """
        v_r = params["v_r"][None, :, None]
        return v_r - q_vh


class QIF(neuron_model):
    """
    Individual neuron dynamics and transitions

    Transition functions r() take state values at a trial and time step of shape (dims,)
    """

    def __init__(self, v_t, v_r, tau_s, tau_m):
        super().__init__(tau_s, v_t, v_r, 3)
        self.param_dict["tau_m"] = tau_m
        self.param_dict["v_c"] = v_c

    @partial(jax.jit, static_argnums=(0,))
    def f_vh(self, params, q, I_e):
        """
        q has shape (tr, N, dims)
        """
        tau_m = params["tau_m"][None, :]
        v_c = params["v_c"][None, :]
        return (q[..., 0] * (q[..., 0] - v_c) / tau_m + I_e)[..., None]

    @partial(jax.jit, static_argnums=(0,))
    def r_vh(self, params, q_vh):
        """
        q_vh of shape (tr, N, dims)
        """
        v_r = params["v_r"][None, :, None]
        return v_r - q_vh


class Izhikevich(neuron_model):
    """
    Individual neuron dynamics and transitions
    Note that c is the reset v_r parameter
    """

    def __init__(self, a, b, c, d, v_t, tau_s):
        super().__init__(tau_s, v_t, c, 4)
        self.param_dict["a"] = a
        self.param_dict["b"] = b
        self.eb_param_dict["d"] = d

    @partial(jax.jit, static_argnums=(0,))
    def f_vh(self, params, q, I_e):
        a = params["a"][None, :]
        b = params["b"][None, :]
        return jnp.stack(
            (
                0.04 * q[..., 0] ** 2
                + 5.0 * q[..., 0]
                + 140.0
                - q[..., 1]
                + q[..., -1]
                + I_e,
                (a * (-q[..., 1] + b * q[..., 0])),
            ),
            axis=-1,
        )

    @partial(jax.jit, static_argnums=(0,))
    def r_vh(self, params, q_vh):
        v_r = params["v_r"][None, :]
        d = params["d"][None, :]
        v, h = q_vh[..., 0], q_vh[..., 1]
        return jnp.array([v_r - v, h + d])

    
    
    
    
import torch
import torch.nn as nn

import numpy as np

from tqdm.autonotebook import tqdm



# Continuous models
class Hodgkin_Huxley():
    r"""
    Hodgkin-Huxley model via Euler integration
    """
    def __init__(self, G_na=120, G_k=36, G_l=0.3, E_na=50, E_k=-77, E_l=-54.4):
        r"""
        units are in mV, microS, nF, mA, ms
        """
        self.G_na = G_na
        self.G_k = G_k
        self.G_l = G_l
        self.E_na = E_na
        self.E_k = E_k
        self.E_l = E_l
    
    def euler_int(self, T, runs, I_ext, ic, dt=0.001, prin=1000):
        r"""
        Integrate the HH dynamics, the state array (v, m, h, n) is represented by 4 floating-point values 
        
        :param int T: timesteps to run the simulation for
        :param int runs: number of trials to run (I_ext and i.c. can differ per run)
        :param np.array I_ext: external input current, with shape (runs, timesteps)
        :param np.array ic: neuron initial conditions, with shape (runs, 4)
        :returns: neuron state over the simulation
        :rtype: np.array
        """
        alpha_m = lambda V: (2.5-0.1*(V+65)) / (np.exp(2.5-0.1*(V+65)) -1)
        beta_m = lambda V: 4.0 * np.exp(-(V+65)/18)
        alpha_h = lambda V: 0.07 * np.exp(-(V+65)/20)
        beta_h = lambda V: 1.0 / (np.exp(3.0-0.1*(V+65)) + 1)
        alpha_n = lambda V: (0.1-0.01*(V+65)) / (np.exp(1-0.1*(V+65)) - 1)
        beta_n = lambda V: 0.125 * np.exp(-(V+65)/80)

        state = np.zeros((runs, T, 4)) # vector v, m, h, n
        for k in range(runs):
            state[k, 0, :] = ic[k, :]#[-6.49997224e+01, 5.29342176e-02, 5.96111046e-01, 3.17681168e-01]

        ds = np.zeros((runs, 4))
        iterator = tqdm(range(T-1))
        for t in iterator:
            ds[:, 0] = -(G_l*(state[:, t, 0] - E_l) + \
                   G_k*np.power(state[:, t, 3], 4)*(state[:, t, 0] - E_k) + \
                   G_na*np.power(state[:, t, 1], 3)*state[:, t, 2]*(state[:, t, 0] - E_na)) + I_ext[:, t]
            ds[:, 1] = alpha_m(state[:, t, 0]) * (1 - state[:, t, 1]) - beta_m(state[:, t, 0]) * state[:, t, 1]
            ds[:, 2] = alpha_h(state[:, t, 0]) * (1 - state[:, t, 2]) - beta_h(state[:, t, 0]) * state[:, t, 2]
            ds[:, 3] = alpha_n(state[:, t, 0]) * (1 - state[:, t, 3]) - beta_n(state[:, t, 0]) * state[:, t, 3]

            state[:, t+1] = state[:, t] + ds * dt

        return state
    
    
    
class FitzHugh_Nagumo():
    r"""
    A 2D reduction of the Hodgkin-Huxley model to the phase plane.
    """
    def __init__(self, b_0, b_1, tau_u, tau_w):
        r"""
        units are in mV, microS, nF, mA, ms
        """
        self.b_0 = b_0
        self.b_1 = b_1
        self.tau_u = tau_u
        self.tau_w = tau_w
        
    def euler_int(self, T, runs, I_ext, ic, dt=0.001, prin=1000):
        r"""
        Integrate the HH dynamics, the state array (v, m, h, n) is represented by 4 floating-point values 
        
        :param int T: timesteps to run the simulation for
        :param int runs: number of trials to run (I_ext and i.c. can differ per run)
        :param np.array I_ext: external input current, with shape (runs, timesteps)
        :param np.array ic: neuron initial conditions, with shape (runs, 4)
        :returns: neuron state over the simulation
        :rtype: np.array
        """
        state = np.zeros((runs, T, 2)) # vector u, w
        for k in range(runs):
            state[k, 0, :] = ic[k, :]#[-6.49997224e+01, 5.29342176e-02, 5.96111046e-01, 3.17681168e-01]

        ds = np.zeros((runs, 2))
        iterator = tqdm(range(T-1))
        for t in iterator:
            ds[:, 0] = 1/self.tau_u * (state[:, t, 0] - state[:, t, 0]**3/3. - state[:, t, 1] + I_ext)
            ds[:, 1] = 1/self.tau_w * (self.b_0 + self.b_1*state[:, t, 0] - state[:, t, 1])

            state[:, t+1] = state[:, t] + ds * dt

        return state
    
    
    
class Morris_Lecar():
    r"""
    A 2D reduction of the Hodgkin-Huxley model to the phase plane.
    """
    def __init__(self, G_na=120, G_k=36, G_l=0.3, E_na=50, E_k=-77, E_l=-54.4):
        r"""
        units are in mV, microS, nF, mA, ms
        """
        self.G_na = G_na
        self.G_k = G_k
        self.G_l = G_l
        self.E_na = E_na
        self.E_k = E_k
        self.E_l = E_l

    def euler_int(self, T, runs, I_ext, ic, dt=0.001, prin=1000):
        r"""
        Integrate the HH dynamics, the state array (v, m, h, n) is represented by 4 floating-point values 
        
        :param int T: timesteps to run the simulation for
        :param int runs: number of trials to run (I_ext and i.c. can differ per run)
        :param np.array I_ext: external input current, with shape (runs, timesteps)
        :param np.array ic: neuron initial conditions, with shape (runs, 4)
        :returns: neuron state over the simulation
        :rtype: np.array
        """
        alpha_m = lambda V: (2.5-0.1*(V+65)) / (np.exp(2.5-0.1*(V+65)) -1)
        beta_m = lambda V: 4.0 * np.exp(-(V+65)/18)
        alpha_h = lambda V: 0.07 * np.exp(-(V+65)/20)
        beta_h = lambda V: 1.0 / (np.exp(3.0-0.1*(V+65)) + 1)
        alpha_n = lambda V: (0.1-0.01*(V+65)) / (np.exp(1-0.1*(V+65)) - 1)
        beta_n = lambda V: 0.125 * np.exp(-(V+65)/80)

        state = np.zeros((runs, T, 4)) # vector v, m, h, n
        for k in range(runs):
            state[k, 0, :] = ic[k, :]#[-6.49997224e+01, 5.29342176e-02, 5.96111046e-01, 3.17681168e-01]

        ds = np.zeros((runs, 4))
        iterator = tqdm(range(T-1))
        for t in iterator:
            ds[:, 0] = -(G_l*(state[:, t, 0] - E_l) + \
                   G_k*np.power(state[:, t, 3], 4)*(state[:, t, 0] - E_k) + \
                   G_na*np.power(state[:, t, 1], 3)*state[:, t, 2]*(state[:, t, 0] - E_na)) + I_ext[:, t]
            ds[:, 1] = alpha_m(state[:, t, 0]) * (1 - state[:, t, 1]) - beta_m(state[:, t, 0]) * state[:, t, 1]
            ds[:, 2] = alpha_h(state[:, t, 0]) * (1 - state[:, t, 2]) - beta_h(state[:, t, 0]) * state[:, t, 2]
            ds[:, 3] = alpha_n(state[:, t, 0]) * (1 - state[:, t, 3]) - beta_n(state[:, t, 0]) * state[:, t, 3]

            state[:, t+1] = state[:, t] + ds * dt

        return state
    
    
def count_APs(V, lim=20.0):
    r"""
    Action potential counter
    """
    idx = (V > lim).astype(float)
    idf = np.diff(idx) == 1
    return idf.sum()



# Integrate-and-fire models    
class Izhikevich():
    r""" 
    Biophysically inspired Izhikevich model (2003/2004) [1], a nonlinear integrate-and-fire model.
    
    References:
    
    [1] 
    
    """
    def __init__(self, a, b, c, d):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
    
    def euler_int(self, T, runs, I_ext, ic, dt=0.1, prin=1000):
        r"""
        Euler integration of the dynamics, with state array (v, u)
        """
        state = np.zeros((runs, T, 2)) # vector v, u
        spiketrain = np.zeros((runs, T))
        reset_state = np.empty((runs, 2))
        reset_state[:, 0].fill(self.c)
        
        for k in range(runs):
            state[k, 0, :] = ic[k, :]

        ds = np.zeros((runs, 2))
        iterator = tqdm(range(T-1))
        for t in iterator:
            ds[:, 0] = 0.04*state[:, t, 0]**2 + 5.*state[:, t, 0] + 140. - state[:, t, 1] + I_ext[:, t]
            ds[:, 1] = self.a*(self.b*state[:, t, 0] - state[:, t, 1])

            reset = (state[:, t, 0] >= 30.)
            if reset.sum() > 0:
                reset_state[:, 1] = (state[:, t, 1] + self.d)
                state[:, t+1] = reset[:, None]*reset_state + (1-reset)[:, None]*(state[:, t] + ds * dt)
                spiketrain[:, t+1] = reset
            else:
                state[:, t+1] = state[:, t] + ds * dt

        return state, spiketrain
    
    
    
class AdExIF():
    r"""
    Adaptive exponential integrate-and-fire model. [1]
    
    References:
    
    [1] `Neuronal Dynamics`, Wulfram Gerstner, Werner M. Kistler, Richard Naud and Liam Paninski.
    
    """
    def __init__(self, a, b, c, d):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        
    def euler_int(self, T, runs, I_ext, ic, dt=0.001, prin=1000):
        r"""
        Euler integration of the dynamics, with state array (v, u)
        """
        state = np.zeros((runs, T, 2)) # vector v, u
        spiketrain = np.zeros((runs, T))
        reset_state = np.empty((runs, 2))
        reset_state[:, 0].fill(self.c)
        
        for k in range(runs):
            state[k, 0, :] = ic[k, :]

        ds = np.zeros((runs, 2))
        iterator = tqdm(range(T-1))
        for t in iterator:
            ds[:, 0] = 0.04*state[:, t, 0]**2 + 5.*state[:, t, 0] + 140. - state[:, t, 1] + I_ext[:, t]
            ds[:, 1] = self.a*(self.b*state[:, t, 0] - state[:, t, 1])

            reset = (state[:, t, 0] >= 30.)
            if reset.sum() > 0:
                reset_state[:, 1] = (state[:, t, 1] + self.d)
                state[:, t+1] = reset[:, None]*reset_state + (1-reset)[:, None]*(state[:, t] + ds * dt)
                spiketrain[:, t+1] = reset
            else:
                state[:, t+1] = state[:, t] + ds * dt

        return state, spiketrain
    
    
    
def neuron_model(dynamics, model_type):
    r"""
    Neuronal dynamics library of parameter values.
    Izhikevich parameters from [1].
    
    References:
    
    [1] `Capturing the Dynamical Repertoire of Single Neurons with Generalized Linear Models`,
        Alison I. Weber & Jonathan W. Pillow
        
    """
    
    if model_type == 'Izhikevich': # dt in ms
        if dynamics == 'tonic_spiking':
            model = Izhikevich(0.02, 0.2, -65, 6)
            I = 14
            dt = 0.1
        elif dynamics == 'phasic_spiking':
            model = Izhikevich(0.02, 0.2, -65, 6)
            I = 0.5
            dt = 0.1
        elif dynamics == 'tonic_bursting':
            model = Izhikevich(0.02, 0.2, -50, 2)
            I = 10
            dt = 0.1
        elif dynamics == 'phasic_bursting':
            model = Izhikevich(0.02, 0.25, -55, 0.05)
            I = 0.6
            dt = 0.1
        elif dynamics == 'mixed':
            model = Izhikevich(0.02, 0.2, -55, 4)
            I = 10
            dt = 0.1
        elif dynamics == 'frequency_adaptation':
            model = Izhikevich(0.01, 0.2, -65, 5)
            I = 20
            dt = 0.1
        elif dynamics == 'type_I':
            model = Izhikevich(0.02, -0.1, -55, 6)
            I = 25
            dt = 1.
        elif dynamics == 'type_II':
            model = Izhikevich(0.2, 0.26, -65, 0)
            I = 0.5
            dt = 1.
        elif dynamics == 'spike_latency':
            model = Izhikevich(0.02, 0.2, -65, 6)
            I = 3.49
            dt = 0.1
        elif dynamics == 'resonator':
            model = Izhikevich(0.1, 0.26, -60, -1)
            I = 0.3
            dt = 0.5
        elif dynamics == 'integrator':
            model = Izhikevich(0.02, -0.1, -66, 6)
            I = 27.4
            dt = 0.5
        elif dynamics == 'rebound_spike':
            model = Izhikevich(0.03, 0.25, -60, 4)
            I = -5.
            dt = 0.1
        elif dynamics == 'rebound_burst':
            model = Izhikevich(0.03, 0.25, -52, 0)
            I = -5.
            dt = 0.1
        elif dynamics == 'threshold_variability':
            model = Izhikevich(0.03, 0.25, -60, 4)
            I = 2.3
            dt = 1.
        elif dynamics == 'bistability_I':
            model = Izhikevich(1., 1.5, -60, 0)
            I = 30.
            dt = 0.05
        elif dynamics == 'bistability_II':
            model = Izhikevich(1., 1.5, -60, 0)
            I = 40.
            dt = 0.05
        else:
            raise NotImplementedError
        
        return model, I, dt
    
    else:
        raise NotImplementedError