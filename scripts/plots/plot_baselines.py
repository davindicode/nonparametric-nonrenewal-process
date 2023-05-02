import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.special as sps

import jax

import sys
sys.path.append("../../../GaussNeuro")
from gaussneuro import utils
import gaussneuro as lib



def spike_history_filters(rng, prng_state, jitter, array_type):
    # GLM
    a, c = 6., 30.  # inverse width of the bumps, uniformity of the bump placement
    phi_lower, phi_upper = 17., 36.
    B = 16
    filter_length = 500
    obs_dims = B
    num_samps = 10
    

    ini_var = 1.0
    a = a * np.ones((obs_dims, 1))
    c = c * np.ones((obs_dims, 1))
    phi_h = np.linspace(phi_lower, phi_upper, B)[:, None, None].repeat(obs_dims, axis=1)
    w_h = np.zeros_like((np.sqrt(ini_var) * rng.normal(size=(B, obs_dims)))[..., None])
    for o in range(obs_dims):
        w_h[o, o, :] = 1.

    flt = lib.filters.RaisedCosineBumps(
        a,
        c,
        w_h,
        phi_h,
        filter_length,
    )
    
    glm_filter_t = flt.sample_prior(prng_state, 1, None, jitter)
    glm_filter_t = np.array(glm_filter_t[0])  # (filter_length, outs, 1)
    
    # GP
    filter_length = 500
    obs_dims = 9

    num_induc = 10
    x_dims = 1

    # qSVGP inducing points
    induc_locs = np.linspace(0, filter_length, num_induc)[None, :, None].repeat(obs_dims, axis=0)
    u_mu = 0.0*rng.normal(size=(obs_dims, num_induc, 1))
    u_Lcov = 0.1*np.eye(num_induc)[None, ...].repeat(obs_dims, axis=0)

    # kernel
    len_fx = 100.0*np.ones((obs_dims, x_dims))  # GP lengthscale
    beta = 0.0 * np.ones(obs_dims)
    len_beta = 1.5 * len_fx
    var_f = 0.1*np.ones(obs_dims)  # kernel variance

    kern = lib.GP.kernels.DecayingSquaredExponential(
        obs_dims, variance=var_f, lengthscale=len_fx, 
        beta=beta, lengthscale_beta=len_beta, array_type=array_type)
    gp = lib.GP.sparse.qSVGP(kern, induc_locs, u_mu, u_Lcov, RFF_num_feats=0, whitened=False)

    a_r = -6.*np.ones((obs_dims, 1))
    tau_r = 10.*np.ones((obs_dims, 1))

    flt = lib.filters.GaussianProcess(
        gp, 
        a_r,
        tau_r, 
        filter_length,
    )
    
    #gp_filter_t, _ = flt.sample_posterior(prng_state, num_samps, False, None, jitter)
    gp_filter_t = flt.sample_prior(prng_state, num_samps, None, jitter)
    gp_filter_t = np.array(gp_filter_t)
    
    # export
    filters_dict = {
        'gp_filter': gp_filter_t, 
        'glm_filter': glm_filter_t,
    }
    return filters_dict



def rate_rescaling(rng, dt = 0.001, p = 0.003, ts = 3000):
    spikes_at = rng.binomial(1, p, size=(ts,))
    spike_times = np.where(spikes_at > 0)[0]

    time_t = np.arange(ts) * dt
    rates_t = 30. * (np.sin(time_t * (1 + np.exp( -1.5 * (time_t - time_t[-1]/2.) ** 2 )))**2 + \
        np.exp( -0.5 * (time_t - time_t[-1]/3.) ** 2 ))


    rtime_t = np.cumsum(rates_t * dt)
    rspike_times = np.ceil(rtime_t[spike_times] / dt).astype(int)
    
    data_dict = {
        'time': time_t, 
        'rates': rates_t, 
        'spike_times': spike_times, 
        'rescaled_spike_times': rspike_times, 
        'rescaled_time': rtime_t, 
    }
    return data_dict



def main():
    jax.config.update('jax_platform_name', 'cpu')
    #jax.config.update('jax_disable_jit', True)

    double_arrays = True

    if double_arrays:
        jax.config.update("jax_enable_x64", True)
        array_type = "float64"
    else:
        array_type = "float32"

    seed = 123
    rng = np.random.default_rng(seed)
    
    
    
if __name__ == "__main__":
    main()