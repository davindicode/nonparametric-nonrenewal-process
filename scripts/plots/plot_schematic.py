import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.special as sps

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import vmap

import sys
sys.path.append("../../../GaussNeuro")
from gaussneuro import utils
import gaussneuro as lib



def log_time_transform(t, inverse, warp_tau):
    """
    Inverse transform is from tau [0, 1] to t in R

    :param jnp.ndarray t: time of shape (obs_dims,)
    """
    if inverse:
        t_ = -np.log(1 - t) * warp_tau
    else:
        s = np.exp(-t / warp_tau)
        t_ = 1 - s

    return t_


def generate_behaviour(prng_state, evalsteps, L):
    # animal position
    x_dims = 2
    num_samps = 1
    jitter = 1e-6

    var_x = 1.0*np.ones((x_dims))  # GP variance
    len_x = 2.0*np.ones((x_dims, 1))  # GP lengthscale

    kernx = lib.GP.kernels.Matern52(
        x_dims, variance=var_x, lengthscale=len_x)

    # site_init
    Tsteps = 0
    site_locs = np.empty(Tsteps)  # s
    site_obs = np.zeros([Tsteps, x_dims, 1])
    site_Lcov = np.eye(x_dims)[None, ...].repeat(Tsteps, axis=0)

    state_space = lib.GP.markovian.GaussianLTI(
        kernx, site_locs, site_obs, site_Lcov, diagonal_site=True, fixed_grid_locs=False)

    t_eval = jnp.linspace(0.0, L, evalsteps)
    pos_prior_samples = state_space.sample_prior(
        prng_state, num_samps, t_eval, jitter)
    
    return pos_prior_samples



def model_inputs(prng_state, rng, evalsteps, L):
    evals = np.arange(evalsteps)
    dt = 0.001

    t_eval = np.zeros(evalsteps)
    y = (rng.normal(size=(evalsteps,)) > 2.2)
    spiketimes = np.where(y > 0)[0]

    pos_sample = generate_behaviour(prng_state, evalsteps, L)

    ISIs = lib.utils.spikes.get_lagged_ISIs(y[:, None], 4, dt)
    uisi = np.unique(ISIs[..., 1])
    uisi = uisi[~np.isnan(uisi)]
    
    warp_tau = np.array([uisi.mean()])
    tISI = log_time_transform(ISIs[..., 0], False, warp_tau)
    tuISIs = log_time_transform(ISIs[..., 1], False, warp_tau)

    tau = np.linspace(0., .4, 100)[:, None]
    tau_tilde = log_time_transform(tau, False, warp_tau)
    
    return evals, spiketimes, pos_sample, ISIs, tISI, tuISIs, tau, tau_tilde



def sample_conditional_ISI(
    bnpp,
    prng_state,
    num_samps,
    tau_eval,
    isi_cond,
    x_cond,
    sel_outdims,
    int_eval_pts=1000,
    num_quad_pts=100,
    prior=True,
    jitter=1e-6,
):
    """
    Compute the instantaneous renewal density with rho(ISI;X) from model
    Uses linear interpolation with Gauss-Legendre quadrature for integrals

    :param jnp.ndarray t_eval: evaluation time points (eval_locs,)
    :param jnp.ndarray isi_cond: past ISI values to condition on (obs_dims, order)
    :param jnp.ndarray x_cond: covariate values to condition on (x_dims,)
    """
    if sel_outdims is None:
        obs_dims = bnpp.gp.kernel.out_dims
        sel_outdims = jnp.arange(obs_dims)

    sigma_pts, weights = utils.linalg.gauss_legendre(1, num_quad_pts)
    sigma_pts, weights = bnpp._to_jax(sigma_pts), bnpp._to_jax(weights)

    # evaluation locs
    tau_tilde, log_dtilde_dt = vmap(
        bnpp._log_time_transform_jac, (1, None), 1
    )(
        tau_eval[None, :], False
    )  # (obs_dims, locs)

    log_rho_tilde, int_rho_tau_tilde, log_normalizer = bnpp._sample_log_ISI_tilde(
        prng_state,
        num_samps,
        tau_tilde,
        isi_cond,
        x_cond,
        sigma_pts,
        weights,
        sel_outdims,
        int_eval_pts,
        prior,
        jitter,
    )

    log_ISI_tilde = log_rho_tilde - int_rho_tau_tilde - log_normalizer
    ISI_density = jnp.exp(log_ISI_tilde + log_dtilde_dt)
    return ISI_density, log_rho_tilde, tau_tilde



def BNPP_samples(prng_state, rng, num_samps, evalsteps, L):
    dt = 1e-3  # s
    obs_dims = 1
    num_induc = 8
    RFF_num_feats = 3000
    cisi_t_eval = jnp.linspace(0.0, L, evalsteps)
    sel_outdims = jnp.arange(obs_dims)
    
    x_cond, isi_cond = None, None
    
    ISI_densities, log_rho_tildes = [], []
    for a_r, m_b in zip([0., -6.], [-1., 0.]):
        # SVGP
        len_t = 1.0 * np.ones((obs_dims, 1))  # GP lengthscale
        var_t = 0.5 * np.ones(obs_dims)  # GP variance
        tau_kernel = lib.GP.kernels.Matern52(obs_dims, variance=var_t, lengthscale=len_t)
        
        induc_locs = np.linspace(0.0, 1.0, num_induc)[None, :, None].repeat(obs_dims, axis=0)
        u_mu = 1.0 * rng.normal(size=(obs_dims, num_induc, 1))
        u_Lcov = 0.1 * np.eye(num_induc)[None, ...].repeat(obs_dims, axis=0)

        svgp = lib.GP.sparse.qSVGP(
            tau_kernel,
            induc_locs,
            u_mu,
            u_Lcov,
            RFF_num_feats=RFF_num_feats,
            whitened=True,
        )
        
        # parameters
        wrap_tau = 1.0 * np.ones((obs_dims,))
        mean_tau = 3e-2 * np.ones((obs_dims,))
        mean_amp = a_r * np.ones((obs_dims,))
        mean_bias = m_b * np.ones((obs_dims,))
        
        bnpp = lib.observations.bnpp.NonparametricPointProcess(
            svgp, wrap_tau, mean_tau, mean_amp, mean_bias, dt)

        ISI_density, log_rho_tilde, tau_tilde = sample_conditional_ISI(
            bnpp, 
            prng_state,
            num_samps,
            cisi_t_eval,
            isi_cond, 
            x_cond,
            sel_outdims, 
            int_eval_pts=1000,
            num_quad_pts=300,
            prior=True,
            jitter=1e-6, 
        )
        
        ISI_densities.append(np.array(ISI_density))
        log_rho_tildes.append(np.array(log_rho_tilde))
    
    return np.array(cisi_t_eval), ISI_densities, np.array(tau_tilde[0]), log_rho_tildes




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
    prng_state = jr.PRNGKey(seed)
    rng = np.random.default_rng(seed)
    
    
    
if __name__ == "__main__":
    main()