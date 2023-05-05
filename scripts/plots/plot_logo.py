import os

import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
import jax.random as jr

import sys
sys.path.append("../../../GaussNeuro")
from gaussneuro import utils
import gaussneuro as lib


def logo_data(prng_state, rng, num_samps, a_m, b_m, mean_tau, L, evalsteps):
    dt = 1e-3  # s
    obs_dims = 1
    num_induc = 8
    RFF_num_feats = 3000
    cisi_t_eval = jnp.linspace(0.0, L, evalsteps)
    sel_outdims = jnp.arange(obs_dims)
    
    x_cond, isi_cond = None, None
    
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
    mean_tau = mean_tau * np.ones((obs_dims,))
    mean_amp = a_m * np.ones((obs_dims,))
    mean_bias = b_m * np.ones((obs_dims,))

    bnpp = lib.observations.bnpp.NonparametricPointProcess(
        svgp, wrap_tau, mean_tau, mean_amp, mean_bias, dt)

    ISI_density = bnpp.sample_conditional_ISI(
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
    
    return np.array(cisi_t_eval), np.array(ISI_density)



def main():
    save_dir = "../saves/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.style.use(["paper.mplstyle"])

    # seed everything
    seed = 123
    prng_state = jr.PRNGKey(seed)
    rng = np.random.default_rng(seed)
    
    ### data ###
    cisi_t_eval, ISI_densities = logo_data(
        prng_state, 
        rng, 
        num_samps = 100, 
        a_m = -6., 
        b_m = 1.5, 
        mean_tau = 3e-1, 
        L = 2., 
        evalsteps = 120, 
    )
    
    ### plot ###
    fig = plt.figure(figsize=(2, 2))
    fig.set_facecolor('white')
    
    widths = [1]
    heights = [1]
    spec = fig.add_gridspec(ncols=len(widths), nrows=len(heights), width_ratios=widths, 
                            height_ratios=heights, top=1.0, bottom=0.0, 
                            left=0.0, right=1.0)

    ax = fig.add_subplot(spec[0, 0])
    ax.plot(cisi_t_eval, ISI_densities[:, 0, :].T, c='k', alpha=0.1)

    ax.set_xlim([cisi_t_eval[0] - 0.03, cisi_t_eval[-1]])
    ax.set_ylim(0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(-0.15, -0.15, r'$0$', fontsize=11)

    ax.set_xlabel('ISI')
    ax.set_ylabel('probablity')
    
    ### export ###
    plt.savefig(save_dir + "logo.png", dpi=100)


if __name__ == "__main__":
    main()
