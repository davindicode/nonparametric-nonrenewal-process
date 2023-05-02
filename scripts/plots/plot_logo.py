import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.special as sps

sys.path.append("../../../GaussNeuro")
from gaussneuro import utils


def logo():
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

        ISI_density = bnpp.sample_conditional_ISI(
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
    
    return np.array(cisi_t_eval), ISI_densities



def main():
    save_dir = "../output/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.style.use(["paper.mplstyle"])

    # plot
    fig = plt.figure(figsize=(4, 4))

    logo(fig)

    plt.savefig(save_dir + "logo.png", dpi=100)


if __name__ == "__main__":
    main()
