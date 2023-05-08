import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.special as sps

import jax

import sys
sys.path.append("../../../GaussNeuro")
from gaussneuro import utils
import gaussneuro as lib



def plot_spike_history_filters(fig, rng, prng_state, jitter, array_type, filter_conf=0):
    # GLM
    if filter_conf == 0:
        a, c = 4.5, 9.  # inverse width of the bumps, uniformity of the bump placement
        phi_lower, phi_upper = 10., 20.
        B = 8
        filter_length = 150
        
    elif filter_conf == 1:
        a, c = 6., 30.  # inverse width of the bumps, uniformity of the bump placement
        phi_lower, phi_upper = 17., 36.
        B = 16
        filter_length = 500
        
    else:
        raise ValueError
        
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
    obs_dims = 9

    num_induc = 10
    x_dims = 1

    # qSVGP inducing points
    induc_locs = np.linspace(0, filter_length, num_induc)[None, :, None].repeat(obs_dims, axis=0)
    u_mu = 0.0*rng.normal(size=(obs_dims, num_induc, 1))
    u_Lcov = 0.1*np.eye(num_induc)[None, ...].repeat(obs_dims, axis=0)

    # kernel
    len_fx = filter_length / 4. * np.ones((obs_dims, x_dims))  # GP lengthscale
    beta = 0.0 * np.ones((obs_dims, x_dims))
    len_beta = 1.5 * len_fx
    var_f = 0.1*np.ones(obs_dims)  # kernel variance

    kern = lib.GP.kernels.DecayingSquaredExponential(
        obs_dims, variance=var_f, lengthscale=len_fx, 
        beta=beta, lengthscale_beta=len_beta, array_type=array_type)
    gp = lib.GP.sparse.qSVGP(kern, induc_locs, u_mu, u_Lcov, RFF_num_feats=0, whitened=False)

    a_r = 0.*np.ones((obs_dims, 1))
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
    
    # plotting
    widths = [1]
    heights = [1, 1]
    spec = fig.add_gridspec(ncols=len(widths), nrows=len(heights), width_ratios=widths, 
                            height_ratios=heights, top=1.0, bottom=0.0, 
                            left=0.5, right=1., wspace=0.1)

    ax = fig.add_subplot(spec[0, 0])
    t = np.arange(glm_filter_t.shape[0])
    ax.plot(t, glm_filter_t[:, :, 0])


    ax = fig.add_subplot(spec[1, 0])
    t = np.arange(gp_filter_t.shape[1])

    tr = 2
    ax.plot(t, gp_filter_t[:, :, 1, 0].T)



def plot_rate_rescaling(fig, rng, dt = 0.001, p = 0.003, ts = 3000):
    """
    rate rescaling schematic
    """
    # data
    spikes_at = rng.binomial(1, p, size=(ts,))
    spike_times = np.where(spikes_at > 0)[0]

    time_t = np.arange(ts) * dt
    rates_t = 30. * (np.sin(time_t * (1 + np.exp( -1.5 * (time_t - time_t[-1]/2.) ** 2 )))**2 + \
        np.exp( -0.5 * (time_t - time_t[-1]/3.) ** 2 ))


    rtime_t = np.cumsum(rates_t * dt)
    rspike_times = np.ceil(rtime_t[spike_times] / dt).astype(int)
    
    # plotting
    widths = [1]
    heights = [1, 1]
    spec = fig.add_gridspec(ncols=len(widths), nrows=len(heights), width_ratios=widths, 
                            height_ratios=heights, top=1.0, bottom=0.85, 
                            left=0.0, right=0.35, hspace=0.4)

    ax = fig.add_subplot(spec[1, 0])
    for st in spike_times:
        ax.plot(np.ones(2)*st, [0, 1], c='k')
    ax.axis('off')
    ax.set_ylim([0, 1])
    ax.set_xlim([spike_times[0]-100, spike_times[-1]+100])

    ax = fig.add_subplot(spec[0, 0])
    for st in rspike_times:   
        ax.plot(np.ones(2)*st, [0, 1], c='gray')
    ax.axis('off')
    ax.set_ylim([0, 1])
    ax.set_xlim([rspike_times[0]-100, rspike_times[-1]+100])


    widths = [1]
    heights = [1]
    spec = fig.add_gridspec(ncols=len(widths), nrows=len(heights), width_ratios=widths, 
                            height_ratios=heights, top=0.8, bottom=0.0, 
                            left=0.0, right=0.35, wspace=0.1)

    ax = fig.add_subplot(spec[0, 0])

    ax.plot(time_t, rates_t)
    ax.plot(time_t, rtime_t)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('time')
    ax.set_ylabel('rescaled time')

    ax.set_xlim([time_t[0], time_t[-1]])
    ax.set_ylim([rtime_t[0], 1.3*rtime_t[-1]])
    for st, rst in zip(spike_times, rspike_times):
        ax.plot(np.ones(2)*st*dt, [0., rtime_t[st]], c='k')
        ax.plot([0., time_t[st]], np.ones(2)*rst*dt, c='lightgray')


def plot_rate_maps(fig, tuning_dict, names, titles):
    # data
    rates = [tuning_dict[n]["GT_rates"] for n in names]
    dx = 0.22
    
    # plot
    for en in range(len(names)):
        fig.text(0.1 + dx * en, 1.04, titles[en], fontsize=13, ha='center')

        widths = [1] * 3
        heights = [1] * 3
        spec = fig.add_gridspec(ncols=len(widths), nrows=len(heights), width_ratios=widths, 
                                height_ratios=heights, top=1.0, bottom=0.5, 
                                left=0.0 + dx * en, right=0.2 + dx * en, wspace=0.1)

        for n in range(3):
            for m in range(3):
                k = 3*n + m
                ax = fig.add_subplot(spec[m, n])

                ax.imshow(rates[en][..., k], vmin=0., 
                          origin='lower', cmap='viridis')
                lib.utils.plots.decorate_ax(ax)
                for spine in ax.spines.values():
                    spine.set_edgecolor(cs[k])
                    spine.set_linewidth(2)
    



def main():
    jax.config.update('jax_platform_name', 'cpu')
    #jax.config.update('jax_disable_jit', True)

    double_arrays = True

    if double_arrays:
        jax.config.update("jax_enable_x64", True)
        array_type = "float64"
    else:
        array_type = "float32"

    save_dir = "../saves/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.style.use(["paper.mplstyle"])
    
    ### data ###
    
    # seed everything
    seed = 123
    rng = np.random.default_rng(seed)
    
    
    ### plot ###
    
    
    ### export ###
    plt.savefig(save_dir + "plot_synthetic.pdf")
    
    
    
if __name__ == "__main__":
    main()