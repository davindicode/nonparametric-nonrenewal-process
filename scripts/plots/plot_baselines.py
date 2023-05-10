import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

import jax
import jax.random as jr

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
    
    # RCB basis
    a = a * np.ones((obs_dims, 1))
    c = c * np.ones((obs_dims, 1))
    phi_h = np.linspace(phi_lower, phi_upper, B)[:, None, None].repeat(obs_dims, axis=1)
    w_h = np.zeros((B, obs_dims, 1))
    for o in range(obs_dims):
        w_h[o, o, :] = 1.

    flt = lib.filters.RaisedCosineBumps(
        a,
        c,
        w_h,
        phi_h,
        filter_length,
    )
    
    glm_basis = flt.sample_prior(prng_state, 1, None, jitter)
    glm_basis = np.array(glm_basis[0])  # (filter_length, outs, 1)
    
    # RCB samples
    ini_var = 1.0 / np.sqrt(B)
    w_h = (np.sqrt(ini_var) * rng.normal(size=(B, obs_dims)))[..., None]

    flt = lib.filters.RaisedCosineBumps(
        a,
        c,
        w_h,
        phi_h,
        filter_length,
    )
    
    glm_filter_t, _ = flt.sample_posterior(prng_state, 1, False, None, jitter)
    glm_filter_t = np.array(glm_filter_t[0])  # (filter_length, outs, 1)
    
    # GP samples
    obs_dims = B

    num_induc = 10
    x_dims = 1

    # qSVGP inducing points
    induc_locs = np.linspace(0, filter_length, num_induc)[None, :, None].repeat(obs_dims, axis=0)
    u_mu = 0.0 * rng.normal(size=(obs_dims, num_induc, 1))
    u_Lcov = 0.1 * np.eye(num_induc)[None, ...].repeat(obs_dims, axis=0)

    # kernel
    len_fx = filter_length / 4. * np.ones((obs_dims, x_dims))  # GP lengthscale
    beta = 0.0 * np.ones((obs_dims, x_dims))
    len_beta = 1.5 * len_fx
    var_f = 1.0 * np.ones(obs_dims)  # kernel variance

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
    t = np.arange(glm_filter_t.shape[0]) - glm_filter_t.shape[0]
    
    widths = [1]
    heights = [0.7, 1]
    spec = fig.add_gridspec(ncols=len(widths), nrows=len(heights), width_ratios=widths, 
                            height_ratios=heights, top=0.99, bottom=0.52, 
                            left=0.45, right=0.75, hspace=0.4)

    ax = fig.add_subplot(spec[0, 0])
    ax.set_title('raised cosine basis', fontsize=12)
    ax.plot(t, glm_basis[:, :, 0])
    ax.set_xlim([t[0], t[-1]])
    ax.set_xticklabels([])
    ax.set_ylabel('amplitude (a.u.)                              ')

    ax = fig.add_subplot(spec[1, 0])
    ax.set_title('random examples', fontsize=12)
    ax.plot(t, glm_filter_t[:, :, 0], c='b', label='RCB', alpha=0.6)
    #ax.plot(t, gp_filter_t[:, :, 1, 0].T, c='r', label='DSE GP', alpha=0.8)
    ax.set_xlim([t[0], t[-1]])
    ax.set_xlabel('lag (ms)')
    
#     handles, labels = ax.get_legend_handles_labels()
#     labels, ids = np.unique(labels, return_index=True)
#     handles = [handles[i] for i in ids]
#     ax.legend(handles, labels, loc='best')
    


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
    heights = [1]
    spec = fig.add_gridspec(ncols=len(widths), nrows=len(heights), width_ratios=widths, 
                            height_ratios=heights, top=1.0, bottom=0.53, 
                            left=0.05, right=0.37, wspace=0.1)

    ax = fig.add_subplot(spec[0, 0])

    ax.plot(time_t, rates_t)
    ax.plot(time_t, rtime_t)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('time')
    ax.set_ylabel('rescaled time', color='gray')

    ax.set_xlim([time_t[0], time_t[-1]])
    ax.set_ylim([rtime_t[0], 1.05*rtime_t[-1]])
    for st, rst in zip(spike_times, rspike_times):
        ax.plot(np.ones(2)*st*dt, [0., rtime_t[st]], c='k')
        ax.plot([0., time_t[st]], np.ones(2)*rst*dt, c='lightgray')
        
    widths = [1]
    heights = [1]
    spec = fig.add_gridspec(ncols=len(widths), nrows=len(heights), width_ratios=widths, 
                            height_ratios=heights, top=0.46, bottom=0.43, 
                            left=0.05, right=0.37, hspace=0.8)

    ax = fig.add_subplot(spec[0, 0])
    for st in spike_times:
        ax.plot(np.ones(2)*st*dt, [0, 1], c='k')
    ax.axis('off')
    ax.set_ylim([0, 1])
    ax.set_xlim([time_t[0], time_t[-1]])
    
    widths = [1]
    heights = [1]
    spec = fig.add_gridspec(ncols=len(widths), nrows=len(heights), width_ratios=widths, 
                            height_ratios=heights, top=1.0, bottom=0.53, 
                            left=0.0, right=0.015, hspace=0.8)

    ax = fig.add_subplot(spec[0, 0])
    for st in rspike_times:   
        ax.plot([0, 1], np.ones(2)*st*dt, c='gray')
    ax.axis('off')
    ax.set_ylim([rtime_t[0], 1.05*rtime_t[-1]])
    ax.set_xlim([0, 1])

        
        
def plot_ARDs(fig, tuning_dict, ard_names, cs):
    """
    Plot kernel lengthscales
    """
    widths = [1]
    heights = [1] * len(ard_names)
    spec = fig.add_gridspec(ncols=len(widths), nrows=len(heights), width_ratios=widths, 
                            height_ratios=heights, top=0.99, bottom=0.59, 
                            left=0.85, right=1.0, hspace=0.4)

    markers = ['o', 's', 'x', '+']

    for en, name in enumerate(ard_names):
        warp_tau = tuning_dict[name]['warp_tau']
        len_tau = tuning_dict[name]['len_tau']
        len_deltas = tuning_dict[name]['len_deltas']
            
        N = len(len_deltas)

        ax = fig.add_subplot(spec[en, 0])
        if en == 0:
            ax.set_title(r'free $\tau_w$', fontsize=12, fontweight='bold')
        else:
            ax.set_title(r'fixed $\tau_w$', fontsize=12, fontweight='bold')
            
        for n in range(N):
            lens = len_deltas[n]
            tlen = len_tau[n]
            ax.scatter(n, tlen, marker=markers[0], c=cs[n], s=20, 
                       label='lag 0' if n == 0 else None)
            for k in range(1, 4):
                ax.scatter(n, lens[k-1], marker=markers[k], c=cs[n], 
                           label='lag {}'.format(k) if n == 0 else None)

            ax.set_yscale('log')
            ax.plot([-.5, N + .5], 3. * np.ones(2), '--', color='gray')
            ax.set_xlim([-.5, N + .5])
            ax.set_xticks(list(range(N)))
            ax.set_xticklabels([])
            
            if en == 1:
                ax.set_ylabel('                         kernel timescales', labelpad=1)
                leg = ax.legend(loc='right', bbox_to_anchor=(0.9, -0.7), 
                                handletextpad=0.2, ncol=2)
                for k in range(4):
                    leg.legend_handles[k].set_color('k')
                
                ax.set_xlabel('neuron index', labelpad=1)
            
        

def plot_rate_maps(fig, tuning_dict, names, titles, cs):
    dx = 0.2
    
    for en in range(len(names)):
        widths = [1] * 3
        heights = [1] * 3
        spec = fig.add_gridspec(ncols=len(widths), nrows=len(heights), width_ratios=widths, 
                                height_ratios=heights, top=0.33, bottom=0.0, 
                                left=0.0 + dx * en, right=0.17 + dx * en, wspace=0.1)

        for n in range(3):
            for m in range(3):
                k = 3*n + m
                
                ax = fig.add_subplot(spec[m, n])
                if m == 0 and n == 1:
                    ax.set_title(titles[en], fontsize=13)
                    
                if en == 0:
                    rs = tuning_dict[names[en]]["pos_rates"][..., k]
                elif en == len(names) - 1:
                    rs = 1 / tuning_dict[names[en]]["pos_mean_ISI"].mean(0)[k]
                else:
                    rs = tuning_dict[names[en]]["pos_rates"][k]
                    
                    
                ax.imshow(rs, vmin=0., origin='lower', cmap='viridis')
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
    
    name = 'synthetic_results'
    results = pickle.load(
        open(save_dir + name + ".p", "rb")
    )
    
    jitter = 1e-5
    dt = 0.001

    tuning_dict, regression_dict = results["tuning"], results["regression"]
    reg_config_names = list(regression_dict.keys())

    titles = [
        'ground truth',
        'Poisson', 
        'rescaled gamma', 
        'conditional Poisson', 
        'nonparametric', 
    ]
    tuning_names = ['GT'] + reg_config_names[:3] + reg_config_names[-1:]
    ard_names = reg_config_names[-2:]

    cs = [
        'tab:blue',
        'tab:orange', 
        'tab:green',
        'tab:red',
        'tab:purple',
        'tab:brown',
        'tab:pink',
        'tab:gray',
        'tab:cyan', 
    ]  # cell ID colour


    # seed everything
    seed = 123
    rng = np.random.default_rng(seed)
    prng_state = jr.PRNGKey(seed)
    
    ### plot ###
    fig = plt.figure(figsize=(10, 5))



    fig.text(-0.02, 1.01, 'A', fontsize=15, ha='center', fontweight='bold')
    plot_rate_rescaling(fig, rng, dt, p = 0.003, ts = 3000)


    fig.text(0.4, 1.01, 'B', fontsize=15, ha='center', fontweight='bold')
    plot_spike_history_filters(fig, rng, prng_state, jitter, array_type, filter_conf=0)


    fig.text(0.8, 1.01, 'C', fontsize=15, ha='center', fontweight='bold')
    plot_ARDs(fig, tuning_dict, ard_names, cs)


    fig.text(-0.02, 0.36, 'D', fontsize=15, ha='center', fontweight='bold')
    plot_rate_maps(fig, tuning_dict, tuning_names, titles, cs)
    
    ### export ###
    plt.savefig(save_dir + "baselines.pdf")
    
    
    
if __name__ == "__main__":
    main()