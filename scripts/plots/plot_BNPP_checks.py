import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.append("../../../GaussNeuro")
from gaussneuro import utils




def plot_ARD(fig):
    warp_tau = tuning_dict['warp_tau']
    len_tau = tuning_dict['len_tau']
    len_deltas = tuning_dict['len_deltas']

    ARD_order = []
    for n in range(len_deltas.shape[0]):
        ARD_order.append(np.sum(len_deltas[n] < 3.) + 1)

    # plot
    widths = [1, 0.2]
    heights = [1]
    spec = fig.add_gridspec(ncols=len(widths), nrows=len(heights), width_ratios=widths, 
                            height_ratios=heights, top=0.3 + Y, bottom=0.0 + Y, 
                            left=0.0 + X, right=0.27 + X, wspace=0.2)

    markers = ['o', 's', 'x', '+']
    N = len(len_deltas)

    ax = fig.add_subplot(spec[0, 1])
    ax.hist(ARD_order, bins=np.linspace(0, 4, 5) + .5, color='lightgray')
    ax.set_yticks([])
    ax.set_xticks([1, 2, 3, 4])
    ax.set_ylabel('frequency', labelpad=1)
    ax.set_xlabel('order', labelpad=1)

    ax = fig.add_subplot(spec[0, 0])
    for n in range(N):
        lens = len_deltas[n]
        tlen = len_tau[n]
        ax.scatter(n, tlen, marker=markers[0], c='gray', s=20, 
                   label='lag 0' if n == 0 else None)
        for k in range(1, 4):
            ax.scatter(n, lens[k-1], marker=markers[k], c='gray', 
                       label='lag {}'.format(k) if n == 0 else None)

        ax.set_yscale('log')
        ax.set_ylabel('kernel timescales', labelpad=1)
        ax.plot([-.5, N + .5], 3. * np.ones(2), '--', color='gray')
        ax.set_xlim([-.5, N + .5])
        ax.set_xlabel('neuron index', labelpad=1)
        ax.set_xticks(list(range(N)))
        ax.set_xticklabels([])
        leg = ax.legend(loc='right', bbox_to_anchor=(1.5, 0.7), handletextpad=0.2)
        for k in range(4):
            leg.legend_handles[k].set_color('k')



def plot_BNPPs(fig):
    widths = [1]
    heights = [1, 1]

    spec = fig.add_gridspec(ncols=len(widths), nrows=len(heights), width_ratios=widths, 
                            height_ratios=heights, top=1.0, bottom=0.6, 
                            left=0.0, right=0.1, wspace=0.1)


    use_reg_config_names = [reg_config_names[k] for k in [4, 6, 7, 8, 9]]

    mdls = len(use_reg_config_names)
    test_lls = np.array([regression_dict[n]['test_ells'] for n in use_reg_config_names])
    train_lls = np.array([regression_dict[n]['train_ell'] for n in use_reg_config_names])

    ax = fig.add_subplot(spec[0, 0])
    means = test_lls.mean(-1)
    sems = test_lls.std(-1) / np.sqrt(test_lls.shape[-1])
    ax.errorbar(
        .5 + np.arange(mdls), means, yerr=sems, 
        alpha=0.5, color='black', capsize=10, linestyle='', 
        marker='o', markersize=4,
    )
    
    
      
def main():
    save_dir = "../saves/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.style.use(["paper.mplstyle"])

    ### load ###
    

    ### plot ###
    

    ### export ###
    plt.savefig(save_dir + "plot_synthetic.pdf")


if __name__ == "__main__":
    main()
