import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.append("../../../GaussNeuro")
from gaussneuro import utils
import gaussneuro as lib



def plot_ground_truth(fig, tuning_dict, cs):
    unit_ISI_t_eval = tuning_dict["unit_ISI_t_eval"]
    GT_unit_renewals = tuning_dict["GT"]["GT_unit_renewals"]
    GT_rates = tuning_dict["GT"]["GT_rates"]
    
    ### ground truth ###
    fig.text(0.1, 1.04, 'true rate maps', fontsize=13, ha='center')

    widths = [1] * 3
    heights = [1] * 3
    spec = fig.add_gridspec(ncols=len(widths), nrows=len(heights), width_ratios=widths, 
                            height_ratios=heights, top=1.0, bottom=0.5, 
                            left=0.0, right=0.2, wspace=0.1)

    for n in range(3):
        for m in range(3):
            k = 3*n + m
            ax = fig.add_subplot(spec[m, n])

            ax.imshow(GT_rates[..., k], vmin=0., 
                      origin='lower', cmap='viridis')
            lib.utils.plots.decorate_ax(ax)
            for spine in ax.spines.values():
                spine.set_edgecolor(cs[k])
                spine.set_linewidth(2)


    widths = [1] * 3
    heights = [1] * 3
    spec = fig.add_gridspec(ncols=len(widths), nrows=len(heights), width_ratios=widths, 
                            height_ratios=heights, top=0.37, bottom=0.0, 
                            left=0.0, right=0.2, wspace=0.1)

    for n in range(3):
        for m in range(3):
            k = 3*n + m

            ax = fig.add_subplot(spec[m, n])
            ax.plot(unit_ISI_t_eval, GT_unit_renewals[:, k], c=cs[k])
            ax.set_xlim([-0.1, unit_ISI_t_eval[-1]])
            ax.set_ylim([0, 2.])
            ax.set_xticks([])
            ax.set_yticks([])

            if m == 0 and n == 1:
                ax.set_title('true renewal shapes', fontsize=13)

            if m == 1 and n == 0:
                ax.set_ylabel('density')

            if m == 2 and n == 1:
                ax.set_xlabel('ISI (a.u.)')



def plot_tuning_and_ISIs(fig, tuning_dict, name, cs):
    pos_mean_ISI = tuning_dict[name]["pos_mean_ISI"]
    pos_est_rate = 1 / pos_mean_ISI.mean(0)

    ISI_t_eval = tuning_dict["ISI_t_eval"]
    ISI_densities = tuning_dict[name]["ISI_densities"]
    ISI_neuron_conds = tuning_dict["ISI_neuron_conds"]
    GT_ISI_densities = tuning_dict["GT"]["GT_ISI_densities"]
    
    ### tuning curves ###
    fig.text(0.37, 1.04, 'inferred rates', fontsize=13, ha='center')

    widths = [1] * 3
    heights = [1] * 3
    spec = fig.add_gridspec(ncols=len(widths), nrows=len(heights), width_ratios=widths, 
                            height_ratios=heights, top=1.0, bottom=0.5, 
                            left=0.27, right=0.47, wspace=0.1)

    for n in range(3):
        for m in range(3):
            k = 3*n + m
            ax = fig.add_subplot(spec[m, n])

            ax.imshow(pos_est_rate[k], vmin=0., 
                      origin='lower', cmap='viridis')
            lib.utils.plots.decorate_ax(ax)
            for spine in ax.spines.values():
                spine.set_edgecolor(cs[k])
                spine.set_linewidth(2)

    ### conditional ISI distributions ###
    widths = [1] * 3
    heights = [1]
    spec = fig.add_gridspec(ncols=len(widths), nrows=len(heights), width_ratios=widths, 
                            height_ratios=heights, top=0.35, bottom=0.0, 
                            left=0.3, right=0.76, wspace=0.1)


    for en, ne in enumerate(ISI_neuron_conds):
        ax = fig.add_subplot(spec[0, en])
        dens = ISI_densities[:, en, :]
        gt_dens = GT_ISI_densities[en, :, ne]
        ax.plot(ISI_t_eval, gt_dens, c=cs[ne], alpha=1.0, 
                label='truth' if en == 1 else None)
        ax.plot(ISI_t_eval, dens.T, c='k', alpha=0.03, 
                label='posterior samples' if en == 1 else None)

        ax.set_yticks([])
        if en == 0:
            ax.set_ylabel('probability', labelpad=2)
            ax.set_xlabel('ISI (s)', labelpad=-6)
        else:
            ax.set_xticklabels([])


    widths = [1]
    heights = [1]
    spec = fig.add_gridspec(ncols=len(widths), nrows=len(heights), width_ratios=widths, 
                            height_ratios=heights, top=1.0, bottom=0.0, 
                            left=0.3, right=0.6, wspace=0.1)

    ax = fig.add_subplot(spec[0, 0])

    ax.annotate("", xy=(0.1, 0.3), xytext=(0.05, 0.6), rotation=np.pi/2., xycoords='axes fraction', 
        arrowprops=dict(arrowstyle="simple, head_width=0.7, head_length=1.0, tail_width=0.1", 
                        color='gray', alpha=0.3), 
        annotation_clip=False)

    ax.annotate("", xy=(0.5, 0.3), xytext=(0.24, 0.56), rotation=np.pi/2., xycoords='axes fraction', 
        arrowprops=dict(arrowstyle="simple, head_width=0.7, head_length=1.0, tail_width=0.1", 
                        color='gray', alpha=0.3), 
        annotation_clip=False)

    ax.annotate("", xy=(1.03, 0.3), xytext=(0.425, 0.575), rotation=np.pi/2., xycoords='axes fraction', 
        arrowprops=dict(arrowstyle="simple, head_width=0.7, head_length=1.0, tail_width=0.1", 
                        color='gray', alpha=0.3), 
        annotation_clip=False)

    ax.axis('off')


def plot_kernel_lens(fig, data_dict, cs):
    warp_tau = data_dict['warp_tau']
    len_tau = data_dict['len_tau']
    len_deltas = data_dict['len_deltas']
    
    ARD_order = []
    for n in range(len_deltas.shape[0]):
        ARD_order.append(np.sum(len_deltas[n] < 3.) + 1)
    
    ### kernel lengthscales ###
    widths = [1, 0.5]
    heights = [1]
    spec = fig.add_gridspec(ncols=len(widths), nrows=len(heights), width_ratios=widths, 
                            height_ratios=heights, top=1.0, bottom=0.6, 
                            left=0.56, right=0.76, wspace=0.4)

    markers = ['o', 's', 'x', '+']

    ax = fig.add_subplot(spec[0, 0])
    for n in range(9):
        lens = len_deltas[n]
        tlen = len_tau[n]
        ax.scatter(n, tlen, marker=markers[0], c=cs[n], s=20, 
                   label='lag 0' if n == 0 else None)
        for k in range(1, 4):
            ax.scatter(n, lens[k-1], marker=markers[k], c=cs[n], 
                       label='lag {}'.format(k) if n == 0 else None)

        ax.set_yscale('log')
        ax.set_ylabel('kernel timescales', labelpad=1)
        ax.plot([-.5, 9.5], 3. * np.ones(2), '--', color='gray')
        ax.set_xlim([-.5, 9.5])
        ax.set_xlabel('neuron index', labelpad=1)
        ax.set_xticks(list(range(9)))
        ax.set_xticklabels([1, '', '', '', 5, '', '', '', 9])
        leg = ax.legend(loc='right', bbox_to_anchor=(1.9, -0.75), handletextpad=0.2)
        for k in range(4):
            leg.legend_handles[k].set_color('k')

    ax = fig.add_subplot(spec[0, 1])
    ax.hist(ARD_order, bins=np.linspace(0, 4, 5) + .5, color='lightgray')
    ax.set_yticks([])
    ax.set_xticks([1, 2, 3, 4])
    ax.set_ylabel('frequency', labelpad=1)
    ax.set_xlabel('order', labelpad=1)

    

def plot_QQ(fig, use_reg_config_names, use_names, regression_dict, cs):
    ### KS statistics ###
    widths = [1, 1]
    heights = [1, 1]
    spec = fig.add_gridspec(ncols=len(widths), nrows=len(heights), width_ratios=widths, 
                            height_ratios=heights, top=1.0, bottom=0.05, 
                            left=0.85, right=1.2, hspace=0.3, wspace=0.1)

    for en, n in enumerate(use_reg_config_names):
        ax = fig.add_subplot(spec[en // 2, en % 2])
        ax.set_title(use_names[en], fontsize=12, fontweight='bold')

        dd = regression_dict[n]
        sort_cdfs = dd['KS_quantiles']

        for n in range(len(sort_cdfs)):
            if sort_cdfs[n] is not None:
                ax.plot(sort_cdfs[n], np.linspace(0., 1., len(sort_cdfs[n])), c=cs[n], alpha=0.8)
        ax.plot(np.linspace(0., 1., 100), np.linspace(0., 1., 100), c='k')
        ax.set_xlim([-0.03, 1.03])
        ax.set_xticks([0, 1])
        if en // 2 == 0:
            ax.set_xticklabels([])
        ax.set_ylim([-0.03, 1.03])
        ax.set_yticks([0, 1])
        if en % 2 == 1:
            ax.set_yticklabels([])
        ax.set_aspect(1.0)

    fig.text(1.025, -0.075, 'quantile', ha='center', fontsize=11)
    fig.text(0.815, 0.525, 'empirical distribution function', rotation=90, va='center', fontsize=11)




def main():
    save_dir = "../saves/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.style.use(["paper.mplstyle"])

    ### load ###
    datadir = '../saves/'
    name = 'synthetic_results'

    results = pickle.load(
        open(datadir + name + ".p", "rb")
    )

    regression_dict = results["regression"]
    reg_config_names = list(regression_dict.keys())

    name = reg_config_names[-1]
    tuning_dict = results["tuning"]

    ### plot ###
    use_reg_config_names = [reg_config_names[k] for k in [0, 1, 2, 4]]
    use_names = [
        'Poisson', 
        'conditional Poisson', 
        'rescaled gamma', 
        'nonparametric', 
    ]
    
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
    
    
    fig = plt.figure(figsize=(8, 3))

    fig.text(-0.03, 1.05, 'A', fontsize=15, ha='center', fontweight='bold')
    plot_ground_truth(fig, tuning_dict, cs)
    
    fig.text(0.26, 1.05, 'B', fontsize=15, ha='center', fontweight='bold')
    plot_tuning_and_ISIs(fig, tuning_dict, name, cs)
    plot_kernel_lens(fig, tuning_dict[name], cs)
    
    fig.text(0.81, 1.05, 'C', fontsize=15, ha='center', fontweight='bold')
    plot_QQ(fig, use_reg_config_names, use_names, regression_dict, cs)

    fig.text(1.23, 1.05, 'S', fontsize=15, alpha=0., ha='center')  # space


    ### export ###
    plt.savefig(save_dir + "synthetic.pdf")


if __name__ == "__main__":
    main()
