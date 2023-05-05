import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

sys.path.append("../../../GaussNeuro")
from gaussneuro import utils



def plot_fit_stats(fig):
    ### KS and likelihood statistics ###
    widths = [1, 1]
    heights = [1]
    spec = fig.add_gridspec(ncols=len(widths), nrows=len(heights), width_ratios=widths, 
                            height_ratios=heights, top=1.0, bottom=0.5, 
                            left=0.03, right=0.3, wspace=0.1) 

    mdls = len(use_reg_config_names)
    p_vals = [regression_dict[n]['KS_p_value'] for n in use_reg_config_names]

    test_lls = np.array([
        [tll / ts * 1e3 for tll, ts in zip(regression_dict[n]['test_ells'], regression_dict[n]['test_datas_ts'])] 
        for n in use_reg_config_names
    ])
    train_lls = np.array([
        regression_dict[n]['train_ell'] / regression_dict[n]['train_data_ts'] * 1e3
        for n in use_reg_config_names
    ])


    ax = fig.add_subplot(spec[0, 0])
    for m in range(mdls):
        violin_parts = ax.violinplot(
            p_vals[m], [m], points=20, widths=0.9,
            showmeans=True, showextrema=True, showmedians=True, 
            bw_method='silverman', vert=False, 
        )

        # Make all the violin statistics marks red:
        for partname in ('cbars','cmins','cmaxes','cmeans','cmedians'):
            vp = violin_parts[partname]
            vp.set_edgecolor(cs[m])
            vp.set_linewidth(1)

        # Make the violin body blue with a red border:
        for vp in violin_parts['bodies']:
            vp.set_facecolor(cs[m])
            vp.set_edgecolor('gray')
            vp.set_linewidth(1)
            vp.set_alpha(0.3)

    ax.set_xlabel('KS $p$-values', labelpad=1)
    #ax.set_xlim([0, 1])
    #ax.set_xticks([0, 1])
    ax.set_ylim([-.5, mdls-.5])
    ax.set_yticks(np.arange(mdls))
    ax.set_yticklabels(use_names)
    [t.set_color(cs[m]) for m, t in enumerate(ax.yaxis.get_ticklabels())]


    ax = fig.add_subplot(spec[0, 1])
    means = test_lls.mean(-1)
    sems = test_lls.std(-1) / np.sqrt(test_lls.shape[-1])
    for m in range(mdls):
        ax.errorbar(
            means[m], .5 + m, xerr=sems[m], capsize=5, linestyle='', 
            marker='o', markersize=4, color=cs[m] 
        )
    ax.set_ylim([0, mdls])
    ax.set_yticks(.5 + np.arange(mdls))
    ax.set_yticklabels([])
    ax.set_xlabel('test ELL (nats/s)', labelpad=2)
    ax.xaxis.grid(True)
    
    
    
def plot_QQ(fig):
    ### QQ plots ###
    widths = [1] * len(use_reg_config_names)
    heights = [1]

    spec = fig.add_gridspec(ncols=len(widths), nrows=len(heights), width_ratios=widths, 
                            height_ratios=heights, top=1.0, bottom=0.81, 
                            left=0.35, right=1.0, wspace=0.1)    

    for en, name in enumerate(use_reg_config_names):
        quantiles = regression_dict[name]['KS_quantiles']

        ax = fig.add_subplot(spec[0, en])
        for n in range(len(quantiles)):
            if quantiles[n] is not None:
                ax.plot(quantiles[n], np.linspace(0., 1., len(quantiles[n])), 
                        c=cs[en], alpha=0.7)
        ax.plot(np.linspace(0., 1., 100), np.linspace(0., 1., 100), c='k')
        ax.set_aspect(1.)

        if en == 0:
            ax.set_xlabel('quantile', fontsize=11, labelpad=-8)
            ax.set_ylabel('EDF', fontsize=11, labelpad=-8)
        else:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
    
    
def plot_posteriors(fig):
    ### posterior visualizations ###
    widths = [1] * (len(visualize_names))
    heights = [0.2, 1, 1]

    spec = fig.add_gridspec(ncols=len(widths), nrows=len(heights), width_ratios=widths, 
                            height_ratios=heights, top=0.70, bottom=0.45, 
                            left=0.35, right=1.0, hspace=0.02, wspace=0.3)

    test_fold = 0
    ne = 7
    Ts, Te = 5000, 8000
    for en, n in enumerate(visualize_names):
        pred_ts = regression_dict[n]['pred_ts']

        sts = regression_dict[n]['pred_spiketimes'][test_fold][ne]
        ax = fig.add_subplot(spec[0, en])
        for st in sts:
            ax.plot(st*np.ones(2), np.linspace(0, 1, 2), c='k')
        ax.set_xlim([pred_ts[Ts], pred_ts[Te]])
        ax.axis('off')

        pred_lint = regression_dict[n]['pred_log_intensities']
        ax = fig.add_subplot(spec[1, en])
        ax.plot(pred_ts, pred_lint[test_fold][0, ne, :], c=cs[visualize_inds[en]])
        ax.set_xlim([pred_ts[Ts], pred_ts[Te]])
        ax.set_ylim([-3.5, 3.5])
        ax.set_yticks([])
        if en == 0:
            ax.set_ylabel('log CIF')
        ax.set_xticks([])

        stss = regression_dict[n]['sample_spiketimes'][test_fold][ne]
        ax = fig.add_subplot(spec[2, en])
        for num, st_samples in enumerate(stss):
            for st in st_samples:
                ax.plot(st*np.ones(2), num + np.linspace(0, 1, 2), c=cs[visualize_inds[en]])
        ax.set_xlim([pred_ts[Ts], pred_ts[Te]])
        ax.set_yticks([])
        ax.spines[['left', 'bottom']].set_visible(False)
        ax.set_xticks([])
        if en == 0:
            ax.set_xlabel('time', labelpad=0)

    widths = [1]
    heights = [1]

    spec = fig.add_gridspec(ncols=len(widths), nrows=len(heights), width_ratios=widths, 
                            height_ratios=heights, top=0.5, bottom=0.4, 
                            left=0.35, right=0.5)

    ax = fig.add_subplot(spec[0, 0])
    ax.text(1.0, 0.0, '1 s')
    ax.plot([0.0, 0.8], [0.0, 0.0], lw=2, c='k')
    
    
    
def plot_kernel_lens(fig):
    ### kernel lengthscales ###
    widths = [1, 0.2]
    heights = [1]
    spec = fig.add_gridspec(ncols=len(widths), nrows=len(heights), width_ratios=widths, 
                            height_ratios=heights, top=0.3, bottom=0.0, 
                            left=0.0, right=0.27, wspace=0.2)

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
            
            
            
def plot_instantaneous(fig):
    ### moment-by-moment ###
    widths = [1] * len(plot_units)
    heights = [1]

    spec = fig.add_gridspec(ncols=len(widths), nrows=len(heights), width_ratios=widths, 
                            height_ratios=heights, top=0.3, bottom=0.2, 
                            left=0.75, right=1.0, wspace=0.3, hspace=0.8)

    skip = 10
    for en, ne in enumerate(plot_units):
        ax = fig.add_subplot(spec[0, en])
        ax.set_title('neuron {}'.format(ne + 1), fontsize=11)
        ax.scatter(imISI[ne, ::skip], cvISI[ne, ::skip], marker='.', c='gray', s=4, alpha=0.3)
        ax.set_ylim([0, cvISI.max() * 1.1])
        ax.set_yticks([0, 1, 2])


    widths = [1, 1, 1]
    heights = [1]

    spec = fig.add_gridspec(ncols=len(widths), nrows=len(heights), width_ratios=widths, 
                            height_ratios=heights, top=0.15, bottom=0.0, 
                            left=0.75, right=1.0, wspace=0.3, hspace=0.8)

    ax = fig.add_subplot(spec[0, 0])
    ax.scatter(variability_dict["mean_ISI"].mean(0).mean(1), variability_dict["CV_ISI"].mean(0).mean(1), 
               marker='.', s=8, c='gray')


    ax = fig.add_subplot(spec[0, 1])
    ax.scatter(variability_dict["linear_slope"], variability_dict["linear_R2"], 
               marker='.', s=8, c='gray')#variability_dict["linear_slope"])


    ax = fig.add_subplot(spec[0, 2])

    R2 = [variability_dict["linear_R2"], variability_dict["GP_R2"]]
    ax.violinplot(R2, np.arange(2), points=20, widths=0.9,
        showmeans=True, showextrema=True, showmedians=True, 
        bw_method='silverman', 
    )

    for en, r2 in enumerate(R2):
        ax.scatter(en + 0.1 * rng.normal(size=r2.shape), r2, c='gray')
            
            

def plot_th1_tuning(fig):
    ### tuning ###
    widths = [1] * len(plot_units)
    heights = [1, 1]
    spec = fig.add_gridspec(ncols=len(widths), nrows=len(heights), width_ratios=widths, 
                            height_ratios=heights, top=0.3, bottom=0.0, 
                            left=0.35, right=0.55)

    for en, n in enumerate(plot_units):
        ax = fig.add_subplot(spec[0, en])
        line, = ax.plot(eval_locs[:, 0], rmedian[n, :], 'gray', label='posterior mean')
        ax.fill_between(eval_locs[:, 0], rlower[n, :], rupper[n, :], 
                        color=line.get_color(), alpha=0.2, label='95% confidence')
        ax.plot(eval_locs[:, 0], rsamples[:, n, :].T, 'gray', alpha=0.2)

        ax = fig.add_subplot(spec[1, en])
        line, = ax.plot(eval_locs[:, 0], cvmedian[n, :], 'gray', label='posterior mean')
        ax.fill_between(eval_locs[:, 0], cvlower[n, :], cvupper[n, :], 
                        color=line.get_color(), alpha=0.2, label='95% confidence')
        ax.plot(eval_locs[:, 0], cvsamples[:, n, :].T, 'gray', alpha=0.2)


    widths = [1]
    heights = [1, 1]
    spec = fig.add_gridspec(ncols=len(widths), nrows=len(heights), width_ratios=widths, 
                            height_ratios=heights, top=0.3, bottom=0.0, 
                            left=0.6, right=0.7)

    ne_inds = [20, 27]
    pts_inds = [0, 2]
    for en in range(2):
        ax = fig.add_subplot(spec[en, 0])
        ax.plot(tuning_dict['ISI_t_eval'], 
                tuning_dict['ISI_densities'][pts_inds[en], :, ne_inds[en], :].T, 
                alpha=0.02, color='k')


def plot_hc3_tuning(fig):
    ### tuning ###
    widths = [1] * len(plot_units)
    heights = [1, 1]
    spec = fig.add_gridspec(ncols=len(widths), nrows=len(heights), width_ratios=widths, 
                            height_ratios=heights, top=0.3, bottom=0.0, 
                            left=0.35, right=0.55)

    for en, n in enumerate(plot_units):
        ax = fig.add_subplot(spec[0, en])
        ax.imshow(pos_invEisi[n], origin='lower', cmap='gray')

        ax = fig.add_subplot(spec[1, en])
        ax.imshow(pos_CV[n], origin='lower', cmap='gray')



    widths = [1]
    heights = [1, 1]
    spec = fig.add_gridspec(ncols=len(widths), nrows=len(heights), width_ratios=widths, 
                            height_ratios=heights, top=0.3, bottom=0.0, 
                            left=0.6, right=0.7)

    ne_inds = [20, 27]
    pts_inds = [0, 2]
    for en in range(2):
        ax = fig.add_subplot(spec[en, 0])
        ax.plot(tuning_dict['ISI_t_eval'], 
                tuning_dict['ISI_densities'][pts_inds[en], :, ne_inds[en], :].T, 
                alpha=0.02, color='k')

        
        

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
