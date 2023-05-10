import os
import pickle

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

import sys
sys.path.append("../../../GaussNeuro")
from gaussneuro import utils



# computation functions
def get_tuning_curves(samples, percentiles, sm_filter, padding_modes):
    percentiles = utils.stats.percentiles_from_samples(
        samples, percentiles=percentiles)
    
    return [
        utils.stats.smooth_histogram(p, sm_filter, padding_modes) for p in percentiles
    ]


# plot functions
def plot_th1_tuning(fig, tuning_dict):
    # data
    eval_locs = tuning_dict["plot_hd_x_locs"]
    
    rsamples = 1 / tuning_dict["hd_mean_ISI"]
    cvsamples = tuning_dict["hd_CV_ISI"]
    
    sm_filter = np.ones(5) / 5
    padding_modes = ['periodic']
    percentiles = [0.025, 0.5, 0.975]
    
    rlower, rmedian, rupper = get_tuning_curves(rsamples, percentiles, sm_filter, padding_modes)
    cvlower, cvmedian, cvupper = get_tuning_curves(cvsamples, percentiles, sm_filter, padding_modes)
    
    # plot
    W, H = 9, 4
    widths = [1] * W
    heights = [1] * H

    spec = fig.add_gridspec(ncols=len(widths), nrows=len(heights), width_ratios=widths, 
                            height_ratios=heights, top=1.0, bottom=0.1, 
                            left=0.0, right=1.0, wspace=0.3, hspace=2.2)

    for n in range(W):
        for m in range(H):
            ne = n + m*W
            if ne >= rmedian.shape[0]:
                continue

            ax = fig.add_subplot(spec[m, n])
            ax.set_title('neuron {}'.format(ne + 1), fontsize=11)
            line, = ax.plot(eval_locs[:, 0], rmedian[ne, :], 'gray', label='posterior mean')
            ax.fill_between(eval_locs[:, 0], rlower[ne, :], rupper[ne, :], 
                            color=line.get_color(), alpha=0.2, label='95% confidence')
            ax.plot(eval_locs[:, 0], rsamples[:, ne, :].T, 'gray', alpha=0.2)
            ax.set_ylim(0)
            ax.set_xticks([0, 2*np.pi])
            ax.set_xticklabels([])
            if ne == 0:
                ax.set_ylabel('rate (Hz)')
                
            ax.yaxis.get_major_locator().set_params(integer=True)
                
            
    spec = fig.add_gridspec(ncols=len(widths), nrows=len(heights), width_ratios=widths, 
                            height_ratios=heights, top=0.9, bottom=0.0, 
                            left=0.0, right=1.0, wspace=0.3, hspace=2.2)

    for n in range(W):
        for m in range(H):
            ne = n + m*W
            if ne >= cvmedian.shape[0]:
                continue

            ax = fig.add_subplot(spec[m, n])
            line, = ax.plot(eval_locs[:, 0], cvmedian[ne, :], 'gray', label='posterior mean')
            ax.fill_between(eval_locs[:, 0], cvlower[ne, :], cvupper[ne, :], 
                            color=line.get_color(), alpha=0.2, label='95% confidence')
            ax.plot(eval_locs[:, 0], cvsamples[:, ne, :].T, 'gray', alpha=0.2)
            ax.set_ylim(0)
            ax.set_xticks([0, 2*np.pi])
            if ne == 0:
                ax.set_ylabel('CV')
                ax.set_xticklabels([r'$0$', r'$2\pi$'])
            else:
                ax.set_xticklabels([])
                
            ax.yaxis.get_major_locator().set_params(integer=True)
            
    fig.text(0.5, -0.05, 'head direction (radians)', fontsize=11, ha='center')

    
    
def plot_hc3_tuning(fig, tuning_dict, direction):
    # data
    eval_locs = tuning_dict["plot_xt_x_locs"]
    
    xt_rate = 1 / tuning_dict["xt{}_mean_ISI".format(direction)].mean(0)
    xt_CV = tuning_dict["xt{}_CV_ISI".format(direction)].mean(0)
    
    # plot
    white = "#ffffff"
    black = "#000000"
    red = "#ff0000"
    blue = "#0000ff"
    weight_map = utils.plots.make_cmap([blue, white, red], "weight_map")
    
    W, H = 9, 4
    widths = [1] * W
    heights = [1] * H

    spec = fig.add_gridspec(ncols=len(widths), nrows=len(heights), width_ratios=widths, 
                            height_ratios=heights, top=1.0, bottom=0.1, 
                            left=0.0, right=1.0, wspace=0.3, hspace=2.5)

    for n in range(W):
        for m in range(H):
            ne = n + m*W
            if ne >= xt_rate.shape[0]:
                continue

            g = xt_rate[ne].max()
            ax = fig.add_subplot(spec[m, n])
            ax.text(0.5, 1.4, 'neuron {}'.format(ne + 1), ha='center', fontsize=11, transform=ax.transAxes)
            ax.set_title("{:.1f} Hz".format(g), fontsize=10, pad=3)
            im = ax.imshow(xt_rate[ne], origin='lower', cmap='viridis', vmin=0., vmax=g, aspect='auto')
            ax.set_xticks([])
            if ne == 0:
                ylims = ax.get_ylim()
                ax.set_yticks(ylims)
                ax.set_yticklabels([r'$0$', r'$2\pi$'])
            else:
                ax.set_yticks([])
            ax.spines.right.set_visible(True)
            ax.spines.top.set_visible(True)

    spec = fig.add_gridspec(ncols=len(widths), nrows=len(heights), width_ratios=widths, 
                            height_ratios=heights, top=0.9, bottom=0.0, 
                            left=0.0, right=1.0, wspace=0.3, hspace=2.5)
    
    cspec = fig.add_gridspec(
        ncols=1,
        nrows=1,
        width_ratios=[1],
        height_ratios=[1],
        left=1.015,
        right=1.02,
        bottom=0.65,
        top=0.85,
    )
    ax = fig.add_subplot(cspec[0, 0])
    ax.set_title("     rate", fontsize=12, pad=10)
    utils.plots.add_colorbar(
        (fig, ax),
        im,
        ticktitle=r"",
        ticks=[0, g],
        ticklabels=["0", "max"],
        cbar_format=None,
        cbar_ori="vertical",
    )
    
    for n in range(W):
        for m in range(H):
            ne = n + m*W
            if ne >= xt_rate.shape[0]:
                continue
                
            field = np.log(xt_CV[ne])
            g = max(-field.min(), field.max())

            ax = fig.add_subplot(spec[m, n])
            ax.text(
                0.3, 1.04, "{:.1f}".format(np.exp(g)), ha="center", fontsize=10, color="red", 
                transform=ax.transAxes, 
            )
            ax.text(
                0.7, 1.04, "{:.1f}".format(np.exp(-g)), ha="center", fontsize=10, color="blue", 
                transform=ax.transAxes, 
            )
            im = ax.imshow(field, origin='lower', cmap=weight_map, vmin=-g, vmax=g, aspect='auto')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines.right.set_visible(True)
            ax.spines.top.set_visible(True)
            
    cspec = fig.add_gridspec(
        ncols=1,
        nrows=1,
        width_ratios=[1],
        height_ratios=[1],
        left=1.015,
        right=1.02,
        bottom=0.15,
        top=0.35,
    )
    ax = fig.add_subplot(cspec[0, 0])
    ax.set_title("     CV", fontsize=12, pad=10)
    utils.plots.add_colorbar(
        (fig, ax),
        im,
        ticktitle=r"",
        ticks=[-g, 0, g],
        ticklabels=["min", "0", "max"],
        cbar_format=None,
        cbar_ori="vertical",
    )
    
    spec = fig.add_gridspec(
        ncols=1,
        nrows=1,
        width_ratios=[1],
        height_ratios=[1],
        left=0.61,
        right=0.63,
        bottom=-0.02,
        top=-0.01,
    )
    ax = fig.add_subplot(spec[0, 0])
    ax.plot([0, 1], np.zeros(2), c='k', lw=3)
    ax.text(1.25, 0.0, '50 cm', va='center')
    ax.axis('off')
    
    fig.text(0.5, -0.04, '$x$ position along linear track', fontsize=11, ha='center')
    fig.text(-0.04, 0.5, r'LFP $\theta$-phase', fontsize=11, va='center', rotation=90)
    
    

def plot_instantaneous(fig, variability_dict):
    # data
    imISI, cvISI = 1. / variability_dict["mean_ISI"].mean(0), variability_dict["CV_ISI"].mean(0)
    skip = 2
    
    # plot
    W, H = 9, 4
    widths = [1] * W
    heights = [1] * H

    spec = fig.add_gridspec(ncols=len(widths), nrows=len(heights), width_ratios=widths, 
                            height_ratios=heights, top=1.0, bottom=0.0, 
                            left=0.0, right=1.0, wspace=0.3, hspace=0.6)

    for n in range(W):
        for m in range(H):
            ne = n + m*W
            if ne >= imISI.shape[0]:
                continue

            ax = fig.add_subplot(spec[m, n])
            ax.set_title('neuron {}'.format(ne + 1), fontsize=11)
            ax.scatter(imISI[ne, ::skip], cvISI[ne, ::skip], marker='.', c='gray', s=4, alpha=0.3)
            ax.set_ylim([0, cvISI.max() * 1.1])
            ax.set_yticks([0, 1, 2])
            
            xx = variability_dict["GP_post_locs"]
            lin_func = variability_dict["linear_slope"][:, None] * xx + \
                variability_dict["linear_intercept"][:, None]
            post_means = variability_dict["GP_post_mean"]

            ax.plot(xx[ne], post_means[ne], c='r', label='GP')
            ax.plot(xx[ne], lin_func[ne], c='b', label='linear')
            
            #ax.yaxis.get_major_locator().set_params(integer=True)
            #ax.yaxis.get_major_locator().set_params(integer=True)
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            
            if ne == imISI.shape[0] - 1:
                ax.legend(loc='right', bbox_to_anchor=(2.2, 0.8))
    
    fig.text(0.5, -0.1, 'rate (Hz)', fontsize=11, ha='center')
    fig.text(-0.05, 0.5, 'coefficient of variation', fontsize=11, va='center', rotation=90)
        
        
        
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
