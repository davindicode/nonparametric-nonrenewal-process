import os
import pickle

import matplotlib.pyplot as plt
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

    
    
def plot_hc3_tuning(fig, tuning_dict):
    # data
    eval_locs = tuning_dict["plot_pos_x_locs"]
    
    pos_Einvisi = tuning_dict["xtLR_mean_invISI"].mean(0)
    
    # plot
    W, H = 8, 6
    widths = [1] * W
    heights = [1] * H

    spec = fig.add_gridspec(ncols=len(widths), nrows=len(heights), width_ratios=widths, 
                            height_ratios=heights, top=1.0, bottom=0.0, 
                            left=0.0, right=1.0, wspace=0.3, hspace=0.8)

    for n in range(W):
        for m in range(H):
            ne = n + m*W
            if ne >= rmedian.shape[0]:
                continue

            ax = fig.add_subplot(spec[m, n])
            line, = ax.plot(eval_locs[:, 0], rmedian[ne, :], 'b', label='posterior mean')
            ax.fill_between(eval_locs[:, 0], rlower[ne, :], rupper[ne, :], 
                            color=line.get_color(), alpha=0.2, label='95% confidence')
            ax.plot(eval_locs[:, 0], rsamples[:, ne, :].T, 'b', alpha=0.2)
            ax.set_ylim(0)
    
    

def plot_instantaneous(fig, variability_dict):
    # data
    imISI, cvISI = 1. / variability_dict["mean_ISI"].mean(0), variability_dict["CV_ISI"].mean(0)
    skip = 2
    
    # plot
    W, H = 9, 5
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

            ax.plot(xx[ne], post_means[ne], c='r')
            ax.plot(xx[ne], lin_func[ne], c='b')
            
            ax.yaxis.get_major_locator().set_params(integer=True)

#     ax = fig.add_subplot(spec[-1, -1])
#     ax.scatter(variability_dict["linear_slope"], variability_dict["linear_R2"], 
#                marker='.', s=8, c='gray')
    
    
        
        
        
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
