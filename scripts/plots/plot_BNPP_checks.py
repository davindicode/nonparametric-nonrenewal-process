import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.append("../../../GaussNeuro")
from gaussneuro import utils




def plot_BNPPs(fig):
    # test ELLs
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
    
    # KS p-values
    widths = [1]
    heights = [1]
    spec = fig.add_gridspec(ncols=len(widths), nrows=len(heights), width_ratios=widths, 
                        height_ratios=heights, top=0.4, bottom=0.0, 
                        left=0.0, right=1., wspace=0.1)

    ax = fig.add_subplot(spec[0, 0])

    mdls = len(reg_config_names[:])
    p_vals = [regression_dict[n]['KS_p_value'] for n in reg_config_names[:]]
    p_vals = [[v for v in d if v is not None] for d in p_vals]

    ax.violinplot(
        list(np.log(p_vals)), list(range(mdls)), points=20, widths=0.9,
        showmeans=True, showextrema=True, showmedians=True, 
        bw_method='silverman', 
    )
    #ax.set_yscale('log')

    
    
      
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
