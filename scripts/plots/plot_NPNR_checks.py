import os
import pickle

import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append("../../../GaussNeuro")
from gaussneuro import utils


def plot_NPNRs(
    fig, X, Y, rng, th1_npnr_dict, hc3_npnr_dict, th1_names, hc3_names, kernel_names
):
    # reorder
    th1_names = [
        th1_names[0],
        th1_names[2],
        th1_names[4],
        th1_names[6],
        th1_names[1],
        th1_names[3],
        th1_names[5],
        th1_names[7],
    ]
    hc3_names = [
        hc3_names[0],
        hc3_names[2],
        hc3_names[4],
        hc3_names[6],
        hc3_names[1],
        hc3_names[3],
        hc3_names[5],
        hc3_names[7],
    ]
    use_names = kernel_names + kernel_names

    # th1 plot
    fig.text(
        0.235 + X,
        1.02 + Y,
        "mouse thalamus",
        fontsize=12,
        ha="center",
        fontweight="bold",
    )
    fig.text(
        0.765 + X,
        1.02 + Y,
        "rat hippocampus",
        fontsize=12,
        ha="center",
        fontweight="bold",
    )

    fig.text(
        -0.23 + X,
        0.25 + Y,
        r"learned $\tau_w$",
        fontsize=12,
        va="center",
        rotation=90,
        color="tab:blue",
    )
    fig.text(
        -0.23 + X,
        0.75 + Y,
        r"fixed $\tau_w = \langle ISI \rangle$",
        fontsize=12,
        va="center",
        rotation=90,
        color="tab:green",
    )

    widths = [0.9, 1]
    heights = [1]
    spec = fig.add_gridspec(
        ncols=len(widths),
        nrows=len(heights),
        width_ratios=widths,
        height_ratios=heights,
        top=1.0 + Y,
        bottom=0.0 + Y,
        left=0.0 + X,
        right=0.47 + X,
        wspace=0.1,
    )

    mdls = len(th1_names)
    cs = ["tab:blue"] * len(kernel_names) + ["tab:green"] * len(kernel_names)
    lp_vals = np.log10(
        np.maximum(np.array([th1_npnr_dict[n]["KS_p_value"] for n in th1_names]), 1e-12)
    )

    test_lls = np.array(
        [
            [
                tll / ts * 1e3
                for tll, ts in zip(
                    th1_npnr_dict[n]["test_ells"], th1_npnr_dict[n]["test_datas_ts"]
                )
            ]
            for n in th1_names
        ]
    )
    train_lls = np.array(
        [
            th1_npnr_dict[n]["train_ell"] / th1_npnr_dict[n]["train_data_ts"] * 1e3
            for n in th1_names
        ]
    )

    ax = fig.add_subplot(spec[0, 0])
    for m in range(mdls):
        violin_parts = ax.violinplot(
            lp_vals[m],
            [m],
            points=20,
            widths=0.9,
            showmeans=True,
            showextrema=True,
            showmedians=True,
            bw_method="silverman",
            vert=False,
        )
        ax.scatter(
            lp_vals[m],
            0.1 * rng.normal(size=lp_vals[m].shape) + m,
            marker=".",
            s=10,
            c=cs[m],
            alpha=0.3,
        )

        # Make all the violin statistics marks red:
        for partname in ("cbars", "cmins", "cmaxes", "cmeans", "cmedians"):
            vp = violin_parts[partname]
            vp.set_edgecolor(cs[m])
            vp.set_linewidth(1)

        # Make the violin body blue with a red border:
        for vp in violin_parts["bodies"]:
            vp.set_facecolor(cs[m])
            vp.set_edgecolor("gray")
            vp.set_linewidth(1)
            vp.set_alpha(0.3)

    ax.set_xlabel("KS $p$-values", labelpad=1)
    ax.set_xlim([-8, 0])
    tick_range = np.arange(-8, 1)
    ax.set_xticks(tick_range)
    ax.set_xticklabels([r"$10^{-8}$", "", "", "", r"$10^{-4}$", "", "", "", r"$10^0$"])
    ax.set_ylim([-0.5, mdls - 0.5])
    ax.set_yticks(np.arange(mdls))
    ax.set_yticklabels(use_names)
    [t.set_color(cs[m]) for m, t in enumerate(ax.yaxis.get_ticklabels())]

    ax = fig.add_subplot(spec[0, 1])
    means = test_lls.mean(-1)
    sems = test_lls.std(-1) / np.sqrt(test_lls.shape[-1])
    for m in range(mdls):
        ax.errorbar(
            means[m],
            0.5 + m,
            xerr=sems[m],
            capsize=5,
            linestyle="",
            marker="o",
            markersize=4,
            color=cs[m],
        )
    ax.set_ylim([0, mdls])
    ax.set_yticks(0.5 + np.arange(mdls))
    ax.set_yticklabels([])
    ax.set_xlabel("test ELL (nats/s)", labelpad=2)
    ax.xaxis.grid(True)

    # hc3 plot
    widths = [0.9, 1]
    heights = [1]
    spec = fig.add_gridspec(
        ncols=len(widths),
        nrows=len(heights),
        width_ratios=widths,
        height_ratios=heights,
        top=1.0 + Y,
        bottom=0.0 + Y,
        left=0.53 + X,
        right=1.0 + X,
        wspace=0.1,
    )

    lp_vals = np.log10(
        np.maximum(np.array([hc3_npnr_dict[n]["KS_p_value"] for n in hc3_names]), 1e-12)
    )

    test_lls = np.array(
        [
            [
                tll / ts * 1e3
                for tll, ts in zip(
                    hc3_npnr_dict[n]["test_ells"], hc3_npnr_dict[n]["test_datas_ts"]
                )
            ]
            for n in hc3_names
        ]
    )
    train_lls = np.array(
        [
            hc3_npnr_dict[n]["train_ell"] / hc3_npnr_dict[n]["train_data_ts"] * 1e3
            for n in hc3_names
        ]
    )

    ax = fig.add_subplot(spec[0, 0])
    for m in range(mdls):
        violin_parts = ax.violinplot(
            lp_vals[m],
            [m],
            points=20,
            widths=0.9,
            showmeans=True,
            showextrema=True,
            showmedians=True,
            bw_method="silverman",
            vert=False,
        )
        ax.scatter(
            lp_vals[m],
            0.1 * rng.normal(size=lp_vals[m].shape) + m,
            marker=".",
            s=10,
            c=cs[m],
            alpha=0.3,
        )

        # Make all the violin statistics marks red:
        for partname in ("cbars", "cmins", "cmaxes", "cmeans", "cmedians"):
            vp = violin_parts[partname]
            vp.set_edgecolor(cs[m])
            vp.set_linewidth(1)

        # Make the violin body blue with a red border:
        for vp in violin_parts["bodies"]:
            vp.set_facecolor(cs[m])
            vp.set_edgecolor("gray")
            vp.set_linewidth(1)
            vp.set_alpha(0.3)

    ax.set_xlabel("KS $p$-values", labelpad=1)
    ax.set_xlim([-8, 0])
    tick_range = np.arange(-8, 1)
    ax.set_xticks(tick_range)
    ax.set_xticklabels([r"$10^{-8}$", "", "", "", r"$10^{-4}$", "", "", "", r"$10^0$"])
    ax.set_ylim([-0.5, mdls - 0.5])
    ax.set_yticks(np.arange(mdls))
    ax.set_yticklabels([])
    [t.set_color(cs[m]) for m, t in enumerate(ax.yaxis.get_ticklabels())]

    ax = fig.add_subplot(spec[0, 1])
    means = test_lls.mean(-1)
    sems = test_lls.std(-1) / np.sqrt(test_lls.shape[-1])
    for m in range(mdls):
        ax.errorbar(
            means[m],
            0.5 + m,
            xerr=sems[m],
            capsize=5,
            linestyle="",
            marker="o",
            markersize=4,
            color=cs[m],
        )
    ax.set_ylim([0, mdls])
    ax.set_yticks(0.5 + np.arange(mdls))
    ax.set_yticklabels([])
    ax.set_xlabel("test ELL (nats/s)", labelpad=2)
    ax.xaxis.grid(True)


def main():
    save_dir = "../saves/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.style.use(["paper.mplstyle"])

    # seed
    seed = 123
    rng = np.random.default_rng(seed)

    ### load ###
    th1_npnr_dict = pickle.load(open(save_dir + "th1_regression" + ".p", "rb"))

    hc3_npnr_dict = pickle.load(open(save_dir + "hc3_regression" + ".p", "rb"))

    th1_names = list(th1_npnr_dict.keys())
    hc3_names = list(hc3_npnr_dict.keys())

    kernel_names = [
        r"matern$\frac{1}{2}$-matern$\frac{1}{2}$",
        r"matern$\frac{1}{2}$-matern$\frac{3}{2}$",
        r"matern$\frac{3}{2}$-matern$\frac{5}{2}$",
        r"matern$\frac{3}{2}$-matern$\frac{3}{2}$",
    ]

    name = "synthetic_tuning"
    tuning_dict = pickle.load(open(save_dir + name + ".p", "rb"))

    name = "synthetic_regression"
    regression_dict = pickle.load(open(save_dir + name + ".p", "rb"))
    reg_config_names = list(regression_dict.keys())

    ### plot ###
    fig = plt.figure(figsize=(6, 3))
    fig.set_facecolor("white")

    plot_NPNRs(
        fig,
        0.25,
        0.0,
        rng,
        th1_npnr_dict,
        hc3_npnr_dict,
        th1_names,
        hc3_names,
        kernel_names,
    )

    ### export ###
    plt.savefig(save_dir + "NPNRs.pdf")


if __name__ == "__main__":
    main()
