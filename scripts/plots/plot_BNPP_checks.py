import os
import pickle

import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append("../../../GaussNeuro")
from gaussneuro import utils


def plot_BNPPs(
    fig, X, Y, th1_bnpp_dict, hc3_bnpp_dict, th1_names, hc3_names, kernel_names
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
    fig.text(0.235, 1.02, "mouse thalamus", fontsize=12, ha="center", fontweight="bold")
    fig.text(
        0.765, 1.02, "rat hippocampus", fontsize=12, ha="center", fontweight="bold"
    )

    fig.text(
        -0.27,
        0.25,
        r"learned $\tau_w$",
        fontsize=12,
        va="center",
        rotation=90,
        color="tab:blue",
    )
    fig.text(
        -0.27,
        0.75,
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
    p_vals = [th1_bnpp_dict[n]["KS_p_value"] for n in th1_names]

    test_lls = np.array(
        [
            [
                tll / ts * 1e3
                for tll, ts in zip(
                    th1_bnpp_dict[n]["test_ells"], th1_bnpp_dict[n]["test_datas_ts"]
                )
            ]
            for n in th1_names
        ]
    )
    train_lls = np.array(
        [
            th1_bnpp_dict[n]["train_ell"] / th1_bnpp_dict[n]["train_data_ts"] * 1e3
            for n in th1_names
        ]
    )

    ax = fig.add_subplot(spec[0, 0])
    for m in range(mdls):
        violin_parts = ax.violinplot(
            p_vals[m],
            [m],
            points=20,
            widths=0.9,
            showmeans=True,
            showextrema=True,
            showmedians=True,
            bw_method="silverman",
            vert=False,
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
    # ax.set_xlim([0, 1])
    # ax.set_xticks([0, 1])
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

    p_vals = [hc3_bnpp_dict[n]["KS_p_value"] for n in hc3_names]

    test_lls = np.array(
        [
            [
                tll / ts * 1e3
                for tll, ts in zip(
                    hc3_bnpp_dict[n]["test_ells"], hc3_bnpp_dict[n]["test_datas_ts"]
                )
            ]
            for n in hc3_names
        ]
    )
    train_lls = np.array(
        [
            hc3_bnpp_dict[n]["train_ell"] / hc3_bnpp_dict[n]["train_data_ts"] * 1e3
            for n in hc3_names
        ]
    )

    ax = fig.add_subplot(spec[0, 0])
    for m in range(mdls):
        violin_parts = ax.violinplot(
            p_vals[m],
            [m],
            points=20,
            widths=0.9,
            showmeans=True,
            showextrema=True,
            showmedians=True,
            bw_method="silverman",
            vert=False,
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
    # ax.set_xlim([0, 1])
    # ax.set_xticks([0, 1])
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

    ### load ###
    th1_bnpp_dict = pickle.load(open(save_dir + "th1_regression" + ".p", "rb"))

    hc3_bnpp_dict = pickle.load(open(save_dir + "hc3_regression" + ".p", "rb"))

    th1_names = list(th1_bnpp_dict.keys())
    hc3_names = list(hc3_bnpp_dict.keys())

    kernel_names = [
        "matern12-matern12",
        "matern12-matern32",
        "matern32-matern52",
        "matern32-matern32",
    ]

    ### plot ###
    fig = plt.figure(figsize=(6, 3))
    fig.set_facecolor("white")

    plot_BNPPs(
        fig, 0.0, 0.0, th1_bnpp_dict, hc3_bnpp_dict, th1_names, hc3_names, kernel_names
    )

    ### export ###
    plt.savefig(save_dir + "BNPPs.pdf")


if __name__ == "__main__":
    main()
