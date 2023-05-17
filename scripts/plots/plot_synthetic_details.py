import os
import pickle

import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append("../../../GaussNeuro")
import gaussneuro as lib
from gaussneuro import utils


def plot_ARDs(fig, X, Y, tuning_dict, ard_names, cs):
    """
    Plot kernel lengthscales
    """
    widths = [1]
    heights = [1] * len(ard_names)
    spec = fig.add_gridspec(
        ncols=len(widths),
        nrows=len(heights),
        width_ratios=widths,
        height_ratios=heights,
        top=0.97 + Y,
        bottom=0.15 + Y,
        left=-0.3 + X,
        right=-0.07 + X,
        hspace=0.4,
    )

    markers = ["o", "s", "x", "+"]

    for en, name in enumerate(ard_names):
        warp_tau = tuning_dict[name]["warp_tau"]
        len_tau = tuning_dict[name]["len_tau"]
        len_deltas = tuning_dict[name]["len_deltas"]

        N = len(len_deltas)

        ax = fig.add_subplot(spec[en, 0])
        if en == 0:
            ax.set_title(r"learned $\tau_w$", fontsize=12, fontweight="bold")
        else:
            ax.set_title(
                r"fixed $\tau_w = \langle ISI \rangle$", fontsize=12, fontweight="bold"
            )

        for n in range(N):
            lens = len_deltas[n]
            tlen = len_tau[n]
            ax.scatter(
                n,
                tlen,
                marker=markers[0],
                c=cs[n],
                s=20,
                label="lag 0" if n == 0 else None,
            )
            for k in range(1, 4):
                ax.scatter(
                    n,
                    lens[k - 1],
                    marker=markers[k],
                    c=cs[n],
                    label="lag {}".format(k) if n == 0 else None,
                )

            ax.set_yscale("log")
            ax.plot([-0.5, N + 0.5], 3.0 * np.ones(2), "--", color="gray")
            ax.set_xlim([-0.5, N + 0.5])
            ax.set_xticks(list(range(N)))
            ax.set_xticklabels([])

            if en == 1:
                ax.set_ylabel(
                    "                                kernel timescales", labelpad=1
                )
                leg = ax.legend(
                    loc="right", bbox_to_anchor=(0.9, -0.6), handletextpad=0.2, ncol=2
                )
                for k in range(4):
                    leg.legend_handles[k].set_color("k")

                ax.set_xlabel("neuron index", labelpad=1)


def plot_tunings(fig, tuning_dict, names, titles, cs):
    ### tuning curves ###
    GT_rates = tuning_dict["GT"]["pos_rates"]

    widths = [1] * 3
    heights = [1] * 3
    spec = fig.add_gridspec(
        ncols=len(widths),
        nrows=len(heights),
        width_ratios=widths,
        height_ratios=heights,
        top=0.98,
        bottom=-0.02,
        left=0.25,
        right=0.7,
        wspace=3.0,
        hspace=1.0,
    )

    a = 1.0
    for n in range(3):
        for m in range(3):
            k = 3 * n + m
            ax = fig.add_subplot(spec[m, n])
            ax.imshow(
                GT_rates[..., k], vmin=0.0, origin="lower", cmap="viridis", alpha=a
            )
            lib.utils.plots.decorate_ax(ax)
            for spine in ax.spines.values():
                spine.set_edgecolor(cs[k])
                spine.set_linewidth(2)
                spine.set_alpha(a)

            if m == 2 and n == 0:
                ax.text(
                    0.1,
                    0.1,
                    "GT",
                    fontsize=10,
                    ha="left",
                    c="white",
                )

    dx, dy = 0.2, 0.4

    for n in range(3):
        for m in range(3):
            k = 3 * n + m

            widths = [1] * 2
            heights = [1] * 2
            spec = fig.add_gridspec(
                ncols=len(widths),
                nrows=len(heights),
                width_ratios=widths,
                height_ratios=heights,
                top=0.22 + dy * (2 - m),
                bottom=-0.07 + dy * (2 - m),
                left=0.31 + dx * n,
                right=0.42 + dx * n,
            )

            for en in range(len(names)):
                enx, eny = en // 2, en % 2

                ax = fig.add_subplot(spec[eny, enx])
                if en == len(names) - 1:
                    rs = 1 / tuning_dict[names[en]]["pos_mean_ISI"].mean(0)[k]
                else:
                    rs = tuning_dict[names[en]]["pos_rates"][k]

                ax.imshow(
                    rs,
                    vmin=0.0,
                    origin="lower",
                    cmap="viridis",
                    vmax=GT_rates[..., k].max(),
                )
                lib.utils.plots.decorate_ax(ax)
                for spine in ax.spines.values():
                    spine.set_edgecolor(cs[k])
                    spine.set_linewidth(2)

                if m == 2 and n == 0:
                    ax.text(
                        0.1,
                        0.1,
                        titles[en],
                        fontsize=10,
                        ha="left",
                        c="white",
                    )


def main():
    save_dir = "../saves/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.style.use(["paper.mplstyle"])

    ### load ###
    datadir = "../saves/"

    name = "synthetic_regression"
    regression_dict = pickle.load(open(datadir + name + ".p", "rb"))
    reg_config_names = list(regression_dict.keys())

    name = "synthetic_tuning"
    tuning_dict = pickle.load(open(datadir + name + ".p", "rb"))

    ### plot ###
    cs = [
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red",
        "tab:purple",
        "tab:brown",
        "tab:pink",
        "tab:gray",
        "tab:cyan",
    ]  # cell ID colour

    titles = [
        "P",
        "G",
        "CP",
        "NPNR",
    ]
    tuning_names = reg_config_names[:3] + reg_config_names[-1:]
    ard_names = reg_config_names[-2:]

    # plot
    fig = plt.figure(figsize=(8, 3))

    fig.text(-0.11, 1.02, "A", fontsize=15, ha="center", fontweight="bold")
    plot_ARDs(fig, 0.25, 0.0, tuning_dict, ard_names, cs)

    fig.text(0.22, 1.02, "B", fontsize=15, ha="center", fontweight="bold")
    plot_tunings(fig, tuning_dict, tuning_names, titles, cs)

    fig.text(0.8, 1.02, "S", fontsize=15, alpha=0.0, ha="center")  # space

    ### export ###
    plt.savefig(save_dir + "synthetic_details.pdf")


if __name__ == "__main__":
    main()
