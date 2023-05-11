import os
import pickle

import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as scstats

sys.path.append("../../../GaussNeuro")
from gaussneuro import utils


# computation functions
def get_tuning_curves(samples, percentiles, sm_filter, padding_modes):
    percentiles = utils.stats.percentiles_from_samples(samples, percentiles=percentiles)

    return [
        utils.stats.smooth_histogram(p, sm_filter, padding_modes) for p in percentiles
    ]


# plot functions
def plot_fit_stats(fig, X, Y, regression_dict, use_reg_config_names, use_names, cs):
    """
    KS and likelihood statistics
    """
    widths = [0.9, 1]
    heights = [1]
    spec = fig.add_gridspec(
        ncols=len(widths),
        nrows=len(heights),
        width_ratios=widths,
        height_ratios=heights,
        top=1.0 + Y,
        bottom=0.5 + Y,
        left=0.03 + X,
        right=0.28 + X,
        wspace=0.1,
    )

    mdls = len(use_reg_config_names)
    p_vals = [regression_dict[n]["KS_p_value"] for n in use_reg_config_names]

    test_lls = np.array(
        [
            [
                tll / ts * 1e3
                for tll, ts in zip(
                    regression_dict[n]["test_ells"], regression_dict[n]["test_datas_ts"]
                )
            ]
            for n in use_reg_config_names
        ]
    )
    train_lls = np.array(
        [
            regression_dict[n]["train_ell"] / regression_dict[n]["train_data_ts"] * 1e3
            for n in use_reg_config_names
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


def plot_QQ(fig, X, Y, regression_dict, use_reg_config_names, cs):
    ### QQ plots ###
    widths = [1] * len(use_reg_config_names)
    heights = [1]

    spec = fig.add_gridspec(
        ncols=len(widths),
        nrows=len(heights),
        width_ratios=widths,
        height_ratios=heights,
        top=1.0 + Y,
        bottom=0.81 + Y,
        left=0.35 + X,
        right=1.0 + X,
        wspace=0.1,
    )

    for en, name in enumerate(use_reg_config_names):
        quantiles = regression_dict[name]["KS_quantiles"]

        ax = fig.add_subplot(spec[0, en])
        for n in range(len(quantiles)):
            if quantiles[n] is not None:
                ax.plot(
                    quantiles[n],
                    np.linspace(0.0, 1.0, len(quantiles[n])),
                    c=cs[en],
                    alpha=0.7,
                )
        ax.plot(np.linspace(0.0, 1.0, 100), np.linspace(0.0, 1.0, 100), c="k")
        ax.set_aspect(1.0)

        if en == 0:
            ax.set_xlabel("quantile", fontsize=11, labelpad=-10)
            ax.set_ylabel("EDF", fontsize=11, labelpad=-8)
        else:
            ax.set_xticklabels([])
            ax.set_yticklabels([])


def plot_posteriors(fig, X, Y, regression_dict, visualize_names, visualize_inds, cs):
    test_fold = 0
    ne = 7
    Ts, Te = 5000, 8000

    ### posterior visualizations ###
    widths = [1] * (len(visualize_names))
    heights = [0.1, 1, 1]

    spec = fig.add_gridspec(
        ncols=len(widths),
        nrows=len(heights),
        width_ratios=widths,
        height_ratios=heights,
        top=0.725 + Y,
        bottom=0.44 + Y,
        left=0.35 + X,
        right=1.0 + X,
        hspace=0.02,
        wspace=0.3,
    )

    for en, n in enumerate(visualize_names):
        pred_ts = regression_dict[n]["pred_ts"]

        sts = regression_dict[n]["pred_spiketimes"][test_fold][ne]
        ax = fig.add_subplot(spec[0, en])
        for st in sts:
            ax.plot(st * np.ones(2), np.linspace(0, 1, 2), c="k")
        ax.set_xlim([pred_ts[Ts], pred_ts[Te]])
        ax.axis("off")

        pred_lint = regression_dict[n]["pred_log_intensities"]
        ax = fig.add_subplot(spec[1, en])
        ax.plot(pred_ts, pred_lint[test_fold][0, ne, :], c=cs[visualize_inds[en]])
        ax.set_xlim([pred_ts[Ts], pred_ts[Te]])
        ax.set_ylim([-3.0, 4.0])
        ax.set_yticks([])
        if en == 0:
            ax.set_ylabel("log CIF", labelpad=2)
        ax.set_xticks([])

        stss = regression_dict[n]["sample_spiketimes"][test_fold][ne]
        ax = fig.add_subplot(spec[2, en])
        for num, st_samples in enumerate(stss):
            for st in st_samples:
                ax.plot(
                    st * np.ones(2),
                    num + np.linspace(0, 1, 2),
                    c=cs[visualize_inds[en]],
                )
        ax.set_xlim([pred_ts[Ts], pred_ts[Te]])
        ax.set_yticks([])
        ax.spines[["left", "bottom"]].set_visible(False)
        ax.set_xticks([])
        if en == 0:
            ax.set_xlabel("time", labelpad=2)
            ax.set_ylabel("trials", labelpad=2)

    widths = [1]
    heights = [1]

    spec = fig.add_gridspec(
        ncols=len(widths),
        nrows=len(heights),
        width_ratios=widths,
        height_ratios=heights,
        top=0.42 + Y,
        bottom=0.41 + Y,
        left=0.485 + X,
        right=0.52 + X,
    )

    ax = fig.add_subplot(spec[0, 0])
    ax.text(1.2, 0.0, "1 s", va="center", fontsize=11)
    ax.plot([0.0, 1.0], [0.0, 0.0], lw=2, c="k")
    ax.axis("off")


def plot_kernel_lens(fig, X, Y, tuning_dict):
    """
    kernel lengthscales
    """
    warp_tau = tuning_dict["warp_tau"]
    len_tau = tuning_dict["len_tau"]
    len_deltas = tuning_dict["len_deltas"]

    ARD_order = []
    for n in range(len_deltas.shape[0]):
        ARD_order.append(np.sum(len_deltas[n] < 3.0) + 1)

    # plot
    widths = [1, 0.3, 0.1, 0.3]
    heights = [1, 0.02]
    spec = fig.add_gridspec(
        ncols=len(widths),
        nrows=len(heights),
        width_ratios=widths,
        height_ratios=heights,
        top=0.3 + Y,
        bottom=-0.04 + Y,
        left=0.0 + X,
        right=0.27 + X,
        wspace=0.1,
    )

    cs = ["r", "g", "b", "orange"]
    markers = ["o", "s", "x", "+"]
    N = len(len_deltas)

    ax = fig.add_subplot(spec[0, -1])
    ax.hist(ARD_order, bins=np.linspace(0, 4, 5) + 0.5, color="lightgray")
    ax.set_yticks([])
    ax.set_xticks([1, 2, 3, 4])
    ax.set_ylabel("frequency", labelpad=1)
    ax.set_xlabel("order", labelpad=1)

    ax = fig.add_subplot(spec[:, 0])
    for n in range(N):
        lens = len_deltas[n]
        tlen = len_tau[n]
        ax.scatter(
            n, tlen, marker=markers[0], c=cs[0], s=20, label="lag 0" if n == 0 else None
        )
        for k in range(1, 4):
            ax.scatter(
                n,
                lens[k - 1],
                marker=markers[k],
                c=cs[k],
                label="lag {}".format(k) if n == 0 else None,
            )

        ax.set_yscale("log")
        ax.set_ylabel("kernel timescales", labelpad=1)
        ax.plot([-0.5, N + 0.5], 3.0 * np.ones(2), "--", color="gray")
        ax.set_xlim([-0.5, N + 0.5])
        ax.set_xlabel("neuron index", labelpad=1)
        ax.set_xticks(list(range(N)))
        ax.set_xticklabels([])
        leg = ax.legend(loc="right", bbox_to_anchor=(2.08, 0.7), handletextpad=0.2)
        for k in range(4):
            leg.legend_handles[k].set_color(cs[k])

    yl = ax.get_ylim()

    ax = fig.add_subplot(spec[:, 1])
    for k in range(4):
        if k > 0:
            lens = len_deltas[:, k - 1]
        else:
            lens = len_tau

        ax.hist(
            np.log(lens),
            color=cs[k],
            orientation="horizontal",
            bins=np.linspace(np.log(yl[0]), np.log(yl[1]), 20),
            alpha=0.3,
            density=True,
        )
        xx = np.linspace(np.log(yl[0]), np.log(yl[1]), 100)
        kde = scstats.gaussian_kde(np.log(lens))
        ax.plot(kde(xx), xx, c=cs[k])
        ax.set_ylim(np.log(yl[0]), np.log(yl[1]))

        xl = ax.get_xlim()
        ax.plot(xl, np.log(3.0) * np.ones(2), color="gray", linestyle="--")
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_xlabel("density", labelpad=3)


def plot_instantaneous(fig, X, Y, rng, variability_dict, plot_units):
    """
    moment-by-moment spike train statistics
    """
    imISI = 1 / variability_dict["mean_ISI"].mean(0)
    # imISI = variability_dict["mean_invISI"].mean(0)
    cvISI = variability_dict["CV_ISI"].mean(0)

    # plot
    widths = [1, 0.2]
    heights = [0.2, 1]

    spec = fig.add_gridspec(
        ncols=len(widths),
        nrows=len(heights),
        width_ratios=widths,
        height_ratios=heights,
        top=0.32 + Y,
        bottom=0.0 + Y,
        left=0.37 + X,
        right=0.48 + X,
        wspace=0.02,
        hspace=0.02,
    )

    ax = fig.add_subplot(spec[1, 0])
    ax.scatter(
        (1 / variability_dict["mean_ISI"].mean(0)).mean(1),
        variability_dict["CV_ISI"].mean(0).mean(1),
        marker=".",
        s=8,
        c="gray",
    )
    ax.set_xlabel("average rate (Hz)", labelpad=2)
    ax.set_ylabel("average CV", labelpad=2)
    ax.set_yticks([1, 2])

    ax = fig.add_subplot(spec[0, 0])
    dd = variability_dict["mean_ISI"].mean(0).mean(1)
    ax.hist(dd, color="lightgray", density=True)
    xl = ax.get_xlim()
    xx = np.linspace(xl[0], xl[1], 100)
    kde = scstats.gaussian_kde(dd)
    ax.plot(xx, kde(xx), c="gray")
    ax.set_xlim(xl)
    ax.set_xticks([])
    ax.set_yticks([])

    ax = fig.add_subplot(spec[1, 1])
    dd = variability_dict["mean_ISI"].mean(0).mean(1)
    ax.hist(dd, orientation="horizontal", color="lightgray", density=True)
    xl = ax.get_ylim()
    xx = np.linspace(xl[0], xl[1], 100)
    kde = scstats.gaussian_kde(dd)
    ax.plot(kde(xx), xx, c="gray")
    ax.set_ylim(xl)
    ax.set_xticks([])
    ax.set_yticks([])

    widths = [1]
    heights = [1]
    spec = fig.add_gridspec(
        ncols=len(widths),
        nrows=len(heights),
        width_ratios=widths,
        height_ratios=heights,
        top=0.32 + Y,
        bottom=0.0 + Y,
        left=0.52 + X,
        right=0.58 + X,
    )

    ax = fig.add_subplot(spec[0, 0])
    R2 = np.array([variability_dict["linear_R2"], variability_dict["GP_R2"]])
    xx = 0.05 * rng.normal(size=R2.shape)

    for x, vals in zip(xx.T, R2.T):
        ax.plot(np.arange(2) + x, vals, c="lightgray")

    for en, r2 in enumerate(R2):
        ax.scatter(en + xx[en], r2, c="gray")

    ax.set_xlim([-0.3, 1.3])
    ax.set_xticks([0, 1])
    ax.set_ylim([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["linear", "GP"])
    ax.set_ylabel(r"CV-rate $R^2$", labelpad=-6)


def plot_th1_tuning(fig, X, Y, tuning_dict, plot_units):
    """
    tuning of th1 head direction cells
    """
    percentiles = [0.025, 0.5, 0.975]
    sm_filter = np.ones(5) / 5
    padding_modes = ["periodic"]

    eval_locs = tuning_dict["hd_x_locs"]

    rsamples = 1 / tuning_dict["hd_mean_ISI"]
    # rsamples = tuning_dict["hd_mean_invISI"]
    cvsamples = tuning_dict["hd_CV_ISI"]

    rlower, rmedian, rupper = get_tuning_curves(
        rsamples, percentiles, sm_filter, padding_modes
    )
    cvlower, cvmedian, cvupper = get_tuning_curves(
        cvsamples, percentiles, sm_filter, padding_modes
    )

    # plot
    pts_inds = [2, 3]
    cs = ["r", "b"]

    widths = [1] * len(plot_units)
    heights = [1, 1]
    spec = fig.add_gridspec(
        ncols=len(widths),
        nrows=len(heights),
        width_ratios=widths,
        height_ratios=heights,
        top=0.3 + Y,
        bottom=0.0 + Y,
        left=0.66 + X,
        right=0.87 + X,
        wspace=0.4,
    )

    for en, n in enumerate(plot_units):
        ax = fig.add_subplot(spec[0, en])
        ax.set_title("neuron {}".format(n + 1), fontsize=11)
        (line,) = ax.plot(
            eval_locs[:, 0], rmedian[n, :], "gray", label="posterior mean"
        )
        ax.fill_between(
            eval_locs[:, 0],
            rlower[n, :],
            rupper[n, :],
            color=line.get_color(),
            alpha=0.2,
            label="95% confidence",
        )
        ax.plot(eval_locs[:, 0], rsamples[:, n, :].T, "gray", alpha=0.2)
        ax.set_xlim([0, 2 * np.pi])
        ax.set_xticks([0, 2 * np.pi])
        ax.set_xticklabels([])

        ylims = ax.get_ylim()
        ax.plot(
            np.ones(2) * tuning_dict["ISI_xs_conds"][pts_inds[en], 0],
            ylims,
            c=cs[en],
            linestyle="--",
        )

        if en == 0:
            ax.set_ylabel("rate (Hz)", labelpad=1)

        ax = fig.add_subplot(spec[1, en])
        (line,) = ax.plot(
            eval_locs[:, 0], cvmedian[n, :], "gray", label="posterior mean"
        )
        ax.fill_between(
            eval_locs[:, 0],
            cvlower[n, :],
            cvupper[n, :],
            color=line.get_color(),
            alpha=0.2,
            label="95% confidence",
        )
        ax.plot(eval_locs[:, 0], cvsamples[:, n, :].T, "gray", alpha=0.2)
        ax.set_xlim([0, 2 * np.pi])
        ax.set_xticks([0, 2 * np.pi])
        ax.set_xticklabels([])

        ylims = ax.get_ylim()
        ax.plot(
            np.ones(2) * tuning_dict["ISI_xs_conds"][pts_inds[en], 0],
            ylims,
            c=cs[en],
            linestyle="--",
        )

        if en == 0:
            ax.set_xlabel("head direction", labelpad=1)
            ax.set_xticklabels([r"$0$", r"$2\pi$"])
            ax.set_ylabel("CV", labelpad=1)

    # ISI distributions
    widths = [1]
    heights = [1, 1]
    spec = fig.add_gridspec(
        ncols=len(widths),
        nrows=len(heights),
        width_ratios=widths,
        height_ratios=heights,
        top=0.3 + Y,
        bottom=0.0 + Y,
        left=0.9 + X,
        right=1.0 + X,
    )

    for en in range(2):
        ax = fig.add_subplot(spec[en, 0])
        ax.plot(
            tuning_dict["ISI_t_eval"],
            tuning_dict["ISI_densities"][pts_inds[en], :, plot_units[en], :].T,
            alpha=0.02,
            color=cs[en],
        )
        ax.set_yticks([])
        ax.set_xticks([tuning_dict["ISI_t_eval"][0], tuning_dict["ISI_t_eval"][-1]])

        if en == 1:
            ax.set_ylabel("             probability", labelpad=2)
            ax.set_xlabel("ISI (s)", labelpad=1)
        else:
            ax.set_xticklabels([])


def plot_hc3_tuning(fig, X, Y, tuning_dict, direction, plot_units):
    """
    tuning of hc3 linear track cells
    """
    # data
    eval_locs = tuning_dict["plot_xt_x_locs"]

    xt_rate = 1 / tuning_dict["xt{}_mean_ISI".format(direction)].mean(0)
    xt_CV = tuning_dict["xt{}_CV_ISI".format(direction)].mean(0)

    # plot
    pts_inds = [2, 3]
    cs = ["r", "b"]

    white = "#ffffff"
    black = "#000000"
    red = "#ff0000"
    blue = "#0000ff"
    weight_map = utils.plots.make_cmap([blue, white, red], "weight_map")

    widths = [1] * len(plot_units)
    heights = [1, 1]
    spec = fig.add_gridspec(
        ncols=len(widths),
        nrows=len(heights),
        width_ratios=widths,
        height_ratios=heights,
        top=0.27 + Y,
        bottom=-0.05 + Y,
        left=0.65 + X,
        right=0.83 + X,
        hspace=0.4,
    )

    for en, ne in enumerate(plot_units):
        g = xt_rate[ne].max()
        ax = fig.add_subplot(spec[0, en])
        ax.text(
            0.5,
            1.4,
            "neuron {}".format(ne + 1),
            ha="center",
            fontsize=11,
            transform=ax.transAxes,
        )
        ax.set_title("{:.1f} Hz".format(g), fontsize=10, pad=3)
        im = ax.imshow(
            xt_rate[ne], origin="lower", cmap="viridis", vmin=0.0, vmax=g, aspect="auto"
        )
        ax.set_xticks([])
        if en == 0:
            ylims = ax.get_ylim()
            ax.set_yticks(ylims)
            ax.set_yticklabels(["", r"$2\pi$"])
        else:
            ax.set_yticks([])
        ax.spines.right.set_visible(True)
        ax.spines.top.set_visible(True)

        # ax.plot(np.ones(2) * tuning_dict["ISI_xs_conds"][pts_inds[en], 0], ylims, c=cs[en], linestyle='--')

    cspec = fig.add_gridspec(
        ncols=1,
        nrows=1,
        width_ratios=[1],
        height_ratios=[1],
        left=0.84,
        right=0.845,
        bottom=0.15,
        top=0.25,
    )
    ax = fig.add_subplot(cspec[0, 0])
    ax.set_title("     rate", fontsize=11, pad=7)
    cb = utils.plots.add_colorbar(
        (fig, ax),
        im,
        ticktitle=r"",
        ticks=[0, g],
        ticklabels=["0", "max"],
        cbar_format=None,
        cbar_ori="vertical",
    )
    cb.ax.yaxis.set_tick_params(color="gray")
    plt.setp(plt.getp(cb.ax.axes, "yticklabels"), color="gray")

    for en, ne in enumerate(plot_units):
        field = np.log(xt_CV[ne])
        g = max(-field.min(), field.max())

        ax = fig.add_subplot(spec[1, en])
        ax.text(
            0.3,
            1.06,
            "{:.1f}".format(np.exp(g)),
            ha="center",
            fontsize=10,
            color="red",
            transform=ax.transAxes,
        )
        ax.text(
            0.7,
            1.06,
            "{:.1f}".format(np.exp(-g)),
            ha="center",
            fontsize=10,
            color="blue",
            transform=ax.transAxes,
        )
        im = ax.imshow(
            field, origin="lower", cmap=weight_map, vmin=-g, vmax=g, aspect="auto"
        )
        ax.set_xticks([])
        if en == 0:
            ylims = ax.get_ylim()
            ax.set_yticks(ylims)
            ax.set_yticklabels([r"$0$", ""])
        else:
            ax.set_yticks([])
        ax.spines.right.set_visible(True)
        ax.spines.top.set_visible(True)

    cspec = fig.add_gridspec(
        ncols=1,
        nrows=1,
        width_ratios=[1],
        height_ratios=[1],
        left=0.84,
        right=0.845,
        bottom=-0.05,
        top=0.05,
    )
    ax = fig.add_subplot(cspec[0, 0])
    ax.set_title("     CV", fontsize=11, pad=7)
    cb = utils.plots.add_colorbar(
        (fig, ax),
        im,
        ticktitle=r"",
        ticks=[-g, 0, g],
        ticklabels=["min", "1", "max"],
        cbar_format=None,
        cbar_ori="vertical",
    )
    cb.ax.yaxis.set_tick_params(color="gray")
    plt.setp(plt.getp(cb.ax.axes, "yticklabels"), color="gray")

    spec = fig.add_gridspec(
        ncols=1,
        nrows=1,
        width_ratios=[1],
        height_ratios=[1],
        left=0.78,
        right=0.8,
        bottom=-0.08,
        top=-0.07,
    )
    ax = fig.add_subplot(spec[0, 0])
    ax.plot([0, 1], np.zeros(2), c="k", lw=3)
    ax.text(1.25, 0.0, "50 cm", va="center")
    ax.axis("off")

    fig.text(0.74, -0.1, "$x$ position", fontsize=11, ha="center")
    fig.text(0.62, 0.11, r"LFP $\theta$-phase", fontsize=11, va="center", rotation=90)

    # ISI distributions
    widths = [1]
    heights = [1, 1]
    spec = fig.add_gridspec(
        ncols=len(widths),
        nrows=len(heights),
        width_ratios=widths,
        height_ratios=heights,
        top=0.3 + Y,
        bottom=0.0 + Y,
        left=0.9 + X,
        right=1.0 + X,
    )

    for en in range(2):
        ax = fig.add_subplot(spec[en, 0])
        ax.plot(
            tuning_dict["ISI_t_eval"],
            tuning_dict["ISI_densities"][pts_inds[en], :, plot_units[en], :].T,
            alpha=0.02,
            color=cs[en],
        )
        ax.set_yticks([])
        ax.set_xticks([tuning_dict["ISI_t_eval"][0], tuning_dict["ISI_t_eval"][-1]])

        if en == 1:
            ax.set_ylabel("             probability", labelpad=2)
            ax.set_xlabel("ISI (s)", labelpad=1)
        else:
            ax.set_xticklabels([])


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
