import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as scstats

sys.path.append("../../../GaussNeuro")
from gaussneuro import utils




# plot both mean and ISI tuning maps, and then show examples of individual conditional ISIs



def ind_to_pair(ind, N):
    a = ind
    k = 1
    while a >= 0:
        a -= N - k
        k += 1

    n = k - 1
    m = N - n + a
    return n - 1, m



def model_icons(fig):
    delX = 0.5
    Xoff = -0.1
    Yoff = 0.4
    for l in range(2):
        widths = [1]
        heights = [1]
        spec = fig.add_gridspec(
            ncols=len(widths),
            nrows=len(heights),
            width_ratios=widths,
            height_ratios=heights,
            left=0.0 + delX * l,
            right=0.3 + delX * l,
            bottom=0.1,
            top=1.1,
        )

        ax = fig.add_subplot(spec[0, 0])

        pgm = utils.plots.daft_init_figax(fig, ax, shape=(2, 2), node_unit=0.7)

        pgm.add_node("y", r"$y_n$", 0.7 + Xoff, 0.6 + Yoff, observed=True, fontsize=12)
        if l == 0:
            pgm.add_node(
                "x", r"$X$", 0.7 + Xoff, 1.2 + Yoff, observed=True, fontsize=12
            )
            pgm.add_edge("x", "y")
        elif l == 1:
            pgm.add_node(
                "z", r"$Z$", 0.7 + Xoff, 1.2 + Yoff, observed=False, fontsize=12
            )
            pgm.add_edge("z", "y")

        pgm.add_plate(
            [0.3 + Xoff, 0.2 + Yoff, 0.8, 0.75],
            label=r"$N$",
            position="bottom right",
            shift=0.1,
        )

        utils.plots.daft_render(pgm)


def regression_scores(fig, regression_dict, variability_dict):
    ### data ###
    T_KS = variability_dict["T_KS"]
    significance_KS = variability_dict["significance_KS"]
    RG_cv_ll = regression_dict["RG_cv_ll"]

    ### plot ###
    widths = [1]
    heights = np.ones(2)
    spec = fig.add_gridspec(
        ncols=len(widths),
        nrows=len(heights),
        width_ratios=widths,
        hspace=0.4,
        height_ratios=heights,
        top=0.2,
        bottom=-0.4,
        left=0.07,
        right=0.17,
    )

    eps = 0.4
    Ncases = T_KS.shape[0] - 1

    # RG
    ax = fig.add_subplot(spec[0, 0])
    fact = 10**2
    ax.set_xlim(-eps, Ncases + eps)
    scores = RG_cv_ll
    score_err = scores.std(-1) / np.sqrt(scores.shape[-1] - 1)
    rel_score = (scores - scores[0:1, :]) / fact
    
    ax.plot(
        np.arange(scores.shape[0])[:, None].repeat(scores.shape[1], axis=1),
        rel_score,
        color="gray",
        marker=".",
        markersize=4,
        alpha=0.5,
    )
    ax.errorbar(
        np.arange(scores.shape[0])[1:],
        rel_score.mean(-1)[1:],
        linestyle="",
        marker="+",
        markersize=4,
        capsize=3,
        yerr=rel_score.std(-1, ddof=1)[1:] / np.sqrt(rel_score.shape[-1]),
        c="k",
    )

    ax.set_xticks(np.arange(RG_cv_ll.shape[0]))
    ax.set_xticklabels([])
    # ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    ax.set_ylabel(r"$\Delta$cvLL ($10^2$)", fontsize=10)

    # KS
    ax = fig.add_subplot(spec[1, 0])
    ax.set_xlim(-eps, Ncases + eps)
    for en, r in enumerate(T_KS.mean(0)):
        ax.scatter(
            en * np.ones(len(r)) + np.random.rand(len(r)) * eps / 2 - eps / 4,
            r,
            color="gray",
            marker="+",
        )

    xl, xu = ax.get_xlim()
    ax.fill_between(
        np.linspace(xl, xu, 2), 0, np.ones(2) * significance_KS, color="k", alpha=0.2
    )
    # ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    ax.set_ylabel(r"$T_{KS}$", fontsize=10)
    ax.set_xticks(np.arange(RG_cv_ll.shape[0]))
    ax.set_xticklabels(["Poisson", "hNB", "U (GP)", "U (ANN)"], rotation=90)  # , 'GT'])


def count_tuning(fig, regression_dict):
    ### data ###
    eval_hd_inds = regression_dict["eval_hd_inds"]
    gt_P_count = regression_dict["gt_P_count"]
    UCM_P_count = regression_dict["UCM_P_count"]
    max_count = UCM_P_count.shape[-1] - 1
    neurons = gt_P_count.shape[1]

    covariates_hd = regression_dict["covariates_hd"]
    gt_mean = regression_dict["gt_mean"]
    gt_FF = regression_dict["gt_FF"]

    cntlower, cntmedian, cntupper = regression_dict["cnt_percentiles"]
    avglower, avgmedian, avgupper = regression_dict["avg_percentiles"]
    FFlower, FFmedian, FFupper = regression_dict["FF_percentiles"]

    ### plot ###
    pick_neurons = list(range(neurons))

    cx = np.arange(max_count + 1)
    plot_cnt = 11

    delx = 0.1
    fig.text(0.34, 1.05, "Universal (GP)", fontsize=12, ha="center")

    sel_neurons = [6, 16]
    for en, n in enumerate(sel_neurons):

        widths = [1]
        heights = [0.8, 1, 0.7, 1, 0.4, 1]
        spec = fig.add_gridspec(
            ncols=len(widths),
            nrows=len(heights),
            width_ratios=widths,
            hspace=0.0,
            height_ratios=heights,
            top=0.85,
            bottom=-0.45,
            left=0.25 + delx * en,
            right=0.33 + delx * en,
        )

        ax = fig.add_subplot(spec[0, 0])
        fig.text(
            0.29 + delx * en,
            0.91,
            "neuron {}".format(pick_neurons[n] + 1),
            fontsize=11,
            ha="center",
        )

        c = ["g", "b", "r"]
        for enn, hd_n in enumerate(eval_hd_inds):
            if enn == 1:
                continue

            l = "truth" if enn == 0 else None
            ax.plot(
                cx[:plot_cnt], gt_P_count[enn, n, :plot_cnt], "--", c=c[enn], label=l
            )

            for pp in range(plot_cnt):
                l = "fit" if (enn == 0 and pp == 0) else None
                XX = np.linspace(cx[pp] - 0.5, cx[pp] + 0.5, 2)
                YY = np.ones(2) * cntmedian[n, enn, pp]
                YY_l = np.ones(2) * cntlower[n, enn, pp]
                YY_u = np.ones(2) * cntupper[n, enn, pp]
                (line,) = ax.plot(XX, YY, c=c[enn], label=l, alpha=0.3)
                ax.fill_between(XX, YY_l, YY_u, color=line.get_color(), alpha=0.3)

        if en == 1:
            leg = ax.legend(
                handlelength=1.0, bbox_to_anchor=(1.5, 1.3), loc="upper right"
            )
            leg.legendHandles[0]._color = "k"
            leg.legendHandles[1]._color = "k"
            leg.get_lines()[1].set_linewidth(3)

        ax.set_xticklabels([])
        if en == 0:
            ax.set_ylabel("prob.", fontsize=10)
        ax.set_xlim([-0.5, plot_cnt - 1 + 0.5])
        ax.set_ylim(0)
        ax.set_yticks([])
        ax.set_xticks([])  # np.arange(plot_cnt))
        # ax.set_xlabel('count')

        ax = fig.add_subplot(spec[1, 0])
        im = ax.imshow(
            UCM_P_count[n, :, :plot_cnt],
            origin="lower",
            cmap="gray_r",
            vmin=0,
            vmax=UCM_P_count[n, :, :plot_cnt].max(),
            interpolation="nearest",
        )

        # arrows
        if en == 0:
            c = ["g", "b", "r"]
            for enn, hd_n in enumerate(eval_hd_inds):
                if enn == 1:
                    continue

                ax.annotate(
                    text="",
                    xy=(
                        0.0,
                        hd_n,
                    ),
                    zorder=1,
                    color="tab:blue",
                    va="center",
                    xytext=(-3.0, hd_n),
                    arrowprops=dict(arrowstyle="->", color=c[enn]),
                )

        utils.plots.decorate_ax(ax, spines=[False, False, False, False])
        # ax.set_xlim([0, plot_cnt-1])
        ax.set_xticks(np.arange(plot_cnt))

        ax.set_yticklabels([])
        if en == 0:
            ax.set_ylabel(r"$x$", fontsize=10, labelpad=5)
            ax.set_xticklabels([0, "", "", "", "", 5, "", "", "", "", ""])
        else:
            ax.set_xticklabels(["", "", "", "", "", 5, "", "", "", "", 10])

        ax = fig.add_subplot(spec[3, 0])
        (line,) = ax.plot(covariates_hd, avgmedian[n, :], color="tab:blue")
        ax.fill_between(
            covariates_hd,
            avglower[n, :],
            avgupper[n, :],
            color=line.get_color(),
            alpha=0.3,
        )
        ax.plot(covariates_hd, gt_mean[n, :], "k--")
        ax.set_ylim(0, 5.0)
        if en == 0:
            ax.set_ylabel("mean", fontsize=10)
        else:
            ax.set_yticklabels([])
        ax.set_xlim([0, 2 * np.pi])
        ax.set_xticklabels([])
        ax.set_xticks([0, 2 * np.pi])

        ax = fig.add_subplot(spec[5, 0])
        (line,) = ax.plot(covariates_hd, FFmedian[n, :], color="tab:blue")
        ax.fill_between(
            covariates_hd,
            FFlower[n, :],
            FFupper[n, :],
            color=line.get_color(),
            alpha=0.3,
        )
        ax.set_ylim(0.4, 1.4)
        ax.plot(covariates_hd, gt_FF[n, :], "k--")
        ax.set_xticks([0, 2 * np.pi])
        if en == 0:
            ax.set_ylabel("FF", fontsize=10, labelpad=0)
            ax.set_xticklabels([r"$0$", r"$2\pi$"])
        else:
            ax.set_yticklabels([])
            ax.set_xticklabels([])
        ax.set_xlim([0, 2 * np.pi])

    fig.text(0.34, 0.25, "count", ha="center", fontsize=10)
    fig.text(0.34, -0.675, r"head direction $x$", ha="center", fontsize=10)


def latent_variables(fig, regression_dict, latent_dict):
    ### data ###
    covariates_hd = regression_dict["covariates_hd"]
    gt_mean = regression_dict["gt_mean"]
    gt_FF = regression_dict["gt_FF"]

    gt_hd = latent_dict["gt_hd"]
    latent_mu = latent_dict["latent_mu"]
    latent_std = latent_dict["latent_std"]

    covariates_aligned = latent_dict["covariates_aligned"]
    avg_percentiles = latent_dict["avg_percentiles"]
    FF_percentiles = latent_dict["FF_percentiles"]

    ### plot ###
    widths = np.ones(1)
    heights = np.ones(1)
    spec = fig.add_gridspec(
        ncols=len(widths),
        nrows=len(heights),
        width_ratios=widths,
        height_ratios=heights,
        top=1.0,
        bottom=0.7,
        left=0.75,
        right=0.9,
    )

    ax = fig.add_subplot(spec[0, 0])

    tbin = 0.1
    T = 300
    T_start = 0

    ax.set_xlim([0, tbin * T])
    ax.set_xticks([])
    ax.set_xlabel("time", labelpad=5)

    ax.set_ylim([0, 2 * np.pi])
    ax.set_yticks([0, 2 * np.pi])
    ax.set_yticklabels([r"$0$", r"$2\pi$"])

    ax.set_title(r"posterior $q_{\varphi}(z)$", fontsize=12, pad=7)
    utils.plots.plot_circ_posterior(
        ax,
        tbin * np.arange(T),
        gt_hd[T_start : T_start + T] % (2 * np.pi),
        None,
        col="k",
        linewidth=1.0,
        step=3,
        l="truth",
    )

    utils.plots.plot_circ_posterior(
        ax,
        tbin * np.arange(T),
        latent_mu[0][T_start : T_start + T],
        latent_std[0][T_start : T_start + T],
        col="tab:blue",
        linewidth=0.7,
        step=1,
        alpha=0.5,
        line_alpha=0.5,
        l="GP",
    )  # , l_std='var. post.')

    utils.plots.plot_circ_posterior(
        ax,
        tbin * np.arange(T),
        latent_mu[1][T_start : T_start + T],
        latent_std[1][T_start : T_start + T],
        col="tab:green",
        linewidth=0.7,
        step=1,
        alpha=0.5,
        line_alpha=0.5,
        l="ANN",
    )  # , l_std='var. post.')

    leg = ax.legend(bbox_to_anchor=(1.05, 1.2), handlelength=0.8)
    for l in leg.get_lines()[1:]:
        l.set_linewidth(3)
    leg.get_lines()[0].set_linestyle("--")

    widths = np.ones(2)
    heights = [1, 1, 1]
    spec = fig.add_gridspec(
        ncols=len(widths),
        nrows=len(heights),
        width_ratios=widths,
        wspace=0.3,
        hspace=0.3,
        height_ratios=heights,
        top=0.4,
        bottom=-0.45,
        left=0.75,
        right=0.95,
    )

    delx = 0.4
    n = 6
    col_ = ["tab:blue", "tab:green"]
    for l in range(2):

        ax = fig.add_subplot(spec[0, l])
        if l == 0:
            ax.set_title("U (GP)", fontsize=11)
        else:
            ax.set_title("U (ANN)", fontsize=11)
            
        ax.set_aspect(1)
        ax.scatter(
            gt_hd[: latent_mu[l].shape[0]],
            latent_mu[l],
            marker=".",
            s=1,
            alpha=0.5,
            color=col_[l],
        )
        ax.set_xticks([0, 2 * np.pi])
        ax.set_xticklabels([])
        ax.set_yticks([0, 2 * np.pi])
        if l > 0:
            ax.set_yticklabels([])
        else:
            ax.set_yticklabels(["0", r"$2\pi$"])
        if l == 0:
            ax.set_ylabel(r"$z$", fontsize=10)

        ax = fig.add_subplot(spec[1, l])
        lower, mean, upper = avg_percentiles[l]
        (line,) = ax.plot(covariates_aligned, mean[n], color=col_[l])
        ax.fill_between(
            covariates_aligned, lower[n], upper[n], color=line.get_color(), alpha=0.3
        )
        ax.plot(covariates_hd, gt_mean[n, :], "k--")
        if l == 0:
            ax.set_ylabel("mean", fontsize=10)
        ax.set_xlim([0, 2 * np.pi])
        ax.set_ylim([0, 5.5])
        ax.set_xticklabels([])
        ax.set_xticks([0, 2 * np.pi])
        if l > 0:
            ax.set_yticklabels([])

        ax = fig.add_subplot(spec[2, l])
        lower, mean, upper = FF_percentiles[l]
        (line,) = ax.plot(covariates_aligned, mean[n, :], color=col_[l])
        ax.fill_between(
            covariates_aligned,
            lower[n, :],
            upper[n, :],
            color=line.get_color(),
            alpha=0.3,
        )
        ax.plot(covariates_hd, gt_FF[n, :], "k--")
        if l == 0:
            ax.set_ylabel("FF", fontsize=10)
        ax.set_xticks([0, 2 * np.pi])
        ax.set_xlim([0, 2 * np.pi])
        ax.set_ylim([0.4, 2.3])
        if l > 0:
            ax.set_yticklabels([])
            ax.set_xticklabels([])
        else:
            ax.set_xticklabels([r"$0$", r"$2\pi$"])

    fig.text(0.85, -0.675, "head direction (truth)", ha="center", fontsize=10)
    fig.text(
        0.97,
        -0.15,
        "neuron {}".format(n + 1),
        va="center",
        ha="center",
        fontsize=11,
        rotation=90,
    )


def LVM_scores(fig, latent_dict):
    ### data ###
    LVM_cv_ll = latent_dict["LVM_cv_ll"]
    RMS_cv = latent_dict["RMS_cv"]
    neurons = LVM_cv_ll.shape[-1]
    
    ### plot ###
    pick_neurons = list(range(neurons))
    
    eps = 0.4
    Ncases = LVM_cv_ll.shape[0] - 1
    
    widths = np.ones(1)
    heights = np.ones(2)
    spec = fig.add_gridspec(
        ncols=len(widths),
        nrows=len(heights),
        width_ratios=widths,
        wspace=0.5,
        hspace=0.4,
        height_ratios=heights,
        top=0.2,
        bottom=-0.4,
        left=0.565,
        right=0.665,
    )

    # LVM
    ax = fig.add_subplot(spec[0, 0])
    fact = 10**4
    ax.set_xlim(-eps, Ncases + eps)
    scores = np.transpose(LVM_cv_ll, (1, 0, 2)).mean(-1)
    scores_err = scores.std(-1) / np.sqrt(scores.shape[-1] - 1)
    rel_score = (scores - scores[0:1, :]) / fact * (len(pick_neurons) / 5)

    ax.plot(
        np.arange(scores.shape[0])[:, None].repeat(scores.shape[1], axis=1),
        rel_score,
        color="gray",
        marker=".",
        markersize=4,
        alpha=0.5,
    )
    ax.errorbar(
        np.arange(scores.shape[0])[1:],
        rel_score.mean(-1)[1:],
        linestyle="",
        marker="+",
        markersize=4,
        capsize=3,
        yerr=rel_score.std(-1, ddof=1)[1:] / np.sqrt(rel_score.shape[-1]),
        c="k",
    )

    ax.set_xticks(np.arange(scores.shape[0]))
    ax.set_xticklabels([])
    ax.set_ylabel(r"$\Delta$cvLL ($10^4$)", fontsize=10, labelpad=3)

    # RMS
    ax = fig.add_subplot(spec[1, 0])
    ax.set_xlim(-eps, Ncases + eps)
    cvs = RMS_cv.shape[0]
    cvtrials = RMS_cv.shape[1]
    yerr = RMS_cv.std(1, ddof=1) / np.sqrt(cvtrials)
    ax.bar(
        np.arange(cvs),
        RMS_cv.mean(1),
        yerr=yerr,
        capsize=3,
        color=[0.5, 0.5, 0.5],
        width=0.5,
    )

    ax.set_ylim(0)
    ax.set_xticks(np.arange(cvs))
    ax.set_xticklabels(["Poisson", "hNB", "U (GP)", "U (ANN)"], rotation=90)
    ax.set_ylabel("RMSE", fontsize=10)


def noise_correlations(fig, variability_dict):
    ### data ###
    R = variability_dict["R"]
    Fisher_Z = variability_dict["Fisher_Z"]
    Z_scores = variability_dict["Z_scores"]
    T_DS = variability_dict["T_DS"]
    neurons = T_DS.shape[-1]

    R_Poisson_X = variability_dict["R_Poisson_X"]
    R_Universal_X = variability_dict["R_Universal_X"]
    R_Universal_XZ = variability_dict["R_Universal_XZ"]
    datas = [R_Poisson_X, R_Universal_X, R_Universal_XZ]

    ### plot ###
    pick_neurons = list(range(neurons))
    
    names = ["Poisson", "Universal (X)", "Universal (X,Z)"]
    delX = 0.22
    Yoff = -0.25
    Xoff = -0.075
    for l in range(3):
        widths = [1]
        heights = [1]
        spec = fig.add_gridspec(
            ncols=len(widths),
            nrows=len(heights),
            width_ratios=widths,
            height_ratios=heights,
            left=0.0 + delX * l,
            right=0.3 + delX * l,
            bottom=-1.6,
            top=-0.6,
        )

        ax = fig.add_subplot(spec[0, 0])
        ax.text(
            1.3 + Xoff + (1 - l) * 0.05,
            2.5,
            names[l],
            fontsize=12,
            rotation=0,
            ha="center",
        )

        pgm = utils.plots.daft_init_figax(fig, ax, shape=(2, 2), node_unit=0.7)

        pgm.add_node("y", r"$y_n$", 0.7 + Xoff, 0.6 + Yoff, observed=True, fontsize=12)
        if l == 0 or l == 1:
            pgm.add_node(
                "x", r"$X$", 0.7 + Xoff, 1.2 + Yoff, observed=True, fontsize=12
            )
            pgm.add_edge("x", "y")
        elif l == 2:
            pgm.add_node(
                "x", r"$X$", 0.45 + Xoff, 1.2 + Yoff, observed=True, fontsize=12
            )
            pgm.add_edge("x", "y")
            pgm.add_node(
                "z", r"$Z$", 0.95 + Xoff, 1.2 + Yoff, observed=False, fontsize=12
            )
            pgm.add_edge("x", "y")
            pgm.add_edge("z", "y")

        pgm.add_plate(
            [0.3 + Xoff, 0.2 + Yoff, 0.8, 0.75],
            label=r"$N$",
            position="bottom right",
            shift=0.1,
        )

        utils.plots.daft_render(pgm)

    Xoff = 0.03
    aa = [np.argsort(R[0])[-2]]
    delY = 0.5
    delX = 0.22
    J = 10  # jump skip
    for en, a in enumerate(aa):
        for l in range(3):  # kcv=2

            widths = [4, 1]
            heights = [1, 4]
            spec = fig.add_gridspec(
                ncols=len(widths),
                nrows=len(heights),
                width_ratios=widths,
                height_ratios=heights,
                left=0.05 + delX * l + Xoff,
                right=0.16 + delX * l + Xoff,
                bottom=-2.15 - delY * en,
                top=-1.75 - delY * en,
            )

            n, m = ind_to_pair(a, len(pick_neurons))
            m_here = m + n + 1
            L = 4.0

            ax = fig.add_subplot(spec[1, 0])
            ax.scatter(
                Z_scores[l][n][::J],
                Z_scores[l][m_here][::J],
                marker=".",
                c="tab:blue",
                alpha=0.3,
            )
            ax.set_aspect(1)
            ax.set_xlim(-L, L)
            ax.set_ylim(-L, L)
            utils.plots.decorate_ax(ax, spines=[False, False, False, False])
            if l == 0:
                ax.set_xlabel(r"$\xi_1$", labelpad=-1, fontsize=10)
                ax.set_ylabel(r"$\xi_2$", labelpad=-1, fontsize=10)

            ax = fig.add_subplot(spec[0, 0])
            Z = Z_scores[l][m_here]
            ax.hist(Z, bins=np.linspace(-L, L, 20), density=True, color="tab:blue")
            xx = np.linspace(-L, L, 100)
            yy = scstats.norm.pdf(xx)
            ax.plot(xx, yy, "r")
            ax.set_xlim(-L, L)
            ax.set_xticks([])
            ax.set_yticks([])

            ax = fig.add_subplot(spec[1, 1])
            Z = Z_scores[l][n]
            ax.hist(
                Z,
                bins=np.linspace(-L, L, 20),
                density=True,
                orientation="horizontal",
                color="tab:blue",
            )
            xx = np.linspace(-L, L, 100)
            yy = scstats.norm.pdf(xx)
            ax.plot(yy, xx, "r")
            ax.set_ylim(-L, L)
            ax.set_xticks([])
            ax.set_yticks([])

            widths = [1]
            heights = [1]
            spec = fig.add_gridspec(
                ncols=len(widths),
                nrows=len(heights),
                width_ratios=widths,
                height_ratios=heights,
                left=0.0 + delX * l + Xoff,
                right=0.025 + delX * l + Xoff,
                bottom=-2.15 - delY * en,
                top=-1.75 - delY * en,
            )
            L = 0.1
            r = T_DS[0, l]
            ax = fig.add_subplot(spec[0, 0])
            ax.hist(
                r,
                density=True,
                bins=np.linspace(-L, L, 20),
                orientation="horizontal",
                color="gray",
            )
            ax.set_xticks([])
            if l == 0:
                ax.set_ylabel(r"$T_{DS}$", fontsize=10, labelpad=-4)
            else:
                ax.set_yticks([])

            samples = len(Z_scores[0][0])  # number of quantiles
            std = np.sqrt(2 / (samples - 1))
            xx = np.linspace(-L, L, 100)
            yy = scstats.norm.pdf(xx / std) / std
            ax.plot(yy, xx, "r")
            ax.set_ylim(-L, L)

    white = "#ffffff"
    lightgray = "#D3D3D3"
    black = "#000000"
    red = "#ff0000"
    blue = "#0000ff"
    weight_map = utils.plots.make_cmap([blue, white, red], "weight_map")

    Xoff = 0.03

    g = max(-np.stack(datas).min(), np.stack(datas).max()) * 1.0
    for en, r in enumerate(Fisher_Z[:3]):
        widths = [0.25, 1]
        heights = [1]
        spec = fig.add_gridspec(
            ncols=len(widths),
            nrows=len(heights),
            width_ratios=widths,
            height_ratios=heights,
            wspace=0.6,
            left=0.0 + delX * en + Xoff,
            right=0.14 + delX * en + Xoff,
            bottom=-2.7,
            top=-2.3,
        )

        ax = fig.add_subplot(spec[0, 0])
        ax.hist(
            r,
            bins=np.linspace(-5, 5, 20),
            density=True,
            orientation="horizontal",
            color="gray",
        )
        ax.set_xticks([])
        if en == 0:
            ax.set_ylabel(r"Fisher $Z$", fontsize=10, labelpad=4)
        else:
            ax.set_yticks([])

        xx = np.linspace(-5, 5, 100)
        yy = scstats.norm.pdf(xx)
        ax.plot(yy, xx, "r")
        ax.set_ylim(-5, 5)

        # show correlations
        ax = fig.add_subplot(spec[0, 1])

        data = datas[en]
        im = ax.imshow(
            data[:, ::-1],
            origin="lower",
            cmap=weight_map,
            vmin=-g,
            vmax=g,
            aspect="equal",
        )
        utils.plots.decorate_ax(ax, spines=[False, True, False, True])
        ax.yaxis.set_label_position("right")
        ax.set_xticks([])
        ax.set_yticks([])
        if en == 0:
            ax.set_xlabel(r"neuron $j$", fontsize=10, labelpad=3)
            ax.set_ylabel(r"neuron $i$", fontsize=10, labelpad=3)
            ax.text(-6.0, 12.0, r"$r_{ij}$ $(i<j)$", fontsize=12, rotation=45)

    widths = [1]
    heights = [1]
    spec = fig.add_gridspec(
        ncols=len(widths),
        nrows=len(heights),
        width_ratios=widths,
        height_ratios=heights,
        left=0.588 + Xoff,
        right=0.593 + Xoff,
        bottom=-2.65,
        top=-2.35,
    )
    ax = fig.add_subplot(spec[0, 0])
    ax.set_title(r"  $r_{ij}$", fontsize=12)
    utils.plots.add_colorbar(
        (fig, ax), im, ticktitle="", ticks=[-0.1, 0, 0.1], ticklabels=[-0.1, 0, 0.1]
    )

    # lines
    Yoff = 0.0

    widths = [1]
    heights = [1]
    spec = fig.add_gridspec(
        ncols=len(widths),
        nrows=len(heights),
        width_ratios=widths,
        height_ratios=heights,
        left=0.205,
        right=0.445,
        bottom=-2.7 + Yoff,
        top=-1.0 + Yoff,
    )
    ax = fig.add_subplot(spec[0, 0])
    yy = np.linspace(0, 1.0, 10)
    ax.plot(1.0 * np.ones_like(yy), yy, "k", linewidth=0.9)
    ax.plot(2.0 * np.ones_like(yy), yy, "k", linewidth=0.9)
    ax.axis("off")


def latent_observed_tuning(fig, latent_observed_dict):
    ### data
    cv_Ell = latent_observed_dict["cv_Ell"]
    neurons = cv_Ell.shape[-1]
    
    covariates_a = latent_observed_dict["covariates_a"]
    avglower, avgmedian, avgupper = latent_observed_dict["avg_percentiles"]
    FFlower, FFmedian, FFupper = latent_observed_dict["FF_percentiles"]
    X_c = latent_observed_dict["X_c"]
    X_s = latent_observed_dict["X_s"]

    gt_hd = latent_observed_dict["gt_hd"]
    gt_a = latent_observed_dict["gt_a"]
    
    gt_mean = latent_observed_dict["gt_mean"]
    gt_FF = latent_observed_dict["gt_FF"]

    ### XZ tuning ###
    tbin = 0.04  # 40 ms
    pick_neurons = list(range(neurons))
    
    widths = [1]
    heights = [1]
    spec = fig.add_gridspec(
        ncols=len(widths),
        nrows=len(heights),
        width_ratios=widths,
        height_ratios=heights,
        left=0.775,
        right=0.95,
        bottom=-1.3,
        top=-0.95,
    )

    ax = fig.add_subplot(spec[0, 0])
    fact = 10**3
    scores = cv_Ell.mean(-1)  # reshape(cv_pll.shape[0], -1)
    scores_err = scores.std(-1) / np.sqrt(scores.shape[-1] - 1)
    rel_score = (scores - scores[0:1, :]) / fact * (len(pick_neurons) / 5)
    
    ax.plot(
        np.arange(scores.shape[0])[:, None].repeat(scores.shape[1], axis=1),
        rel_score,
        color="gray",
        marker=".",
        markersize=4,
        alpha=0.5,
    )
    ax.errorbar(
        np.arange(scores.shape[0])[1:],
        rel_score.mean(-1)[1:],
        linestyle="",
        marker="+",
        markersize=4,
        capsize=3,
        yerr=rel_score.std(-1, ddof=1)[1:] / np.sqrt(rel_score.shape[-1]),
        c="k",
    )

    xlims = ax.get_xlim()
    ax.plot(np.linspace(xlims[0], xlims[1], 2), np.zeros(2), "gray")
    ax.set_xlim(xlims)

    ax.set_xticks(np.arange(scores.shape[0]))
    ax.set_xticklabels(["Poisson", "U (X)", "U (X,Z)"])
    ax.set_ylabel(r"$\Delta$cvLL ($10^3$)", fontsize=10)

    fig.text(0.86, -1.55, "Universal (X,Z)", fontsize=12, ha="center")
    widths = [1]
    heights = [1, 1]
    spec = fig.add_gridspec(
        ncols=len(widths),
        nrows=len(heights),
        width_ratios=widths,
        height_ratios=heights,
        left=0.74,
        right=0.85,
        hspace=0.3,
        bottom=-2.15,
        top=-1.6,
    )

    ax = fig.add_subplot(spec[0, 0])
    T = 501
    ts = np.arange(T) * tbin
    (line,) = ax.plot(ts, X_c[:T], alpha=0.5)
    ax.fill_between(
        ts,
        X_c[:T] - X_s[:T],
        X_c[:T] + X_s[:T],
        color=line.get_color(),
        alpha=0.3,
        label="inferred",
    )
    ax.plot(ts, gt_a[:T], "k--", label="truth")
    ax.set_ylabel(r"$z$", labelpad=5, fontsize=10)
    ax.set_xlim([0, ts[-1]])
    ax.set_xticks([0, ts[-1]])
    ax.set_xticklabels([])
    ax.set_yticks([-1, 0, 1])
    ax.set_yticklabels([-1, "", 1])
    leg = ax.legend(handlelength=1.0, bbox_to_anchor=(2.1, 1.2), loc="upper right")
    for l in leg.get_lines()[1:]:
        l.set_linewidth(3)

    T = 501
    ts = np.arange(T) * tbin
    ax = fig.add_subplot(spec[1, 0])
    utils.plots.plot_circ_posterior(
        ax,
        tbin * np.arange(T),
        gt_hd[:T] % (2 * np.pi),
        None,
        col="k",
        linewidth=0.7,
        step=1,
    )
    ax.set_ylim([0, 2 * np.pi])
    ax.set_yticks([0, 2 * np.pi])
    ax.set_yticklabels(["0", r"$2\pi$"])
    ax.set_ylabel(r"$x$", labelpad=0, fontsize=10)
    ax.set_xlim([0, ts[-1]])
    ax.set_xticks([0, ts[-1]])
    ax.set_xticklabels([])
    ax.set_xlabel("time ({} s)".format(int(ts[-1])), labelpad=0, color="k", fontsize=10)

    widths = [1]
    heights = [1]
    spec = fig.add_gridspec(
        ncols=len(widths),
        nrows=len(heights),
        width_ratios=widths,
        height_ratios=heights,
        left=0.91,
        right=0.98,
        bottom=-2.2,
        top=-1.7,
    )

    ax = fig.add_subplot(spec[0, 0])
    ax.scatter(gt_a, X_c, marker=".", alpha=0.5)
    Ra = np.linspace(gt_a.min() - 0.3, gt_a.max() + 0.3, 2)
    ax.set_xlim(Ra[0], Ra[-1])
    ax.plot(Ra, Ra, "k")
    ax.set_aspect("equal")
    ax.set_ylabel("inferred", fontsize=10, labelpad=3)
    ax.set_yticks([])
    ax.set_xlabel("truth", fontsize=10, labelpad=3)
    ax.set_xticks([])

    widths = [1, 1, 1]
    heights = [1]
    spec = fig.add_gridspec(
        ncols=len(widths),
        nrows=len(heights),
        width_ratios=widths,
        height_ratios=heights,
        left=0.74,
        right=0.97,
        bottom=-2.6,
        top=-2.3,
    )

    for en, n in enumerate([30, 6, 15]):
        ax = fig.add_subplot(spec[0, en])

        (line,) = ax.plot(covariates_a, avgmedian[n, :])
        ax.fill_between(
            covariates_a,
            avglower[n, :],
            avgupper[n, :],
            color=line.get_color(),
            alpha=0.3,
        )
        ax.plot(covariates_a, gt_mean[n, :], "k--")
        if en == 0:
            ax.set_ylabel("mean", fontsize=10)

        ax.set_xticks([])
        ax.set_yticks([])
        if en == 1:
            ax.set_xlabel(r"$z$", fontsize=10)


def main():
    save_dir = "../output/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.style.use(["paper.mplstyle"])

    # load
    datarun = pickle.load(open(save_dir + "hCMP_results.p", "rb"))

    regression_hCMP = datarun["regression"]
    variability_hCMP = datarun["variability"]
    latent_dict_hCMP = datarun["latent"]

    datarun = pickle.load(open(save_dir + "modIP_results.p", "rb"))

    latent_observed_modIP = datarun["latent_observed"]
    variability_modIP = datarun["variability"]

    # plot
    fig = plt.figure(figsize=(8, 2))
    fig.text(-0.04, 1.1, "A", fontsize=15, fontweight="bold")
    fig.text(-0.04, -0.95, "B", fontsize=15, fontweight="bold")

    ### regression ###
    model_icons(fig)
    regression_scores(fig, regression_hCMP, variability_hCMP)
    count_tuning(fig, regression_hCMP)

    # line
    Yoff = 0.0

    widths = [1]
    heights = [1]
    spec = fig.add_gridspec(
        ncols=len(widths),
        nrows=len(heights),
        width_ratios=widths,
        height_ratios=heights,
        left=0.45,
        right=0.5,
        bottom=-0.7 + Yoff,
        top=1.0 + Yoff,
    )
    ax = fig.add_subplot(spec[0, 0])
    yy = np.linspace(0, 1.0, 10)
    ax.plot(1.0 * np.ones_like(yy), yy, "k", linewidth=0.9)
    ax.axis("off")

    ### LVM ###
    latent_variables(fig, regression_hCMP, latent_dict_hCMP)
    LVM_scores(fig, latent_dict_hCMP)

    ### noise correlations ###
    noise_correlations(fig, variability_modIP)
    latent_observed_tuning(fig, latent_observed_modIP)

    plt.savefig(save_dir + "plot_synthetic.pdf")


if __name__ == "__main__":
    main()
