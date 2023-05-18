import os

import sys

import jax
import jax.numpy as jnp
import jax.random as jr

import matplotlib.pyplot as plt
import numpy as np
from jax import vmap

sys.path.append("../../")
import lib
from lib import utils


# computation functions
def log_time_transform(t, inverse, warp_tau):
    """
    Inverse transform is from tau [0, 1] to t in R

    :param jnp.ndarray t: time of shape (obs_dims,)
    """
    if inverse:
        t_ = -np.log(1 - t) * warp_tau
    else:
        s = np.exp(-t / warp_tau)
        t_ = 1 - s

    return t_


def generate_behaviour(prng_state, evalsteps, len_x, L):
    # animal position
    x_dims = 2
    num_samps = 1
    jitter = 1e-6

    var_x = np.ones((x_dims))  # GP variance
    len_x = len_x * np.ones((x_dims, 1))  # GP lengthscale

    kernx = lib.GP.kernels.Matern52(x_dims, variance=var_x, lengthscale=len_x)

    # site_init
    Tsteps = 0
    site_locs = np.empty(Tsteps)  # s
    site_obs = np.zeros([Tsteps, x_dims, 1])
    site_Lcov = np.eye(x_dims)[None, ...].repeat(Tsteps, axis=0)

    state_space = lib.GP.markovian.GaussianLTI(
        kernx, site_locs, site_obs, site_Lcov, diagonal_site=True, fixed_grid_locs=False
    )

    t_eval = jnp.linspace(0.0, L, evalsteps)
    pos_prior_samples = state_space.sample_prior(prng_state, num_samps, t_eval, jitter)

    return pos_prior_samples


def model_inputs(prng_state, rng, evalsteps, L):
    evals = np.arange(evalsteps)
    dt = 0.001

    t_eval = np.zeros(evalsteps)
    y = rng.normal(size=(evalsteps,)) > 2.2
    spiketimes = np.where(y > 0)[0]

    pos_sample = generate_behaviour(prng_state, evalsteps, 1.0, L)

    ISIs = lib.utils.spikes.get_lagged_ISIs(y[:, None], 4, dt)
    uisi = np.unique(ISIs[..., 1])
    uisi = uisi[~np.isnan(uisi)]

    warp_tau = np.array([uisi.mean()])
    tISI = log_time_transform(ISIs[..., 0], False, warp_tau)
    tuISIs = log_time_transform(ISIs[..., 1], False, warp_tau)

    tau = np.linspace(0.0, 0.4, 100)[:, None]
    tau_tilde = log_time_transform(tau, False, warp_tau)

    return evals, spiketimes, pos_sample, ISIs, tISI, tuISIs, tau, tau_tilde


def sample_conditional_ISI(
    bnpp,
    prng_state,
    num_samps,
    tau_eval,
    isi_cond,
    x_cond,
    sel_outdims,
    int_eval_pts=1000,
    num_quad_pts=100,
    prior=True,
    jitter=1e-6,
):
    """
    Compute the instantaneous renewal density with rho(ISI;X) from model
    Uses linear interpolation with Gauss-Legendre quadrature for integrals

    :param jnp.ndarray t_eval: evaluation time points (eval_locs,)
    :param jnp.ndarray isi_cond: past ISI values to condition on (obs_dims, order)
    :param jnp.ndarray x_cond: covariate values to condition on (x_dims,)
    """
    if sel_outdims is None:
        obs_dims = bnpp.gp.kernel.out_dims
        sel_outdims = jnp.arange(obs_dims)

    sigma_pts, weights = utils.linalg.gauss_legendre(1, num_quad_pts)
    sigma_pts, weights = bnpp._to_jax(sigma_pts), bnpp._to_jax(weights)

    # evaluation locs
    tau_tilde, log_dtilde_dt = vmap(bnpp._log_time_transform_jac, (1, None), 1)(
        tau_eval[None, :], False
    )  # (obs_dims, locs)

    log_rho_tilde, int_rho_tau_tilde, log_normalizer = bnpp._sample_log_ISI_tilde(
        prng_state,
        num_samps,
        tau_tilde,
        isi_cond,
        x_cond,
        sigma_pts,
        weights,
        sel_outdims,
        int_eval_pts,
        prior,
        jitter,
    )

    log_ISI_tilde = log_rho_tilde - int_rho_tau_tilde - log_normalizer
    ISI_density = jnp.exp(log_ISI_tilde + log_dtilde_dt)
    return ISI_density, log_rho_tilde, tau_tilde


def BNPP_samples(prng_state, rng, num_samps, evalsteps, L):
    dt = 1e-3  # s
    obs_dims = 1
    num_induc = 8
    RFF_num_feats = 3000
    cisi_t_eval = jnp.linspace(0.0, L, evalsteps)
    sel_outdims = jnp.arange(obs_dims)

    x_cond, isi_cond = None, None

    ISI_densities, log_rho_tildes = [], []
    for a_m, b_m in zip([0.0, -6.0], [-1.0, 0.0]):
        # SVGP
        len_t = 1.0 * np.ones((obs_dims, 1))  # GP lengthscale
        var_t = 0.5 * np.ones(obs_dims)  # GP variance
        tau_kernel = lib.GP.kernels.Matern52(
            obs_dims, variance=var_t, lengthscale=len_t
        )

        induc_locs = np.linspace(0.0, 1.0, num_induc)[None, :, None].repeat(
            obs_dims, axis=0
        )
        u_mu = 1.0 * rng.normal(size=(obs_dims, num_induc, 1))
        u_Lcov = 0.1 * np.eye(num_induc)[None, ...].repeat(obs_dims, axis=0)

        svgp = lib.GP.sparse.qSVGP(
            tau_kernel,
            induc_locs,
            u_mu,
            u_Lcov,
            RFF_num_feats=RFF_num_feats,
            whitened=True,
        )

        # parameters
        wrap_tau = 1.0 * np.ones((obs_dims,))
        mean_tau = 3e-2 * np.ones((obs_dims,))
        mean_amp = a_m * np.ones((obs_dims,))
        mean_bias = b_m * np.ones((obs_dims,))

        bnpp = lib.observations.bnpp.NonparametricPointProcess(
            svgp, wrap_tau, mean_tau, mean_amp, mean_bias, dt
        )

        ISI_density, log_rho_tilde, tau_tilde = sample_conditional_ISI(
            bnpp,
            prng_state,
            num_samps,
            cisi_t_eval,
            isi_cond,
            x_cond,
            sel_outdims,
            int_eval_pts=1000,
            num_quad_pts=300,
            prior=True,
            jitter=1e-6,
        )

        ISI_densities.append(np.array(ISI_density))
        log_rho_tildes.append(np.array(log_rho_tilde))

    return np.array(cisi_t_eval), ISI_densities, np.array(tau_tilde[0]), log_rho_tildes


# plotting functions
def plot_inputs(fig, Tst, T, Te, evals, spiketimes, pos_sample, ISIs):
    """
    spike train and input time series
    """
    widths = [1]
    heights = [0.3, 1, 1, 1]
    spec = fig.add_gridspec(
        ncols=len(widths),
        nrows=len(heights),
        width_ratios=widths,
        height_ratios=heights,
        top=1.0,
        bottom=0.5,
        left=0.0,
        right=0.2,
        wspace=0.1,
    )

    ax = fig.add_subplot(spec[0, 0])
    spkts = spiketimes[(spiketimes > Tst) & (spiketimes < T)]
    for st in spkts:
        ax.plot(evals[st] * np.ones(2), np.linspace(0, 1, 2), c="k")
        spkts = spiketimes[(T <= spiketimes) & (spiketimes < Te)]
    for st in spkts:
        ax.plot(evals[st] * np.ones(2), np.linspace(0, 1, 2), c="k", alpha=0.3)
    ax.set_xlim([evals[Tst], evals[Te]])
    ax.set_ylim([0, 1])
    ax.axis("off")

    ax = fig.add_subplot(spec[1, 0])

    ax.plot(evals[Tst:T], ISIs[Tst:T, 0, 0], c="gray")
    ax.plot(evals[T:Te], ISIs[T:Te, 0, 0], c="gray", alpha=0.3)
    ax.set_xlim([evals[Tst], evals[Te]])
    ax.set_ylabel(r"$\tau$")
    ax.set_yticks([])
    ax.set_xticks([])

    for k in range(2):
        ax = fig.add_subplot(spec[k + 2, 0])
        ax.plot(evals[Tst:T], ISIs[Tst:T, 0, k + 1], c="gray")
        ax.plot(evals[T:Te], ISIs[T:Te, 0, k + 1], c="gray", alpha=0.3)
        ax.set_xlim([evals[Tst], evals[Te]])
        ax.set_xticks([])
        ax.set_ylim(0)
        ax.set_yticks([])
        ax.set_ylabel("$\Delta_{}$".format(k + 1))

    widths = [1]
    heights = [1]
    spec = fig.add_gridspec(
        ncols=len(widths),
        nrows=len(heights),
        width_ratios=widths,
        height_ratios=heights,
        top=0.475,
        bottom=0.375,
        left=0.0,
        right=0.2,
        wspace=0.0,
    )
    ax = fig.add_subplot(spec[0, 0])
    ax.scatter(np.zeros(3), np.arange(3), c="k", marker=".")
    ax.set_ylim([-0.5, 3.5])
    ax.axis("off")

    widths = [1]
    heights = [1, 1]
    spec = fig.add_gridspec(
        ncols=len(widths),
        nrows=len(heights),
        width_ratios=widths,
        height_ratios=heights,
        top=0.35,
        bottom=0.1,
        left=0.0,
        right=0.2,
        wspace=0.0,
    )

    for d in range(2):
        ax = fig.add_subplot(spec[d, 0])
        ax.plot(evals[Tst:T], pos_sample[0, Tst:T, d, 0], c="gray")
        ax.plot(evals[T:Te], pos_sample[0, T:Te, d, 0], alpha=0.3, c="gray")
        ax.set_xlim([evals[Tst], evals[Te]])
        ax.set_ylabel("$x_{}$".format(d + 1))
        ax.set_xticks([])
        ax.set_yticks([])

    ax.set_xlabel("time", labelpad=20)

    widths = [1]
    heights = [1]
    spec = fig.add_gridspec(
        ncols=len(widths),
        nrows=len(heights),
        width_ratios=widths,
        height_ratios=heights,
        top=0.1,
        bottom=0.0,
        left=0.0,
        right=0.2,
        wspace=0.0,
    )
    ax = fig.add_subplot(spec[0, 0])
    ax.scatter(np.zeros(3), np.arange(3), c="k", marker=".")
    ax.set_ylim([-0.5, 3.5])
    ax.axis("off")


def plot_time_warping(fig, Tst, Te, evals, ISIs, tISI, tuISIs, tau, tau_tilde):
    """
    time warping schematic
    """
    widths = [1, 0.4]
    heights = [1, 1]
    spec = fig.add_gridspec(
        ncols=len(widths),
        nrows=len(heights),
        width_ratios=widths,
        height_ratios=heights,
        top=1.0,
        bottom=0.0,
        left=0.29,
        right=0.49,
        hspace=2.1,
    )

    ax = fig.add_subplot(spec[0, 0])
    ax.plot(evals[Tst:Te], 1e-3 * ISIs[Tst:Te, 0, 0] * 1e3, c="gray")
    ax.set_xticks([])
    ax.set_xlim([evals[Tst], evals[Te]])
    ax.set_xlabel("time", labelpad=4)
    ax.set_ylabel(r"$\tau$ (s)", labelpad=-2)
    ax.set_ylim([0, 0.3])
    ax.set_yticks([0, 0.3])

    ax = fig.add_subplot(spec[0, 1])
    ax.hist(ISIs[:, 0, 0], orientation="horizontal", color="lightgray", density=True)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylim(0)
    ax.set_xlabel("density", labelpad=3)

    ax = fig.add_subplot(spec[1, 0])
    ax.plot(evals[Tst:Te], tISI[Tst:Te, 0], c="gray")
    ax.set_xticks([])
    ax.set_xlim([evals[Tst], evals[Te]])
    ax.set_xlabel("time", labelpad=4)
    ax.set_ylabel(r"$\tilde{\tau}$ (a.u.)", labelpad=5)
    ax.set_ylim([0, 1])
    ax.set_yticks([0, 1])

    ax = fig.add_subplot(spec[1, 1])
    ax.hist(tISI[:, 0], orientation="horizontal", color="lightgray", density=True)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylim([0, 1])
    ax.set_xlim([0, 3])
    ax.set_xlabel("density", labelpad=3)

    widths = [1]
    heights = [1]
    spec = fig.add_gridspec(
        ncols=len(widths),
        nrows=len(heights),
        width_ratios=widths,
        height_ratios=heights,
        top=0.6,
        bottom=0.35,
        left=0.31,
        right=0.39,
    )

    ax = fig.add_subplot(spec[0, 0])
    ax.plot(tau[:, 0], tau_tilde[:, 0], c="k", alpha=0.3)
    ax.set_xticks([])
    ax.set_yticks([0, 1])
    ax.set_xlim([tau[0, 0], tau[-1, 0]])
    ax.set_ylim([0, 1])
    ax.set_xlabel(r"$\tau$", labelpad=4, alpha=0.3)
    ax.set_ylabel(r"$\tilde{\tau}$", labelpad=-5, alpha=0.3)

    ax.tick_params(color="gray", labelcolor="gray")
    for spine in ax.spines.values():
        spine.set_edgecolor("gray")

    ax.text(
        0.57,
        0.5,
        r"$\tau_w = \langle$" + "ISI" + r"$\rangle$",
        fontsize=12,
        va="center",
        alpha=0.4,
    )
    ax.annotate(
        "",
        xy=(1.33, -0.2),
        xytext=(1.33, 1.2),
        rotation=np.pi / 2.0,
        xycoords="axes fraction",
        arrowprops=dict(
            arrowstyle="simple, head_width=1.6, head_length=1.0, tail_width=0.5",
            alpha=0.2,
            color="gray",
        ),
        annotation_clip=False,
    )


def plot_intensities(fig, cisi_t_eval, ISI_densities, cisi_tau_tilde, log_rho_tildes):
    """
    log intensity and ISI
    """
    n = 0
    f_dim = 0

    widths = [1, 1]
    heights = [1, 1]
    spec = fig.add_gridspec(
        ncols=len(widths),
        nrows=len(heights),
        width_ratios=widths,
        height_ratios=heights,
        top=0.93,
        bottom=0.0,
        left=0.6,
        right=1.0,
        hspace=0.6,
        wspace=0.3,
    )

    ax = fig.add_subplot(spec[0, 0])
    ax.set_title(
        r"non-refractory prior" + "\n" + r"( $a_m = 0$ )",
        fontsize=12,
        fontweight="bold",
    )
    ax.plot(cisi_tau_tilde, log_rho_tildes[0][:, f_dim, :].T, c="k", alpha=0.3)
    ax.set_ylim([-3.5, 0.5])
    ax.set_yticks([-3, 0])
    ax.set_ylabel("log CIF", labelpad=1)
    ax.set_xlabel(r"$\tilde{\tau}$ (a.u.)", labelpad=-11)
    ax.set_xlim([0, 1])
    ax.set_xticks([0, 1])

    ax = fig.add_subplot(spec[0, 1])
    ax.set_title(
        r"refractory prior" + "\n" + r"( $a_m = -6$ )", fontsize=12, fontweight="bold"
    )
    ax.plot(cisi_tau_tilde, log_rho_tildes[1][:, f_dim, :].T, c="k", alpha=0.3)
    ax.set_xticks([0, 1])
    ax.set_xlim([0, 1])

    ax = fig.add_subplot(spec[1, 0])
    ax.plot(cisi_t_eval, ISI_densities[0][:, f_dim, :].T, c="k", alpha=0.3)
    ax.set_xticks([0, 3])
    ax.set_yticks([])
    ax.set_ylabel("ISI distribution", labelpad=10)
    ax.set_xlabel(r"$\tau$ (s)", labelpad=-11)
    ax.set_xlim([cisi_t_eval[0], cisi_t_eval[-1]])
    ax.set_ylim(0)

    ax.annotate(
        "",
        xy=(0.5, 0.8),
        xytext=(0.5, 1.2),
        rotation=np.pi / 2.0,
        xycoords="axes fraction",
        arrowprops=dict(
            arrowstyle="simple, head_width=1.0, head_length=0.7, tail_width=0.3",
            color="gray",
        ),
        annotation_clip=False,
    )

    ax = fig.add_subplot(spec[1, 1])
    ax.plot(cisi_t_eval, ISI_densities[1][:, f_dim, :].T, c="k", alpha=0.3)
    ax.set_xticks([0, 3])
    ax.set_yticks([])
    ax.set_xlim([cisi_t_eval[0], cisi_t_eval[-1]])
    ax.set_ylim(0)

    ax.annotate(
        "",
        xy=(0.5, 0.8),
        xytext=(0.5, 1.2),
        rotation=np.pi / 2.0,
        xycoords="axes fraction",
        arrowprops=dict(
            arrowstyle="simple, head_width=1.0, head_length=0.7, tail_width=0.3",
            color="gray",
        ),
        annotation_clip=False,
    )


def main():
    save_dir = "../saves/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.style.use(["paper.mplstyle"])

    ### data ###

    # seed everything
    seed = 123
    prng_state = jr.PRNGKey(seed)
    rng = np.random.default_rng(seed)

    evalsteps = 10000
    L = 100.0

    evals, spiketimes, pos_sample, ISIs, tISI, tuISIs, tau, tau_tilde = model_inputs(
        prng_state, rng, evalsteps, L
    )

    num_samps = 10
    evalsteps = 500
    L = 3.0

    cisi_t_eval, ISI_densities, cisi_tau_tilde, log_rho_tildes = BNPP_samples(
        prng_state, rng, num_samps, evalsteps, L
    )

    Tst = 500
    T = 1200
    Te = 1500

    ### plot ###
    fig = plt.figure(figsize=(10, 2))
    fig.set_facecolor("white")

    fig.text(-0.03, 1.05, "A", fontsize=15, ha="center", fontweight="bold")
    plot_inputs(fig, Tst, T, Te, evals, spiketimes, pos_sample, ISIs)

    fig.text(0.245, 1.05, "B", fontsize=15, ha="center", fontweight="bold")
    plot_time_warping(fig, Tst, Te, evals, ISIs, tISI, tuISIs, tau, tau_tilde)

    fig.text(0.555, 1.05, "C", fontsize=15, ha="center", fontweight="bold")
    plot_intensities(fig, cisi_t_eval, ISI_densities, cisi_tau_tilde, log_rho_tildes)

    fig.text(1.03, 1.05, "S", fontsize=15, alpha=0.0, ha="center")  # space

    ### export ###
    plt.savefig(save_dir + "schematic.pdf")


if __name__ == "__main__":
    main()
