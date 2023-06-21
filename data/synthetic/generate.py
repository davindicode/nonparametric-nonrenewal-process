import argparse
import os
import pickle

import sys

import jax
import jax.numpy as jnp
import jax.random as jr

import numpy as np
from jax import lax, vmap
from jax.scipy.special import gammainc, gammaln

sys.path.append("../../")
import animal
import lib


# tuning curves
def quadratic_2D_GLM(x, w, nonlin=jnp.exp):
    """
    Quadratic GLM for position

    :param np.array x: input series of shape (2,)
    :param np.array w: weights of shape (neurons, 6)
    :return:
        rate array of shape (trials, time, neurons)
    """
    x, y = x[0], x[1]
    g = jnp.stack((jnp.ones_like(x), x, y, x**2, y**2, x * y), axis=-1)
    return nonlin((g[None, :] * w).sum(-1))


def w_to_gaussian_2D(w):
    """
    Get Gaussian and orthogonal theta parameterization from the GLM parameters.

    :param np.array w: input GLM parameters of shape (neurons, dims), dims labelling (w_1, w_x,
                       w_y, w_xx, w_yy, w_xy)
    """
    neurons = mu.shape[0]
    prec = np.empty((neurons, 3))  # xx, yy and xy/yx
    mu = np.empty((neurons, 2))  # x and y
    prec[:, 0] = -2 * w_spat[:, 3]
    prec[:, 1] = -2 * w_spat[:, 4]
    prec[:, 2] = -w_spat[:, 5]
    prec_mat = []
    for n in range(neurons):
        prec_mat.append([[prec[n, 0], prec[n, 2]], [prec[n, 2], prec[n, 1]]])
    prec_mat = np.array(prec_mat)
    denom = prec[:, 0] * prec[:, 1] - prec[:, 2] ** 2
    mu[:, 0] = (w_spat[:, 1] * prec[:, 1] - w_spat[:, 2] * prec[:, 2]) / denom
    mu[:, 1] = (w_spat[:, 2] * prec[:, 0] - w_spat[:, 1] * prec[:, 2]) / denom
    rate_0 = np.exp(
        w_spat[:, 0] + 0.5 * (mu * np.einsum("nij,nj->ni", prec_mat, mu)).sum(1)
    )

    return mu, prec, rate_0


def gaussian_2D_to_w(mu, prec, rate_0):
    """
    Get GLM parameters from Gaussian and orthogonal theta parameterization

    :param np.array mu: mean of the Gaussian field of shape (neurons, 2)
    :param np.array prec: precision matrix elements xx, yy, and xy of shape (neurons, 3)
    :param np.array rate_0: rate amplitude of shape (neurons)

    """
    neurons = mu.shape[0]
    prec_mat = []
    for n in range(neurons):
        prec_mat.append([[prec[n, 0], prec[n, 2]], [prec[n, 2], prec[n, 1]]])
    prec_mat = np.array(prec_mat)
    w = np.empty((neurons, 6))
    w[:, 0] = np.log(rate_0) - 0.5 * (mu * np.einsum("nij,nj->ni", prec_mat, mu)).sum(1)
    w[:, 1] = mu[:, 0] * prec[:, 0] + mu[:, 1] * prec[:, 2]
    w[:, 2] = mu[:, 1] * prec[:, 1] + mu[:, 0] * prec[:, 2]
    w[:, 3] = -0.5 * prec[:, 0]
    w[:, 4] = -0.5 * prec[:, 1]
    w[:, 5] = -prec[:, 2]
    return w


# smooth signals
def rbf_kernel(x):
    return np.exp(-0.5 * (x**2))


def stationary_GP_trajectories(Tl, dt, trials, tau_list, eps, kernel_func, jitter=1e-9):
    """
    generate smooth GP input
    """
    tau_list_ = tau_list * trials
    out_dims = len(tau_list_)

    l = np.array(tau_list_)[:, None]
    v = np.ones(out_dims)

    T = np.arange(Tl)[None, :] * dt / l
    dT = T[:, None, :] - T[..., None]  # (tr, T, T)
    K = kernel_func(dT)
    K.reshape(out_dims, -1)[:, :: Tl + 1] += jitter

    L = np.linalg.cholesky(K)
    v = (L @ eps[..., None])[..., 0]
    a_t = v.reshape(trials, -1, Tl)
    return a_t  # trials, tau_arr, time


def generate_data(arena, mode, sample_bin, track_samples, seed):
    np.random.seed(seed)

    if mode == "straight":
        an = animal.animal_straight_lines(sample_bin, track_samples, arena)
        sim_samples, x_t, y_t, s_t, dir_t, hd_t, theta_t = an.sample(
            0.01, [50.0, 50.0], 100.0, 0.14, 0.0
        )

    elif mode == "Lever":
        an = animal.animal_Lever(sample_bin, track_samples, arena)
        sim_samples, x_t, y_t, s_t, dir_t, hd_t, theta_t = an.sample(
            np.array([50.0, 50.0]),
            10.0,
            0.0,
            6.1,
            0.14,
            0.0,
            10.0,
            0.5,
            wall_exit_angle=0,
            switch_avg=100000,
        )

    else:
        raise ValueError

    return track_samples, x_t, y_t, s_t, hd_t, theta_t, dir_t


def get_renewal(renewal_type, param, neurons, dt):
    if renewal_type == "gamma":
        alpha = param
        renewal = lib.likelihoods.GammaRenewal(
            neurons,
            dt,
            alpha,
        )

    elif renewal_type == "lognorm":
        sigma = param
        renewal = lib.likelihoods.LogNormalRenewal(
            neurons,
            dt,
            sigma,
        )

    elif renewal_type == "invgauss":
        mu = param
        renewal = lib.likelihoods.InverseGaussianRenewal(
            neurons,
            dt,
            mu,
        )

    elif renewal_type == "exponential":
        renewal = lib.likelihoods.ExponentialRenewal(
            neurons,
            dt,
        )

    return renewal


def get_model():
    dt = 0.001

    # rate
    mu = jnp.array(
        [
            [250.0, 250.0],
            [100.0, 100.0],
            [400.0, 300.0],
            [50.0, 350.0],
            [320.0, 90.0],
            [360.0, 270.0],
            [50.0, 200.0],
            [160.0, 170.0],
            [200.0, 230.0],
        ]
    )
    prec = jnp.array(
        [
            [0.0001, 0.0001, 0.00005],
            [0.0002, 0.0002, 0.0],
            [0.0001, 0.0002, 0.00005],
            [0.0001, 0.0002, 0.0],
            [0.0001, 0.0001, 0.0],
            [0.0001, 0.0001, -0.00005],
            [0.0001, 0.0001, 0.00001],
            [0.00015, 0.00025, 0.0],
            [0.0001, 0.0002, 0.00002],
        ]
    )
    rate_0 = jnp.array(
        [
            11.0,
            20.0,
            13.0,
            17.0,
            32.0,
            25.0,
            40.0,
            16.0,
            8.0,
        ]
    )  # Hz
    w = gaussian_2D_to_w(mu, prec, rate_0)

    ratefunc = lambda x: quadratic_2D_GLM(x, w, jnp.exp)
    N = rate_0.shape[0]

    S = 3

    renewals_sample = []
    renewals_ll = []
    rts = [
        ["gamma", 0.5],
        ["gamma", 1.0],
        ["gamma", 1.5],
        ["lognorm", 0.5],
        ["lognorm", 1.0],
        ["lognorm", 1.5],
        ["invgauss", 0.5],
        ["invgauss", 1.0],
        ["invgauss", 1.5],
    ]
    for rt in rts:
        rtype, p = rt
        rm = get_renewal(rtype, np.array([p]), 1, dt)

        renewals_sample.append(rm.sample_ISI)
        renewals_ll.append(rm.log_density)

    return ratefunc, renewals_sample, renewals_ll, N, dt


def main():
    ### parser ###
    parser = argparse.ArgumentParser(
        usage="%(prog)s [options]",
        description="Generating synthetic spike train data.",
    )

    parser.add_argument("--seed", default=123, type=int)
    parser.add_argument("--savedir", default="../saves/", type=str)

    args = parser.parse_args()

    ### setup ###
    jax.config.update("jax_platform_name", "cpu")

    save_dir = args.savedir

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    rng = np.random.default_rng(args.seed)
    prng_state = jr.PRNGKey(args.seed)

    ### generate ###
    track_samples = 1000000
    arena = animal.get_arena("box")

    sample_bin = 0.02  # s
    track_samples, x_t, y_t, s_t, hd_t, theta_t, dir_t = generate_data(
        arena, "Lever", sample_bin, track_samples, args.seed
    )
    pos_t = np.stack([x_t, y_t], axis=-1)

    ratefunc, renewals_sample, renewals_ll, obs_dims, dt = get_model()

    ts = pos_t.shape[0]
    ini_t_step = jnp.zeros(obs_dims)
    rates_t = jax.vmap(ratefunc)(pos_t)

    spikes = []
    for en, sample_ISI in enumerate(renewals_sample):
        prng_state, prng_key = jr.split(prng_state)
        spikes.append(
            lib.likelihoods.distributions.sample_rate_rescaled_renewal(
                prng_key,
                sample_ISI,
                ini_t_step[en : en + 1],
                rates_t[:, en : en + 1].T,
                dt,
            )
        )

    spikes = np.concatenate(spikes, axis=-1)  # (ts, N)

    # ISI
    ISI_orders = 5
    ISIs = lib.utils.spikes.get_lagged_ISIs(spikes, ISI_orders, dt)
    ISIs = np.array(ISIs)  # (ts, N, orders)

    ### export ###
    tr = 0
    Ns = [0, 1, 2, 3, 4]

    savefile = "syn_data_seed{}".format(args.seed)
    np.savez_compressed(
        save_dir + savefile,
        spktrain=spikes.T,
        ISIs=ISIs,
        x_t=x_t,
        y_t=y_t,
        tbin=dt,
    )  # auto conversion to numpy


if __name__ == "__main__":
    main()
