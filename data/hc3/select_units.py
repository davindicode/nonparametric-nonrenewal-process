import argparse
import os
import pickle

import jax

jax.config.update("jax_platform_name", "cpu")

import sys

import numpy as np
from scipy.signal import hilbert

sys.path.append("../../")
import tools
from lib import utils as utils


def load_into_arrays(data_dict, rotate_angle):

    tbin = data_dict["sample_bin"]
    timesamples = len(data_dict["ts"])

    # covariates
    _x_t, _y_t = (
        data_dict["covariates"]["x"],
        data_dict["covariates"]["y"],
    )

    x_t = _x_t * np.cos(rotate_angle) + _y_t * np.sin(rotate_angle)
    y_t = -_x_t * np.sin(rotate_angle) + _y_t * np.cos(rotate_angle)

    hd_t = data_dict["covariates"]["hd"] - rotate_angle
    eeg_t = data_dict["covariates"]["eeg"]

    _, _, spktrain, _ = utils.spikes.bin_data(
        1,
        tbin,
        data_dict["neural"]["spike_time_inds"],
        timesamples,
        None,
    )

    return tbin, timesamples, spktrain, x_t, y_t, hd_t, eeg_t


def histogram_analysis(
    tbin,
    spktrain,
    x_t,
    theta_t,
    min_time_spent,
    bins_x,
    bins_theta,
    filter_win_x,
    filter_win_theta,
    sigma_smooth_x,
    sigma_smooth_theta,
):
    # binning of covariates and basic properties
    left_x, right_x = x_t.min(), x_t.max()
    arena = np.array([left_x, right_x])
    bin_x = np.linspace(left_x, right_x + 1e-3, bins_x + 1)
    bin_theta = np.linspace(0, 2 * np.pi + 1e-3, bins_theta + 1)

    (
        sp_rate,
        sp_occup_time,
        sp_tot_spikes,
    ) = utils.stats.occupancy_normalized_histogram(
        tbin, min_time_spent, (x_t, theta_t), (bin_x, bin_theta), activities=spktrain
    )  # min_time_spent in s
    sp_prob = sp_occup_time / sp_occup_time.sum()
    sp_MI = utils.spikes.spike_var_MI(sp_rate, sp_prob)

    centre_win_x = filter_win_x // 2
    xfilter = np.exp(
        -0.5 * (np.arange(filter_win_x) - centre_win_x) ** 2 / sigma_smooth_x**2
    )
    centre_win_theta = filter_win_theta // 2
    tfilter = np.exp(
        -0.5
        * (np.arange(filter_win_theta) - centre_win_theta) ** 2
        / sigma_smooth_theta**2
    )
    sfilter = xfilter[:, None] * tfilter[None, :]  # outer product to 2D
    sfilter /= sfilter.sum()

    smth_rate = utils.stats.smooth_histogram(sp_rate, sfilter, ["repeat", "periodic"])
    coherence, sparsity = utils.spikes.geometric_tuning(sp_rate, smth_rate, sp_prob)

    return arena, smth_rate, sp_MI, coherence, sparsity


def main():
    ### parser ###
    parser = argparse.ArgumentParser(
        usage="%(prog)s [options]",
        description="Refine preprocessed hc-3 datasets.",
    )

    parser.add_argument("--session_id", type=str)
    parser.add_argument("--rec_id", type=str)
    parser.add_argument("--ISI_order", default=5, type=int)

    parser.add_argument("--datadir", type=str)
    parser.add_argument("--savedir", default="../saves/", type=str)

    args = parser.parse_args()

    savedir = args.savedir
    datadir = args.datadir

    if not os.path.exists(savedir):
        os.makedirs(savedir)

    # selected mouse and session properties
    rec_id = args.rec_id
    session_id = args.session_id
    ISI_order = args.ISI_order

    data_dict = pickle.load(open(datadir + "{}_{}.p".format(rec_id, session_id), "rb"))

    tbin, timesamples, spktrain, x_t, y_t, hd_t, eeg_t = load_into_arrays(
        data_dict, -0.014 * np.pi
    )  # preprocessed at 1 ms

    mean_eeg_t = eeg_t.mean(0)

    # hilbert signal of filtered EEG
    eeg_filt = tools.filter_signal(mean_eeg_t, 5.0, 11.0, tbin)
    analytic_signal = hilbert(eeg_filt)
    hilbert_theta = np.unwrap(np.angle(analytic_signal))
    theta_t = hilbert_theta % (2 * np.pi)

    # ISI
    ISIs = utils.spikes.get_lagged_ISIs(spktrain.T, ISI_order, tbin)
    ISIs = np.array(ISIs)

    # analysis
    arena, smth_rate, sp_MI, coherence, sparsity = histogram_analysis(
        tbin,
        spktrain,
        x_t,
        theta_t,
        min_time_spent=0.025,
        bins_x=40,
        bins_theta=30,
        filter_win_x=31,
        filter_win_theta=31,
        sigma_smooth_x=2,
        sigma_smooth_theta=2,
    )

    # select cells based on criterion
    units = spktrain.shape[0]
    refract_viol = data_dict["neural"]["refract_viol"]
    tot_spikes = spktrain.sum(-1)
    Ti = 100000
    early_spikes = spktrain[:, :Ti].sum(-1)
    print("time bin: ", tbin, " units: ", units)

    sel_unit = np.zeros(units).astype(bool)
    for u in range(units):
        if (
            coherence[u] > 0.4
            and sparsity[u] > 0.6
            and refract_viol[u] < 0.02
            and tot_spikes[u] > 9e2
            and early_spikes[u] > 5
        ):
            sel_unit[u] = True

    unit_used = np.where(sel_unit)[0]
    print("num of selected cells:", (sel_unit).sum())

    np.savez_compressed(
        savedir + "{}_{}_isi{}".format(rec_id, session_id, ISI_order),
        spktrain=spktrain[sel_unit],
        ISIs=ISIs[:, sel_unit],
        arena=arena,
        tbin=tbin,
        x_t=x_t,
        y_t=y_t,
        theta_t=theta_t,
        hd_t=hd_t,
    )


if __name__ == "__main__":
    main()
