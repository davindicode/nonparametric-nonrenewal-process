import argparse
import os
import pickle

import jax

jax.config.update("jax_platform_name", "cpu")

import sys

import numpy as np

sys.path.append("../../")
from lib import utils as utils

brain_regions = {"ANT": 0, "PoS": 1, "CA1": 2, "mPFC": 3}


def load_into_arrays(data_dict, bin_size):
    """
    Load and resample the data at valid time indices
    """
    covariates = (
        data_dict["covariates"]["x"],
        data_dict["covariates"]["y"],
        np.unwrap(data_dict["covariates"]["hd"]),
    )

    tbin, resamples, spktrain, (rx_t, ry_t, rhd_t,) = utils.spikes.bin_data(
        bin_size,
        data_dict["sample_bin"],
        data_dict["neural"]["spike_time_inds"],
        data_dict["use_sample_num"],
        covariates,
        average_behav=True,
    )

    rhd_t = rhd_t % (2 * np.pi)

    # remove invalid data
    inval_behav = data_dict["covariates"]["invalid_behaviour"]
    print("invalid time intervals: ", inval_behav)
    if len(inval_behav["HD"]) > 0:  # assume XY invalids is subset
        if (
            inval_behav["HD"][-1]["index"] + inval_behav["HD"][-1]["length"]
            == resamples
        ):  # remove invalid region at start
            end_cut = int(np.ceil(inval_behav["HD"][-1]["length"] / bin_size))
            resamples -= end_cut
            spktrain = spktrain[:, :-end_cut]
            rx_t = rx_t[:-end_cut]
            ry_t = ry_t[:-end_cut]
            rhd_t = rhd_t[:-end_cut]

        if inval_behav["HD"][0]["index"] == 0:  # remove invalid region at start
            start_ind = int(np.ceil(inval_behav["HD"][0]["length"] / bin_size))
            resamples -= start_ind
            spktrain = spktrain[:, start_ind:]
            rx_t = rx_t[start_ind:]
            ry_t = ry_t[start_ind:]
            rhd_t = rhd_t[start_ind:]

    return tbin, resamples, spktrain, rx_t, ry_t, rhd_t


def histogram_analysis(tbin, spktrain, rhd_t, bins_hd, filter_win, sigma_smooth):
    # binning of covariates and analysis
    bin_hd = np.linspace(0, 2 * np.pi + 1e-3, bins_hd + 1)
    (
        hd_rate,
        hd_occup_time,
        hd_tot_spikes,
    ) = utils.stats.occupancy_normalized_histogram(
        tbin, 0.0, (rhd_t,), (bin_hd,), activities=spktrain
    )
    hd_prob = hd_occup_time / hd_occup_time.sum()
    hd_MI = utils.spikes.spike_var_MI(hd_rate, hd_prob)

    centre_win = filter_win // 2
    sfilter = np.exp(
        -0.5 * (np.arange(filter_win) - centre_win) ** 2 / sigma_smooth**2
    )
    sfilter = sfilter / sfilter.sum()

    sm_tun = utils.stats.smooth_histogram(hd_rate, sfilter, ["periodic"])
    coherence, sparsity = utils.spikes.geometric_tuning(hd_rate, sm_tun, hd_prob)

    return sm_tun, hd_MI, coherence, sparsity


def main():
    ### parser ###
    parser = argparse.ArgumentParser(
        usage="%(prog)s [options]",
        description="Refine preprocessed th-1 datasets.",
    )

    parser.add_argument("--mouse_id", type=str)
    parser.add_argument("--session_id", type=str)
    parser.add_argument("--phase", type=str)
    parser.add_argument("--ISI_order", default=5, type=int)

    parser.add_argument("--datadir", type=str)
    parser.add_argument("--savedir", default="../saves/", type=str)

    args = parser.parse_args()

    savedir = args.savedir
    datadir = args.datadir

    if not os.path.exists(savedir):
        os.makedirs(savedir)

    # selected mouse and session properties
    mouse_id = args.mouse_id
    session_id = args.session_id
    phase = args.phase
    ISI_order = args.ISI_order

    data_dict = pickle.load(
        open(datadir + "/th1_{}_{}_{}.p".format(mouse_id, session_id, phase), "rb")
    )

    tbin, resamples, spktrain, rx_t, ry_t, rhd_t = load_into_arrays(
        data_dict, bin_size=20
    )  # original data at 20k Hz
    sm_tun, hd_MI, coherence, sparsity = histogram_analysis(
        tbin, spktrain, rhd_t, bins_hd=60, filter_win=41, sigma_smooth=3
    )

    # ISI statistics
    ISI, LV = utils.spikes.compute_ISI_LV(tbin, data_dict["neural"]["spike_time_inds"])

    # select cells based on criterion
    units = data_dict["neural"]["units"]
    refract_viol = data_dict["neural"]["refract_viol"]  # violates 2 ms ISI
    print("time bin: ", tbin, " units: ", units)

    hdc_unit = np.zeros(units).astype(bool)
    for u in range(units):
        if hd_MI[u] > 0.5 and sparsity[u] > 0.2 and refract_viol[u] < 0.02:
            hdc_unit[u] = True

    print("num of HD cells:", (hdc_unit).sum())

    # put region ID for each unit
    neuron_groups = data_dict["neural"]["neuron_groups"]
    neuron_regions = np.empty(units)
    for key in neuron_groups.keys():
        neuron_regions[neuron_groups[key]] = brain_regions[key]

    left_x = rx_t.min()
    right_x = rx_t.max()
    bottom_y = ry_t.min()
    top_y = ry_t.max()

    arena = np.array([left_x, right_x, bottom_y, top_y])

    # ISI
    ISIs = utils.spikes.get_lagged_ISIs(spktrain.T, ISI_order, tbin)
    ISIs = np.array(ISIs)

    np.savez_compressed(
        savedir + "{}_{}_{}_isi{}".format(mouse_id, session_id, phase, ISI_order),
        spktrain=spktrain,
        ISIs=ISIs,
        hdc_unit=hdc_unit,
        neuron_regions=neuron_regions,
        arena=arena,
        tbin=tbin,
        x_t=rx_t,
        y_t=ry_t,
        hd_t=rhd_t,
    )


if __name__ == "__main__":
    main()
