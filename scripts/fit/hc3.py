import argparse

import template

import numpy as np



def counts_dataset(session_name, bin_size, path, select_fracs=None):
    filename = session_name + ".npz"
    data = np.load(path + filename)

    # spike counts
    spktrain = data["spktrain"]
    sample_bin = data["tbin"]  # s
    # sample_bin = 0.001
    track_samples = spktrain.shape[1]

    # cut out section
    if select_fracs is None:
        select_fracs = [0.0, 1.0]
    t_ind = start_time

    # covariates
    x_t = data["x_t"]
    y_t = data["y_t"]
    hd_t = data["hd_t"]

    tbin, resamples, rc_t, (rhd_t, rx_t, ry_t) = utils.neural.bin_data(
        bin_size,
        sample_bin,
        spktrain,
        track_samples,
        (np.unwrap(hd_t), x_t, y_t),
        average_behav=True,
        binned=True,
    )

    # recompute velocities
    rw_t = (rhd_t[1:] - rhd_t[:-1]) / tbin
    rw_t = np.concatenate((rw_t, rw_t[-1:]))

    rvx_t = (rx_t[1:] - rx_t[:-1]) / tbin
    rvy_t = (ry_t[1:] - ry_t[:-1]) / tbin
    rs_t = np.sqrt(rvx_t**2 + rvy_t**2)
    rs_t = np.concatenate((rs_t, rs_t[-1:]))

    timestamps = np.arange(resamples) * tbin

    rcov = {
        "hd": rhd_t % (2 * np.pi),
        "omega": rw_t,
        "speed": rs_t,
        "x": rx_t,
        "y": ry_t,
        "time": timestamps,
    }

    metainfo = {
        "neuron_regions": neuron_regions,
    }
    name = data_type
    units_used = rc_t.shape[0]
    max_count = int(rc_t.max())

    # export
    props = {
        "max_count": max_count,
        "tbin": tbin,
        "name": name,
        "neurons": units_used,
        "metainfo": metainfo,
    }

    dataset_dict = {
        "covariates": rcov,
        "ISIs": ISIs,
        "spiketrains": rc_t,
        "timestamps": timestamps,
        "properties": props,
    }
    return dataset_dict


def spikes_dataset(session_name, path, max_ISI_order, select_fracs):
    """
    :param int max_ISI_order: selecting the starting time based on the
                              given ISI lag for which it is defined by data
    :param List select_fracs: boundaries for data subselection based on covariates timesteps
    """
    assert len(select_fracs) == 2
    filename = session_name + ".npz"
    data = np.load(path + filename)

    # spikes
    spktrain = data["spktrain"]
    spktrain[spktrain > 1.0] = 1.0  # ensure binary train
    tbin = data["tbin"]  # seconds
    track_samples = spktrain.shape[1]

    # ISIs
    ISIs = data["ISIs"][:, :, :max_ISI_order]  # (ts, neurons, order)
    order_computed_at = np.empty_like(ISIs[0, :, 0]).astype(int)
    for n in range(order_computed_at.shape[0]):
        order_computed_at[n] = np.where(
            ISIs[:, n, max_ISI_order - 1] == ISIs[:, n, max_ISI_order - 1]
        )[0][0]

    # cut out start of covariates based on ISIs, leave spike trains
    start_ind_covariates = max(order_computed_at)
    valid_samples = track_samples - start_ind_covariates
    start = start_ind_covariates + int(valid_samples * select_fracs[0])
    end = start_ind_covariates + int(valid_samples * select_fracs[1])
    cov_timesamples = end - start
    subselect = slice(start, end)

    # covariates
    x_t = data["x_t"][subselect]
    y_t = data["y_t"][subselect]
    theta_t = data["hd_t"][subselect]
    hd_t = data["hd_t"][subselect]

    # compute velocities
    w_t = (hd_t[1:] - hd_t[:-1]) / tbin
    w_t = np.concatenate((w_t, w_t[-1:]))

    vx_t = (x_t[1:] - x_t[:-1]) / tbin
    vy_t = (y_t[1:] - y_t[:-1]) / tbin
    s_t = np.sqrt(vx_t**2 + vy_t**2)
    s_t = np.concatenate((s_t, s_t[-1:]))

    timestamps = np.arange(start, end) * tbin

    rcov = {
        "theta": theta_t % (2 * np.pi),
        "hd": hd_t % (2 * np.pi),
        "omega": w_t,
        "speed": s_t,
        "x": x_t,
        "y": y_t,
        "time": timestamps,
    }

    ISIs = ISIs[subselect]

    metainfo = {}
    name = (
        session_name + "ISI{}".format(max_ISI_order) + "sel{}to{}".format(*select_fracs)
    )
    units_used = spktrain.shape[0]

    # export
    props = {
        "tbin": tbin,
        "name": name,
        "neurons": units_used,
        "metainfo": metainfo,
    }

    dataset_dict = {
        "covariates": rcov,
        "ISIs": ISIs,
        "spiketrains": spktrain,
        "timestamps": timestamps,
        "align_start_ind": start,
        "properties": props,
    }
    return dataset_dict


def observed_kernel_dict_induc_list(rng, obs_covs, num_induc, out_dims, covariates):
    """
    Get kernel dictionary and inducing point locations for dataset covariates
    """
    induc_list = []
    kernel_dicts = []

    ones = np.ones(out_dims)

    obs_covs_comps = obs_covs.split("-")
    for comp in obs_covs_comps:
        if comp == "":  # empty
            continue

        order_arr = rng.permuted(
            np.tile(np.arange(num_induc), out_dims).reshape(out_dims, num_induc), 
            axis=1, 
        )
        
        if comp == "theta":
            induc_list += [
                np.linspace(0, 2 * np.pi, num_induc + 1)[order_arr][..., None]
            ]
            kernel_dicts += [
                {
                    "type": "periodic",
                    "in_dims": 1,
                    "var": ones,
                    "len": 2.0 * np.ones((out_dims, 1)),
                }
            ]
            
        elif comp == "hd":
            induc_list += [
                np.linspace(0, 2 * np.pi, num_induc + 1)[order_arr][..., None]
            ]
            kernel_dicts += [
                {
                    "type": "periodic",
                    "in_dims": 1,
                    "var": ones,
                    "len": 5.0 * np.ones((out_dims, 1)),
                }
            ]

        elif comp == "omega":
            scale = covariates["omega"].std()
            induc_list += [scale * np.linspace(-1., 1., num_induc)[order_arr][..., None]]
            ls = scale * np.ones(out_dims)
            kernel_dicts += [
                {
                    "type": "SE",
                    "in_dims": 1,
                    "var": ones,
                    "len": 10.0 * np.ones((out_dims, 1)),
                }
            ]

        elif comp == "speed":
            scale = covariates["speed"].std()
            induc_list += [np.linspace(0, scale, num_induc, 1)[order_arr][..., None]]
            kernel_dicts += [
                {
                    "type": "SE",
                    "in_dims": 1,
                    "var": ones,
                    "len": scale * np.ones((out_dims, 1)),
                }
            ]

        elif comp == "x":
            left_x = covariates["x"].min()
            right_x = covariates["x"].max()
            induc_list += [np.linspace(left_x, right_x, num_induc, 1)[order_arr][..., None]]
            ls = (right_x - left_x) / 10.0
            kernel_dicts += [
                {
                    "type": "SE",
                    "in_dims": 1,
                    "var": ones,
                    "len": ls * np.ones((out_dims, 1)),
                }
            ]

        elif comp == "y":
            bottom_y = covariates["y"].min()
            top_y = covariates["y"].max()
            induc_list += [np.linspace(bottom_y, top_y, num_induc)[order_arr][..., None]]
            ls = (top_y - bottom_y) / 10.0
            kernel_dicts += [
                {
                    "type": "SE",
                    "in_dims": 1,
                    "var": ones,
                    "len": ls * np.ones((out_dims, 1)),
                }
            ]

        elif comp == "time":
            scale = covariates["time"].max()
            induc_list += [
                np.linspace(0, scale, num_induc)[order_arr][..., None]
            ]
            kernel_dicts += [
                {
                    "type": "SE",
                    "in_dims": 1,
                    "var": ones,
                    "len": scale / 2.0 * np.ones((out_dims, 1)),
                }
            ]

        else:
            raise ValueError("Invalid covariate type")

    return kernel_dicts, induc_list


def gen_name(parser_args, dataset_dict):

    name = dataset_dict["properties"]["name"] + "_{}_{}_{}_X[{}]_Z[{}]".format(
        parser_args.likelihood,
        parser_args.filter_type,
        parser_args.observations,
        parser_args.observed_covs,
        parser_args.latent_covs,
    )
    return name


def main():
    parser = argparse.ArgumentParser("%(prog)s [options]", "Fit model to data")
    subparsers = parser.add_subparsers(dest="datatype")

    parser_counts = subparsers.add_parser("counts", help="Fit model to count data.")
    parser_spikes = subparsers.add_parser("spikes", help="Fit model to spikes data.")

    parser_counts = template.standard_parser(parser_counts)
    parser_spikes = template.standard_parser(parser_spikes)

    parser_counts.add_argument("--data_path", action="store", type=str)
    parser_counts.add_argument("--session_name", action="store", type=str)
    parser_counts.add_argument(
        "--select_fracs", default=[0.0, 1.0], nargs="+", type=float
    )
    parser_counts.add_argument("--bin_size", default=10, type=int)

    parser_spikes.add_argument("--data_path", action="store", type=str)
    parser_spikes.add_argument("--session_name", action="store", type=str)
    parser_spikes.add_argument(
        "--select_fracs", default=[0.0, 1.0], nargs="+", type=float
    )
    parser_spikes.add_argument("--max_ISI_order", default=5, type=int)

    args = parser.parse_args()

    print("Loading data...")
    if args.datatype == "counts":
        assert args.observations.split("-")[0] == "factorized_gp"
        dataset_dict = counts_dataset(
            args.session_name, args.data_path, args.bin_size, args.select_fracs
        )
    elif args.datatype == "spikes":
        dataset_dict = spikes_dataset(
            args.session_name, args.data_path, args.max_ISI_order, args.select_fracs
        )
    else:
        raise ValueError

    print("Setting up model...")
    save_name = gen_name(args, dataset_dict)
    template.fit_and_save(
        args, dataset_dict, observed_kernel_dict_induc_list, save_name
    )


if __name__ == "__main__":
    main()
