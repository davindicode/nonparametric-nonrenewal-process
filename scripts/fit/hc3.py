import argparse

import numpy as np

import template


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
    theta_t = data["theta_t"][subselect]
    hd_t = data["hd_t"][subselect]

    # compute velocities
    w_t = (hd_t[1:] - hd_t[:-1]) / tbin
    w_t = np.concatenate((w_t, w_t[-1:]))

    vx_t = (x_t[1:] - x_t[:-1]) / tbin
    vy_t = (y_t[1:] - y_t[:-1]) / tbin
    s_t = np.sqrt(vx_t**2 + vy_t**2)
    s_t = np.concatenate((s_t, s_t[-1:]))

    timestamps = np.arange(start, end) * tbin
    print("Selected number of data time points: {}".format(len(timestamps)))

    rcov = {
        "theta": theta_t % (2 * np.pi),
        "hd": hd_t % (2 * np.pi),
        "omega": w_t,
        "speed": s_t,
        "x": x_t,
        "y": y_t,
        "time": timestamps - timestamps[0],
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
            induc_list += [
                scale * np.linspace(-1.0, 1.0, num_induc)[order_arr][..., None]
            ]
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
            induc_list += [
                np.linspace(left_x, right_x, num_induc, 1)[order_arr][..., None]
            ]
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
            induc_list += [
                np.linspace(bottom_y, top_y, num_induc)[order_arr][..., None]
            ]
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
            induc_list += [np.linspace(0, scale, num_induc)[order_arr][..., None]]
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


def main():
    parser = argparse.ArgumentParser("%(prog)s [options]", "Fit model to data")
    parser = template.standard_parser(parser)

    parser.add_argument("--data_path", action="store", type=str)
    parser.add_argument("--session_name", action="store", type=str)
    parser.add_argument(
        "--select_fracs", default=[0.0, 1.0], nargs="+", type=float
    )
    parser.add_argument("--max_ISI_order", default=5, type=int)

    args = parser.parse_args()

    print("Loading data...")
    dataset_dict = spikes_dataset(
        args.session_name, args.data_path, args.max_ISI_order, args.select_fracs
    )

    print("Setting up model...")
    save_name = template.gen_name(args, dataset_dict)
    template.fit_and_save(
        args, dataset_dict, observed_kernel_dict_induc_list, save_name
    )


if __name__ == "__main__":
    main()
