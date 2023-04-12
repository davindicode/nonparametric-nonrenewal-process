import argparse

import sys

import gplvm_template

import numpy as np

sys.path.append("..")

import lib


def get_dataset(session_name, max_ISI_order, select_fracs=None):
    data = np.load("../data/hc5_{}_ext.npz".format(session_name))
    cov = data["covariates"]  # time, dims, neurons
    cov_tuple = tuple(np.transpose(cov, (1, 2, 0))[..., None])

    timesamples = cov.shape[0]
    tbin = 0.001

    data = np.load("../data/hc5_{}.npz".format(session_name))
    spktrain = data["spktrain"]  # full
    arena = data["arena"]

    bin_size = 1  # resolution of data provided
    max_count = 1
    spktrain[spktrain > 1.0] = 1.0
    units_used = spktrain.shape[-2]

    model_name = "CA{}_".format(dataset)
    metainfo = (arena,)
    return (
        cov_tuple,
        units_used,
        tbin,
        timesamples,
        spktrain,
        max_count,
        bin_size,
        metainfo,
        model_name,
    )


def observed_kernel_dict_induc_list(obs_covs, num_induc, out_dims, covariates):
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

        if comp == "hd":
            induc_list += [np.linspace(0, 2 * np.pi, num_induc + 1)[:-1]]
            kernel_dicts += [
                {"type": "circSE", "var": ones, "len": 5.0 * np.ones((out_dims, 1))}
            ]

        elif comp == "theta":
            induc_list += [np.linspace(0, 2 * np.pi, num_induc + 1)[:-1]]
            kernel_dicts += [
                {"type": "circSE", "var": ones, "len": 5.0 * np.ones((out_dims, 1))}
            ]

        elif comp == "speed":
            scale = covariates["speed"].std()
            induc_list += [np.random.uniform(0, scale, size=(num_induc,))]
            kernel_dicts += [
                {"type": "SE", "var": ones, "len": scale * np.ones((out_dims, 1))}
            ]

        elif comp == "x":
            left_x = covariates["x"].min()
            right_x = covariates["x"].max()
            induc_list += [np.random.uniform(left_x, right_x, size=(num_induc,))]
            ls = (right_x - left_x) / 10.0
            kernel_dicts += [
                {"type": "SE", "var": ones, "len": ls * np.ones((out_dims, 1))}
            ]

        elif comp == "y":
            bottom_y = covariates["y"].min()
            top_y = covariates["y"].max()
            induc_list += [np.random.uniform(bottom_y, top_y, size=(num_induc,))]
            ls = (top_y - bottom_y) / 10.0
            kernel_dicts += [
                {"type": "SE", "var": ones, "len": ls * np.ones((out_dims, 1))}
            ]

        elif comp == "time":
            scale = covariates["time"].max()
            induc_list += [np.linspace(0, scale, num_induc)]
            kernel_dicts += [
                {"type": "SE", "var": ones, "len": scale / 2.0 * np.ones((out_dims, 1))}
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
        # parser_args.bin_size,
    )
    return name


def main():
    parser = template.standard_parser("%(prog)s [options]", "Fit model to data.")
    parser.add_argument("--data_path", action="store", type=str)
    parser.add_argument("--data_type", action="store", type=str)

    args = parser.parse_args()

    # dataset
    dataset_dict = get_dataset(session_id, phase)
    lib.models.fit_model(
        dev,
        args,
        dataset_dict,
    )

    batches = args.batches
    dataset = lib.utils.Dataset(inp, target, batches)

    dataset = args.dataset
    dataset_dict = get_dataset(
        args.data_type, args.bin_size, args.single_spikes, args.data_path
    )

    template.fit_model(dev, args, dataset_dict, dataset_specifics, args.checkpoint_dir)


if __name__ == "__main__":
    main()
