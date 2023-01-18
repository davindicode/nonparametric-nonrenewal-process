import argparse

import numpy as np

import gplvm_template

import sys
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




def dataset_specifics(model_dict, covariates, learn_mean):
    """
    Dataset specific information for GP
    """
    ll_mode, map_mode, x_mode, z_mode = (
        model_dict["ll_mode"],
        model_dict["map_mode"],
        model_dict["x_mode"],
        model_dict["z_mode"],
    )
    jitter, tensor_type = model_dict["jitter"], model_dict["tensor_type"]
    neurons, in_dims = (
        model_dict["neurons"],
        model_dict["map_xdims"] + model_dict["map_zdims"],
    )
    
    
    left_x = x_t.min()
    right_x = x_t.max()
    bottom_y = y_t.min()
    top_y = y_t.max()

    v = var * torch.ones(out_dims)

    l = 10.0 * np.ones(out_dims)
    l_ang = 5.0 * np.ones(out_dims)
    l_s = 10.0 * np.ones(out_dims)
    l_time = time_t.max() / 2.0 * np.ones(out_dims)
    l_one = np.ones(out_dims)

    upperT = 2.0  # e^2
    l_ISI = 3.0 * np.ones(out_dims)
    tau_0 = 5.0 * np.ones((out_dims, max_Horder))
    # tau_0 = np.empty((out_dims, max_Horder))
    # for k in range(1, 1+max_Horder):
    #    tau_0[:, -k] = cov_tuple[-k].mean(1)[:, 0]

    # x
    if x_mode[:-1] == "th-pos-isi":
        H_isi = int(x_mode[-1])
        ind_list = [
            np.linspace(0, 2 * np.pi, num_induc + 1)[:-1],
            np.random.uniform(left_x, right_x, size=(num_induc,)),
            np.random.uniform(bottom_y, top_y, size=(num_induc,)),
        ]
        for h in range(H_isi):
            ind_list += [
                np.random.uniform(0.0, tau_0[:, h].mean(), size=(num_induc,))
            ]

        kernel_tuples = [
            ("variance", v),
            ("SE", "torus", torch.tensor([l_ang])),
            ("SE", "euclid", torch.tensor([l, l])),
            ("tSE", "euclid", torch.tensor([l_ISI] * H_isi), tau_0[:, :H_isi]),
        ]

    elif x_mode == "th-pos":
        ind_list = [
            np.linspace(0, 2 * np.pi, num_induc + 1)[:-1],
            np.random.uniform(left_x, right_x, size=(num_induc,)),
            np.random.uniform(bottom_y, top_y, size=(num_induc,)),
        ]

        kernel_tuples = [
            ("variance", v),
            ("SE", "torus", torch.tensor([l_ang])),
            ("SE", "euclid", torch.tensor([l, l])),
        ]

    elif x_mode[:-1] == "th-hd-pos-s-t-isi":
        ind_list = [
            np.linspace(0, 2 * np.pi, num_induc + 1)[:-1],
            np.random.rand(num_induc) * 2 * np.pi,
            np.random.uniform(left_x, right_x, size=(num_induc,)),
            np.random.uniform(bottom_y, top_y, size=(num_induc,)),
            np.random.uniform(0, s_t.std(), size=(num_induc,)),
            np.linspace(0, time_t.max(), num_induc),
        ]
        kernel_tuples = [
            ("variance", v),
            ("SE", "torus", torch.tensor([l_ang, l_ang])),
            ("SE", "euclid", torch.tensor([l, l])),
            ("SE", "euclid", torch.tensor([l_s])),
            ("SE", "euclid", torch.tensor([l_time])),
        ]


    out_dims = model_dict["map_outdims"]
    mean = torch.zeros((out_dims)) if learn_mean else 0  # not learnable vs learnable

    map_mode_comps = map_mode.split("-")
    x_mode_comps = x_mode.split("-")

    def get_inducing_locs_and_ls(comp):
        if comp == "hd":
            locs = np.linspace(0, 2 * np.pi, num_induc + 1)[:-1]
            ls = 5.0 * np.ones(out_dims)
        elif comp == "omega":
            scale = covariates["omega"].std()
            locs = scale * np.random.randn(num_induc)
            ls = scale * np.ones(out_dims)
        elif comp == "speed":
            scale = covariates["speed"].std()
            locs = np.random.uniform(0, scale, size=(num_induc,))
            ls = 10.0 * np.ones(out_dims)
        elif comp == "x":
            left_x = covariates["x"].min()
            right_x = covariates["x"].max()
            locs = np.random.uniform(left_x, right_x, size=(num_induc,))
            ls = (right_x - left_x) / 10.0 * np.ones(out_dims)
        elif comp == "y":
            bottom_y = covariates["y"].min()
            top_y = covariates["y"].max()
            locs = np.random.uniform(bottom_y, top_y, size=(num_induc,))
            ls = (top_y - bottom_y) / 10.0 * np.ones(out_dims)
        elif comp == "time":
            scale = covariates["time"].max()
            locs = np.linspace(0, scale, num_induc)
            ls = scale / 2.0 * np.ones(out_dims)
        else:
            raise ValueError("Invalid covariate type")

        return locs, ls, (comp == "hd")

    if map_mode_comps[0] == "svgp":
        num_induc = int(map_mode_comps[1])

        var = 1.0  # initial kernel variance
        v = var * torch.ones(out_dims)

        ind_list = []
        kernel_tuples = [("variance", v)]
        ang_ls, euclid_ls = [], []

        # x
        for xc in x_mode_comps:
            if xc == "":
                continue

            locs, ls, angular = get_inducing_locs_and_ls(xc)

            ind_list += [locs]
            if angular:
                ang_ls += [ls]
            else:
                euclid_ls += [ls]

        if len(ang_ls) > 0:
            ang_ls = np.array(ang_ls)
            kernel_tuples += [("SE", "torus", torch.tensor(ang_ls))]
        if len(euclid_ls) > 0:
            euclid_ls = np.array(euclid_ls)
            kernel_tuples += [("SE", "euclid", torch.tensor(euclid_ls))]

        # z
        latent_k, latent_u = template.latent_kernel(z_mode, num_induc, out_dims)
        kernel_tuples += latent_k
        ind_list += latent_u

    return kernel_dicts, induc_list



def gen_name(parser_args, dataset_dict):

    name = dataset_dict["properties"]["name"] + "_{}_{}_{}_X[{}]_Z[{}]".format(
        parser_args.likelihood,
        parser_args.filter_type,
        parser_args.observations,
        parser_args.observed_covs,
        parser_args.latent_covs,
        #parser_args.bin_size,
    )
    return name



def main():
    parser = template.standard_parser("%(prog)s [options]", "Fit model to data.")
    parser.add_argument("--data_path", action="store", type=str)
    parser.add_argument("--data_type", action="store", type=str)

    args = parser.parse_args()
    
    
    
    # dataset
    dataset_dict = get_dataset(session_id, phase)
    lib.models.fit_model(dev, args, dataset_dict, )

    
    batches = args.batches
    dataset = lib.utils.Dataset(inp, target, batches)

    dataset = args.dataset
    dataset_dict = get_dataset(
        args.data_type, args.bin_size, args.single_spikes, args.data_path
    )

    template.fit_model(dev, args, dataset_dict, dataset_specifics, args.checkpoint_dir)


if __name__ == "__main__":
    main()
