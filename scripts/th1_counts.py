import argparse

import numpy as np

import sys
sys.path.append("..")

import lib



def get_dataset(session_name, bin_size, path, select_fracs=None):
    filename = session_name + ".npz"
    data = np.load(path + filename)

    # select units
    if data_type == "th1":
        sel_unit = data["hdc_unit"]
    else:
        sel_unit = ~data["hdc_unit"]
    neuron_regions = data["neuron_regions"][sel_unit]  # 1 is ANT, 0 is PoS
    
    # spike counts
    spktrain = data["spktrain"][sel_unit, :]
    sample_bin = data["tbin"]  # s
    #sample_bin = 0.001
    track_samples = spktrain.shape[1]
            
    # cut out section
    if select_fracs is None:
        select_fracs = [0., 1.]
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




def dataset_specifics(config, covariates):
    """ """
    ll_mode, map_mode, x_mode, z_mode = (
        model_dict["ll_mode"],
        model_dict["map_mode"],
        model_dict["x_mode"],
        model_dict["z_mode"],
    )
    jitter, tensor_type = model_dict["jitter"], model_dict["tensor_type"]
    neurons = config.neurons

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

    return kernel_tuples, induc_list



def main():
    parser = template.standard_parser("%(prog)s [OPTION] [FILE]...", "Fit model to data.")
    parser.add_argument("--data_path", action="store", type=str)
    parser.add_argument("--data_type", action="store", type=str)

    parser.add_argument("--bin_size", type=int)
    
    args = parser.parse_args()

    # session_name = "Mouse28_140313_wake"
    dataset_dict = get_dataset(
        args.data_type, args.bin_size, args.single_spikes, args.data_path
    )

    template.fit(args, dataset_dict, dataset_specifics, args.checkpoint_dir)


if __name__ == "__main__":
    main()
