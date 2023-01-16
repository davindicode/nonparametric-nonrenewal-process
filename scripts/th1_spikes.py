import argparse

import numpy as np

import sys
sys.path.append("..")

import lib



def get_dataset(session_name, path, max_ISI_order, select_fracs=None):
    filename = session_name + ".npz"
    data = np.load(path + filename)

    # select units
    if data_type == "th1":
        sel_unit = data["hdc_unit"]
    else:
        sel_unit = ~data["hdc_unit"]
    neuron_regions = data["neuron_regions"][sel_unit]  # 1 is ANT, 0 is PoS
    
    # spikes
    spktrain = data["spktrain"][sel_unit, :]
    spktrain[spktrain > 1.0] = 1.0  # ensure binary train
    sample_bin = data["tbin"]  # s
    #sample_bin = 0.001
    track_samples = spktrain.shape[1]
    
    # ISIs
    ISIs = data["ISIs"][:, sel_unit, :]  # (ts, neurons, order)
    order_computed_at = np.empty_like(ISIs[0, ...]).astype(int)
    for n in range(order_computed_at.shape[0]):
        for k in range(order_computed_at.shape[1]):
            order_computed_at[n, k] = np.where(ISIs[:, n, k] == ISIs[:, n, k])[0][0]
            
    # cut out section
    if select_fracs is None:
        select_fracs = [0., 1.]
    start_time = max(order_computed_at[:, max_ISI_order])
    valid_samples = track_samples - start_time
    t_ind = start_time
    
    
    # covariates
    x_t = data["x_t"]
    y_t = data["y_t"]
    hd_t = data["hd_t"]

    # compute velocities
    w_t = (hd_t[1:] - hd_t[:-1]) / tbin
    w_t = np.concatenate((w_t, w_t[-1:]))

    vx_t = (x_t[1:] - x_t[:-1]) / tbin
    vy_t = (y_t[1:] - y_t[:-1]) / tbin
    s_t = np.sqrt(vx_t**2 + vy_t**2)
    s_t = np.concatenate((s_t, s_t[-1:]))
    
    timestamps = np.arange(resamples) * tbin

    rcov = {
        "hd": hd_t % (2 * np.pi),
        "omega": w_t,
        "speed": s_t,
        "x": x_t,
        "y": y_t,
        "time": timestamps,
    }

    metainfo = {
        "neuron_regions": neuron_regions,
    }
    name = data_type
    units_used = rc_t.shape[0]
    
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
        "spiketrains": rc_t,
        "timestamps": timestamps,
        "properties": props, 
    }
    return dataset_dict




def observed_kernel_dict_induc_list(x_mode, num_induc, covariates):
    """
    Get kernel dictionary and inducing point locations for dataset covariates
    """
    induc_list = []
    kernel_dicts = []
    
    ones = np.ones(out_dims)

    x_mode_comps = x_mode.split("-")
    for comp in x_mode_comps:
        if comp == "":  # empty
            continue

        if comp == "hd":
            induc_list += [np.linspace(0, 2 * np.pi, num_induc + 1)[:-1]]
            kernel_dicts += [
                {"type": "circSE", "var": ones, "len": 5.0 * np.ones((out_dims, 1))}]
            
        elif comp == "omega":
            scale = covariates["omega"].std()
            induc_list += [scale * np.random.randn(num_induc)]
            ls = scale * np.ones(out_dims)
            kernel_dicts += [
                {"type": "SE", "var": ones, "len": 10.0 * np.ones((out_dims, 1))}]
            
        elif comp == "speed":
            scale = covariates["speed"].std()
            induc_list += [np.random.uniform(0, scale, size=(num_induc,))]
            kernel_dicts += [
                {"type": "SE", "var": ones, "len": scale * np.ones((out_dims, 1))}]
            
        elif comp == "x":
            left_x = covariates["x"].min()
            right_x = covariates["x"].max()
            induc_list += [np.random.uniform(left_x, right_x, size=(num_induc,))]
            ls = (right_x - left_x) / 10.0
            kernel_dicts += [
                {"type": "SE", "var": ones, "len": ls * np.ones((out_dims, 1))}]
            
        elif comp == "y":
            bottom_y = covariates["y"].min()
            top_y = covariates["y"].max()
            induc_list += [np.random.uniform(bottom_y, top_y, size=(num_induc,))]
            ls = (top_y - bottom_y) / 10.0
            kernel_dicts += [
                {"type": "SE", "var": ones, "len": ls * np.ones((out_dims, 1))}]
            
        elif comp == "time":
            scale = covariates["time"].max()
            induc_list += [np.linspace(0, scale, num_induc)]
            kernel_dicts += [
                {"type": "SE", "var": ones, "len": scale / 2.0 * np.ones((out_dims, 1))}]
            
        else:
            raise ValueError("Invalid covariate type")

    return kernel_dicts, induc_list



def gen_name(model_dict, ):

    name = model_dict[
        "model_name"
    ] + "_{}_{}H{}_{}_X[{}]_Z[{}]".format(
        config.ll_mode,
        config.filter,
        config.filter_length,
        config.mapping,
        config.observed_cov,
        config.latent_cov,
        #config.bin_size,
    )
    return name



def main():
    parser = template.standard_parser("%(prog)s [OPTION] [FILE]...", "Fit model to data.")
    parser.add_argument("--data_path", action="store", type=str)
    parser.add_argument("--session_name", action="store", type=str)
    
    parser.add_argument("--select_fracs", default=[0., 1.], nargs="+", type=float)
    
    args = parser.parse_args()

    # session_name = "Mouse28_140313_wake"
    dataset_dict = get_dataset(
        args.session_name, args.data_path, args.select_fracs
    )

    save_name = gen_name(args, dataset_dict)
    template.fit(args, dataset_dict, observed_kernel_dict_induc_list, fix_param_names, save_name)


if __name__ == "__main__":
    main()
