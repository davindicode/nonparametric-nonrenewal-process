import numpy as np

import torch
import torch.nn as nn

import template

import sys
sys.path.append("../lib/")
import neuroprob as nprb
from neuroprob import utils


### data ###
def get_dataset(data_type, bin_size, single_spikes, path):

    if data_type == "hCMP" or data_type == "IP":  # synthetic
        assert bin_size == 1
        metainfo = {}

        if data_type == "hCMP":
            syn_data = np.load(path + "hCMP_HDC.npz")
            rcov = {
                "hd": syn_data["rhd_t"],
            }

        elif data_type == "IP":
            syn_data = np.load(path + "IP_HDC.npz")
            rcov = {"hd": syn_data["rhd_t"], "a": syn_data["ra_t"]}

        rc_t = syn_data["spktrain"]
        tbin = syn_data["tbin"].item()

        resamples = rc_t.shape[1]
        units_used = rc_t.shape[0]

    elif data_type == "th1" or data_type == "th1leftover":
        data = np.load(path + "Mouse28_140313_wake.npz")

        if data_type == "th1":
            sel_unit = data["hdc_unit"]
        else:
            sel_unit = ~data["hdc_unit"]

        neuron_regions = data["neuron_regions"][sel_unit]  # 1 is ANT, 0 is PoS
        spktrain = data["spktrain"][sel_unit, :]

        x_t = data["x_t"]
        y_t = data["y_t"]
        hd_t = data["hd_t"]

        sample_bin = 0.001
        track_samples = spktrain.shape[1]

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
        rtime_t = np.arange(resamples) * tbin

        rcov = {
            "hd": rhd_t % (2 * np.pi),
            "omega": rw_t,
            "speed": rs_t,
            "x": rx_t,
            "y": ry_t,
            "time": rtime_t,
        }

        metainfo = {
            "neuron_regions": neuron_regions,
        }

    name = data_type

    # export
    if single_spikes is True:
        rc_t[rc_t > 1.0] = 1.0

    units_used = rc_t.shape[0]
    max_count = int(rc_t.max())

    dataset_dict = {
        "name": name,
        "covariates": rcov,
        "spiketrains": rc_t,
        "neurons": units_used,
        "metainfo": metainfo,
        "tbin": tbin,
        "timesamples": resamples,
        "max_count": max_count,
        "bin_size": bin_size,
        "trial_sizes": None,
    }
    return dataset_dict


### model ###
class FFNN_model(nn.Module):
    """
    Multi-layer perceptron class
    """

    def __init__(self, layers, angle_dims, euclid_dims, hist_len, out_dims):
        """
        Assumes angular dimensions to be ordered first in the input of shape dimensionsxhist_len.
        dim_fill is hist_len or hist_len*neurons when parallel neurons
        """
        super().__init__()
        self.angle_dims = angle_dims
        self.in_dims = 2 * angle_dims + euclid_dims
        net = utils.pytorch.MLP(
            layers, self.in_dims, out_dims, nonlin=utils.pytorch.Siren(), out=None
        )
        self.add_module("net", net)

    def forward(self, input):
        """
        Input of shape (samplesxtime, dims)
        """
        embed = torch.cat(
            (
                torch.cos(input[:, : self.angle_dims]),
                torch.sin(input[:, : self.angle_dims]),
                input[:, self.angle_dims :],
            ),
            dim=-1,
        )
        return self.net(embed)


def enc_used(model_dict, covariates, learn_mean):
    """ """
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

        # objects
        kernelobj, constraints = template.create_kernel(kernel_tuples, "exp", tensor_type)

        Xu = torch.tensor(np.array(ind_list)).T[None, ...].repeat(out_dims, 1, 1)
        inpd = Xu.shape[-1]
        inducing_points = nprb.mappings.inducing_points(
            out_dims, Xu, constraints, tensor_type=tensor_type
        )

        mapping = nprb.mappings.SVGP(
            in_dims,
            out_dims,
            kernelobj,
            inducing_points=inducing_points,
            jitter=jitter,
            whiten=True,
            mean=mean,
            learn_mean=learn_mean,
            tensor_type=tensor_type,
        )

    elif map_mode_comps[0] == "ffnn":  # feedforward neural network mapping
        rate_model = FFNN_params(in_dims, enc_layers, angle_dims, inner_dims, inv_link)
        hist_len = 1

        mu_ANN = FFNN_model(
            enc_layers, angle_dims, tot_dims - angle_dims, hist_len, neurons
        )
        mapping = mdl.parametrics.FFNN(
            tot_dims, neurons, inv_link, mu_ANN, sigma_ANN=None, tens_type=torch.float
        )

    else:
        raise ValueError

    return mapping


### main ###
def main():
    parser = template.standard_parser("%(prog)s [OPTION] [FILE]...", "Fit model to data.")
    parser.add_argument("--data_path", action="store", type=str)
    parser.add_argument("--data_type", action="store", type=str)

    args = parser.parse_args()

    dev = nprb.inference.get_device(gpu=args.gpu)
    dataset_dict = get_dataset(
        args.data_type, args.bin_size, args.single_spikes, args.data_path
    )

    template.train_model(dev, args, dataset_dict, enc_used, args.checkpoint_dir)


if __name__ == "__main__":
    main()
