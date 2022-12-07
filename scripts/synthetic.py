import pickle

import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append("../../neuroppl/")
sys.path.append("../lib/")


import lib
import neuroppl as nppl
from neuroppl import utils

from synthetic_data import max_Horder


### data
def get_dataset(datatype):
    bin_size = 1

    if datatype == 0:  # single neuron
        dataname = "SN"

    elif datatype == 1:  # head direction cell
        dataname = "HDC"

        data = np.load("../data/HDC_gamma.npz")
        spktrain = data["spktrain"]
        cov = data["cov"]
        tbin = 0.001

        units_used = spktrain.shape[0]
        resamples = cov.shape[0]
        cov_tuple = tuple(np.transpose(cov, (1, 2, 0))[..., None])

        max_count = int(spktrain.max())
        model_name = "syn{}_".format(dataname)
        metainfo = ()

    elif datatype == 2:  # place cells
        dataname = "CA"

    else:
        raise ValueError

    return (
        cov_tuple,
        units_used,
        tbin,
        resamples,
        spktrain,
        max_count,
        bin_size,
        metainfo,
        model_name,
    )


### model
def inputs_used_(datatype, x_mode, z_mode, cov_tuple, batch_info, tensor_type):
    """
    Create the used covariates list.
    """
    timesamples = cov_tuple[0].shape[0]
    b = [torch.from_numpy(b) for b in cov_tuple]
    if datatype == 0:
        I_t, isi = b
    elif datatype == 1:
        hd_t, isi1, isi2, isi3, isi4, isi5 = b
    elif datatype == 2:
        x_t, y_t, th_t, isi = b

    # x
    if x_mode[:-1] == "hd-isi":
        H_isi = int(x_mode[-1])
        input_data = [hd_t]
        for h in range(-max_Horder, -max_Horder + H_isi):
            input_data += [b[h]]

    elif x_mode == "hd":
        input_data = [hd_t]

    elif x_mode[:-1] == "th-pos-isi":
        H_isi = int(x_mode[-1])
        input_data = [th_t, x_t, y_t]
        for h in range(-max_Horder, -max_Horder + H_isi):
            input_data += [b[h]]

    elif x_mode == "I":
        input_data = [nppl.inference.filtered_input()]

    elif x_mode == "":
        input_data = []

    else:
        raise ValueError

    d_x = len(input_data)

    latents, d_z = lib.helper.latent_objects(z_mode, d_x, timesamples, tensor_type)
    input_data += latents
    return input_data, d_x, d_z


def enc_used_(
    datatype,
    map_mode,
    ll_mode,
    x_mode,
    z_mode,
    cov_tuple,
    in_dims,
    inner_dims,
    jitter,
    tensor_type,
):
    """ """

    def get_angle_dims(x_mode, z_mode):
        angle_dims = []
        if x_mode[:-1] == "hd-isi":
            angle_dims += [0]
        return angle_dims

    def kernel_used_(
        datatype, x_mode, z_mode, cov_tuple, num_induc, out_dims, var, tensor_type
    ):
        """
        Get kernel and inducing points
        """
        if datatype == 0:
            I_t, isi = cov_tuple

        elif datatype == 1:
            hd_t, isi1, isi2, isi3, isi4, isi5 = cov_tuple

            l_ang = 3.0 * np.ones(out_dims)

        elif datatype == 2:
            x_t, y_t, th_t, isi = cov_tuple

            left_x = x_t.min()
            right_x = x_t.max()
            bottom_y = y_t.min()
            top_y = y_t.max()

        v = var * torch.ones(out_dims)

        l_ISI = 2.0 * np.ones(out_dims)
        tau_0 = 5.0 * np.ones((out_dims, max_Horder))
        # tau_0 = np.empty((out_dims, max_Horder))
        # for k in range(1, max_Horder+1):
        # tau_0[:, -k] = cov_tuple[-k].mean(1)[:, 0]

        # x
        if x_mode[:-1] == "hd-isi":
            H_isi = int(x_mode[-1])
            ind_list = [np.linspace(0, 2 * np.pi, num_induc + 1)[:-1]]
            for h in range(H_isi):
                ind_list += [
                    np.random.uniform(0.0, tau_0[:, h].mean(), size=(num_induc,))
                ]

            kernel_tuples = [
                ("variance", v),
                ("RBF", "torus", torch.tensor([l_ang])),
                ("tRBF", "euclid", torch.tensor([l_ISI] * H_isi), tau_0[:, :H_isi].T),
            ]

        elif x_mode == "hd":
            ind_list = [np.linspace(0, 2 * np.pi, num_induc + 1)[:-1]]

            kernel_tuples = [("variance", v), ("RBF", "torus", torch.tensor([l_ang]))]

        elif x_mode[:-1] == "th-pos-isi":
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
                ("RBF", "torus", torch.tensor([l_ang])),
                ("RBF", "euclid", torch.tensor([l, l])),
                ("RBF", "euclid", torch.tensor([l_ISI] * H_isi)),
            ]

        elif x_mode == "":  # for GP filters
            ind_list = []
            kernel_tuples = [("variance", v)]

        else:
            raise ValueError

        # z
        latent_k, latent_u = lib.helper.latent_kernel(z_mode)
        kernel_tuples += latent_k
        ind_list += latent_u

        # objects
        kernelobj, constraints = lib.helper.create_kernel(
            kernel_tuples, "exp", tensor_type
        )

        Xu = torch.tensor(ind_list).T[None, ...].repeat(out_dims, 1, 1)
        inpd = Xu.shape[-1]
        inducing_points = nppl.kernels.kernel.inducing_points(
            out_dims, Xu, constraints, tensor_type=tensor_type
        )

        return kernelobj, inducing_points

    if map_mode[:4] == "svgp":
        num_induc = int(map_mode[4:])

        mean = (
            0 if ll_mode == "U" else torch.zeros((inner_dims))
        )  # not learnable vs learnable
        learn_mean = ll_mode != "U"

        var = 1.0  # initial kernel variance
        kernelobj, inducing_points = kernel_used_(
            datatype, x_mode, z_mode, cov_tuple, num_induc, inner_dims, var, tensor_type
        )

        mapping = nppl.mappings.SVGP(
            in_dims,
            inner_dims,
            kernelobj,
            inducing_points=inducing_points,
            jitter=jitter,
            whiten=True,
            mean=mean,
            learn_mean=learn_mean,
            tensor_type=tensor_type,
        )

        # heteroscedastic likelihoods
        if ll_mode[0] == "h":
            kernelobj, inducing_points = kernel_used_(
                datatype,
                x_mode,
                z_mode,
                cov_tuple,
                num_induc,
                inner_dims,
                var,
                tensor_type,
            )

            hgp = nppl.mappings.SVGP(
                in_dims,
                inner_dims,
                kernelobj,
                inducing_points=inducing_points,
                jitter=jitter,
                whiten=True,
                mean=torch.zeros((inner_dims)),
                learn_mean=True,
                tensor_type=tensor_type,
            )

        else:
            hgp = None

    elif map_mode == "ffnn":
        enc = lib.rnn.enc_model(layers, angle_dims, hist_len, in_dims, neurons, nonlin)
        ffnn = nppl.mappings.FFNN(
            input_dim,
            out_dims,
            mu_ANN,
            sigma_ANN=None,
            tensor_type=torch.float,
            active_dims=None,
        )
        mapping = lib.helper.ANN(enc, ffnn)

        if ll_mode[0] == "h":  # heteroscedastic likelihoods
            enc = lib.rnn.enc_model(
                layers, angle_dims, hist_len, in_dims, neurons, nonlin
            )
            ffnn = nppl.mappings.FFNN(
                input_dim,
                out_dims,
                mu_ANN,
                sigma_ANN=None,
                tensor_type=torch.float,
                active_dims=None,
            )
            hgp = lib.helper.ANN(enc, ffnn)

        else:
            hgp = None

    elif map_mode == "rnn":
        encoder = lib.rnn.enc_used()
        mapping = lib.rnn.cumul_RNN()

    else:
        raise ValueError

    return mapping, hgp


### main
def main():
    parser = lib.models.standard_parser(
        "%(prog)s [OPTION] [FILE]...", "Fit model to data."
    )
    parser.add_argument("--datatype", type=int)
    args = parser.parse_args()

    dev = utils.pytorch.get_device(gpu=args.gpu)

    datatype = args.datatype
    dataset_tuple = get_dataset(datatype)
    inputs_used = lambda *args: inputs_used_(datatype, *args)
    enc_used = lambda *args: enc_used_(datatype, *args)

    lib.models.training(dev, args, dataset_tuple, inputs_used, enc_used)


if __name__ == "__main__":
    main()
