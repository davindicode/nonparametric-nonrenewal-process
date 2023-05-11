import argparse

import os
import sys

sys.path.append("../fit/")
import synthetic
import template

sys.path.append("../../../GaussNeuro")
import gaussneuro as lib

sys.path.append("../../data/synthetic")
import pickle

import generate

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

import utils
from jax import jit, vmap


def tuning(
    checkpoint_dir,
    reg_config_names,
    dataset_dict,
    rng,
    prng_state,
    neuron_list,
    num_samps,
    int_eval_pts,
    num_quad_pts,
    batch_size,
    outdims_per_batch,
):
    """
    Compute tuning curves of BNPP model
    """
    # data
    tbin = dataset_dict["properties"]["tbin"]
    neurons = dataset_dict["properties"]["neurons"]

    covs_dims = 2
    x = dataset_dict["covariates"]["x"]
    y = dataset_dict["covariates"]["y"]
    ISIs = dataset_dict["ISIs"]

    mean_ISIs = utils.mean_ISI(ISIs)

    ### plotting mesh ###
    pos_x_locs = np.meshgrid(
        np.linspace(x.min(), x.max(), 30),
        np.linspace(y.min(), y.max(), 30),
    )
    pos_x_locs = np.stack(pos_x_locs, axis=-1)
    or_shape = pos_x_locs.shape[:-1]
    pos_x_locs = pos_x_locs.reshape(-1, covs_dims)  # (evals, x_dim)
    eval_pts = pos_x_locs.shape[0]

    # conditional ISIs
    evalsteps = 200
    uisi_t_eval = np.linspace(0.0, 5.0, evalsteps)
    cisi_t_eval = np.linspace(0.0, 1.0, evalsteps)

    pts = 3
    neuron_conds = [2, 5, 8]
    x_conds = [
        np.array([400.0, 300.0]),
        np.array([300.0, 200.0]),
        np.array([250.0, 250.0]),
    ]  # (covs_dims,)

    tunings = {}

    ### ground truth ###
    mdl = generate.model("Gaussian")
    ratefunc, _, renewals_ll, _, _ = mdl.get_model()

    GT_rates = vmap(ratefunc)(jnp.array(pos_x_locs))
    GT_rates = np.array(GT_rates.reshape(*or_shape, -1))

    # ground truth conditional ISIs
    GT_rate_conds = vmap(ratefunc)(x_conds)
    GT_ISI_densities, GT_unit_renewals = [], []
    for pn in range(pts):
        rate_cond = GT_rate_conds[pn]
        gt_isi = []
        for ne, rll in enumerate(renewals_ll):
            gt_isi.append(
                rate_cond[ne]
                * np.exp(vmap(rll)(rate_cond[ne] * jnp.array(cisi_t_eval)[:, None]))
            )

        gt_isi = np.concatenate(gt_isi, axis=1)  # (pts, obs_dims)
        GT_ISI_densities.append(gt_isi)

    GT_ISI_densities = np.stack(GT_ISI_densities, axis=0)

    for rll in renewals_ll:
        GT_unit_renewals.append(np.exp(vmap(rll)(jnp.array(uisi_t_eval)[:, None])))
    GT_unit_renewals = np.concatenate(GT_unit_renewals, axis=1)  # (pts, obs_dims)

    tunings["GT"] = {
        "unit_renewals": GT_unit_renewals,
        "pos_rates": GT_rates,
        "ISI_densities": GT_ISI_densities,
    }

    for model_name in reg_config_names:
        print("Analyzing tuning for {}...".format(model_name))

        # config
        model, config = utils.load_model_and_config(
            checkpoint_dir + model_name,
            dataset_dict,
            synthetic.observed_kernel_dict_induc_list,
            rng,
        )
        obs_type = config.observations.split("-")[0]
        jitter = config.jitter

        if obs_type == "factorized_gp" or obs_type == "rate_renewal_gp":
            ### tuning ###
            print("Position tuning...")

            pos_rates, _, _ = model.obs_model.gp_posterior(
                pos_x_locs[None, None],
                mean_only=True,
                diag_cov=True,
                compute_KL=False,
                jitter=jitter,
                sel_outdims=jnp.array(neuron_list),
            )  # (1, out_dims, eval_locs)
            pos_rates = np.array(pos_rates.reshape(-1, *or_shape))
            len_xs = np.array(model.obs_model.gp.kernel.kernels[0].lengthscale)

            # export
            results = {
                "pos_rates": pos_rates,
                "len_xs": len_xs,
            }

            if model.obs_model.spikefilter is not None:
                filter_t, _ = model.obs_model.spikefilter.sample_posterior(
                    prng_state, 1, False, None, jitter
                )
                results["spike_filter"] = np.array(
                    filter_t[0]
                )  # (filter_length, outs, 1)
                prng_state, _ = jr.split(prng_state)

            tunings[model_name] = results

        elif obs_type == "nonparam_pp_gp":
            ISI_order = int(config.likelihood[3:])
            pos_isi_locs = mean_ISIs[:, None] * np.ones(
                (eval_pts, neurons, ISI_order - 1)
            )

            ### tuning ###
            print("Position tuning...")

            pos_mean_ISI, pos_mean_invISI, pos_CV_ISI = utils.compute_ISI_stats(
                prng_state,
                num_samps,
                pos_x_locs,
                pos_isi_locs,
                model.obs_model,
                jitter,
                neuron_list,
                int_eval_pts=int_eval_pts,
                num_quad_pts=num_quad_pts,
                batch_size=batch_size,
            )
            prng_state, _ = jr.split(prng_state)

            # reshape
            pos_mean_ISI = pos_mean_ISI.reshape(num_samps, -1, *or_shape)
            pos_mean_invISI = pos_mean_invISI.reshape(num_samps, -1, *or_shape)
            pos_CV_ISI = pos_CV_ISI.reshape(num_samps, -1, *or_shape)

            ### conditional ISI distribution ###
            print("Conditional ISI densities...")

            isi_conds = [mean_ISIs[:, None] * np.ones((neurons, ISI_order - 1))] * pts
            ISI_densities = []
            for pn in range(pts):
                ISI_density = utils.compute_ISI_densities(
                    prng_state,
                    num_samps,
                    cisi_t_eval,
                    isi_conds[pn][None],
                    x_conds[pn][None],
                    model.obs_model,
                    jitter,
                    outdims_list=[neuron_conds[pn]],
                    int_eval_pts=int_eval_pts,
                    num_quad_pts=num_quad_pts,
                    outdims_per_batch=outdims_per_batch,
                )  # (1, mc, 1, ts)
                ISI_densities.append(ISI_density[0])

            ISI_densities = np.concatenate(ISI_densities, axis=1)  # (mc, pts, ts))

            # ISI kernel ARD
            warp_tau = np.exp(model.obs_model.log_warp_tau)
            len_tau, len_deltas, len_xs = utils.extract_lengthscales(
                model.obs_model.gp.kernel.kernels, ISI_order
            )

            # export
            results = {
                "pos_isi_locs": pos_isi_locs,
                "pos_mean_ISI": pos_mean_ISI,
                "pos_mean_invISI": pos_mean_invISI,
                "pos_CV_ISI": pos_CV_ISI,
                "ISI_densities": ISI_densities,
                "ISI_deltas_conds": isi_conds,
                "warp_tau": warp_tau,
                "len_tau": len_tau,
                "len_deltas": len_deltas,
                "len_xs": len_xs,
            }
            tunings[model_name] = results

    # reshape
    pos_x_locs = pos_x_locs.reshape(*or_shape, *pos_x_locs.shape[1:])
    pos_isi_locs = pos_isi_locs.reshape(*or_shape, *pos_isi_locs.shape[1:])

    tuning_dict = {
        "pos_x_locs": pos_x_locs,
        "ISI_xs_conds": x_conds,
        "ISI_neuron_conds": neuron_conds,
        "ISI_t_eval": cisi_t_eval,
        "unit_ISI_t_eval": uisi_t_eval,
        **tunings,
    }
    return tuning_dict


def main():
    ### parser ###
    parser = argparse.ArgumentParser(
        usage="%(prog)s [options]",
        description="Analysis of synthetic regression models.",
    )

    parser.add_argument("--seed", default=123, type=int)
    parser.add_argument("--data_seed", default=123, type=int)
    parser.add_argument("--savedir", default="../saves/", type=str)
    parser.add_argument("--datadir", default="../../data/synthetic/", type=str)
    parser.add_argument("--checkpointdir", default="../checkpoint/", type=str)

    parser.add_argument("--batch_size", default=50000, type=int)

    parser.add_argument("--device", default=0, type=int)
    parser.add_argument("--cpu", dest="cpu", action="store_true")
    parser.set_defaults(cpu=False)

    args = parser.parse_args()

    ### setup ###
    if args.cpu:
        jax.config.update("jax_platform_name", "cpu")
    else:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.device}"

    save_dir = args.savedir
    data_path = args.datadir
    checkpoint_dir = args.checkpointdir

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    rng = np.random.default_rng(args.seed)
    prng_state = jr.PRNGKey(args.seed)
    batch_size = args.batch_size

    data_seed = args.data_seed

    ### names ###
    ISI_order = 4
    reg_config_names = [
        "syn_data_seed123ISI4sel0.0to1.0_PP-log__factorized_gp-16-1000_X[x-y]_Z[]_freeze[]",
        "syn_data_seed123ISI4sel0.0to1.0_gamma-log__rate_renewal_gp-16-1000_X[x-y]_Z[]_freeze[]",
        "syn_data_seed123ISI4sel0.0to1.0_PP-log_rcb-8-10.-20.-4.5-9.-self-H150_factorized_gp-16-1000_"
        + "X[x-y]_Z[]_freeze[obs_model0spikefilter0a-obs_model0spikefilter0log_c-obs_model0spikefilter0phi]",
        "syn_data_seed123ISI4sel0.0to1.0_isi4__nonparam_pp_gp-48-matern32-matern32-1000-n2._"
        + "X[x-y]_Z[]_freeze[]",
        "syn_data_seed123ISI4sel0.0to1.0_isi4__nonparam_pp_gp-48-matern32-matern32-1000-n2._"
        + "X[x-y]_Z[]_freeze[obs_model0log_warp_tau]",
    ]

    ### load dataset ###
    session_name = "syn_data_seed{}".format(data_seed)
    max_ISI_order = 4

    select_fracs = [0.0, 1.0]
    dataset_dict = synthetic.spikes_dataset(
        session_name, data_path, max_ISI_order, select_fracs
    )
    neurons = dataset_dict["properties"]["neurons"]

    ### analysis ###
    regression_dict, tuning_dict = {}, {}
    tuning_neuron_list = list(range(neurons))

    process_steps = [0, 1]
    for k in process_steps:  # save after finishing each dict
        if k == 0:
            regression_dict = utils.evaluate_regression_fits(
                checkpoint_dir,
                reg_config_names,
                synthetic.observed_kernel_dict_induc_list,
                dataset_dict,
                [],
                rng,
                prng_state,
                batch_size,
                0,
            )

        elif k == 1:
            tuning_dict = tuning(
                checkpoint_dir,
                reg_config_names,
                dataset_dict,
                rng,
                prng_state,
                tuning_neuron_list,
                num_samps=30,
                int_eval_pts=1000,
                num_quad_pts=100,
                batch_size=100,
                outdims_per_batch=2,
            )

        ### export ###
        data_run = {
            "regression": regression_dict,
            "tuning": tuning_dict,
        }
        pickle.dump(data_run, open(save_dir + "synthetic_results.p", "wb"))


if __name__ == "__main__":
    main()
