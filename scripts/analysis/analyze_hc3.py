import argparse

import os
import sys

sys.path.append("../fit/")
import template
import hc3

sys.path.append("../../../GaussNeuro")
import gaussneuro as lib

import utils

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

import pickle

    
    
def tuning(
    checkpoint_dir, 
    model_name, 
    dataset_dict, 
    rng, 
    prng_state, 
    neuron_list, 
    num_samps = 30, 
    int_eval_pts = 1000, 
    num_quad_pts = 100, 
    batch_size = 1000, 
):
    """
    Compute tuning curves of BNPP model
    """
    
    ### evaluation ###
    locs_eval = jnp.linspace(0, 2*np.pi, 30)[:, None]

    n = 0
    sel_outdims = jnp.array([n])
    isi_cond = jnp.ones((len(sel_outdims), ISI_orders-1))

    def m(x_cond, n):
        mean_ISI = model.obs_model.sample_conditional_ISI_expectation(
            prng_state,
            num_samps,
            lambda x: x,
            isi_cond, 
            x_cond,
            sel_outdims, 
            int_eval_pts=1000,
            f_num_quad_pts=100,
            isi_num_quad_pts=100, 
            prior=False,
            jitter=jitter, 
        )

        secmom_ISI = model.obs_model.sample_conditional_ISI_expectation(
            prng_state,
            num_samps,
            lambda x: x**2,
            isi_cond, 
            x_cond,
            sel_outdims, 
            int_eval_pts=1000,
            f_num_quad_pts=100,
            isi_num_quad_pts=100, 
            prior=False,
            jitter=jitter, 
        )

        var_ISI = (secmom_ISI - mean_ISI**2)
        CV_ISI = jnp.sqrt(var_ISI) / (mean_ISI + 1e-12)

    mean_ISI, var_ISI, CV_ISI = jax.vmap(m, (0, None))(locs_eval, sel_outdims)

    ### conditional ISI distribution ###
    evalsteps = 120
    covs_dims = covs_t.shape[-1]

    cisi_t_eval = jnp.linspace(0.0, 5., evalsteps)
    isi_cond = jnp.ones((neurons, ISI_orders-1))

    x_cond = 1.*jnp.ones(covs_dims)
    ISI_density = model.obs_model.sample_conditional_ISI(
        prng_state,
        num_samps,
        cisi_t_eval,
        isi_cond, 
        x_cond,
        sel_outdims=None, 
        int_eval_pts=1000,
        num_quad_pts=100,
        prior=False,
        jitter=jitter, 
    )

    
    tuning_dict = {
        'mean_ISI': mean_ISI, 
        'CV_ISI': CV_ISI, 
    }
    return tuning_dict
    
    
    
    
def main():
    ### parser ###
    parser = argparse.ArgumentParser(
        usage="%(prog)s [options]",
        description="Analysis of hc3 regression models.",
    )

    parser.add_argument("--seed", default=123, type=int)
    parser.add_argument("--savedir", default="../saves/", type=str)
    parser.add_argument("--datadir", default="../../data/hc3/", type=str)
    parser.add_argument("--checkpointdir", default="../checkpoint/", type=str)

    parser.add_argument("--batch_size", default=10000, type=int)
    
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

    
    ### names ###
    reg_config_names = [
        'ec014.29_ec014.468_L-to-RISI5sel0.0to0.5_PP-log__factorized_gp-32-1000_X[x-theta]_Z[]', 
        'ec014.29_ec014.468_L-to-RISI5sel0.0to0.5_PP-log_rcb-8-3.-1.-1.-self-H500_factorized_gp-32-1000_X[x-theta]_Z[]', 
        'ec014.29_ec014.468_L-to-RISI5sel0.0to0.5_gamma-log__rate_renewal_gp-32-1000_X[x-theta]_Z[]', 
        'ec014.29_ec014.468_L-to-RISI5sel0.0to0.5_lognorm-log__rate_renewal_gp-32-1000_X[x-theta]_Z[]', 
        'ec014.29_ec014.468_L-to-RISI5sel0.0to0.5_invgauss-log__rate_renewal_gp-32-1000_X[x-theta]_Z[]', 
        'ec014.29_ec014.468_L-to-RISI5sel0.0to0.5_isi4__nonparam_pp_gp-64-matern32-1000-12._X[x-theta]_Z[]', 
    ]

    tuning_model_name = (
        "th1_U-el-3_svgp-64_X[hd-omega-speed-x-y-time]_Z[]_40K11_0d0_10f-1"
    )

    ### load dataset ###
    session_name = 'ec014.29_ec014.468_L-to-R'
    max_ISI_order = 4

    select_fracs = [0.0, 0.5]
    dataset_dict = hc3.spikes_dataset(session_name, data_path, max_ISI_order, select_fracs)

    test_select_fracs = [
        [0.5, 0.6], 
        [0.6, 0.7], 
        [0.7, 0.8], 
        [0.8, 0.9], 
        [0.9, 1.0], 
    ]
    test_dataset_dicts = [
        hc3.spikes_dataset(session_name, data_path, max_ISI_order, tf) for tf in test_select_fracs
    ]

    ### analysis ###
    regression_dict = regression(
        checkpoint_dir, reg_config_names, dataset_dict, test_dataset_dicts, rng, prng_state, batch_size
    )

    ### export ###
    data_run = {
        "regression": regression_dict,
    }

    pickle.dump(data_run, open(save_dir + "hc3_results.p", "wb"))


if __name__ == "__main__":
    main()
