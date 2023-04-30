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
    tbin = dataset_dict["properties"]["tbin"]
    neurons = dataset_dict["properties"]["neurons"]

    print('Analyzing tuning for {}...'.format(model_name))

    # config
    model, config = utils.load_model_and_config(
        checkpoint_dir + model_name, 
        dataset_dict, 
        th1.observed_kernel_dict_induc_list, 
        rng, 
    )
    obs_type = config.observations.split('-')[0]
    jitter = config.jitter
    ISI_order = int(config.likelihood[3:])
    
    ### evaluation ###
    locs_eval = jnp.linspace(0, 2*np.pi, 30)[:, None]

    n = 0
    sel_outdims = jnp.array([n])
    isi_cond = jnp.ones((len(sel_outdims), ISI_order-1))

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
    isi_cond = jnp.ones((neurons, ISI_order-1))

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

    
    ### names ###
    reg_config_names = [
        'ec014.29_ec014.468_isi5ISI5sel0.0to0.5_PP-log__factorized_gp-32-1000_X[x-hd-theta]_Z[]_freeze[]', 
        'ec014.29_ec014.468_isi5ISI5sel0.0to0.5_PP-log_rcb-8-17.-36.-6.-30.-self-H500_factorized_gp-32-1000_' + \
        'X[x-hd-theta]_Z[]_freeze[obs_model0spikefilter0a-obs_model0spikefilter0log_c-obs_model0spikefilter0phi]', 
        'ec014.29_ec014.468_isi5ISI5sel0.0to0.5_gamma-log__rate_renewal_gp-32-1000_X[x-hd-theta]_Z[]_freeze[]', 
        'ec014.29_ec014.468_isi5ISI5sel0.0to0.5_invgauss-log__rate_renewal_gp-32-1000_X[x-hd-theta]_Z[]_freeze[]', 
        'ec014.29_ec014.468_isi5ISI5sel0.0to0.5_lognorm-log__rate_renewal_gp-32-1000_X[x-hd-theta]_Z[]_freeze[]', 
        'ec014.29_ec014.468_isi5ISI5sel0.0to0.5_isi4__nonparam_pp_gp-64-matern32-matern32-1000-n2._' + \
        'X[x-hd-theta]_Z[]_freeze[obs_model0log_warp_tau]', 
    ]

    tuning_model_name = reg_config_names[-1]

    ### load dataset ###
    session_name = 'ec014.29_ec014.468_isi5'
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
    regression_dict = utils.evaluate_regression_fits(
        checkpoint_dir, reg_config_names, hc3.observed_kernel_dict_induc_list, 
        dataset_dict, test_dataset_dicts, rng, prng_state, batch_size
    )
    
    variability_dict = utils.analyze_variability_stats(
        checkpoint_dir, tuning_model_name, hc3.observed_kernel_dict_induc_list, 
        dataset_dict, rng, prng_state, 
        num_samps = 3,#30, 
        dilation = 50000, 
        int_eval_pts = 10,#1000, 
        num_quad_pts = 3,#100, 
        batch_size = 100, 
        jitter = 1e-6, 
    )
    
    tuning_dict = tuning(
        checkpoint_dir, tuning_model_name, dataset_dict, rng, prng_state, batch_size
    )

    ### export ###
    data_run = {
        "regression": regression_dict,
        "variability": variability_dict, 
        "tuning": tuning_dict, 
    }

    pickle.dump(data_run, open(save_dir + "hc3_results.p", "wb"))


if __name__ == "__main__":
    main()
