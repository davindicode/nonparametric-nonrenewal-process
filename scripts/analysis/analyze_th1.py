import argparse

import os
import sys

sys.path.append("../fit/")
import template
import th1

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
    num_samps, 
    int_eval_pts, 
    num_quad_pts, 
    batch_size, 
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
    
    # data
    hd = dataset_dict['covariates']['hd']
    x = dataset_dict['covariates']['x']
    y = dataset_dict['covariates']['y']
    speed = dataset_dict['covariates']['speed']
    ISIs = dataset_dict['ISIs']
    
    mean_ISIs = utils.mean_ISI(ISIs)
    
    ### tuning ###
    
    # head direction
    print('Head direction tuning...')
    
    pts = 100
    hd_x_locs = np.stack([
        np.linspace(0, 2*np.pi, pts)
    ], axis=1)  # (evals, x_dim)
    hd_isi_locs = mean_ISIs[:, None] * np.ones((pts, neurons, ISI_order-1))
    
    hd_mean_ISI, hd_mean_invISI, hd_CV_ISI = utils.compute_ISI_stats(
        prng_state, 
        num_samps, 
        hd_x_locs, 
        hd_isi_locs, 
        model.obs_model, 
        jitter, 
        neuron_list, 
        int_eval_pts = int_eval_pts, 
        num_quad_pts = num_quad_pts, 
        batch_size = batch_size, 
    )  # (mc, eval_pts, out_dims)
    prng_state, _ = jr.split(prng_state)
    
    ### conditional ISI densities ###
    print('Conditional ISI densities...')
    
    evalsteps = 100
    cisi_t_eval = np.linspace(0.0, 5., evalsteps)
    
    pts = 8
    isi_conds = mean_ISIs[:, None] * np.ones((pts, neurons, ISI_order-1))
    x_conds = np.stack([
        np.linspace(0, 2*np.pi, pts), 
    ], axis=-1)
    
    ISI_densities = utils.compute_ISI_densities(
        prng_state, 
        num_samps, 
        cisi_t_eval, 
        isi_conds, 
        x_conds, 
        model.obs_model, 
        jitter, 
        outdims_list = neuron_list, 
        int_eval_pts = int_eval_pts, 
        num_quad_pts = num_quad_pts, 
        outdims_per_batch = 2, 
    )  # (eval_pts, mc, out_dims, ts)
    
    ### ISI kernel ARD ###
    warp_tau = np.exp(model.obs_model.log_warp_tau)
    len_tau, len_deltas, len_xs = utils.extract_lengthscales(
        model.obs_model.gp.kernel.kernels, ISI_order)
    
    # export
    tuning_dict = {
        'neuron_list': neuron_list, 
        'hd_x_locs': hd_x_locs, 
        'hd_isi_locs': hd_isi_locs, 
        'hd_mean_ISI': hd_mean_ISI, 
        'hd_mean_invISI': hd_mean_invISI, 
        'hd_CV_ISI': hd_CV_ISI, 
        'ISI_t_eval': cisi_t_eval, 
        'ISI_deltas_conds': isi_conds, 
        'ISI_xs_conds': x_conds, 
        'ISI_densities': ISI_densities, 
        'warp_tau': warp_tau, 
        'len_tau': len_tau, 
        'len_deltas': len_deltas, 
        'len_xs': len_xs, 
    }
    return tuning_dict

    
    
def main():
    ### parser ###
    parser = argparse.ArgumentParser(
        usage="%(prog)s [options]",
        description="Analysis of th1 regression models.",
    )

    parser.add_argument("--seed", default=123, type=int)
    parser.add_argument("--savedir", default="../saves/", type=str)
    parser.add_argument("--datadir", default="../../data/th1/", type=str)
    parser.add_argument("--checkpointdir", default="../checkpoint/", type=str)

    parser.add_argument("--batch_size", default=100000, type=int)
    
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
        # exponential and renewal
        'Mouse28_140313_wakeISI5sel0.0to0.5_PP-log__factorized_gp-8-1000_X[hd]_Z[]', 
        'Mouse28_140313_wakeISI5sel0.0to0.5_gamma-log__rate_renewal_gp-8-1000_X[hd]_Z[]', 
        'Mouse28_140313_wakeISI5sel0.0to0.5_lognorm-log__rate_renewal_gp-8-1000_X[hd]_Z[]', 
        'Mouse28_140313_wakeISI5sel0.0to0.5_invgauss-log__rate_renewal_gp-8-1000_X[hd]_Z[]', 
        # conditional
        'Mouse28_140313_wake_isi5ISI5sel0.0to0.5_PP-log_rcb-16-17.-36.-6.-30.-self-H500_' + \
        'factorized_gp-8-1000_X[hd]_Z[]_freeze[obs_model0spikefilter0a-' + \
        'obs_model0spikefilter0log_c-obs_model0spikefilter0phi]', 
        'Mouse28_140313_wake_isi5ISI5sel0.0to0.5_lognorm-log_rcb-16-17.-36.-6.-30.-self-H500_' + \
        'rate_renewal_gp-8-1000_X[hd]_Z[]_freeze[]',
        # BNPP
        'Mouse28_140313_wake_isi5ISI5sel0.0to0.5_isi4__nonparam_pp_gp-40-matern12-matern32-1000-n2._' + \
        'X[hd]_Z[]_freeze[]', 
        'Mouse28_140313_wake_isi5ISI5sel0.0to0.5_isi4__nonparam_pp_gp-40-matern12-matern32-1000-n2._' + \
        'X[hd]_Z[]_freeze[obs_model0log_warp_tau]', 
        'Mouse28_140313_wake_isi5ISI5sel0.0to0.5_isi4__nonparam_pp_gp-40-matern32-matern32-1000-n2._' + \
        'X[hd]_Z[]_freeze[]', 
        'Mouse28_140313_wake_isi5ISI5sel0.0to0.5_isi4__nonparam_pp_gp-40-matern32-matern32-1000-n2._' + \
        'X[hd]_Z[]_freeze[obs_model0log_warp_tau]', 
    ]

    tuning_model_name = reg_config_names[-1]

    ### load dataset ###
    session_name = 'Mouse28_140313_wake_isi5'
    max_ISI_order = 4

    select_fracs = [0.0, 0.5]
    dataset_dict = th1.spikes_dataset(session_name, data_path, max_ISI_order, select_fracs)
    neurons = dataset_dict["properties"]["neurons"]
    
    test_select_fracs = [
        [0.5, 0.6], 
        [0.6, 0.7], 
        [0.7, 0.8], 
        [0.8, 0.9], 
        [0.9, 1.0], 
    ]
    test_dataset_dicts = [
        th1.spikes_dataset(session_name, data_path, max_ISI_order, tf) for tf in test_select_fracs
    ]

    ### analysis ###
    regression_dict, variability_dict, tuning_dict = {}, {}, {}
    tuning_neuron_list = list(range(neurons))
    
    process_steps = 3
    for k in range(process_steps):  # save after finishing each dict
        
        if k == 0:
            regression_dict = utils.evaluate_regression_fits(
                checkpoint_dir, reg_config_names, th1.observed_kernel_dict_induc_list, 
                dataset_dict, test_dataset_dicts, rng, prng_state, batch_size
            )
        
        elif k == 1:
            variability_dict = utils.analyze_variability_stats(
                checkpoint_dir, tuning_model_name, th1.observed_kernel_dict_induc_list, 
                dataset_dict, rng, prng_state, 
                num_samps = 30, 
                dilation = 100, 
                int_eval_pts = 1000, 
                num_quad_pts = 100, 
                batch_size = 100, 
                num_induc = 8, 
                jitter = 1e-6, 
            )

        elif k == 2:
            tuning_dict = tuning(
                checkpoint_dir, 
                tuning_model_name, 
                dataset_dict, 
                rng, 
                prng_state, 
                tuning_neuron_list, 
                num_samps = 30, 
                int_eval_pts = 1000, 
                num_quad_pts = 100, 
                batch_size = 1000, 
                outdims_per_batch = 2, 
            )

        ### export ###
        data_run = {
            "regression": regression_dict,
            "variability": variability_dict, 
            "tuning": tuning_dict, 
        }
        pickle.dump(data_run, open(save_dir + "results_th1.p", "wb"))


if __name__ == "__main__":
    main()
