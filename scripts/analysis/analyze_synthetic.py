import argparse

import os
import sys

sys.path.append("../fit/")
import template
import synthetic

sys.path.append("../../../GaussNeuro")
import gaussneuro as lib

sys.path.append("../../data/synthetic")
import generate

import utils

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import jit, vmap
import numpy as np

import pickle

import utils



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
    outdims_per_batch, 
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
        synthetic.observed_kernel_dict_induc_list, 
        rng, 
    )
    obs_type = config.observations.split('-')[0]
    covs_dims = len(config.observed_covs.split('-'))
    jitter = config.jitter
    ISI_order = int(config.likelihood[3:])
    
    # data
    x = dataset_dict['covariates']['x']
    y = dataset_dict['covariates']['y']
    ISIs = dataset_dict['ISIs']
    
    mean_ISIs = utils.mean_ISI(ISIs)
    
    ### tuning ###
    print('Position tuning...')
    
    # ground_truth
    mdl = generate.model('Gaussian')
    ratefunc, _, renewals_ll, _, _ = mdl.get_model()
    
    pos_x_locs = np.meshgrid(
        np.linspace(x.min(), x.max(), 30), 
        np.linspace(y.min(), y.max(), 30), 
    )
    pos_x_locs = np.stack(pos_x_locs, axis=-1)
    or_shape = pos_x_locs.shape[:-1]
    pos_x_locs = pos_x_locs.reshape(-1, covs_dims)  # (evals, x_dim)
    eval_pts = pos_x_locs.shape[0]
    
    pos_isi_locs = mean_ISIs[:, None] * np.ones((eval_pts, neurons, ISI_order-1))    
    pos_mean_ISI, pos_mean_invISI, pos_CV_ISI = utils.compute_ISI_stats(
        prng_state, 
        num_samps, 
        pos_x_locs, 
        pos_isi_locs, 
        model.obs_model, 
        jitter, 
        neuron_list, 
        int_eval_pts = int_eval_pts, 
        num_quad_pts = num_quad_pts, 
        batch_size = batch_size, 
    )
    prng_state, _ = jr.split(prng_state)
    
    # ground truth rate maps
    GT_rates = vmap(ratefunc)(jnp.array(pos_x_locs))
    
    # reshape
    pos_x_locs = pos_x_locs.reshape(*or_shape, *pos_x_locs.shape[1:])
    pos_isi_locs = pos_isi_locs.reshape(*or_shape, *pos_isi_locs.shape[1:])
    pos_mean_ISI = pos_mean_ISI.reshape(num_samps, -1, *or_shape)
    pos_mean_invISI = pos_mean_invISI.reshape(num_samps, -1, *or_shape)
    pos_CV_ISI = pos_CV_ISI.reshape(num_samps, -1, *or_shape)
    GT_rates = np.array(GT_rates.reshape(*or_shape, -1))
    
    ### conditional ISI distribution ###
    print('Conditional ISI densities...')
    
    evalsteps = 120
    cisi_t_eval = jnp.linspace(0.0, 5., evalsteps)
    
    pts = 10
    isi_conds = mean_ISIs[:, None] * np.ones((pts, neurons, ISI_order-1))
    x_conds = 1.*np.ones((pts, covs_dims))
    
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
        outdims_per_batch = outdims_per_batch, 
    )  # (eval_pts, mc, out_dims, ts)
    
    # ground truth conditional ISIs
    GT_rate_conds = vmap(ratefunc)(x_conds)
    GT_ISI_densities, GT_unit_renewals = [], []
    for pn in range(pts):
        rate_cond  = GT_rate_conds[pn]
        gt_isi = []
        for ne, rll in enumerate(renewals_ll):
            gt_isi.append(rate_cond[ne] * np.exp(vmap(rll)(rate_cond[ne] * cisi_t_eval[:, None])))
            
        gt_isi = np.concatenate(gt_isi, axis=1)  # (pts, obs_dims)
        GT_ISI_densities.append(gt_isi)
        
    for rll in renewals_ll:
        GT_unit_renewals.append(np.exp(vmap(rll)(cisi_t_eval[:, None])))
    GT_unit_renewals = np.concatenate(GT_unit_renewals, axis=1)  # (pts, obs_dims)
        
    # arrays
    cisi_t_eval = np.array(cisi_t_eval)
    GT_ISI_densities = np.stack(GT_ISI_densities, axis=0)
    
    # ISI kernel ARD
    warp_tau = np.exp(model.obs_model.log_warp_tau)
    len_tau, len_deltas, len_xs = utils.extract_lengthscales(
        model.obs_model.gp.kernel.kernels, ISI_order)
            
            
    # export
    tuning_dict = {
        'pos_x_locs': pos_x_locs, 
        'pos_isi_locs': pos_isi_locs, 
        'pos_mean_ISI': pos_mean_ISI, 
        'pos_mean_invISI': pos_mean_invISI, 
        'pos_CV_ISI': pos_CV_ISI, 
        'GT_rates': GT_rates, 
        'ISI_t_eval': cisi_t_eval, 
        'ISI_deltas_conds': isi_conds, 
        'ISI_xs_conds': x_conds, 
        'ISI_densities': ISI_densities, 
        'GT_ISI_densities': GT_ISI_densities, 
        'GT_unit_renewals': GT_unit_renewals, 
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
        description="Analysis of synthetic regression models.",
    )

    parser.add_argument("--seed", default=123, type=int)
    parser.add_argument("--data_seed", default=123, type=int)
    parser.add_argument("--savedir", default="../saves/", type=str)
    parser.add_argument("--datadir", default="../../data/synthetic/", type=str)
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
    
    data_seed = args.data_seed

    ### names ###
    ISI_order = 4
    reg_config_names = [
        'syn_data_seed123ISI4sel0.0to1.0_PP-log__factorized_gp-16-1000_X[x-y]_Z[]', 
        'syn_data_seed123ISI4sel0.0to1.0_gamma-log__rate_renewal_gp-16-1000_X[x-y]_Z[]_freeze[]', 
        'syn_data_seed123ISI4sel0.0to1.0_PP-log_rcb-8-17.-36.-6.-30.-self-H500_factorized_gp-16-1000_' + \
        'X[x-y]_Z[]_freeze[obs_model0spikefilter0a-obs_model0spikefilter0log_c-obs_model0spikefilter0phi]', 
        'syn_data_seed123ISI4sel0.0to1.0_isi4__nonparam_pp_gp-48-matern32-matern32-1000-n2._' + \
        'X[x-y]_Z[]_freeze[]', 
        'syn_data_seed123ISI4sel0.0to1.0_isi4__nonparam_pp_gp-48-matern32-matern32-1000-n2._' + \
        'X[x-y]_Z[]_freeze[obs_model0log_warp_tau]', 
    ]

    tuning_model_name = reg_config_names[-1]

    ### load dataset ###
    session_name = 'syn_data_seed{}'.format(data_seed)
    max_ISI_order = 4

    select_fracs = [0.0, 1.0]
    dataset_dict = synthetic.spikes_dataset(session_name, data_path, max_ISI_order, select_fracs)
    neurons = dataset_dict["properties"]["neurons"]
    
    ### analysis ###
    regression_dict, tuning_dict = {}, {}
    tuning_neuron_list = list(range(neurons))
    
    process_steps = 2
    for k in range(process_steps):  # save after finishing each dict
        
        if k == 0:
            regression_dict = utils.evaluate_regression_fits(
                checkpoint_dir, reg_config_names, synthetic.observed_kernel_dict_induc_list, 
                dataset_dict, [], rng, prng_state, batch_size
            )

        elif k == 1:
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
                batch_size = 100, 
                outdims_per_batch = 2, 
            )

        ### export ###
        data_run = {
            "regression": regression_dict,
            "tuning": tuning_dict, 
        }
        pickle.dump(data_run, open(save_dir + "results_synthetic.p", "wb"))


if __name__ == "__main__":
    main()
