import argparse

import os
import sys

sys.path.append("../fit/")
import template
import synthetic

sys.path.append("../../../GaussNeuro")
import gaussneuro as lib

import utils

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import jit, vmap
import numpy as np

import pickle



def regression(
    checkpoint_dir, reg_config_names, dataset_dict, rng, prng_state, batch_size
):
    tbin = dataset_dict["properties"]["tbin"]
    neurons = dataset_dict["properties"]["neurons"]
    
    lik_int_method = {
        "type": "GH", 
        "approx_pts": 50, 
    }

    regression_dict = {}
    for model_name in reg_config_names:
        print('Analyzing regression for {}...'.format(model_name))

        # config
        model, config = utils.load_model_and_config(
            checkpoint_dir + model_name, 
            dataset_dict, 
            synthetic.observed_kernel_dict_induc_list, 
            rng, 
        )
        obs_type = config.observations.split('-')[0]
        jitter = config.jitter

        # data
        timestamps, covs_t, ISIs, observations, filter_length = template.select_inputs(
            dataset_dict, config)
        
        ys = observations[:, filter_length:]
        ys_filt = observations[:, :]
        data = (timestamps, covs_t, ISIs, ys, ys_filt, filter_length)
        
        # train likelihoods and time rescaling
        train_ell = utils.likelihood_metric(
            prng_state, data, model.obs_model, obs_type, lik_int_method, jitter, log_predictive=False)
        prng_state, _ = jr.split(prng_state)
        
        sort_cdfs, T_KSs, sign_KSs, p_KSs = utils.time_rescaling_statistics(
            data, model.obs_model, obs_type, jitter, list(range(neurons)))

        # export
        results = {
            "train_ells": np.array(train_ell), 
            "KS_quantiles": sort_cdfs, 
            "KS_statistic": T_KSs,
            "KS_significance": sign_KSs,
            "KS_p_value": p_KSs,
        }
        regression_dict[model_name] = results

    return regression_dict


def tuning(
    checkpoint_dir, model_name, dataset_dict, rng, prng_state, batch_size
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
    
    ### tuning ###
    num_samps = 30
    
    pos_x_locs = np.meshgrid(
        np.linspace(x.min(), x.max(), 4), 
        np.linspace(y.min(), y.max(), 4), 
    )
    pos_x_locs = np.stack(pos_x_locs, axis=-1)
    or_shape = pos_x_locs.shape[:-1]
    pos_x_locs = pos_x_locs.reshape(-1, covs_dims)  # (evals, x_dim)
    eval_pts = pos_x_locs.shape[0]
    
    pos_isi_locs = np.ones((eval_pts, neurons, ISI_order-1))    
    pos_mean_ISI, pos_CV_ISI = utils.compute_ISI_stats(
        prng_state, 
        num_samps, 
        pos_x_locs, 
        pos_isi_locs, 
        model.obs_model, 
        jitter, 
        list(range(neurons)), 
        int_eval_pts = 100, 
        num_quad_pts = 10, 
        batch_size = 100, 
    )
    prng_state, _ = jr.split(prng_state)
    
    pos_x_locs = pos_x_locs.reshape(*or_shape, -1)
    pos_isi_locs = pos_isi_locs.reshape(*or_shape, -1)
    pos_mean_ISI = pos_mean_ISI.reshape(-1, *or_shape)
    pos_CV_ISI = pos_CV_ISI.reshape(-1, *or_shape)
    
    ### conditional ISI distribution ###
    evalsteps = 120
    cisi_t_eval = jnp.linspace(0.0, 5., evalsteps)
    
    pts = 10
    isi_conds = np.ones((pts, neurons, ISI_order-1))
    x_conds = 1.*np.ones((pts, covs_dims))
    
    ISI_densities = []
    for pn in range(pts):
        
        ISI_density = model.obs_model.sample_conditional_ISI(
            prng_state,
            num_samps,
            cisi_t_eval,
            jnp.array(isi_conds[pn]), 
            jnp.array(x_conds[pn]),
            sel_outdims=None, 
            int_eval_pts=100,
            num_quad_pts=10,
            prior=False,
            jitter=jitter, 
        )
        prng_state, _ = jr.split(prng_state)
        
        ISI_densities.append(np.array(ISI_density))
        
    cisi_t_eval = np.array(cisi_t_eval)
    ISI_densities = np.stack(ISI_densities, axis=0)
    
    # ISI kernel ARD
    warp_tau = np.exp(model.obs_model.log_warp_tau)
    len_tau = np.array(model.obs_model.gp.kernel.kernels[0].lengthscale[:, 0])
    len_deltas = np.array(model.obs_model.gp.kernel.kernels[1].kernels[0].lengthscale[:, :ISI_order-1])
    len_xs = np.array(model.obs_model.gp.kernel.kernels[1].kernels[0].lengthscale[:, -covs_dims:])
    
    # export
    tuning_dict = {
        'pos_x_locs': pos_x_locs, 
        'pos_isi_locs': pos_isi_locs, 
        'pos_mean_ISI': pos_mean_ISI, 
        'pos_CV_ISI': pos_CV_ISI, 
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
        description="Analysis of synthetic regression models.",
    )

    parser.add_argument("--seed", default=123, type=int)
    parser.add_argument("--data_seed", default=123, type=int)
    parser.add_argument("--savedir", default="../saves/", type=str)
    parser.add_argument("--datadir", default="../../data/synthetic/", type=str)
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
    
    data_seed = args.data_seed

    ### names ###
    ISI_order = 4
    reg_config_names = [
        'syn_data_seed123ISI4sel0.0to1.0_PP-log__factorized_gp-16-1000_X[x-y]_Z[]', 
        'syn_data_seed123ISI4sel0.0to1.0_isi4__nonparam_pp_gp-16-matern32-1000-1._X[x-y]_Z[]'.format(ISI_order), 
    ]
    
    tuning_model_name = reg_config_names[-1]

    ### load dataset ###
    session_name = 'syn_data_seed{}'.format(data_seed)
    max_ISI_order = 4

    select_fracs = [0.0, 1.0]
    dataset_dict = synthetic.spikes_dataset(session_name, data_path, max_ISI_order, select_fracs)

    ### analysis ###
    regression_dict = regression(
        checkpoint_dir, reg_config_names, dataset_dict, rng, prng_state, batch_size
    )

    ### export ###
    data_run = {
        "regression": regression_dict,
    }

    pickle.dump(data_run, open(save_dir + "synthetic_results.p", "wb"))


if __name__ == "__main__":
    main()
