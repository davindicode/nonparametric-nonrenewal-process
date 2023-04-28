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



def regression(
    checkpoint_dir, reg_config_names, dataset_dict, test_dataset_dicts, rng, prng_state, batch_size
):
    tbin = dataset_dict["properties"]["tbin"]
    neurons = dataset_dict["properties"]["neurons"]
    
    lik_int_method = {
        "type": "GH", 
        "approx_pts": 50, 
    }

    num_samps = 1
    pred_start, pred_end = 0, 10000
    pred_ts = np.arange(pred_start, pred_end) * tbin
    regression_dict = {}
    for model_name in reg_config_names:
        print('Analyzing regression for {}...'.format(model_name))
        
        # config
        model, config = utils.load_model_and_config(
            checkpoint_dir + model_name, 
            dataset_dict, 
            th1.observed_kernel_dict_induc_list, 
            rng, 
        )
        obs_type = config.observations.split('-')[0]
        jitter = config.jitter

        # data
        timestamps, covs_t, ISIs, observations, filter_length = template.select_inputs(
            dataset_dict, config)
        
        ys = observations[:, filter_length:]
        ys_filt = observations[:, :-1]
        data = (timestamps, covs_t, ISIs, ys, ys_filt, filter_length)
        
        # train likelihoods and rate/time rescaling
        train_ell = utils.likelihood_metric(
            prng_state, data, model.obs_model, obs_type, lik_int_method, jitter, log_predictive=False)
        prng_state, _ = jr.split(prng_state)
        
        train_lpd = None
#         train_lpd = utils.likelihood_metric(
#             prng_state, data, model.obs_model, obs_type, lik_int_method, jitter, log_predictive=True)
#         prng_state, _ = jr.split(prng_state)
        
        sort_cdfs, T_KSs, sign_KSs, p_KSs = utils.time_rescaling_statistics(
            data, model.obs_model, obs_type, jitter, list(range(neurons)))

        # test data
        test_lpds, test_ells = [], []
        pred_log_intensities, pred_spiketimes = [], []
        sample_log_rhos, sample_spiketimes = [], []
        for test_dataset_dict in test_dataset_dicts:  # test
            timestamps, covs_t, ISIs, observations, filter_length = template.select_inputs(
                test_dataset_dict, config)
            
            test_ys = observations[:, filter_length:]
            test_ys_filt = observations[:, :-1]
            data = (timestamps, covs_t, ISIs, test_ys, test_ys_filt, filter_length)
            
            test_ell = utils.likelihood_metric(
                prng_state, data, model.obs_model, obs_type, lik_int_method, jitter, log_predictive=False)
            prng_state, _ = jr.split(prng_state)
            
            test_lpd = None
#             test_lpd = utils.likelihood_metric(
#                 prng_state, data, model.obs_model, obs_type, lik_int_method, jitter, log_predictive=True)
#             prng_state, _ = jr.split(prng_state)
            
            test_ells.append(test_ell)
            test_lpds.append(test_lpd)
            
            pred_ys, pred_log_intens = utils.conditional_intensity(
                prng_state, 
                data, 
                model.obs_model, 
                obs_type, 
                pred_start, 
                pred_end, 
                jitter, 
                list(range(neurons)), 
                sampling=False, 
            )
            pred_spkts = np.where(pred_ys > 0)[0] + pred_start * tbin
            
            pred_log_intensities.append(pred_log_intens)
            pred_spiketimes.append(pred_spkts)
            
            sample_ys, log_rho_ts = utils.sample_activity(
                prng_state, 
                num_samps, 
                data, 
                model.obs_model, 
                obs_type, 
                pred_start, 
                pred_end, 
                jitter, 
            )
            sample_spkts = [
                np.where(sample_ys[tr] > 0)[0] + pred_start * tbin 
                for tr in range(num_samps)
            ]
            
            sample_spiketimes.append(sample_spkts)
            sample_log_rhos.append(log_rho_ts)

        # export
        results = {
            "train_ell": train_ell, 
            "train_lpd": train_lpd, 
            "test_ells": test_ells,
            "test_lpds": test_lpds,
            "pred_ts": pred_ts, 
            "pred_log_intensities": pred_log_intensities, 
            "pred_spiketimes": pred_spiketimes, 
            "sample_log_rhos": sample_log_rhos, 
            "sample_spiketimes": sample_spiketimes,
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
        th1.observed_kernel_dict_induc_list, 
        rng, 
    )
    obs_type = config.observations.split('-')[0]
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
        int_eval_pts = 1000, 
        num_quad_pts = 100, 
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
            int_eval_pts=1000,
            num_quad_pts=100,
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
    

    
def variability(
    checkpoint_dir, model_name, dataset_dict, rng, prng_state, batch_size
):
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
    timestamps, covs_t, ISIs, observations, filter_length = template.select_inputs(
        dataset_dict, config)
    
    ### evaluation ###
    num_samps = 30
    dilation = 10
    
    x_locs = covs_t[::dilation, :]
    isi_locs = ISIs[:, ::dilation, :]
    eval_pts = x_locs.shape[0]
    
    isi_locs = np.ones((eval_pts, neurons, ISI_order-1))    
    mean_ISI, CV_ISI = utils.compute_ISI_stats(
        prng_state, 
        num_samps, 
        x_locs, 
        isi_locs, 
        model.obs_model, 
        jitter, 
        list(range(neurons)), 
        int_eval_pts = 1000, 
        num_quad_pts = 100, 
        batch_size = 100, 
    )  # (eval_pts, out_dims)
    prng_state, _ = jr.split(prng_state)
    
    ### stats ###
    X, Y = 1 / mean_ISI.T, CV_ISI.T
    a, b = utils.linear_regression(X, Y)  # (out_dims, pts)
    Y_fit = a[:, None] * X + b[:, None]
    R2_lin = 1 - (Y - Y_fit).var(-1) / Y.var(-1)  # R^2 of fit (out_dims,)
    
    # export
    variability_dict = {
        'mean_ISI': mean_ISI, 
        'CV_ISI': CV_ISI, 
        'linear_slope': a, 
        'linear_intercept': b, 
        'linear_R2': R2_lin, 
    }
    return variability_dict
    
    
    
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
        'Mouse28_140313_wakeISI5sel0.0to0.5_PP-log__factorized_gp-8-1000_X[hd]_Z[]', 
        'Mouse28_140313_wakeISI5sel0.0to0.5_PP-log_rcb-8-3.-1.-1.-self-H500_factorized_gp-8-1000_X[hd]_Z[]', 
        'Mouse28_140313_wakeISI5sel0.0to0.5_gamma-log__rate_renewal_gp-8-1000_X[hd]_Z[]', 
        'Mouse28_140313_wakeISI5sel0.0to0.5_lognorm-log__rate_renewal_gp-8-1000_X[hd]_Z[]', 
        'Mouse28_140313_wakeISI5sel0.0to0.5_invgauss-log__rate_renewal_gp-8-1000_X[hd]_Z[]', 
        'Mouse28_140313_wakeISI5sel0.0to0.5_isi4__nonparam_pp_gp-40-matern32-1000-12._X[hd]_Z[]', 
    ]

    tuning_model_name = reg_config_names[-1]

    ### load dataset ###
    session_name = 'Mouse28_140313_wake'
    max_ISI_order = 4

    select_fracs = [0.0, 0.5]
    dataset_dict = th1.spikes_dataset(session_name, data_path, max_ISI_order, select_fracs)

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
    regression_dict = regression(
        checkpoint_dir, reg_config_names, dataset_dict, test_dataset_dicts, rng, prng_state, batch_size
    )

    ### export ###
    data_run = {
        "regression": regression_dict,
    }

    pickle.dump(data_run, open(save_dir + "th1_results.p", "wb"))


if __name__ == "__main__":
    main()
