import argparse

import os
import sys

sys.path.append("../fit/")
import template
import th1

sys.path.append("../../../GaussNeuro")
import gaussneuro as lib

import equinox as eqx
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
    #pick_neurons = list(range(neurons))
    sel_outdims = None
    
    lik_int_method = {
        "type": "GH", 
        "approx_pts": 50, 
    }

    regression_dict = {}
    for name in reg_config_names:
        print('Analyzing regression for {}...'.format(name))
        
        # config
        results = pickle.load(
            open(checkpoint_dir + name + ".p", "rb")
        )
        config = results["config"]
        obs_type = config.observations.split('-')[0]
        jitter = 1e-6

        # data
        timestamps, covs_t, ISIs, observations, filter_length = template.select_inputs(
            dataset_dict, config)
        
        xs = covs_t  # (ts, x_dims)
        deltas = ISIs[..., :ISI_orders] if ISIs is not None else ISIs
        ys = observations[:, filter_length:]
        ys_filt = observations[:, :]
        
        # model
        jax.config.update("jax_enable_x64", config.double_arrays)
        obs_covs_dims = covs_t.shape[-1]
        
        model = template.build_model(
            config, dataset_dict, th1.observed_kernel_dict_induc_list, rng, timestamps, obs_covs_dims)
        model = eqx.tree_deserialise_leaves(checkpoint_dir + name + ".eqx", model)
        
        # rate/time rescaling
        max_intervals = int(ys.sum(-1).max())
        post_mean_rho = model.obs_model.evaluate_posterior_mean(
            prng_state, xs, ys_filt, jitter, sel_outdims)
        
        if obs_type == 'factorized_gp' or obs_type == 'nonparam_pp_gp':
            rescaled_intervals = lib.utils.spikes.time_rescale(
                ys.T, post_mean_rho.T, model.obs_model.likelihood.dt, max_intervals=max_intervals)

            pp = lib.likelihoods.Exponential(
                model.obs_model.likelihood.obs_dims, model.obs_model.likelihood.dt)

            qs = np.array(pp.cum_density(rescaled_intervals))[:, 1:]
            
        elif obs_type == 'rate_renewal_gp':
            rescaled_intervals = lib.utils.spikes.time_rescale(
                ys.T, post_mean_rho.T, model.obs_model.renewal.dt, max_intervals=max_intervals)

            qs = np.array(jax.vmap(model.obs_model.renewal.cum_density)(rescaled_intervals.T).T)
            
        else:
            raise ValueError('Invalid observation model type')
            
        sort_cdfs, T_KSs, sign_KSs, p_KSs = [], [], [], []
        for n in range(qs.shape[0]):
            ul = np.where(qs[n, :] != qs[n, :])[0]
            if len(ul) == 0:
                sort_cdf, T_KS, sign_KS, p_KS = None, None, None, None
            else:
                sort_cdf, T_KS, sign_KS, p_KS = lib.utils.stats.KS_test(qs[n, :ul[0]], alpha=0.05)
            sort_cdfs.append(sort_cdf)
            T_KSs.append(T_KS)
            sign_KSs.append(sign_KS)
            p_KSs.append(p_KS)

        # train likelihoods
        timesteps = len(timestamps)
        train_ell, _ = model.obs_model.variational_expectation(
            prng_state, jitter, xs[None, None], ys, ys_filt, False, timesteps, lik_int_method, False)

        # test data
        test_lpds, pred_log_intensities, test_spikes = [], [], []
        for test_dataset_dict in test_dataset_dicts:  # test
            timestamps, covs_t, ISIs, observations, filter_length = template.select_inputs(
                test_dataset_dict, config)
            
            test_xs = covs_t
            test_deltas = ISIs[:, st:, :ISI_orders] if ISIs is not None else ISIs
            test_ys = observations[:, filter_length:]
            test_ys_filt = observations

            # test likelihoods
            test_timesteps = len(timestamps)
            test_lpd, _ = model.obs_model.variational_expectation(
                prng_state, jitter, test_xs[None, None], test_ys, test_ys_filt, 
                False, test_timesteps, lik_int_method, False, 
            )
            test_lpds.append(test_lpd)
            
            # predicted intensities
            if obs_type == 'factorized_gp':
                pred_log_intens, _, _ = model.obs_model.filtered_gp_posterior(
                    prng_state, test_xs, ys_filt=test_ys_filt, 
                    mean_only=True, diag_cov=True, 
                    compute_KL=False, jitter=jitter, sel_outdims=None
                )

            elif obs_type == 'rate_renewal_gp':
                ini_t_tilde = jnp.zeros(neurons)
                pred_log_intens = model.obs_model.sample_log_conditional_intensity(
                    prng_state, ini_t_tilde, test_xs, test_ys, test_ys_filt, jitter, unroll=10
                )

            elif obs_type == 'nonparam_pp_gp':
                pred_log_intens, _ = model.obs_model.evaluate_log_conditional_intensity(
                    test_xs, test_deltas, jitter, sel_outdims)
                
            pred_log_intensities.append(pred_log_intens)
            test_spikes.append(test_ys)

        results = {
            "train_ells": train_ell, 
            "test_lpds": test_lpds,
            "pred_log_intensities": pred_log_intensities, 
            "test_spikes": test_spikes, 
            "T_KS": T_KS,
            "significance_KS": sign_KS,
            "p_KS": p_KS,
        }
        regression_dict[name] = results

    return regression_dict
    
    
    
def tuning(
    checkpoint_dir, reg_config_names, subset_config_names, dataset_dict, prng_state, batch_size
):
    
    quantile_factors = [scipy.stats.norm.ppf(0.025), 0.5, scipy.stats.norm.ppf(0.975)]

    pre_rates_mean, pre_rates_var, _ = model.obs_model.gp_posterior(
        x_eval_tun[:1, ...],
        mean_only=False,
        diag_cov=True,
        compute_KL=False,
        jitter=jitter,
        sel_outdims=sel_outdims,
    )  # (num_samps, out_dims, ts, 1)
    pre_rates_mean, pre_rates_std = pre_rates_mean[0, ..., 0], jnp.sqrt(pre_rates_var[0, ..., 0])

    post_quantiles = [model.obs_model.renewal.inverse_link(
        pre_rates_mean + f*pre_rates_std) for f in quantile_factors]

    
    tuning_dict = {
        
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
        'Mouse28_140313_wakeISI5sel0.0to0.5_isi4__nonparam_pp_gp-32-matern32-1000-12._X[hd]_Z[]', 
    ]

    tuning_model_name = (
        "th1_U-el-3_svgp-64_X[hd-omega-speed-x-y-time]_Z[]_40K11_0d0_10f-1"
    )

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
