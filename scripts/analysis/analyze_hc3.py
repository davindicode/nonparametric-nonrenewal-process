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



def regression(
    checkpoint_dir, reg_config_names, dataset_dict, test_dataset_dicts, rng, prng_state, batch_size
):
    tbin = dataset_dict["properties"]["tbin"]
    neurons = dataset_dict["properties"]["neurons"]
    
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
            config, dataset_dict, hc3.observed_kernel_dict_induc_list, rng, timestamps, obs_covs_dims)
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
        sample_spikes, sample_log_rhos = [], []
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
            
            # predicted intensities and posterior sampling
            num_samps = 10
            if obs_type == 'factorized_gp':
                pred_log_intens, _, _ = model.obs_model.filtered_gp_posterior(
                    prng_state, test_xs[None, None], ys_filt=test_ys_filt, 
                    mean_only=True, diag_cov=True, 
                    compute_KL=False, jitter=jitter, sel_outdims=None
                )
                
                x_eval = test_xs[None, None].repeat(num_samps, axis=0)
                ini_Y = test_ys_filt[None, :, :filter_length].repeat(num_samps, axis=0)
                sample_ys, log_rho_ts = model.obs_model.sample_posterior(
                    prng_state, x_eval, ini_Y=ini_Y, jitter=jitter)

            elif obs_type == 'rate_renewal_gp':
                ini_t_tilde = jnp.zeros(neurons)
                pred_log_intens = model.obs_model.sample_log_conditional_intensity(
                    prng_state, ini_t_tilde, test_xs[None, None], test_ys, test_ys_filt, jitter, unroll=10
                )
                
                ini_t_tilde = jnp.zeros((num_samps, neurons))
                sample_ys, log_rho_ts = model.obs_model.sample_posterior(
                    prng_state, x_eval, ini_spikes=None, ini_t_tilde=ini_t_tilde, jitter=jitter)

            elif obs_type == 'nonparam_pp_gp':
                pred_log_intens, _ = model.obs_model.evaluate_log_conditional_intensity(
                    test_xs[None, None], test_deltas[None], jitter, sel_outdims)
                
                ini_t_tilde = jnp.zeros((num_samps, neurons))
                y_samples, log_rho_ts = model.obs_model.sample_posterior(
                    prng_state, x_eval, ini_t_since=ini_t_since, jitter=jitter)
                
            pred_log_intensities.append(pred_log_intens)
            test_spikes.append(test_ys)
            sample_spikes.append(sample_ys)
            sample_log_rhos.append(log_rho_ts)

        # export
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
