import sys

sys.path.append("../fit/")
import template

sys.path.append("../../../GaussNeuro")
import gaussneuro as lib

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import jit, vmap
import numpy as np

from tqdm.autonotebook import tqdm

import pickle


def load_model_and_config(filename, dataset_dict, mapping_builder, rng):
    results = pickle.load(
        open(filename + ".p", "rb")
    )
    config = results["config"]
    obs_covs_dims = len(config.observed_covs.split('-'))
    timestamps = dataset_dict["timestamps"]
    
    jax.config.update("jax_enable_x64", config.double_arrays)
    model = template.build_model(
        config, dataset_dict, mapping_builder, rng, timestamps, obs_covs_dims)
    model = eqx.tree_deserialise_leaves(filename + ".eqx", model)
    
    return model, config


def compute_ISI_stats(
    prng_state, 
    num_samps, 
    x_locs, 
    isi_locs, 
    obs_model, 
    jitter, 
    sel_outdims, 
    int_eval_pts = 1000, 
    num_quad_pts = 100, 
    batch_size = 100, 
):
    """
    :param np.ndarray x_locs: evaluation locations (pts, x_dims)
    :param np.ndarray isi_locs: evaluation locations (pts, neurons, x_dims)
    :return:
        mean and CV of shape (num_samps, out_dims, pts)
    """
    num_pts = x_locs.shape[0]
    
    def m(prng_key, isi_cond, x_cond, n):
        """
        Compute statistics per neuron
        """
        mean_ISI = obs_model.sample_conditional_ISI_expectation(
            prng_key,
            num_samps,
            lambda x: x,
            isi_cond, 
            x_cond,
            jnp.array([n]), 
            int_eval_pts=int_eval_pts,
            f_num_quad_pts=num_quad_pts,
            isi_num_quad_pts=num_quad_pts, 
            prior=False,
            jitter=jitter, 
        )

        secmom_ISI = obs_model.sample_conditional_ISI_expectation(
            prng_key,
            num_samps,
            lambda x: x**2,
            isi_cond, 
            x_cond,
            jnp.array([n]), 
            int_eval_pts=int_eval_pts,
            f_num_quad_pts=num_quad_pts,
            isi_num_quad_pts=num_quad_pts, 
            prior=False,
            jitter=jitter, 
        )

        var_ISI = (secmom_ISI - mean_ISI**2)
        CV_ISI = jnp.sqrt(var_ISI) / (mean_ISI + 1e-12)

        return mean_ISI[:, 0], CV_ISI[:, 0]  # (num_samps,)

    m_jit = jit(vmap(m, (0, 0, 0, None)))

    mean_ISIs, CV_ISIs = [], []
    batches = int(np.ceil(num_pts / batch_size))
    
    for n in sel_outdims:
        _mean_ISIs, _CV_ISIs = [], []
        iterator = tqdm(range(batches))
        for en in iterator:
            prng_key, prng_state = jr.split(prng_state)
            x_eval = jnp.array(x_locs[en*batch_size:(en+1)*batch_size, :])
            isi_eval = jnp.array(isi_locs[en*batch_size:(en+1)*batch_size, ...])
            
            prng_keys = jr.split(prng_key, x_eval.shape[0])
            mean_ISI, CV_ISI = m_jit(prng_keys, isi_eval, x_eval, n)
            _mean_ISIs.append(mean_ISI)
            _CV_ISIs.append(CV_ISI)
            
        mean_ISIs.append(np.array(jnp.concatenate(_mean_ISIs, axis=-1)))
        CV_ISIs.append(np.array(jnp.concatenate(_CV_ISIs, axis=-1)))
        
    mean_ISIs, CV_ISIs = np.stack(mean_ISIs, axis=1), np.stack(CV_ISIs, axis=1)
    return mean_ISIs, CV_ISIs


def linear_regression(A, B):
    """
    1D linear regression for y = a x + b, with last dimension as observation points
    """
    a = ((A * B).mean(-1) - A.mean(-1) * B.mean(-1)) / A.var(-1)
    b = (B.mean(-1) * (A**2).mean(-1) - (A * B).mean(-1) * A.mean(-1)) / A.var(-1)
    return a, b


def likelihood_metric(
    prng_state, 
    data, 
    obs_model, 
    obs_type, 
    lik_int_method, 
    jitter, 
    log_predictive, 
):
    timestamps, covs_t, ISIs, ys, ys_filt, filter_length = data
 
    timesteps = len(timestamps)
    if obs_type == 'factorized_gp':
        llm, _ = obs_model.variational_expectation(
            prng_state, jitter, covs_t[None, None], ys, ys_filt, False, timesteps, 
            lik_int_method, log_predictive
        )

    elif obs_type == 'rate_renewal_gp':
        llm, _ = obs_model.variational_expectation(
            prng_state, jitter, covs_t[None, None], ys, ys_filt, False, timesteps, 
            lik_int_method, log_predictive
        )

    elif obs_type == 'nonparam_pp_gp':
        llm, _ = obs_model.variational_expectation(
            prng_state, jitter, covs_t[None, None], ISIs, ys, False, timesteps, 
            lik_int_method, log_predictive
        )

    else:
        raise ValueError('Invalid observation model type')
        
    return np.array(llm)



def time_rescaling_statistics(data, obs_model, obs_type, jitter, sel_outdims):
    timestamps, covs_t, ISIs, ys, ys_filt, filter_length = data
    sel_outdims = jnp.array(sel_outdims) if sel_outdims is not None else None
    
    max_intervals = int(ys.sum(-1).max())
    if obs_type == 'factorized_gp':
        post_mean_rho = obs_model.posterior_mean(
            covs_t, ys_filt, jitter, sel_outdims)
        rescaled_intervals = lib.utils.spikes.time_rescale(
            ys.T, post_mean_rho.T, obs_model.likelihood.dt, max_intervals=max_intervals)

        poisson_process = lib.likelihoods.Exponential(
            obs_model.likelihood.obs_dims, obs_model.likelihood.dt)
        qs = np.array(poisson_process.cum_density(rescaled_intervals))[:, 1:]

    elif obs_type == 'rate_renewal_gp':
        post_mean_rho = obs_model.posterior_mean(
            covs_t, ys_filt, jitter, sel_outdims)
        rescaled_intervals = lib.utils.spikes.time_rescale(
            ys.T, post_mean_rho.T, obs_model.renewal.dt, max_intervals=max_intervals)

        qs = np.array(vmap(obs_model.renewal.cum_density)(rescaled_intervals.T).T)

    elif obs_type == 'nonparam_pp_gp':
        post_mean_rho = obs_model.posterior_mean(
            covs_t, ISIs, jitter, sel_outdims)
        rescaled_intervals = lib.utils.spikes.time_rescale(
            ys.T, post_mean_rho.T, obs_model.pp.dt, max_intervals=max_intervals)

        poisson_process = lib.likelihoods.Exponential(
            obs_model.pp.obs_dims, obs_model.pp.dt)
        qs = np.array(poisson_process.cum_density(rescaled_intervals))[:, 1:]

    else:
        raise ValueError('Invalid observation model type')
         
    # KS statistics
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
        
    return sort_cdfs, T_KSs, sign_KSs, p_KSs
        
        
        
def conditional_intensity(
    prng_state, 
    data, 
    obs_model, 
    obs_type, 
    pred_start, 
    pred_end, 
    jitter, 
    sel_outdims, 
    sampling, 
):
    timestamps, covs_t, ISIs, ys, ys_filt, filter_length = data
    neurons = ys.shape[0]
    sel_outdims = jnp.array(sel_outdims) if sel_outdims is not None else None
    
    # predicted intensities and posterior sampling
    pred_window = np.arange(pred_start, pred_end)
    pred_window_filt = pred_window + filter_length
    ini_t_tilde = jnp.zeros((1, neurons))

    if obs_type == 'factorized_gp':
        pred_log_intens = obs_model.log_conditional_intensity(
            prng_state, covs_t[None, None, pred_window], 
            ys_filt[None, :, pred_window_filt], 
            jitter, sel_outdims, sampling=sampling
        )

    elif obs_type == 'rate_renewal_gp':
        pred_log_intens = obs_model.log_conditional_intensity(
            prng_state, ini_t_tilde, covs_t[None, None, pred_window], 
            ys[None, :, pred_window], 
            ys_filt[None, :, pred_window_filt], 
            jitter, sel_outdims, sampling=sampling, unroll=10
        )

    elif obs_type == 'nonparam_pp_gp':
        pred_log_intens = obs_model.log_conditional_intensity(
            prng_state, covs_t[None, None, pred_window], ISIs[None, :, pred_window], 
            jitter, sel_outdims, sampling=sampling
        )
        
    else:
        raise ValueError('Invalid observation model type')
        
    return np.array(pred_log_intens)



def sample_activity(
    prng_state, 
    num_samps, 
    data, 
    obs_model, 
    obs_type, 
    pred_start, 
    pred_end, 
    jitter, 
):
    timestamps, covs_t, ISIs, ys, ys_filt, filter_length = data
    neurons = ys.shape[0]
    
    # predicted intensities and posterior sampling
    pred_window = np.arange(pred_start, pred_end)
    pred_window_filt = pred_window + filter_length
    x_eval = covs_t[None, None, pred_window].repeat(num_samps, axis=0)
    ini_Y = ys_filt[None, :, pred_window_filt[:filter_length]].repeat(num_samps, axis=0)

    if obs_type == 'factorized_gp':
        sample_ys, log_rho_ts = obs_model.sample_posterior(
            prng_state, x_eval, ini_Y=ini_Y, jitter=jitter)

    elif obs_type == 'rate_renewal_gp':
        ini_t_tilde = jnp.zeros((num_samps, neurons))
        sample_ys, log_rho_ts = obs_model.sample_posterior(
            prng_state, x_eval, ini_spikes=ini_Y, ini_t_tilde=ini_t_tilde, jitter=jitter)

    elif obs_type == 'nonparam_pp_gp':
        timesteps, ISI_order = x_eval.shape[2], ISIs.shape[-1]
        ini_t_since = jnp.zeros((num_samps, neurons))
        past_ISIs = jnp.zeros((num_samps, neurons, ISI_order - 1))
        sample_ys, log_rho_ts = obs_model.sample_posterior(
            prng_state, timesteps, x_eval, ini_t_since=ini_t_since, past_ISIs=past_ISIs, jitter=jitter)
        
    else:
        raise ValueError('Invalid observation model type')
        
    return np.array(sample_ys), np.array(log_rho_ts)