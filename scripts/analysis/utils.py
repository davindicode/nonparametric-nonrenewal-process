from functools import partial
from typing import Any
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

import optax

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



def extract_lengthscales(kernels, ISI_order):
    dim_counter, len_deltas, len_xs = 0, [], []
    for k in kernels:  # sort kernel lengthscales
        lens = np.array(k.lengthscale)
        
        for ldim_counter in range(lens.shape[-1]):
            if dim_counter == 0:
                len_tau = lens[:, ldim_counter]
                
            elif dim_counter > 0 and dim_counter < ISI_order:
                len_deltas.append(lens[:, ldim_counter])
                                  
            else:  # covariate dimensions
                len_xs.append(lens[:, ldim_counter])
    
    len_deltas = np.stack(len_deltas, axis=-1)
    len_xs = np.stack(len_xs, axis=-1)
    
    return len_tau, len_deltas, len_xs


def compute_ISI_densities(
    prng_state, 
    num_samps, 
    cisi_t_eval, 
    isi_conds, 
    x_conds, 
    jitter, 
    sel_outdims, 
    int_eval_pts = 1000, 
    num_quad_pts = 100, 
):
    """
    Computes conditional ISI distributions per input location
    
    :param np.ndarray isi_conds: lagged ISI values to condition on (pts, out_dims, orders)
    :param np.ndarray x_conds: covariate values to condition on (pts, x_dims)
    :return:
        ISI densities at evaluated times (pts, mc, out_dims, ts)
    """
    pts = isi_conds.shape[0]
    
    ISI_densities = []
    for pn in range(pts):
        
        ISI_density = obs_model.sample_conditional_ISI(
            prng_state,
            num_samps,
            jnp.array(cisi_t_eval),
            jnp.array(isi_conds[pn]), 
            jnp.array(x_conds[pn]),
            sel_outdims=sel_outdims, 
            int_eval_pts=int_eval_pts,
            num_quad_pts=num_quad_pts,
            prior=False,
            jitter=jitter, 
        )
        prng_state, _ = jr.split(prng_state)
        
        ISI_densities.append(np.array(ISI_density))
        
    return np.stack(ISI_densities, axis=0)


def compute_ISI_stats(
    prng_state, 
    num_samps, 
    x_locs, 
    isi_locs, 
    obs_model, 
    jitter, 
    outdims_list, 
    int_eval_pts = 1000, 
    num_quad_pts = 100, 
    batch_size = 100, 
):
    """
    Computes tuning from conditional ISIs per neuron
    
    :param np.ndarray x_locs: evaluation locations (..., x_dims)
    :param np.ndarray isi_locs: evaluation locations (..., neurons, x_dims)
    :return:
        mean and CV of shape (num_samps, out_dims, ...)
    """
    or_shape = x_locs.shape[:-1]
    x_locs = x_locs.reshape(-1, x_locs.shape[-1])  # (evals, x_dim)
    isi_locs = isi_locs.reshape(-1, *isi_locs.shape[-2:])
    
    num_pts = x_locs.shape[0]
    
    def m(prng_key, isi_cond, x_cond, n):
        """
        Compute statistics per neuron
        """
        fs = [lambda x: x, lambda x: 1 / (x + 1e-12), lambda x: x**2]
        fs_ISI = ()
        for f in fs:
            f_ISI = obs_model.sample_conditional_ISI_expectation(
                prng_key,
                num_samps,
                f,
                isi_cond, 
                x_cond,
                jnp.array([n]), 
                int_eval_pts=int_eval_pts,
                f_num_quad_pts=num_quad_pts,
                isi_num_quad_pts=num_quad_pts, 
                prior=False,
                jitter=jitter, 
            )
            fs_ISI += (f_ISI[:, 0],)  # (num_samps,)
        return fs_ISI

    m_jit = jit(vmap(m, (0, 0, 0, None), (1, 1, 1)))

    mean_ISIs, mean_invISIs, CV_ISIs = [], [], []
    batches = int(np.ceil(num_pts / batch_size))
    
    for n in outdims_list:
        _mean_ISIs, _mean_invISIs, _CV_ISIs = [], [], []
        iterator = tqdm(range(batches))
        for en in iterator:
            prng_key, prng_state = jr.split(prng_state)
            x_eval = jnp.array(x_locs[en*batch_size:(en+1)*batch_size, :])
            isi_eval = jnp.array(isi_locs[en*batch_size:(en+1)*batch_size, ...])
            
            prng_keys = jr.split(prng_key, x_eval.shape[0])
            mean_ISI, mean_invISI, secmom_ISI = m_jit(
                prng_keys, isi_eval, x_eval, n)  # (num_samps, pts)
            
            var_ISI = (secmom_ISI - mean_ISI**2)
            CV_ISI = jnp.sqrt(var_ISI) / (mean_ISI + 1e-12)
        
            _mean_ISIs.append(mean_ISI)
            _mean_invISIs.append(mean_invISI)
            _CV_ISIs.append(CV_ISI)
            
        mean_ISIs.append(np.array(jnp.concatenate(_mean_ISIs, axis=1)))  # (num_samps, ts)
        mean_invISIs.append(np.array(jnp.concatenate(_mean_invISIs, axis=1)))
        CV_ISIs.append(np.array(jnp.concatenate(_CV_ISIs, axis=1)))
        
    mean_ISIs, mean_invISIs, CV_ISIs = (
        np.stack(mean_ISIs, axis=1), np.stack(mean_invISIs, axis=1), np.stack(CV_ISIs, axis=1)
    )  # (num_samps, out_dims, ts)

    mean_ISIs = mean_ISIs.reshape(num_samps, -1, *or_shape)
    mean_invISIs = mean_invISIs.reshape(num_samps, -1, *or_shape)
    CV_ISIs = CV_ISIs.reshape(num_samps, -1, *or_shape)
    return mean_ISIs, mean_invISIs, CV_ISIs



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

        poisson_process = lib.likelihoods.ExponentialRenewal(
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

        poisson_process = lib.likelihoods.ExponentialRenewal(
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
    pred_window_filt = np.arange(pred_start, pred_end + filter_length - 1)
    ini_t_tilde = jnp.zeros((1, neurons))
    
    pred_ys = ys[:, pred_window]

    if obs_type == 'factorized_gp':
        pred_log_intens = obs_model.log_conditional_intensity(
            prng_state, covs_t[None, None, pred_window], 
            ys_filt[None, :, pred_window_filt], 
            jitter, sel_outdims, sampling=sampling
        )

    elif obs_type == 'rate_renewal_gp':
        pred_log_intens = obs_model.log_conditional_intensity(
            prng_state, ini_t_tilde, covs_t[None, None, pred_window], 
            pred_ys[None], 
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
        
    return np.array(pred_ys), np.array(pred_log_intens)



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
    pred_window_filt = np.arange(pred_start, pred_end + filter_length - 1)
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



def linear_regression(X, Y):
    """
    1D linear regression for y = a x + b, with last dimension as observation points
    """
    a = ((X * Y).mean(-1) - X.mean(-1) * Y.mean(-1)) / X.var(-1)
    b = (Y.mean(-1) * (X**2).mean(-1) - (X * Y).mean(-1) * X.mean(-1)) / X.var(-1)
    
    Y_fit = a[:, None] * X + b[:, None]  # (out_dims, pts)
    R2_lin = 1 - (Y - Y_fit).var(-1) / Y.var(-1)  # R^2 of fit (out_dims,)
    
    return a, b, R2_lin



def gp_regression(
    X, 
    Y, 
    rng, 
    ini_len_fx, 
    num_induc, 
    batch_size = 1000, 
    lr_start = 1e-2, 
    lr_decay = 0.9995, 
    lr_end = 1e-3, 
    loss_margin = 0., 
    margin_epochs = 100, 
    max_epochs = 3000, 
    jitter = 1e-6, 
    array_type = "float32", 
):
    """
    1D GP regresssion
    
    :param np.array X: input locs (obs_dims, pts)
    :param np.array Y: observations (obs_dims, pts)
    """
    obs_dims, ts = Y.shape
    batches = int(np.ceil(ts / batch_size))
    
    len_fx = ini_len_fx[:, None]  # GP lengthscale
    var_f = np.ones(obs_dims)  # kernel variance
    kern = lib.GP.kernels.Matern52(obs_dims, variance=var_f, lengthscale=len_fx, array_type=array_type)

    induc_locs = np.stack(
        [np.linspace(X[d].min(), X[d].max(), num_induc) for d in range(obs_dims)]
    )[..., None]

    # qSVGP
    u_mu = 0.1 * rng.normal(size=(obs_dims, num_induc, 1))
    u_Lcov = 1.0 * np.eye(num_induc)[None, ...].repeat(obs_dims, axis=0)

    gp_mean = np.zeros((obs_dims,))
    qsvgp = lib.GP.sparse.qSVGP(kern, induc_locs, u_mu, u_Lcov, RFF_num_feats=0, whitened=True)

    # observation
    variance = np.ones((obs_dims,))
    likelihood = lib.likelihoods.Gaussian(obs_dims, variance, array_type=array_type)
    
    class M(eqx.Module):
        gp: Any
        gp_mean: Any
        likelihood: Any
        
        def __init__(self, gp, gp_mean, likelihood):
            self.gp = gp
            self.gp_mean = jnp.array(gp_mean)
            self.likelihood = likelihood
        
        def apply_constraints(self):
            model = jax.tree_map(lambda p: p, self)  # copy
            model = eqx.tree_at(
                lambda tree: [tree.gp, tree.likelihood],
                model,
                replace_fn=lambda obj: obj.apply_constraints(),
            )

            return model
    
    model = M(qsvgp, gp_mean, likelihood)

    # optimizer
    learning_rate_schedule = optax.exponential_decay(
        init_value=lr_start,
        transition_steps=batches,
        decay_rate=lr_decay,
        transition_begin=0,
        staircase=True,
        end_value=lr_end,
    )
    optim = optax.adam(learning_rate_schedule)

    # loss    
    @partial(eqx.filter_value_and_grad)
    def compute_loss(model, jitter, data, ts, lik_int_method):
        xs, ys = data  # (obs_dims, ts)

        f_mean, f_var, KL, _ = model.gp.evaluate_posterior(
            xs[None, ..., None], mean_only=False, diag_cov=True, 
            compute_KL=True, compute_aux=False, jitter=jitter, sel_outdims=None
        )  # (1, out_dims, time, 1 or time)
        
        f_mean = f_mean[0, ...] + model.gp_mean[:, None, None]  # (out_dims, ts, 1)
        f_var = f_var[0, ...]  # (out_dims, ts, 1)
        
        llf = lambda y, m, c: model.likelihood.variational_expectation(
            y, m, c, None, jitter, lik_int_method
        )
        
        lls = vmap(llf, (1, 1, 1))(ys, f_mean, f_var)  # vmap ts
        Ell = lls.mean()  # take mean over num_samps and ts

        nELBO = - ts * Ell + KL
        return nELBO

    @partial(eqx.filter_jit, device=jax.devices()[0])
    def make_step(
        model, jitter, data, ts, lik_int_method, opt_state, 
    ):
        loss, grads = compute_loss(
            model, jitter, data, ts, lik_int_method, 
        )

        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        model = model.apply_constraints()
        return loss, model, opt_state
    
    # fitting
    lik_int_method = {
        "type": 'MC',
        "approx_pts": 0,
    }  # use exact expression
        
    opt_state = optim.init(model)
    tracker = {
        "train_loss_batches": [],
        "train_loss_epochs": [],
        "learning_rates": []
    }
    
    cnt, minloss = 0, np.inf
    iterator = tqdm(range(max_epochs))
    for epoch in iterator:

        avg_loss = []
        for b in range(batches):
            bslice = slice(b*batch_size, (b+1)*batch_size)
            batch_data = (X[:, bslice], Y[:, bslice])

            loss, model, opt_state = make_step(
                model,
                jitter,
                batch_data,
                ts, 
                lik_int_method,
                opt_state,
            )
            loss = loss.item()
            avg_loss.append(loss)
            
            tracker["train_loss_batches"].append(loss)
            tracker["learning_rates"].append(learning_rate_schedule(epoch).item())

            iterator.set_postfix(loss=loss)

        avgbloss = (
            np.array(avg_loss).mean().item()
        )  # average over batches (subsampled estimator of loss)
        loss_dict = {"train_loss_epochs": avgbloss}
        for n, v in loss_dict.items():
            tracker[n].append(v)

        if avgbloss <= minloss + loss_margin:
            cnt = 0
        else:
            cnt += 1

        if avgbloss < minloss:
            minloss = avgbloss

        if cnt > margin_epochs:
            print("Stopped at epoch {}.".format(epoch + 1))
            break

    xx = X[None, ..., None]

    post_mean, _, _, _ = model.gp.evaluate_posterior(
        xx, mean_only=True, diag_cov=True, compute_KL=False, compute_aux=False, jitter=jitter)
    Y_fit = np.array(post_mean[0, ..., 0] + model.gp_mean[:, None])  # (out_dims, ts)
    
    R2_gp = 1 - (Y - Y_fit).var(-1) / Y.var(-1)  # R^2 of fit (out_dims,)
    
    return model, tracker, R2_gp


# processing
def evaluate_regression_fits(
    checkpoint_dir, reg_config_names, kernel_gen_func, dataset_dict, test_dataset_dicts, rng, prng_state
):
    tbin = dataset_dict["properties"]["tbin"]
    neurons = dataset_dict["properties"]["neurons"]
    
    lik_int_method = {
        "type": "GH", 
        "approx_pts": 50, 
    }

    num_samps = 1  # for sampling activity
    pred_start, pred_end = 0, 10000
    pred_ts = np.arange(pred_start, pred_end) * tbin
    regression_dict = {}
    for model_name in reg_config_names:
        print('Analyzing regression for {}...'.format(model_name))
        
        # config
        model, config = load_model_and_config(
            checkpoint_dir + model_name, 
            dataset_dict, 
            kernel_gen_func, 
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
        print('Training data...')
#        train_ell = None
        train_ell = likelihood_metric(
            prng_state, data, model.obs_model, obs_type, lik_int_method, jitter, log_predictive=False)
        prng_state, _ = jr.split(prng_state)
        
        train_lpd = None
#         train_lpd = likelihood_metric(
#             prng_state, data, model.obs_model, obs_type, lik_int_method, jitter, log_predictive=True)
#         prng_state, _ = jr.split(prng_state)
        
        sort_cdfs, T_KSs, sign_KSs, p_KSs = time_rescaling_statistics(
            data, model.obs_model, obs_type, jitter, list(range(neurons)))

        # test data
        print('Test data...')
        
        test_lpds, test_ells = [], []
        pred_log_intensities, pred_spiketimes = [], []
        sample_log_rhos, sample_spiketimes = [], []
        for test_dataset_dict in test_dataset_dicts:  # test
            timestamps, covs_t, ISIs, observations, filter_length = template.select_inputs(
                test_dataset_dict, config)
            
            test_ys = observations[:, filter_length:]
            test_ys_filt = observations[:, :-1]
            data = (timestamps, covs_t, ISIs, test_ys, test_ys_filt, filter_length)
            
            test_ell = likelihood_metric(
                prng_state, data, model.obs_model, obs_type, lik_int_method, jitter, log_predictive=False)
            prng_state, _ = jr.split(prng_state)
            
            test_lpd = None
#             test_lpd = likelihood_metric(
#                 prng_state, data, model.obs_model, obs_type, lik_int_method, jitter, log_predictive=True)
#             prng_state, _ = jr.split(prng_state)
            
            test_ells.append(test_ell)
            test_lpds.append(test_lpd)
            
            pred_ys, pred_log_intens = conditional_intensity(
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
            prng_state, _ = jr.split(prng_state)
            
            pred_spkts = np.where(pred_ys > 0)[0] + pred_start * tbin
            
            pred_log_intensities.append(pred_log_intens)
            pred_spiketimes.append(pred_spkts)
            
            sample_ys, log_rho_ts = sample_activity(
                prng_state, 
                num_samps, 
                data, 
                model.obs_model, 
                obs_type, 
                pred_start, 
                pred_end, 
                jitter, 
            )
            prng_state, _ = jr.split(prng_state)
            
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


def analyze_variability_stats(
    checkpoint_dir, 
    model_name, 
    kernel_gen_func, 
    dataset_dict, 
    rng, 
    prng_state, 
    num_samps = 30, 
    dilation = 10, 
    int_eval_pts = 1000, 
    num_quad_pts = 100, 
    batch_size = 10, 
    jitter = 1e-6, 
):
    tbin = dataset_dict["properties"]["tbin"]
    neurons = dataset_dict["properties"]["neurons"]

    print('Analyzing tuning for {}...'.format(model_name))

    # config
    model, config = load_model_and_config(
        checkpoint_dir + model_name, 
        dataset_dict, 
        kernel_gen_func, 
        rng, 
    )
    obs_type = config.observations.split('-')[0]
    jitter = config.jitter
    ISI_order = int(config.likelihood[3:])
    
    # data
    timestamps, covs_t, ISIs, observations, filter_length = template.select_inputs(
        dataset_dict, config)
    
    ### evaluation ###
    x_locs = covs_t[::dilation, :]
    isi_locs = ISIs[:, ::dilation, :]
    eval_pts = x_locs.shape[0]
    
    isi_locs = np.ones((eval_pts, neurons, ISI_order-1))    
    mean_ISI, mean_invISI, CV_ISI = compute_ISI_stats(
        prng_state, 
        num_samps, 
        x_locs, 
        isi_locs, 
        model.obs_model, 
        jitter, 
        list(range(2)), 
        int_eval_pts = int_eval_pts, 
        num_quad_pts = num_quad_pts, 
        batch_size = batch_size, 
    )  # (mc, out_dims, eval_pts)
    
    ### stats ###
    X, Y = 1 / mean_ISI.mean(0).T, CV_ISI.mean(0).T  # (out_dims, pts)
    a, b, R2_lin = linear_regression(X, Y)
    gp_model, tracker, R2_gp = gp_regression(
        X, 
        Y, 
        rng, 
        ini_len_fx = X.mean(-1) / 3., 
        num_induc = 10, 
    )
    
    # export
    variability_dict = {
        'mean_ISI': mean_ISI, 
        'CV_ISI': CV_ISI, 
        'linear_slope': a, 
        'linear_intercept': b, 
        'linear_R2': R2_lin, 
        'GP_tracker': tracker, 
        'GP_model': gp_model, 
        'GP_R2': R2_gp, 
    }
    return variability_dict