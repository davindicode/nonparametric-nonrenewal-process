from functools import partial
from typing import Union

import equinox as eqx

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import vmap, lax

import numpy as np

from .base import Observations, FilterObservations

from ..GP.markovian import MultiOutputLTI
from ..GP.sparse import SparseGP
from ..GP.spatiotemporal import KroneckerLTI
from ..likelihoods.base import FactorizedLikelihood, RenewalLikelihood
from ..likelihoods.factorized import PointProcess
from ..utils.linalg import gauss_legendre
from ..utils.jax import safe_log, safe_sqrt




class SparseGPFilterObservations(FilterObservations):
    """
    Uses SVGP to model modulation
    
    Option to learn mean
    """
    
    gp: SparseGP
    gp_mean: Union[jnp.ndarray, None]  # constant mean (obs_dims,), i.e. bias if not None
        
    def _gp_sample(self, prng_state, x_eval, prior, jitter):
        """
        Obtain the log conditional intensity given input path along which to evaluate

        :param jnp.ndarray x_eval: evaluation locs (num_samps, out_dims, eval_locs, in_dims)
        """
        if prior:
            f_samples = self.gp.sample_prior(
                prng_state, x_eval, jitter
            )  # (evals, samp, f_dim)

        else:
            f_samples, _ = self.gp.sample_posterior(
                prng_state, x_eval, jitter, compute_KL=False
            )  # (evals, samp, f_dim)

        return pre_rates



class ModulatedFactorized(FilterObservations):
    """
    Factorization across time points allows one to rely on latent marginals
    """
    
    likelihood: FactorizedLikelihood

    def __init__(self, gp, gp_mean, likelihood, spikefilter=None):
        # checks
        assert likelihood.array_type == gp.array_type
        assert likelihood.f_dims == gp.kernel.out_dims

        super().__init__(spikefilter, gp.array_type)
        self.gp = gp
        self.gp_mean = self._to_jax(gp_mean) if gp_mean is not None else None
        self.likelihood = likelihood
        
    def apply_constraints(self):
        """
        Constrain parameters in optimization
        """
        model = super().apply_constraints()
        model = eqx.tree_at(
            lambda tree: tree.ssgp,
            model,
            replace_fn=lambda obj: obj.apply_constraints(),
        )

        return model
    
    def _sample_Y(self, prng_state, ini_Y, f_samples):
        """
        Sample the spike train autoregressively
        
        For heteroscedastic, couple filters to the rate parameter in each 
        num_f_per_obs group, f_dims = obs_dims x num_f_per_obs
        
        :param jnp.dnarray ini_spikes: initial spike train (num_samps, obs_dims, filter_length)
        """
        num_samps, obs_dims, ts = f_samples.shape
        if self.spikefilter is not None:
            prng_states = jr.split(prng_state, ts*(num_samps+1)).reshape(ts, num_samps+1, -1)

            def step(carry, inputs):
                prng_keys, f = inputs
                past_Y = carry  # (mc, obs_dims, ts)
                
                h, _ = self.spikefilter.apply_filter(prng_keys[-1], past_Y, compute_KL=False)
                f = f.at[:, ::self.likelihood.num_f_per_obs].add(h[..., 0])

                f = f + self.gp_mean[None] if self.gp_mean is not None else f
                y = vmap(self.likelihood.sample_Y)(prng_keys[:-1], f)  # vmap over MC
                past_Y = jnp.concatenate((past_Y[..., 1:], y[..., None]), axis=-1)

                return past_Y, (y, f)

            init = ini_Y
            xs = (prng_states, f_samples.transpose(2, 0, 1))
            _, (Y, f_filtered) = lax.scan(step, init, xs)
            return Y.transpose(1, 2, 0), f_filtered.transpose(1, 2, 0)  # (num_samps, obs_dims, ts)
        
        else:
            prng_states = jr.split(prng_state, ts*num_samps).reshape(num_samps, ts, -1)
            f_samples = f_samples + self.gp_mean[None] if self.gp_mean is not None else f_samples
            Y = vmap(vmap(self.likelihood.sample_Y), (1, 2), 2)(prng_states, f_samples)
            return Y, f_samples

    ### variational inference ###
    def ELBO(
        self,
        prng_state,
        num_samps, 
        x, 
        y, 
        y_filt, 
    ):
        """
        Compute ELBO
        
        :param jnp.ndarray x: inputs (num_samps, out_dims, ts, x_dims)
        """
        f_mean, f_cov, KL_f, _ = self.gp.evaluate_posterior(
            prng_state, x_samples, jitter, compute_KL=True
        )  # (evals, samp, f_dim)

        if self.spikefilter is not None:
            y_filtered, KL_y = self.spikefilter.apply_filter(prng_state, y_filt, compute_KL=True)
            prng_state, _ = jr.split(prng_state)
            f_mean += y_filtered
        
        f_mean = f_mean + self.gp_mean[None] if self.gp_mean is not None else f_mean
        Ell = self.likelihood.variational_expectation(
            prng_state, jitter, y, f_mean, f_cov
        )

        ELBO = Ell - KL_x - KL_f - KL_y
        return ELBO
    
    ### evaluation ###
    def evaluate_pre_conditional_rate(self, prng_state, x_eval, y_filt, jitter):
        """
        evaluate posterior rate
        """
        pre_rates_mean, pre_rates_cov, _, _ = self.gp.evaluate_posterior(
            x_eval, mean_only=False, diag_cov=False, compute_KL=False, compute_aux=False, jitter=jitter)
        pre_rates_mean = pre_rates_mean[..., 0]
        
        if self.spikefilter is not None:
            y_filtered, _ = self.spikefilter.apply_filter(
                prng_state, y_filt, compute_KL=False)
            pre_rates_mean += y_filtered
        
        return pre_rates_mean, pre_rates_cov  # (num_samps, out_dims, ts)
    
    def evaluate_metric(self):
        """
        predictive posterior log likelihood
        log posterior predictive likelihood
        """
        return

    ### sample ###
    def sample_prior(self, prng_state, num_samps, x_samples, ini_Y, jitter):
        """
        Sample from the generative model
        """
        prng_states = jr.split(prng_state, 3)
        
        f_samples = self._gp_sample(prng_states[1], x_samples, True, jitter)  # (samp, evals, f_dim)
        y_samples, filtered_f = self._sample_Y(prng_states[2], ini_Y, f_samples)
        
        return y_samples, filtered_f

    def sample_posterior(self, prng_state, num_samps, x_samples, ini_Y, jitter):
        """
        Sample from posterior predictive
        """
        prng_states = jr.split(prng_state, 3)
        
        f_samples = self._gp_sample(prng_states[1], x_samples, False, jitter)  # (samp, evals, f_dim)
        y_samples, filtered_f = self._sample_Y(prng_states[2], ini_Y, f_samples)
        
        return y_samples, filtered_f


class RateRescaledRenewal(FilterObservations):
    """
    Renewal likelihood GPLVM
    """

    gp: SparseGP
    gp_mean: jnp.ndarray  # constant mean (obs_dims,), i.e. bias if not None
    renewal: RenewalLikelihood

    def __init__(self, gp, gp_mean, renewal, spikefilter=None):
        # checks
        assert renewal.array_type == gp.array_type
        assert renewal.f_dims == gp.kernel.out_dims

        super().__init__(spikefilter, gp.array_type)
        self.gp = gp
        self.gp_mean = self._to_jax(gp_mean)
        self.renewal = renewal

    def _gp_sample(self, prng_state, x_eval, prior, jitter):
        """
        Sample from Gaussian process with mean function

        :param jnp.ndarray x_eval: evaluation locs (num_samps, out_dims, eval_locs, in_dims)
        """
        if prior:
            f_samples = self.gp.sample_prior(
                prng_state, x_eval, jitter
            )  # (samp, f_dim, evals)

        else:
            f_samples, _ = self.gp.sample_posterior(
                prng_state, x_eval, jitter, compute_KL=False
            )  # (samp, f_dim, evals)

        return f_samples + self.gp_mean[None]
    
    def _sample_spikes(self, prng_state, ini_spikes, tau_start, f_samples):
        """
        Sample the spike train autoregressively
        
        :param jnp.dnarray ini_spikes: initial spike train (num_samps, obs_dims, filter_length)
        :param jnp.ndarray tau_start: initial tau values at start (num_samps, obs_dims)
        """
        num_samps, obs_dims, ts = f_samples.shape
        if self.spikefilter is not None:
            prng_states = jr.split(prng_state, ts*2).reshape(ts, 2, -1)
        else:
            prng_states = jr.split(prng_state, ts).reshape(ts, 1, -1)

        def step(carry, inputs):
            prng_state, f = inputs  # (num_samps, obs_dims)
            tau_since, past_spikes, spikes = carry
            
            if self.spikefilter is not None:
                h, _ = self.spikefilter.apply_filter(prng_state[0], past_spikes, False)
                f += h[..., 0]
                
            
            rate = self.renewal.inverse_link(f)
            log_rate = f if self.renewal.link_type == 'log' else safe_log(rate)
            
            tau_since = jnp.where(
                (spikes > 0),  # previous spikes
                rate * self.renewal.dt, 
                tau_since + rate * self.renewal.dt, 
            )  # (num_samps, obs_dims)
            
            log_renewal_density = self.renewal.log_density(tau_since)
            log_survival = self.renewal.log_survival(tau_since)

            log_rho_t = log_rate + log_renewal_density - log_survival
            p_spike = jnp.minimum(jnp.exp(log_rho_t) * self.renewal.dt, 1.)  # approximate by discrete Bernoulli
            spikes = jr.bernoulli(prng_state[-1], p_spike).astype(self.array_type)  # (num_samps, obs_dims)
            
            if self.spikefilter is not None:
                past_spikes = jnp.concatenate((past_spikes[..., 1:], spikes[..., None]), axis=-1)
            
            return (tau_since, past_spikes, spikes), (spikes, log_rho_t)

        init = (tau_start, ini_spikes, jnp.zeros((num_samps, obs_dims)))
        xs = (prng_states, f_samples.transpose(2, 0, 1))
        _, (spikes, log_rho_ts) = lax.scan(step, init, xs)
        return spikes.transpose(1, 2, 0), log_rho_ts.transpose(1, 2, 0)  # (num_samps, obs_dims, ts)

    def _rate_rescale(self, spikes, rates, compute_ll, return_tau):
        """
        rate rescaling, computes the log density on the way
        
        :param jnp.ndarray spikes: (ts, obs_dims)
        :param jnp.ndarray rates: (ts, obs_dims)
        """
        rate_scale = self.renewal.dt / self.renewal.shape_scale()
        
        def step(carry, inputs):
            tau, ll = carry
            rate, spike = inputs

            tau += rate_scale * rate
            if compute_ll:
                ll += self.renewal.log_density(tau)
            
            tau_ = jnp.where(spike, 0., tau)
            return (tau_, ll), tau if return_tau else None

        init = (jnp.zeros(self.renewal.obs_dims), jnp.zeros(self.renewal.obs_dims))
        (_, log_renewals), taus = lax.scan(step, init=init, xs=(rates, spikes))
        return log_renewals, taus
    
    def _log_density(
        self, y, pre_rates, 
    ):
        """
        Pre rates includes spike coupling/history effects
        """
        rates = self.renewal.inverse_link(pre_rates)
        log_rates = pre_rates if self.renewal.link_type == "log" else safe_log(rates)
        spikes = (y.T > 0)
        
        log_renewals, _ = self._rate_rescale(
            spikes, rates.T, compute_ll=True, return_tau=False)
        
        ll = log_rates + log_renewals  # (obs_dims, ts)
        return ll
    
    ### inference ###
    def ELBO(self, prng_state, y, y_filt, pre_rates, x, neuron, num_ISIs):
        """
        Compute the evidence lower bound
        """
        f_samples, KL_f = self.gp.sample_posterior(
            prng_state, x_samples, jitter, compute_KL=True
        )  # (evals, samp, f_dim)

        pre_rates = f_samples + self.gp_mean[None]
        if self.spikefilter is not None:
            y_filtered, KL_y = self.spikefilter.apply_filter(prng_state[0], y_filt, compute_KL)
        pre_rates = f_samples + y_filtered
        
        Ell = self._log_density(
            y, pre_rates, 
        )

        ELBO = Ell - KL_f - KL_y
        return ELBO
    
    ### evaluation ###
    def evaluate_metric(self):
        return

    ### sample ###
    def sample_log_conditional_intensity(self, prng_state, x_eval, y, y_filt, jitter):
        """
        rho(t) = r(t) p(u) / (1 - int p(u) du)
        
        :param jnp.ndarray x_eval: evaluation locations (num_samps, obs_dims, ts, x_dims)
        :param jnp.ndarray y: spike train corresponding to segment (neurons, ts), 
                              y = y_filt[..., filter_length:] + last time step
        """
        f_samples = self._gp_sample(prng_state, x_eval, False, jitter)  # (num_samps, obs_dims, ts)
        num_samps = pre_rates.shape[0]
        y = jnp.broadcast_to(y, (num_samps,) + y.shape[1:])
        
        pre_rates = f_samples + self.gp_mean[None]
        if self.spikefilter is not None:
            h, _ = self.spikefilter.apply_filter(prng_state[0], y_filt, False)
            pre_rates += h
            
        rates = self.renewal.inverse_link(pre_rates).transpose(0, 2, 1)
        
        # rate rescaling, rescaled time since last spike
        spikes = (y > 0).transpose(0, 2, 1)
        _, tau_since_spike = vmap(self._rate_rescale, (0, 0, None, None), (None, 0))(
            spikes, rates, False, True)  # (num_samps, ts, obs_dims)
        
        log_renewal_density = self.renewal.log_density(tau_since_spike)
        log_survival = self.renewal.log_survival(tau_since_spike)
        
        log_rates = pre_rates if self.renewal.link_type == 'log' else safe_log(rates)
        log_rho_t = log_rates + (log_renewal_density - log_survival).transpose(0, 2, 1)
        return log_rho_t  # (num_samps, out_dims, ts)
    
    
    def sample_instantaneous_renewal(
        self, prng_state, t_eval, x_cond, jitter, num_samps=20, prior=True
    ):
        """
        :param jnp.ndarray t_eval: evaluation times since last spike, i.e. ISI (ts,)
        :param jnp.ndarray x_cond: evaluation locations (num_samps, obs_dims, x_dims)
        """
        f_samples = self._gp_sample(
            prng_state, x_cond[..., None, :], prior, jitter
        )  # (num_samps, obs_dims, 1)
        
        f_samples += self.gp_mean[None]
        tau_eval = (
            self.renewal.inverse_link(f_samples) * t_eval / self.renewal.shape_scale()[:, None]
        )
        
        log_dens = vmap(vmap(self.renewal.log_density), 2, 2)(tau_eval)
        return jnp.exp(log_dens)  # (num_samps, obs_dims, ts)
    
    def sample_prior(self, prng_state, num_samps, x_samples, ini_spikes, ini_tau, jitter):
        """
        Sample from the generative model
        Sample spike trains from the modulated renewal process.
        :return:
            pike train of shape (trials, neuron, timesteps)
        """
        prng_states = jr.split(prng_state, 2)
        
        f_samples = self._gp_sample(
            prng_states[0], x_samples, True, jitter
        )  # (num_samps, obs_dims, 1)

        y_samples, log_rho_ts = self._sample_spikes(
            prng_states[1], ini_spikes, ini_tau, f_samples)
        return y_samples, log_rho_ts

    def sample_posterior(self, prng_state, num_samps, x_samples, ini_spikes, ini_tau, jitter):
        """
        Sample from posterior predictive
        """
        prng_states = jr.split(prng_state, 2)
        
        f_samples = self._gp_sample(
            prng_states[0], x_samples, False, jitter
        )  # (num_samps, obs_dims, 1)

        y_samples, log_rho_ts = self._sample_spikes(
            prng_states[1], ini_spikes, ini_tau, f_samples)
        return y_samples, log_rho_ts
    
    
    
class ModulatedRenewal(FilterObservations):
    """
    Renewal likelihood, interaction with GP modulator
    """

    gp: SparseGP
    gp_mean: jnp.ndarray
    renewal: RenewalLikelihood

    def __init__(self, gp, gp_mean, renewal, spikefilter=None):
        # checks
        assert renewal.array_type == gp.array_type
        assert renewal.f_dims == gp.kernel.out_dims

        super().__init__(spikefilter, gp.array_type)
        self.gp = gp
        self.renewal = renewal
        
    def _gp_sample(self, prng_state, x_eval, prior, jitter):
        """
        Obtain the log conditional intensity given input path along which to evaluate

        :param jnp.ndarray x_eval: evaluation locs (num_samps, out_dims, eval_locs, in_dims)
        """
        if prior:
            f_samples = self.gp.sample_prior(
                prng_state, x_eval, jitter
            )  # (samp, f_dim, evals)

        else:
            f_samples, _ = self.gp.sample_posterior(
                prng_state, x_eval, jitter, compute_KL=False
            )  # (samp, f_dim, evals)

        return f_samples
    
    def _log_rho_from_gp_sample(self, prng_state, num_samps, tau_eval, x_eval, prior, jitter):
        """
        Obtain the log conditional intensity given input path along which to evaluate
        
        :param jnp.ndarray tau_eval: evaluation time locs (out_dims, locs)
        :param jnp.ndarray isi_eval: higher order ISIs (out_dims, locs, order)
        :param jnp.ndarray x_eval: external covariates (out_dims, locs, x_dims)
        :return:
            log intensity in rescaled time tau (num_samps, out_dims, locs)
        """
        pre_modulator = self._gp_sample(prng_state, x_eval, prior, jitter)
        log_hazard = self.renewal.log_density(tau_eval) - self.renewal.log_survival(tau_eval)
        
        pre_modulator += self.gp_mean[None]
        log_modulator = pre_modulator if self.renewal.link_type == 'log' else safe_log(
            self.renewal.inverse_link(pre_modulator))
        
        log_rho_t = log_modulator + log_hazard
        return log_rho_t
    
    def _log_rho_from_gp_post(self, prng_state, num_samps, tau_eval, isi_eval, x_eval, mean_only, compute_KL, jitter):
        """
        Obtain the log conditional intensity given input path along which to evaluate
        
        :param jnp.ndarray tau_eval: evaluation time locs (out_dims, locs)
        :param jnp.ndarray isi_eval: higher order ISIs (out_dims, locs, order)
        :param jnp.ndarray x_eval: external covariates (out_dims, locs, x_dims)
        :return:
            log intensity in rescaled time tau (num_samps, out_dims, locs)
        """
        if self.modulated:
            cov_eval = self._combine_input(isi_eval, x_eval)
            
            f_mean, f_cov, KL = self.gp.evaluate_posterior(
                tau_eval, cov_eval, mean_only=False, compute_KL=False, jitter=jitter
            )  # (out_dims, time, 1)

        else:
            f_mean, f_cov, KL = self.gp.evaluate_posterior(
                tau_eval, mean_only=False, compute_KL=False, jitter=jitter
            )  # (out_dims, time, 1)

        m_eval = vmap(self._refractory_mean, 1, 1)(tau_eval)  # (out_dims, locs)
        log_rho_tau_mean = f_mean[..., 0] + m_eval
        return log_rho_tau_mean, f_cov[..., 0], KL
    
    def _sample_spikes(self, prng_state, ini_spikes, t_since, f_samples):
        """
        Sample the spike train autoregressively
        
        :param jnp.dnarray ini_spikes: initial spike train (num_samps, obs_dims, filter_length)
        :param jnp.ndarray t_since: initial tau values at start (num_samps, obs_dims)
        """
        num_samps, obs_dims, ts = f_samples.shape
        if self.spikefilter is not None:
            prng_states = jr.split(prng_state, ts*2).reshape(ts, 2, -1)
        else:
            prng_states = jr.split(prng_state, ts).reshape(ts, 1, -1)

        def step(carry, inputs):
            prng_state, f = inputs  # (num_samps, obs_dims)
            t_since, past_spikes, spikes = carry
            
            if self.spikefilter is not None:
                h, _ = self.spikefilter.apply_filter(prng_state[0], past_spikes, False)
                f += h[..., 0]
                
            f += self.gp_mean[None]
            log_modulator = f if self.renewal.link_type == 'log' else safe_log(
                self.renewal.inverse_link(f))

            t_since = jnp.where(
                (spikes > 0),  # spikes
                self.renewal.dt, 
                t_since + self.renewal.dt, 
            )  # (num_samps, obs_dims)
            
            log_renewal_density = self.renewal.log_density(t_since)
            log_survival = self.renewal.log_survival(t_since)

            log_rho_t = log_modulator + log_renewal_density - log_survival
            p_spike = jnp.minimum(jnp.exp(log_rho_t) * self.renewal.dt, 1.)  # approximate by discrete Bernoulli
            spikes = jr.bernoulli(prng_state[-1], p_spike).astype(self.array_type)  # (num_samps, obs_dims)
            
            if self.spikefilter is not None:
                past_spikes = jnp.concatenate((past_spikes[..., 1:], spikes[..., None]), axis=-1)
            
            return (t_since, past_spikes, spikes), (spikes, log_rho_t)

        init = (t_since, ini_spikes, jnp.zeros((num_samps, obs_dims)))
        xs = (prng_states, f_samples.transpose(2, 0, 1))
        _, (spikes, log_rho_ts) = lax.scan(step, init, xs)
        return spikes.transpose(1, 2, 0), log_rho_ts.transpose(1, 2, 0)  # (num_samps, obs_dims, ts)

    def _log_hazard(
        self, y, pre_rates, 
    ):
        """
        Time since spike component

        :param jnp.ndarray y: binary spike train (obs_dims, ts)
        :param jnp.ndarray pre_rates: pre-link rates (obs_dims, ts)
        :param List spiketimes: list of spike time indices arrays per neuron
        :param jnp.ndarray covariates: covariates time series (mc, obs_dims, ts, in_dims)
        """
        rates = self.renewal.inverse_link(pre_rates)
        log_rates = pre_rates if self.renewal.link_type == "log" else safe_log(rates)
        spikes = (y.T > 0)
        
        def step(carry, inputs):
            prng_state, f = inputs  # (num_samps, obs_dims)
            t_since, past_spikes, spikes = carry
            
            f += self.gp_mean[None]
            log_modulator = f if self.renewal.link_type == 'log' else safe_log(
                self.renewal.inverse_link(f))

            t_since = jnp.where(
                (spikes > 0),  # spikes
                self.renewal.dt, 
                t_since + self.renewal.dt, 
            )  # (num_samps, obs_dims)
            
            log_renewal_density = self.renewal.log_density(t_since)
            log_survival = self.renewal.log_survival(t_since)

            log_rho_t = log_modulator + log_renewal_density - log_survival
            
            
            return (t_since, past_spikes, spikes), (spikes, log_rho_t)
        
        ll = log_rates + log_renewals  # (obs_dims, ts)
        return ll
    
    ### inference ###
    def ELBO(self, prng_state, y, y_filt, pre_rates, x, neuron, num_ISIs):
        """
        Compute the evidence lower bound
        """
        log_rho_tau, KL = self._log_rho_from_gp_post()
        
        Ell = self.pp.variational_expectation(
            prng_state,
            y,
            log_rho_tau_mean,
            log_rho_tau_cov,
            jitter,
            approx_int_method,
        )

        ELBO = Ell - KL_f - KL_y
        return ELBO
    
    ### evaluation ###
    def evaluate_metric(self):
        return
    
    ### sample ###
    def sample_log_conditional_intensity(self, prng_state, x_eval, y, y_filt, jitter):
        """
        rho(t) = r(t) p(u) / (1 - int p(u) du)
        
        :param jnp.ndarray x_eval: evaluation locations (num_samps, obs_dims, ts, x_dims)
        :param jnp.ndarray y: spike train corresponding to segment (neurons, ts), 
                              y = y_filt[..., filter_length:] + last time step
        """
        pre_rates = self._gp_sample(prng_state, x_eval, False, jitter)  # (num_samps, obs_dims, ts)
        num_samps = pre_rates.shape[0]
        y = jnp.broadcast_to(y, (num_samps,) + y.shape[1:])
        
        if self.spikefilter is not None:
            h, _ = self.spikefilter.apply_filter(prng_state[0], y_filt, False)
            pre_rates += h
            
        pre_rates += self.gp_mean[None]
        rates = self.renewal.inverse_link(pre_rates).transpose(0, 2, 1)
        
        # rate rescaling, rescaled time since last spike
        spikes = (y > 0).transpose(0, 2, 1)
        _, tau_since_spike = vmap(self._rate_rescale, (0, 0, None, None), (None, 0))(
            spikes, rates, False, True)  # (num_samps, ts, obs_dims)
        
        log_renewal_density = self.renewal.log_density(tau_since_spike)
        log_survival = self.renewal.log_survival(tau_since_spike)
        
        log_rates = pre_rates if self.renewal.link_type == 'log' else safe_log(rates)
        log_rho_t = log_rates + (log_renewal_density - log_survival).transpose(0, 2, 1)
        return log_rho_t  # (num_samps, out_dims, ts)
    
    
    def sample_instantaneous_renewal(
        self, prng_state, t_eval, x_cond, jitter, num_samps=20, prior=True
    ):
        """
        :param jnp.ndarray t_eval: evaluation times since last spike, i.e. ISI (ts,)
        :param jnp.ndarray x_cond: evaluation locations (num_samps, obs_dims, x_dims)
        """
        f_samples = self._gp_sample(
            prng_state, x_cond[..., None, :], prior, jitter
        )  # (num_samps, obs_dims, 1)
        
        f_samples += self.gp_mean[None]
        tau_eval = (
            self.renewal.inverse_link(f_samples) * t_eval / self.renewal.shape_scale()[:, None]
        )
        
        log_dens = vmap(vmap(self.renewal.log_density), 2, 2)(tau_eval)
        return jnp.exp(log_dens)  # (num_samps, obs_dims, ts)
    
    def sample_prior(self, prng_state, num_samps, x_samples, ini_spikes, ini_tau, jitter):
        """
        Sample from the generative model
        Sample spike trains from the modulated renewal process.
        :return:
            pike train of shape (trials, neuron, timesteps)
        """
        prng_states = jr.split(prng_state, 2)
        
        f_samples = self._gp_sample(
            prng_states[0], x_samples, True, jitter
        )  # (num_samps, obs_dims, 1)

        y_samples, log_rho_ts = self._sample_spikes(
            prng_states[1], ini_spikes, ini_tau, f_samples)
        return y_samples, log_rho_ts

    def sample_posterior(self, prng_state, num_samps, x_samples, ini_spikes, ini_tau, jitter):
        """
        Sample from posterior predictive
        """
        prng_states = jr.split(prng_state, 2)
        
        f_samples = self._gp_sample(
            prng_states[0], x_samples, False, jitter
        )  # (num_samps, obs_dims, 1)

        y_samples, log_rho_ts = self._sample_spikes(
            prng_states[1], ini_spikes, ini_tau, f_samples)
        return y_samples, log_rho_ts

    


class NonparametricPointProcess(Observations):
    """
    Bayesian nonparametric modulated point process likelihood
    """

    gp: Union[MultiOutputLTI, KroneckerLTI]
    pp: PointProcess
        
    modulated: bool

    warp_tau: jnp.ndarray
    refract_tau: jnp.ndarray
    refract_neg: float
    mean_bias: jnp.ndarray

    def __init__(self, gp, warp_tau, refract_tau, refract_neg, mean_bias, dt):
        """
        :param jnp.ndarray warp_tau: time transform timescales of shape (out_dims,)
        :param jnp.ndarray tau: refractory mean timescales of shape (out_dims,)
        """
        super().__init__(gp.array_type)
        self.gp = gp
        self.pp = LogCoxProcess(gp.kernel.out_dims, dt, self.array_type)

        self.warp_tau = self._to_jax(warp_tau)
        self.refract_tau = self._to_jax(refract_tau)
        self.refract_neg = refract_neg
        self.mean_bias = self._to_jax(mean_bias)
        
        self.modulated = False if type(gp) == MultiOutputLTI else True

    ### functions ###
    def _log_time_transform(self, t, inverse):
        """
        Inverse transform is from tau [0, 1] to t in R
        """
        if inverse:
            t_ = -jnp.log(1 - t) * self.warp_tau
        else:
            s = jnp.exp(-t / self.warp_tau)
            t_ = 1 - s
            
        return t_
    
    def _log_time_transform_jac(self, t, inverse):
        """
        Inverse transform is from tau [0, 1] to t in R
        """
        t_ = self._log_time_transform(t, inverse)
        
        if inverse:
            log_jac = jnp.log(self.warp_tau) - jnp.log(1 - t)  # warp_tau / (1 - t)
        else:
            log_jac = -t / self.warp_tau - jnp.log(self.warp_tau)  # s / warp_tau
            
        return t_, log_jac

    def _refractory_mean(self, tau):
        """
        Refractory period implemented by GP mean function
        
        :param jnp.ndarray tau: (out_dims,)
        """
        return self.refract_neg * (tau + 1.) ** (-self.warp_tau / self.refract_tau) + self.mean_bias
    
    def _combine_input(self, isi_eval, x_eval):
        cov_eval = []
        if isi_eval is not None:
            tau_isi_eval = vmap(vmap(self._log_time_transform_jac, (1, None), 1), (1, None), 1)(
                isi_eval, False)  # (out_dims, ts, order)
            cov_eval.append(tau_isi_eval)

        if x_eval is not None:
            cov_eval.append(x_eval)

        return jnp.concatenate(cov_eval, axis=-1)

    def _log_rho_from_gp_sample(self, prng_state, num_samps, tau_eval, isi_eval, x_eval, prior, jitter):
        """
        Obtain the log conditional intensity given input path along which to evaluate
        
        :param jnp.ndarray tau_eval: evaluation time locs (out_dims, locs)
        :param jnp.ndarray isi_eval: higher order ISIs (out_dims, locs, order)
        :param jnp.ndarray x_eval: external covariates (out_dims, locs, x_dims)
        :return:
            log intensity in rescaled time tau (num_samps, out_dims, locs)
        """
        if self.modulated:
            cov_eval = self._combine_input(isi_eval, x_eval)
            
            if prior:
                f_samples = self.gp.sample_prior(
                    prng_state, num_samps, tau_eval, cov_eval, jitter
                )

            else:
                f_samples, _ = self.gp.sample_posterior(
                    prng_state, num_samps, tau_eval, cov_eval, jitter, False
                )  # (tr, time, N, 1)

        else:
            if prior:
                f_samples = self.gp.sample_prior(
                    prng_state, num_samps, tau_eval, jitter
                )

            else:
                f_samples, _ = self.gp.sample_posterior(
                    prng_state, num_samps, tau_eval, jitter, False
                )  # (tr, time, N, 1)

        m_eval = vmap(self._refractory_mean, 1, 1)(tau_eval)  # (out_dims, locs)
        log_rho_tau = f_samples[..., 0] + m_eval
        return log_rho_tau
    
    def _log_rho_from_gp_post(self, prng_state, num_samps, tau_eval, isi_eval, x_eval, mean_only, compute_KL, jitter):
        """
        Obtain the log conditional intensity given input path along which to evaluate
        
        :param jnp.ndarray tau_eval: evaluation time locs (out_dims, locs)
        :param jnp.ndarray isi_eval: higher order ISIs (out_dims, locs, order)
        :param jnp.ndarray x_eval: external covariates (out_dims, locs, x_dims)
        :return:
            log intensity in rescaled time tau (num_samps, out_dims, locs)
        """
        if self.modulated:
            cov_eval = self._combine_input(isi_eval, x_eval)
            
            f_mean, f_cov, KL = self.gp.evaluate_posterior(
                tau_eval, cov_eval, mean_only=False, compute_KL=False, jitter=jitter
            )  # (out_dims, time, 1)

        else:
            f_mean, f_cov, KL = self.gp.evaluate_posterior(
                tau_eval, mean_only=False, compute_KL=False, jitter=jitter
            )  # (out_dims, time, 1)

        m_eval = vmap(self._refractory_mean, 1, 1)(tau_eval)  # (out_dims, locs)
        log_rho_tau_mean = f_mean[..., 0] + m_eval
        return log_rho_tau_mean, f_cov[..., 0], KL
    
    def _sample_spikes(self, prng_state, timesteps, ini_t_since, past_ISIs, x_eval, jitter):
        """
        Sample the spike train autoregressively
        
        # :param jnp.dnarray ini_spikes: initial spike train (num_samps, obs_dims, filter_length)
        :param jnp.ndarray ini_t: initial tau values at start (num_samps, obs_dims)
        :param jnp.ndarray past_ISI: past ISIs (num_samps, obs_dims, order)
        :param jnp.ndarray x_eval: covariates (num_samps, obs_dims, ts, x_dims)
        """
        num_samps = ini_t_since.shape[0]
        prng_state = jr.split(prng_state, num_samps+1)
        prng_gp, prng_state = prng_state[:-1], prng_state[-1]
        
        if self.spikefilter is not None:
            prng_states = jr.split(prng_state, timesteps*2).reshape(timesteps, 2, -1)
        else:
            prng_states = jr.split(prng_state, timesteps).reshape(timesteps, 1, -1)
            
        def step(carry, inputs):
            prng_state, x = inputs  # (num_samps, obs_dims, x_dims)
            t_since, past_ISI, spikes = carry
            
            t_since += self.pp.dt
                
            tau_since, log_dtau_dt = vmap(self._log_time_transform_jac, (0, None), (0, 0))(
                t_since, False)  # (num_samps, out_dims)

            log_rho_tau = vmap(self._log_rho_from_gp_sample, 
                               (0, None, 0, None if past_ISI is None else 0, None if x is None else 0, None, None), 0)(
                prng_gp, 1, tau_since[..., None], past_ISI, x, False, jitter)[..., 0]
            log_rho_t = log_rho_tau + log_dtau_dt  # (num_samps, out_dims)
            
            p_spike = jnp.minimum(jnp.exp(log_rho_t) * self.pp.dt, 1.)  # approximate by discrete Bernoulli
            spikes = jr.bernoulli(prng_state[-1], p_spike).astype(self.array_type)  # (num_samps, obs_dims)
            
            # spike reset
            spike_cond = (spikes > 0)  # (num_samps, obs_dims)
            if past_ISI is not None:
                shift_ISIs = jnp.concatenate((t_since[..., None], past_ISI[..., 0, :-1]), axis=-1)
                past_ISI = jnp.where(spike_cond[..., None], shift_ISIs, past_ISI[..., 0, :])[..., None, :]
            t_since = jnp.where(spike_cond, 0.0, t_since)

            return (t_since, past_ISI, spikes), (spikes, log_rho_t)

        # add dummy time dimension
        if x_eval is not None:
            x_eval = x_eval.transpose(2, 0, 1, 3)[..., None, :]
        if past_ISIs is not None:
            past_ISIs = past_ISIs[..., None, :]
        
        init = (ini_t_since, past_ISIs, jnp.zeros_like(ini_t_since))
        xs = (prng_states, x_eval)
        _, (spikes, log_rho_ts) = lax.scan(step, init, xs)
        return spikes.transpose(1, 2, 0), log_rho_ts.transpose(1, 2, 0)  # (num_samps, obs_dims, ts)
    

    ### inference ###
    def ELBO(self, prng_state, x_samples, jitter, approx_int_method):
        log_rho_tau_mean, log_rho_tau_cov, KL = self._log_rho_from_gp_post()
        
        Ell = self.pp.variational_expectation(
            prng_state,
            y,
            log_rho_tau_mean,
            log_rho_tau_cov,
            jitter,
            approx_int_method,
        )

        ELBO = Ell - KL
        return ELBO
    
    ### evaluation ###
    def evaluate_log_conditional_intensity(self, prng_state, num_samps, t_eval, isi_eval, x_eval, jitter):
        """
        Evaluate the conditional intensity along an input path
        """
        tau_eval, log_dtau_dt_eval = vmap(self._log_time_transform_jac, (1, None), (1, 1))(
            t_eval[None, :], False)  # (out_dims, ts)
        
        log_rho_tau_mean, log_rho_tau_cov, _ = self._log_rho_from_gp_post(
            prng_state, num_samps, tau_eval, isi_eval, x_eval, False, False, jitter)
        log_rho_t_mean = log_rho_tau_mean + log_dtau_dt_eval  # (num_samps, out_dims, ts)
        log_rho_t_cov = log_rho_tau_cov  # additive transform in log space
        
        return log_rho_t_mean, log_rho_t_cov

    def evaluate_metric(self):
        return
    
    def evaluate_conditional_renewal_expectation(
        self, 
        func_of_t, 
        isi_cond, 
        x_cond,
        int_eval_pts=1000,
        num_quad_pts=100,
        jitter=1e-6,
    ):
        return

    ### sample ###
    def sample_conditional_renewal(
        self,
        prng_state, 
        num_samps,
        t_eval,
        isi_cond, 
        x_cond,
        int_eval_pts=1000,
        num_quad_pts=100,
        prior=True,
        jitter=1e-6,
    ):
        """
        Compute the instantaneous renewal density with rho(ISI;X) from model
        Uses linear interpolation with Gauss-Legendre quadrature for integrals

        :param jnp.ndarray t_eval: evaluation time points (eval_locs,)
        """

        def quad_integrate(a, b, sigma_pts, weights):
            # transform interval
            sigma_pts = 0.5 * (sigma_pts + 1) * (b - a) + a
            weights *= jnp.prod(0.5 * (b - a))
            return sigma_pts, weights
        
        vvinterp_ = vmap(vmap(jnp.interp, (None, None, 0), 0), (None, None, 0), 0)  # vmap mc, then out_dims
        vvinterp = vmap(vmap(jnp.interp, (None, 0, 0), 0), (None, None, 0), 0)  # vmap mc, then out_dims
        vvvinterp = vmap(vmap(vmap(jnp.interp, (0, None, 0), 0), 
                              (None, None, 0), 0), 
                         (1, None, None), 2)  # vmap mc, then out_dims, then cubature
        vvquad_integrate = vmap(vmap(quad_integrate, (None, 0, None, None), (0, 0)), 
                                (None, 0, None, None), (0, 0))  # vmap out_dims and locs

        # compute rho(tau) for integral
        tau_pts = jnp.linspace(0.0, 1.0, int_eval_pts)
        t_pts, log_dt_dtau_pts = vmap(self._log_time_transform_jac, (1, None), (1, 1))(
            tau_pts[None, :], True)  # (out_dims, locs)

        log_rho_tau_pts = self._log_rho_from_gp_sample(
            prng_state, num_samps, tau_pts[None, :], isi_cond, x_cond, prior, jitter)
        log_rho_t_pts = log_rho_tau_pts - log_dt_dtau_pts  # (num_samps, out_dims, taus)

        # compute rho(tau) and rho(t)
        rho_tau_pts = jnp.exp(log_rho_tau_pts)
        rho_t_pts = jnp.exp(log_rho_t_pts)

        # evaluation locs
        tau_eval = vmap(self._log_time_transform, (1, None), 1)(
            t_eval[None, :], False)  # (out_dims, locs)
        rho_t_eval = vvinterp(t_eval, t_pts, rho_t_pts)
        
        # compute cumulative intensity int rho(tau) dtau
        sigma_pts, weights = gauss_legendre(1, num_quad_pts)
        locs, ws = vvquad_integrate(
            0.0, tau_eval, sigma_pts, weights
        )  # (out_dims, taus, cub_pts, 1)
        
        quad_rho_tau_eval = vvvinterp(
            locs[..., 0], tau_pts, rho_tau_pts
        )  # num_samps, out_dims, taus, cub_pts
        int_rho_tau_eval = (quad_rho_tau_eval * ws).sum(-1)

        # normalizer
        locs, ws = quad_integrate(0.0, 1.0, sigma_pts, weights)
        quad_rho = vvinterp_(locs[:, 0], tau_pts, rho_tau_pts)
        int_rho = (quad_rho * ws).sum(-1)
        normalizer = 1.0 - jnp.exp(-int_rho)  # (num_samps, out_dims)

        # compute renewal density
        exp_negintrho_t = jnp.exp(-int_rho_tau_eval)  # num_samps, out_dims, taus
        renewal_density = rho_t_eval * exp_negintrho_t / normalizer[..., None]
        return renewal_density
    
    def sample_prior(
        self, 
        prng_state, 
        num_samps: int, 
        timesteps: int, 
        x_samples: Union[None, jnp.ndarray], 
        ini_t_since: jnp.ndarray, 
        past_ISIs: Union[None, jnp.ndarray], 
        jitter: float, 
    ):
        """
        Sample from the generative model
        Sample spike trains from the modulated renewal process.
        :return:
            pike train of shape (trials, neuron, timesteps)
        """
        prng_states = jr.split(prng_state, 2)
        
        y_samples, log_rho_ts = self._sample_spikes(
            prng_states[1], timesteps, ini_t_since, past_ISIs, x_samples, jitter)
        return y_samples, log_rho_ts

    def sample_posterior(
        self, 
        prng_state, 
        num_samps: int, 
        timesteps: int, 
        x_samples: Union[None, jnp.ndarray], 
        ini_t_since: jnp.ndarray, 
        past_ISIs: Union[None, jnp.ndarray], 
        jitter: float, 
    ):
        """
        Sample from posterior predictive
        """
        prng_states = jr.split(prng_state, 2)
        
        y_samples, log_rho_ts = self._sample_spikes(
            prng_states[1], timesteps, ini_t_since, past_ISIs, x_samples, jitter)
        return y_samples, log_rho_ts