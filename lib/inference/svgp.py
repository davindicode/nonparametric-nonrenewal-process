from functools import partial
from typing import Union

import equinox as eqx

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import vmap, lax

import numpy as np

from .base import Observations, FilterObservations
from .timeseries import GaussianLatentObservedSeries

from ..base import module, ArrayTypes_

from ..GP.markovian import IndependentLTI
from ..GP.sparse import SparseGP
from ..GP.spatiotemporal import KroneckerLTI
from ..likelihoods.base import FactorizedLikelihood, RenewalLikelihood, LinkTypes
from ..likelihoods.factorized import PointProcess
from ..utils.linalg import gauss_legendre, gauss_quad_integrate
from ..utils.jax import safe_log, safe_sqrt


class SparseGPFilterObservations(FilterObservations):
    """
    Uses SVGP to model modulation
    
    Option to learn mean
    """
    
    gp: SparseGP
    gp_mean: Union[jnp.ndarray, None]  # constant mean (obs_dims,), i.e. bias if not None
        
    def __init__(self, gp, gp_mean, spikefilter):
        super().__init__(spikefilter, ArrayTypes_[gp.array_type])
        self.gp = gp
        self.gp_mean = self._to_jax(gp_mean) if gp_mean is not None else None
    
    def apply_constraints(self):
        """
        Constrain parameters in optimization
        """
        model = super().apply_constraints()
        model = eqx.tree_at(
            lambda tree: tree.gp,
            model,
            replace_fn=lambda obj: obj.apply_constraints(),
        )

        return model
    
    def _gp_sample(self, prng_state, x_eval, prior, compute_KL, jitter):
        """
        Sample from Gaussian process with mean function

        :param jnp.ndarray x_eval: evaluation locs (num_samps, out_dims, eval_locs, in_dims)
        """
        if prior:
            f_samples = self.gp.sample_prior(
                prng_state, x_eval, jitter
            )  # (num_samps, out_dims, time)
            KL = 0

        else:
            f_samples, KL = self.gp.sample_posterior(
                prng_state, x_eval, jitter, compute_KL
            )  # (num_samps, out_dims, time)

        if self.gp_mean is not None:
            f_samples += self.gp_mean[None, :, None]
        
        return f_samples, KL
    
    def _gp_posterior(self, x_samples, mean_only, diag_cov, compute_KL, jitter):
        """
        Evaluate Gaussian process posterior with mean function
        """
        f_mean, f_cov, KL, _ = self.gp.evaluate_posterior(
            x_samples, mean_only, diag_cov, compute_KL, False, jitter
        )  # (num_samps, out_dims, time, 1 or time)
        
        if self.gp_mean is not None:
            f_mean += self.gp_mean[None, :, None, None]
            
        return f_mean, f_cov, KL



class ModulatedFactorized(SparseGPFilterObservations):
    """
    Factorization across time points allows one to rely on latent marginals
    """
    
    likelihood: FactorizedLikelihood

    def __init__(self, gp, gp_mean, likelihood, spikefilter=None):
        # checks
        assert likelihood.array_type == gp.array_type
        assert likelihood.f_dims == gp.kernel.out_dims

        super().__init__(gp, gp_mean, spikefilter)
        self.likelihood = likelihood
        
    def apply_constraints(self):
        """
        Constrain parameters in optimization
        """
        model = super().apply_constraints()
        model = eqx.tree_at(
            lambda tree: tree.likelihood,
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
                
                y = vmap(self.likelihood.sample_Y)(prng_keys[:-1], f)  # vmap over MC
                past_Y = jnp.concatenate((past_Y[..., 1:], y[..., None]), axis=-1)

                return past_Y, (y, f)

            init = ini_Y
            xs = (prng_states, f_samples.transpose(2, 0, 1))
            _, (Y, f_filtered) = lax.scan(step, init, xs)
            return Y.transpose(1, 2, 0), f_filtered.transpose(1, 2, 0)  # (num_samps, obs_dims, ts)
        
        else:
            prng_states = jr.split(prng_state, ts*num_samps).reshape(num_samps, ts, -1)
            Y = vmap(vmap(self.likelihood.sample_Y), (1, 2), 2)(prng_states, f_samples)
            return Y, f_samples

    ### variational inference ###
    def VI(
        self,
        prng_state,
        num_samps, 
        jitter, 
        xs, 
        ys, 
        ys_filt, 
        lik_int_method, 
    ):
        """
        Compute variational expectation of likelihood and KL divergence
        
        :param jnp.ndarray x: inputs (num_samps, out_dims, ts, x_dims)
        :param jnp.ndarray y: observations (obs_dims, ts)
        """
        f_mean, f_cov, KL = self._gp_posterior(
            xs, mean_only=False, diag_cov=True, compute_KL=True, jitter=jitter
        )  # (num_samps, out_dims, ts, 1)

        if self.spikefilter is not None:
            y_filtered, KL_y = self.spikefilter.apply_filter(prng_state, ys_filt[None, ...], compute_KL=True)
            prng_state, _ = jr.split(prng_state)
            f_mean += y_filtered[..., None]
            KL += KL_y
        
        f_mean = f_mean.transpose(0, 2, 1, 3)  # (num_samps, ts, out_dims, 1)
        f_cov = vmap(vmap(jnp.diag))(f_cov[..., 0].transpose(0, 2, 1))  # (num_samps, ts, out_dims, out_dims)
        ### TODO: add possible block diagonal structure by linear mappings of f_cov
        llf = lambda y, m, c: self.likelihood.variational_expectation(
            y, m, c, prng_state, jitter, lik_int_method)
        Ell = vmap(vmap(llf), (None, 0, 0))(ys.T, f_mean, f_cov).mean()  # vmap and take mean over num_samps and ts

        return Ell, KL
    
    ### evaluation ###
    def evaluate_pre_conditional_rate(self, prng_state, x_eval, y_filt, jitter):
        """
        evaluate posterior rate
        """
        pre_rates_mean, pre_rates_cov, _ = self._gp_posterior(
            x_eval, mean_only=False, diag_cov=False, compute_KL=False, jitter=jitter)
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
        
        f_samples, _ = self._gp_sample(prng_states[1], x_samples, True, False, jitter)  # (samp, evals, f_dim)
        y_samples, filtered_f = self._sample_Y(prng_states[2], ini_Y, f_samples)
        
        return y_samples, filtered_f

    def sample_posterior(self, prng_state, num_samps, x_samples, ini_Y, jitter):
        """
        Sample from posterior predictive
        """
        prng_states = jr.split(prng_state, 3)
        
        f_samples, _ = self._gp_sample(prng_states[1], x_samples, False, False, jitter)  # (samp, evals, f_dim)
        y_samples, filtered_f = self._sample_Y(prng_states[2], ini_Y, f_samples)
        
        return y_samples, filtered_f
    
    
    
class ModulatedRenewal(SparseGPFilterObservations):
    """
    Renewal likelihood, interaction with GP modulator
    """
    
    renewal: RenewalLikelihood
    pp: PointProcess

    def __init__(self, gp, gp_mean, renewal, spikefilter=None):
        # checks
        assert renewal.array_type == gp.array_type
        assert renewal.f_dims == gp.kernel.out_dims

        super().__init__(gp, gp_mean, spikefilter)
        self.renewal = renewal
        self.pp = PointProcess(gp.kernel.out_dims, renewal.dt, "log", ArrayTypes_[gp.array_type])
        
    def apply_constraints(self):
        """
        Constrain parameters in optimization
        """
        model = super().apply_constraints()
        model = eqx.tree_at(
            lambda tree: tree.renewal,
            model,
            replace_fn=lambda obj: obj.apply_constraints(),
        )

        return model
    
    def _sample_spikes(self, prng_state, ini_spikes, ini_tau, f_samples):
        """
        Sample the spike train autoregressively
        
        :param jnp.dnarray ini_spikes: initial spike train (num_samps, obs_dims, filter_length)
        :param jnp.ndarray tau: initial tau values at start (num_samps, obs_dims)
        """
        num_samps, obs_dims, ts = f_samples.shape
        if self.spikefilter is not None:
            prng_states = jr.split(prng_state, ts*2).reshape(ts, 2, -1)
        else:
            prng_states = jr.split(prng_state, ts).reshape(ts, 1, -1)

        def step(carry, inputs):
            prng_state, f = inputs  # (num_samps, obs_dims)
            tau, past_spikes = carry
            
            # compute current tau
            tau = jnp.where(
                (past_spikes[..., -1] > 0),  # spikes
                self.renewal.dt, 
                tau + self.renewal.dt, 
            )  # (num_samps, obs_dims)
            
            # compute intensity
            if self.spikefilter is not None:
                h, _ = self.spikefilter.apply_filter(prng_state[0], past_spikes, False)
                f += h[..., 0]
                
            log_modulator = f if self.renewal.link_type == LinkTypes['log'] else safe_log(
                self.renewal.inverse_link(f))
            log_hazard = self.renewal.log_hazard(tau)
            log_rho_t = log_modulator + log_hazard
            
            # generate spikes
            p_spike = jnp.minimum(jnp.exp(log_rho_t) * self.renewal.dt, 1.)  # approximate by discrete Bernoulli
            spikes = jr.bernoulli(prng_state[-1], p_spike).astype(self.array_dtype())  # (num_samps, obs_dims)
            
            if self.spikefilter is not None:
                past_spikes = jnp.concatenate((past_spikes[..., 1:], spikes[..., None]), axis=-1)
            else:
                past_spikes = spikes[..., None]
            
            return (tau, past_spikes), (spikes, log_rho_t)

        init = (ini_tau, ini_spikes)
        xs = (prng_states, f_samples.transpose(2, 0, 1))
        _, (spikes, log_rho_ts) = lax.scan(step, init, xs)
        return spikes.transpose(1, 2, 0), log_rho_ts.transpose(1, 2, 0)  # (num_samps, obs_dims, ts)
    
    def _log_hazard(
        self, spikes, ini_tau, 
    ):
        """
        Time since spike component
        
        ini_tau zero means spike occured right before start

        :param jnp.ndarray spikes: binary spike train (obs_dims, ts)
        :param jnp.ndarray ini_tau: initial times since (obs_dims,)
        """
        spikes = (y.T > 0)
        
        def step(carry, inputs):
            spikes = inputs  # (obs_dims,)
            tau = carry  # (obs_dims,)
            
            # compute current tau
            tau = jnp.where(
                (spikes > 0),  # spikes
                self.renewal.dt, 
                tau + self.renewal.dt, 
            )  # (num_samps, obs_dims)

            log_hazard = self.renewal.log_hazard(tau)
            return tau, log_hazard
        
        init = ini_tau
        _, log_hazards = lax.scan(step, init=init, xs=spikes)
        return log_hazards
    
    ### inference ###
    def VI(self, prng_state, num_samps, jitter, xs, taus, ys, ys_filt, lik_int_method):
        """
        Compute variational expectation of likelihood and KL divergence
        """
        f_mean, f_cov, KL = self._gp_posterior(
            xs, mean_only=False, diag_cov=True, compute_KL=True, jitter=jitter, 
        )  # (num_samps, out_dims, ts, 1)

        if self.spikefilter is not None:
            y_filtered, KL_y = self.spikefilter.apply_filter(prng_state, ys_filt[None, ...], compute_KL=True)
            prng_state, _ = jr.split(prng_state)
            f_mean += y_filtered[..., None]
            KL += KL_y
            
        f_mean = f_mean.transpose(0, 2, 1, 3)  # (num_samps, ts, out_dims, 1)
        f_cov = vmap(vmap(jnp.diag))(f_cov[..., 0].transpose(0, 2, 1))  # (num_samps, ts, out_dims, out_dims)
        
        log_modulator = f_mean if self.renewal.link_type == LinkTypes['log'] else safe_log(
                self.renewal.inverse_link(f_mean))
        log_hazard = vmap(self.renewal.log_hazard)(taus.T)
        log_rho_t = log_modulator + log_hazard[None, ..., None]
        
        llf = lambda y, m, c: self.pp.variational_expectation(
            y, m, c, prng_state, jitter, lik_int_method)
        Ell = vmap(vmap(llf), (None, 0, 0))(ys.T, log_rho_t, f_cov).mean()  # vmap and take mean over num_samps and ts

        return Ell, KL
    
    ### evaluation ###
    def evaluate_metric(self):
        return
    
    ### sample ###
    def sample_log_conditional_intensity(self, prng_state, ini_tau, x_eval, y, y_filt, jitter):
        """
        lambda(t) = h(tau) * rho(x)
        
        :param jnp.ndarray x_eval: evaluation locations (num_samps, obs_dims, ts, x_dims)
        :param jnp.ndarray y: spike train corresponding to segment (neurons, ts), 
                              y = y_filt[..., filter_length:] + last time step
        """
        pre_modulator, _ = self._gp_sample(prng_state, x_eval, False, False, jitter)  # (num_samps, obs_dims, ts)
        num_samps = pre_rates.shape[0]
        y = jnp.broadcast_to(y, (num_samps,) + y.shape[1:])
        
        if self.spikefilter is not None:
            h, _ = self.spikefilter.apply_filter(prng_state[0], y_filt, False)
            pre_modulator += h
            
        log_modulator = f if self.renewal.link_type == LinkTypes['log'] else safe_log(
                self.renewal.inverse_link(f))
        
        # hazard
        spikes = (y > 0).transpose(0, 2, 1)
        log_hazards = vmap(self._log_hazard, (0, 0))(
            spikes, ini_tau)  # (num_samps, ts, obs_dims)
        
        log_rho_t = log_hazards + log_modulator
        return log_rho_t  # (num_samps, out_dims, ts)
    
    
    def sample_conditional_ISI(
        self, prng_state, t_eval, x_cond, jitter, num_samps=20, prior=True
    ):
        """
        :param jnp.ndarray t_eval: evaluation times since last spike, i.e. ISI (ts,)
        :param jnp.ndarray x_cond: evaluation locations (num_samps, obs_dims, x_dims)
        """
        f_samples, _ = self._gp_sample(
            prng_state, x_cond[..., None, :], prior, False, jitter
        )  # (num_samps, obs_dims, 1)
        
        modulator = self.renewal.inverse_link(f_samples)
        tau_eval = modulator * t_eval
        
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
        
        f_samples, _ = self._gp_sample(
            prng_states[0], x_samples, True, False, jitter
        )  # (num_samps, obs_dims, 1)

        y_samples, log_rho_ts = self._sample_spikes(
            prng_states[1], ini_spikes, ini_tau, f_samples)
        return y_samples, log_rho_ts

    def sample_posterior(self, prng_state, num_samps, x_samples, ini_spikes, ini_tau, jitter):
        """
        Sample from posterior predictive
        """
        prng_states = jr.split(prng_state, 2)
        
        f_samples, _ = self._gp_sample(
            prng_states[0], x_samples, False, False, jitter
        )  # (num_samps, obs_dims, 1)

        y_samples, log_rho_ts = self._sample_spikes(
            prng_states[1], ini_spikes, ini_tau, f_samples)
        return y_samples, log_rho_ts

    
    
class RateRescaledRenewal(SparseGPFilterObservations):
    """
    Renewal likelihood GPLVM
    """

    renewal: RenewalLikelihood

    def __init__(self, gp, gp_mean, renewal, spikefilter=None):
        # checks
        assert renewal.array_type == gp.array_type
        assert renewal.f_dims == gp.kernel.out_dims

        super().__init__(gp, gp_mean, spikefilter)
        self.renewal = renewal

    def apply_constraints(self):
        """
        Constrain parameters in optimization
        """
        model = super().apply_constraints()
        model = eqx.tree_at(
            lambda tree: tree.renewal,
            model,
            replace_fn=lambda obj: obj.apply_constraints(),
        )

        return model
    
    def _sample_spikes(self, prng_state, ini_spikes, ini_t_tilde, f_samples):
        """
        Sample the spike train autoregressively
        
        Initial conditions are values before time points of f_samples
        
        :param jnp.dnarray ini_spikes: initial spike train for history filter (num_samps, obs_dims, filter_length) 
                                       or when no history filter (num_samps, obs_dims, 1)
        :param jnp.ndarray t_tilde_start: initial t_tilde values at start (num_samps, obs_dims)
        """
        num_samps, obs_dims, ts = f_samples.shape
        if self.spikefilter is not None:
            prng_states = jr.split(prng_state, ts*2).reshape(ts, 2, -1)
        else:
            prng_states = jr.split(prng_state, ts).reshape(ts, 1, -1)

        def step(carry, inputs):
            prng_state, f = inputs  # current time step
            tau_tilde, past_spikes = carry  # from previous time step
            
            # compute current t_tilde
            tau_tilde = jnp.where(
                (past_spikes[..., -1] > 0),  # previous spikes
                rate * self.renewal.dt, 
                tau_tilde + rate * self.renewal.dt, 
            )  # (num_samps, obs_dims)
            
            # compute intensity
            if self.spikefilter is not None:
                h, _ = self.spikefilter.apply_filter(prng_state[0], past_spikes, False)
                f += h[..., 0]
                
            rate = self.renewal.inverse_link(f)  # (num_samps, obs_dims)
            log_rate = f if self.renewal.link_type == LinkTypes['log'] else safe_log(rate)
            log_hazard = self.renewal.log_hazard(tau_tilde)
            log_rho_t = log_rate + log_hazard
            
            # generate spikes
            p_spike = jnp.minimum(jnp.exp(log_rho_t) * self.renewal.dt, 1.)  # approximate by discrete Bernoulli
            spikes = jr.bernoulli(prng_state[-1], p_spike).astype(self.array_dtype())  # (num_samps, obs_dims)
            
            if self.spikefilter is not None:
                past_spikes = jnp.concatenate((past_spikes[..., 1:], spikes[..., None]), axis=-1)
            else:
                past_spikes = spikes[..., None]
                
            return (tau_tilde, past_spikes), (spikes, log_rho_t)

        init = (ini_t_tilde, ini_spikes)
        xs = (prng_states, f_samples.transpose(2, 0, 1))
        _, (spikes, log_rho_ts) = lax.scan(step, init, xs)
        return spikes.transpose(1, 2, 0), log_rho_ts.transpose(1, 2, 0)  # (num_samps, obs_dims, ts)

    def _rate_rescale(self, spikes, pre_rates, ini_t_tilde, compute_ll, return_t_tilde):
        """
        rate rescaling, computes the log density on the way
        
        :param jnp.ndarray spikes: (ts, obs_dims)
        :param jnp.ndarray rates: (ts, obs_dims)
        """
        rate_scale = self.renewal.dt * self.renewal.mean_scale()
        
        def step(carry, inputs):
            t_tilde, ll = carry
            pre_rate, spike = inputs

            rate = self.renewal.inverse_link(pre_rate)
            t_tilde += rate_scale * rate
            if compute_ll:
                ll += jnp.where(
                    spike, 
                    (pre_rate if self.renewal.link_type == LinkTypes["log"] else safe_log(rate)) + self.renewal.log_density(t_tilde), 
                    0.,
                )
            
            t_tilde_ = jnp.where(spike, 0., t_tilde)  # reset after spike
            return (t_tilde_, ll), t_tilde if return_t_tilde else None

        init = (ini_t_tilde, jnp.zeros(self.renewal.obs_dims))
        (_, log_lik), t_tildes = lax.scan(step, init=init, xs=(pre_rates, spikes))
        return log_lik, t_tildes
    
    ### inference ###
    def VI(self, prng_state, jitter, xs, ys, ys_filt, lik_int_method):
        """
        Compute variational expectation of likelihood and KL divergence
        """
        pre_rates, KL = self._gp_sample(
            prng_state, xs, prior=False, compute_KL=True, jitter=jitter
        )  # (num_samps, out_dims, time)

        if self.spikefilter is not None:
            prng_state, _ = jr.split(prng_state)
            y_filtered, KL_y = self.spikefilter.apply_filter(prng_state, ys_filt[None, ...], compute_KL=True)
            pre_rates += y_filtered
            KL += KL_y
        
        spikes = (ys.T > 0)
        ini_t_tilde = jnp.zeros(self.renewal.obs_dims)
        rrs = lambda f: self._rate_rescale(
            spikes, f.T, ini_t_tilde, compute_ll=True, return_t_tilde=False)[0]
        
        Ell = vmap(rrs)(pre_rates).mean()  # vmap and take mean over num_samps and ts
        return Ell, KL
    
    ### evaluation ###
    def evaluate_metric(self):
        return

    ### sample ###
    def sample_log_conditional_intensity(self, prng_state, ini_t_tilde, x_eval, y, y_filt, jitter):
        """
        rho(t) = r(t) p(u) / (1 - int p(u) du)
        
        :param jnp.ndarray x_eval: evaluation locations (num_samps, obs_dims, ts, x_dims)
        :param jnp.ndarray y: spike train corresponding to segment (neurons, ts), 
                              y = y_filt[..., filter_length:] + last time step
        """
        f_samples, _ = self._gp_sample(prng_state, x_eval, False, False, jitter)  # (num_samps, obs_dims, ts)
        num_samps = pre_rates.shape[0]
        y = jnp.broadcast_to(y, (num_samps,) + y.shape[1:])
        
        pre_rates = f_samples + self.gp_mean[None]
        if self.spikefilter is not None:
            h, _ = self.spikefilter.apply_filter(prng_state[0], y_filt, False)
            pre_rates += h
            
        rates = self.renewal.inverse_link(pre_rates).transpose(0, 2, 1)
        
        # rate rescaling, rescaled time since last spike
        spikes = (y > 0).transpose(0, 2, 1)
        _, tau_tilde = vmap(self._rate_rescale, (0, 0, 0, None, None), (None, 0))(
            spikes, rates, ini_t_tilde, False, True)  # (num_samps, ts, obs_dims)
        
        log_hazard = self.renewal.log_hazard(tau_tilde)
        log_rates = pre_rates if self.renewal.link_type == LinkTypes['log'] else safe_log(rates)
        log_rho_t = log_rates + log_hazard.transpose(0, 2, 1)
        return log_rho_t  # (num_samps, out_dims, ts)
    
    
    def sample_conditional_ISI(
        self, prng_state, t_eval, x_cond, jitter, num_samps=20, prior=True
    ):
        """
        :param jnp.ndarray t_eval: evaluation times since last spike, i.e. ISI (ts,)
        :param jnp.ndarray x_cond: evaluation locations (num_samps, obs_dims, x_dims)
        """
        f_samples, _ = self._gp_sample(
            prng_state, x_cond[..., None, :], prior, False, jitter
        )  # (num_samps, obs_dims, 1)
        
        rate = self.renewal.inverse_link(f_samples)
        t_tilde_eval = (
            rate * t_eval * self.renewal.mean_scale()[:, None]
        )
        
        log_dens = vmap(vmap(self.renewal.log_density), 2, 2)(t_tilde_eval)
        return jnp.exp(log_dens)  # (num_samps, obs_dims, ts)
    
    def sample_prior(self, prng_state, num_samps, x_samples, ini_spikes, ini_t_tilde, jitter):
        """
        Sample from the generative model
        Sample spike trains from the modulated renewal process.
        :return:
            pike train of shape (trials, neuron, timesteps)
        """
        prng_states = jr.split(prng_state, 2)
        
        f_samples, _ = self._gp_sample(
            prng_states[0], x_samples, True, False, jitter
        )  # (num_samps, obs_dims, 1)

        y_samples, log_rho_ts = self._sample_spikes(
            prng_states[1], ini_spikes, ini_t_tilde, f_samples)
        return y_samples, log_rho_ts

    def sample_posterior(self, prng_state, num_samps, x_samples, ini_spikes, ini_t_tilde, jitter):
        """
        Sample from posterior predictive
        """
        prng_states = jr.split(prng_state, 2)
        
        f_samples, _ = self._gp_sample(
            prng_states[0], x_samples, False, False, jitter
        )  # (num_samps, obs_dims, 1)

        y_samples, log_rho_ts = self._sample_spikes(
            prng_states[1], ini_spikes, ini_t_tilde, f_samples)
        return y_samples, log_rho_ts
    
    
    

class NonparametricPointProcess(Observations):
    """
    Bayesian nonparametric modulated point process likelihood
    """

    gp: SparseGP
    pp: PointProcess

    log_warp_tau: jnp.ndarray
    log_refract_tau: jnp.ndarray
    refract_neg: float
    mean_bias: jnp.ndarray

    def __init__(self, gp, warp_tau, refract_tau, refract_neg, mean_bias, dt):
        """
        :param jnp.ndarray warp_tau: time transform timescales of shape (out_dims,)
        :param jnp.ndarray tau: refractory mean timescales of shape (out_dims,)
        """
        super().__init__(ArrayTypes_[gp.array_type])
        self.gp = gp
        self.pp = PointProcess(gp.kernel.out_dims, dt, "log", ArrayTypes_[gp.array_type])

        self.log_warp_tau = self._to_jax(np.log(warp_tau))
        self.log_refract_tau = self._to_jax(np.log(refract_tau))
        self.refract_neg = refract_neg
        self.mean_bias = self._to_jax(mean_bias)

    def apply_constraints(self):
        """
        Constrain parameters in optimization
        """
        model = super().apply_constraints()
        model = eqx.tree_at(
            lambda tree: tree.gp,
            model,
            replace_fn=lambda obj: obj.apply_constraints(),
        )

        return model
    
    ### functions ###
    def _log_time_transform(self, t, inverse):
        """
        Inverse transform is from tau [0, 1] to t in R
        """
        warp_tau = jnp.exp(self.log_warp_tau)
        
        if inverse:
            t_ = -jnp.log(1 - t) * warp_tau
        else:
            s = jnp.exp(-t / warp_tau)
            t_ = 1 - s
            
        return t_
    
    def _log_time_transform_jac(self, t, inverse):
        """
        Inverse transform is from tau [0, 1] to t in R
        """
        warp_tau = jnp.exp(self.log_warp_tau)
        t_ = self._log_time_transform(t, inverse)
        
        if inverse:
            log_jac = self.log_warp_tau - jnp.log(1 - t)  # warp_tau / (1 - t)
        else:
            log_jac = -t / warp_tau - self.log_warp_tau  # s / warp_tau
            
        return t_, log_jac

    def _refractory_mean(self, tau_tilde):
        """
        Refractory period implemented by GP mean function
        The refractory negative value of the mean at tau = 0 is a fixed, chosen a priori
        
        :param jnp.ndarray tau: (out_dims,)
        """
        div_taus = jnp.exp(self.log_warp_tau - self.log_refract_tau)
        delta = self.refract_neg - self.mean_bias
        return delta * (1. - tau_tilde) ** (div_taus) + self.mean_bias
    
    def _combine_input(self, tau_tilde, isi, x):
        """
        :param jnp.dnarray tau_eval: (out_dims, ts)
        :param jnp.ndarray isi_eval: (out_dims, ts, order)
        :param jnp.ndarray x_eval: (out_dims or 1, ts, x_dims)
        """
        out_dims = self.gp.kernel.out_dims
        
        cov_eval = [tau_tilde[..., None]]
        if isi is not None:
            isi_tilde = vmap(vmap(self._log_time_transform, (1, None), 1), (1, None), 1)(
                isi, False)  # (out_dims, ts, order)
            cov_eval.append(isi_tilde)

        if x is not None:
            cov_eval.append(jnp.broadcast_to(x, (out_dims, *x.shape[-2:])))

        return jnp.concatenate(cov_eval, axis=-1)

    def _log_rho_tilde_sample(self, prng_state, tau_tilde, isi, x, prior, jitter):
        """
        Obtain the log conditional intensity given input path along which to evaluate
        
        :param jnp.ndarray tau_eval: evaluation time locs (out_dims, locs)
        :param jnp.ndarray isi_eval: higher order ISIs (out_dims, locs, order)
        :param jnp.ndarray x_eval: external covariates (num_samps, out_dims or 1, locs, x_dims)
        :return:
            log intensity in rescaled time tau (num_samps, out_dims, locs)
        """
        covariates = vmap(self._combine_input, (None, None, 0))(tau_tilde, isi, x)

        if prior:
            f_samples = self.gp.sample_prior(
                prng_state, covariates, jitter
            )  # (samp, f_dim, evals)

        else:
            f_samples, _ = self.gp.sample_posterior(
                prng_state, covariates, jitter, compute_KL=True
            )  # (samp, f_dim, evals)
        
        m_eval = vmap(self._refractory_mean, 1, 1)(tau_tilde)  # (out_dims, locs)
        log_rho_tilde = f_samples + m_eval[None]
        return log_rho_tilde
    
    def _log_rho_tilde_post(self, prng_state, tau_tilde, isi, x, mean_only, compute_KL, jitter):
        """
        Obtain the log conditional intensity given input path along which to evaluate
        
        :param jnp.ndarray tau_eval: evaluation time locs (out_dims, locs)
        :param jnp.ndarray isi_eval: higher order ISIs (out_dims, locs, order)
        :param jnp.ndarray x_eval: external covariates (num_samps, out_dims or 1, locs, x_dims)
        :return:
            log intensity in rescaled time tau mean and cov (num_samps, out_dims, locs, 1), KL scalar
        """
        covariates = vmap(self._combine_input, (None, None, 0))(tau_tilde, isi, x)

        f_mean, f_cov, KL, _ = self.gp.evaluate_posterior(
            covariates, mean_only, diag_cov=True, compute_KL=compute_KL, compute_aux=False, jitter=jitter
        )  # (num_samps, out_dims, time, 1)

        m_eval = vmap(self._refractory_mean, 1, 1)(tau_tilde)  # (out_dims, locs)
        log_rho_tilde_mean = f_mean + m_eval[None, ..., None]
        return log_rho_tilde_mean, f_cov, KL
    
    def _sample_spikes(self, prng_state, timesteps, ini_tau, past_ISIs, x_eval, jitter):
        """
        Sample the spike train autoregressively
        
        # :param jnp.dnarray ini_spikes: initial spike train (num_samps, obs_dims, filter_length)
        :param jnp.ndarray ini_t: initial tau values at start (num_samps, obs_dims)
        :param jnp.ndarray past_ISI: past ISIs (num_samps, obs_dims, order)
        :param jnp.ndarray x_eval: covariates (num_samps, obs_dims, ts, x_dims)
        """
        num_samps = ini_tau.shape[0]
        prng_state = jr.split(prng_state, num_samps+1)
        prng_gp, prng_state = prng_state[:-1], prng_state[-1]
        
        if self.spikefilter is not None:
            prng_states = jr.split(prng_state, timesteps*2).reshape(timesteps, 2, -1)
        else:
            prng_states = jr.split(prng_state, timesteps).reshape(timesteps, 1, -1)
            
        def step(carry, inputs):
            prng_state, x = inputs  # (num_samps, obs_dims, x_dims)
            tau, past_ISI, spikes = carry
            
            # compute current tau and tau_tilde
            tau += self.pp.dt
            tau_tilde, log_dtilde_dt = vmap(self._log_time_transform_jac, (0, None), (0, 0))(
                tau, False)  # (num_samps, out_dims)

            # compute intensity
            log_rho_tilde = self._log_rho_tilde_sample(
                prng_gp, tau_tilde[..., None], past_ISI, x, False, jitter)[..., 0]
            log_rho_t = log_rho_tilde + log_dtilde_dt  # (num_samps, out_dims)
            
            p_spike = jnp.minimum(jnp.exp(log_rho_t) * self.pp.dt, 1.)  # approximate by discrete Bernoulli
            spikes = jr.bernoulli(prng_state[-1], p_spike).astype(self.array_dtype())  # (num_samps, obs_dims)
            
            # spike reset
            spike_cond = (spikes > 0)  # (num_samps, obs_dims)
            if past_ISI is not None:
                shift_ISIs = jnp.concatenate((tau[..., None], past_ISI[..., 0, :-1]), axis=-1)
                past_ISI = jnp.where(spike_cond[..., None], shift_ISIs, past_ISI[..., 0, :])[..., None, :]
            tau = jnp.where(spike_cond, 0.0, tau)

            return (tau, past_ISI, spikes), (spikes, log_rho_t)

        # add dummy time dimension
        if x_eval is not None:
            x_eval = x_eval.transpose(2, 0, 1, 3)[..., None, :]
        if past_ISIs is not None:
            past_ISIs = past_ISIs[..., None, :]
        
        init = (ini_tau, past_ISIs, jnp.zeros_like(ini_tau))
        xs = (prng_states, x_eval)
        _, (spikes, log_rho_ts) = lax.scan(step, init, xs)
        return spikes.transpose(1, 2, 0), log_rho_ts.transpose(1, 2, 0)  # (num_samps, obs_dims, ts)
    

    ### inference ###
    def _get_rho_post(self, jitter, xs, deltas, compute_KL):
        tau, isi = deltas[..., 0], deltas[..., 1:]
        tau_tilde, log_dtilde_dt = vmap(self._log_time_transform_jac, (1, None), (1, 1))(
                tau, False)  # (out_dims, ts)
        
        log_rho_tilde_mean, log_rho_t_cov, KL = self._log_rho_tilde_post(
            prng_state, tau_tilde, isi, xs, mean_only=False, compute_KL=compute_KL, jitter=jitter)
        log_rho_t_mean = log_rho_tilde_mean + log_dtilde_dt[None, ..., None]  # (num_samps, out_dims, ts)
        return log_rho_t_mean, log_rho_t_cov, KL
        
    def VI(self, prng_state, jitter, xs, deltas, ys, lik_int_method):
        log_rho_t_mean, log_rho_t_cov, KL = self._get_rho_post(jitter, xs, deltas, True)
        
        log_rho_t_mean = log_rho_t_mean.transpose(0, 2, 1, 3)  # (num_samps, ts, out_dims, 1)
        log_rho_t_cov = log_rho_t_cov[..., 0].transpose(0, 2, 1)  # (num_samps, ts, out_dims, 1)
        log_rho_t_cov = vmap(vmap(jnp.diag))(log_rho_t_cov)  # (num_samps, ts, out_dims, out_dims)
        
        llf = lambda m, c: self.pp.variational_expectation(
            ys.T, m, c, prng_state, jitter, lik_int_method)
        Ell = vmap(vmap(llf))(log_rho_t_mean, log_rho_t_cov).mean()  # mean over mc and ts
        
        return Ell, KL
    
    ### evaluation ###
    def evaluate_log_conditional_intensity(self, prng_state, tau_eval, isi_eval, x_eval, jitter):
        """
        Evaluate the conditional intensity along an input path
        """
        tau_tilde, log_dtilde_dt = vmap(self._log_time_transform_jac, (1, None), (1, 1))(
            tau_eval[None, :], False)  # (out_dims, ts)
        
        log_rho_tilde_mean, log_rho_t_cov, _ = self._log_rho_tilde_gp_post(
            prng_state, tau_tilde, isi_eval, x_eval, False, False, jitter)
        log_rho_t_mean = log_rho_tilde_mean + log_dtilde_dt[None, ..., None]  # (num_samps, out_dims, ts, 1)
        log_rho_t_cov = log_rho_t_cov  # additive transform in log space
        
        return log_rho_t_mean, log_rho_t_cov

    def evaluate_metric(self):
        # evaluate log predictive CIF
        log_rho_t_mean, log_rho_t_cov, _ = self._get_rho_post(jitter, xs, deltas, False)
        
        log_rho_t_mean = log_rho_t_mean.transpose(0, 2, 1, 3)  # (num_samps, ts, out_dims, 1)
        log_rho_t_cov = log_rho_t_cov[..., 0].transpose(0, 2, 1)  # (num_samps, ts, out_dims, 1)
        log_rho_t_cov = vmap(vmap(jnp.diag))(log_rho_t_cov)  # (num_samps, ts, out_dims, out_dims)
        
        llf = lambda m, c: self.pp.variational_expectation(
            ys.T, m, c, prng_state, jitter, lik_int_method, log_predictive=True)
        lEl = vmap(vmap(llf))(log_rho_t_mean, log_rho_t_cov).mean()  # mean over mc and ts
        return lEl
    
    def KS_test(self, prng_state, xs, deltas, ys, jitter):
        """
        Use the posterior mean to perform the time rescaling
        """
        # evaluate CIF
        log_rho_t_mean, log_rho_t_cov, _ = self._get_rho_post(jitter, xs, deltas, False)
        log_rho_t_mean, log_rho_t_cov = log_rho_t_mean[..., 0], log_rho_t_cov[..., 0]  # (num_samps, out_dims, ts)
        post_rho_mean = jnp.exp(log_rho_t_mean + log_rho_t_cov / 2.)
        
        # time rescaling
        rescaled_intervals = time_rescale(post_rho_mean, ys, max_intervals)  # (num_intervals, out_dims)
        num_intervals = (~jnp.isnan()).sum(0)  # (out_dims,)
        
        # intervals should be exponentially distributed
        cdfs = 1. - jnp.exp(-rescaled_intervals)
        
        return cdfs, num_intervals

    ### sample ###
    def _sample_log_rho_tilde(
        self, 
        prng_state, 
        num_samps, 
        tau_tilde, 
        isi_cond, 
        x_cond,
        sigma_pts, 
        weights, 
        int_eval_pts=1000,
        prior=True,
        jitter=1e-6,
    ):
        """
        Sample conditional ISI distribution at tau_tilde
        
        :param jnp.ndarray tau_tilde: evaluation time points (out_dims, locs)
        :param jnp.ndarray isi_cond: past ISI values to condition on (out_dims, order)
        :param jnp.ndarray x_cond: covariate values to condition on (x_dims,)
        """
        obs_dims = len(self.log_warp_tau)
        
        vvinterp = vmap(vmap(jnp.interp, (None, 0, 0), 0), (None, None, 0), 0)  # vmap mc, then out_dims
        vvvinterp = vmap(vmap(vmap(
            jnp.interp, (0, None, None), 0)), (None, None, 0), 0)  # vmap mc, then out_dims, then evals
        
        # evaluation locs for integration points
        vvquad_integrate = vmap(vmap(gauss_quad_integrate, (None, 0, None, None), (0, 0)), 
                                (None, 0, None, None), (0, 0))  # vmap out_dims and locs
        locs, ws = vvquad_integrate(
            0.0, tau_tilde, sigma_pts, weights
        )  # (out_dims, eval_locs, cub_pts, 1)
        
        # integral points for interpolation
        tau_tilde_pts = jnp.linspace(0.0, 1.0, int_eval_pts)[None, :].repeat(obs_dims, axis=0)
        
        # compute cumulative intensity int rho(tau_tilde) dtau_tilde
        tau_tilde_cat = jnp.concatenate([tau_tilde_pts, tau_tilde], axis=1)
        cat_pts = tau_tilde_cat.shape[1]
        isi_cond = isi_cond[:, None].repeat(cat_pts, axis=1) if isi_cond is not None else None
        x_cond = jnp.broadcast_to(
            x_cond[None, None, None], 
            (num_samps, obs_dims, cat_pts, x_cond.shape[0]), 
        ) if x_cond is not None else None
        
        log_rho_tilde_cat = self._log_rho_tilde_sample(
            prng_state, tau_tilde_cat, isi_cond, x_cond, prior, jitter)
        log_rho_tilde = log_rho_tilde_cat[..., int_eval_pts:]
        rho_tilde_pts = jnp.exp(log_rho_tilde_cat[..., :int_eval_pts])
        
        # compute integral over rho
        quad_rho_tau_tilde = vvvinterp(
            locs[..., 0], tau_tilde_pts, rho_tilde_pts
        )  # num_samps, out_dims, taus, cub_pts
        int_rho_tau_tilde = (quad_rho_tau_tilde * ws).sum(-1)

        # normalizer
        locs, ws = gauss_quad_integrate(0.0, 1.0, sigma_pts, weights)
        quad_rho = vvinterp(locs[:, 0], tau_tilde_pts, rho_tilde_pts)
        int_rho = (quad_rho * ws).sum(-1)  # num_samps, out_dims, taus
        log_normalizer = safe_log(1.0 - jnp.exp(-int_rho))  # (num_samps, out_dims)
        
        return log_rho_tilde - (int_rho + log_normalizer)[..., None]
    
    def sample_conditional_ISI(
        self,
        prng_state, 
        num_samps,
        tau_eval,
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
        :param jnp.ndarray isi_cond: past ISI values to condition on (out_dims, order)
        :param jnp.ndarray x_cond: covariate values to condition on (x_dims,)
        """
        sigma_pts, weights = gauss_legendre(1, num_quad_pts)
        sigma_pts, weights = self._to_jax(sigma_pts), self._to_jax(weights)
        
        # evaluation locs
        tau_tilde, log_dtilde_dt = vmap(self._log_time_transform_jac, (1, None), 1)(
            tau_eval[None, :], False)  # (out_dims, locs)
        
        log_rho_tilde = self._sample_log_rho_tilde( 
            prng_state, 
            num_samps, 
            tau_tilde, 
            isi_cond, 
            x_cond,
            sigma_pts, 
            weights, 
            int_eval_pts,
            prior,
            jitter,
        )

        ISI_density = jnp.exp(log_rho_tilde + log_dtilde_dt)
        return ISI_density
    
    def sample_conditional_ISI_expectation(
        self, 
        prng_state, 
        num_samps,
        func_of_tau, 
        isi_cond, 
        x_cond,
        int_eval_pts=1000,
        f_num_quad_pts=100,
        isi_num_quad_pts=100, 
        prior=True,
        jitter=1e-6,
    ):
        """
        Compute expectations in warped time space
        """
        obs_dims = len(self.log_warp_tau)
        
        sigma_pts, weights = gauss_legendre(1, f_num_quad_pts)
        sigma_pts, weights = self._to_jax(sigma_pts), self._to_jax(weights)
        
        tau_tilde_pts, ws = gauss_quad_integrate(
            0.0, 1.0, sigma_pts, weights
        )  # (cub_pts, 1), (cub_pts,)
        tau_tilde_pts = tau_tilde_pts.T.repeat(obs_dims, axis=0)
        
        sigma_pts, weights = gauss_legendre(1, isi_num_quad_pts)
        sigma_pts, weights = self._to_jax(sigma_pts), self._to_jax(weights)
        
        tau_pts = vmap(self._log_time_transform, (1, None), 1)(
            tau_tilde_pts, True)  # (out_dims, locs)
        
        f_pts = func_of_tau(tau_pts)
        
        log_rho_tilde_pts = self._sample_log_rho_tilde( 
            prng_state, 
            num_samps, 
            tau_tilde_pts, 
            isi_cond, 
            x_cond,
            sigma_pts, 
            weights, 
            int_eval_pts,
            prior,
            jitter,
        )
        
        return (f_pts * jnp.exp(log_rho_tilde_pts) * ws).sum(-1)
    
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

    
    
    
class GPLVM(module):
    """
    base class for GPLVM
    """

    inp_model: GaussianLatentObservedSeries
    obs_model: Observations

    def __init__(self, inp_model, obs_model):
        """
        Add latent-observed inputs and GP observation model
        """
        assert inp_model.array_type == obs_model.array_type
        assert type(obs_model) in [
            NonparametricPointProcess, ModulatedFactorized, 
            ModulatedRenewal, RateRescaledRenewal, 
        ]
        super().__init__(ArrayTypes_[obs_model.array_type])
        self.inp_model = inp_model
        self.obs_model = obs_model
        
    def apply_constraints(self):
        """
        Constrain parameters in optimization
        """
        model = jax.tree_map(lambda p: p, self)  # copy
        model = eqx.tree_at(
            lambda tree: [tree.inp_model, tree.obs_model],
            model,
            replace_fn=lambda obj: obj.apply_constraints(),
        )

        return model
        
    def ELBO(self, prng_state, num_samps, jitter, tot_ts, data, lik_int_method):
        ts, xs, deltas, ys, ys_filt = data
        prng_key_x, prng_key_o = jr.split(prng_state)
        
        #.mean()  # mean over MC
        if type(self.obs_model) == ModulatedFactorized:
            xs, KL_x = self.inp_model.sample_marginal_posterior(
                prng_key_x, num_samps, ts, xs, jitter, compute_KL=True)
            xs = xs[:, None]  # add dummy out_dims
            
            Ell, KL_o = self.obs_model.VI(
                prng_key_o, jitter, xs, ys, ys_filt, lik_int_method)   
        
        elif type(self.obs_model) == ModulatedRenewal:
            xs, KL_x = self.inp_model.sample_posterior(
                prng_key_x, num_samps, ts, xs, jitter, compute_KL=True)
            xs = xs[:, None]  # add dummy out_dims
            
            taus = deltas[..., 0]
            Ell, KL_o = self.obs_model.VI(
                prng_key_o, jitter, xs, taus, ys, ys_filt, lik_int_method)
        
        elif type(self.obs_model) == RateRescaledRenewal:
            xs, KL_x = self.inp_model.sample_posterior(
                prng_key_x, num_samps, ts, xs, jitter, compute_KL=True)
            xs = xs[:, None]  # add dummy out_dims
            
            Ell, KL_o = self.obs_model.VI(
                prng_key_o, jitter, xs, ys, ys_filt, lik_int_method)
        
        elif type(self.obs_model) == NonparametricPointProcess:
            xs, KL_x = self.inp_model.sample_marginal_posterior(
                prng_key_x, num_samps, ts, xs, jitter, compute_KL=True)
            xs = xs[:, None]  # add dummy out_dims
            
            Ell, KL_o = self.obs_model.VI(
                prng_key_o, jitter, xs, deltas, ys, lik_int_method)
            
        Ell *= tot_ts  # rescale due to temporal batching
        return Ell - KL_o - KL_x