from functools import partial
from typing import Union

import equinox as eqx

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import vmap

import numpy as np

from .base import FilterGPLVM

from ..GP.markovian import MultiOutputLTI
from ..GP.sparse import SparseGP
from ..GP.spatiotemporal import KroneckerLTI
from ..likelihoods.base import FactorizedLikelihood, RenewalLikelihood
from ..likelihoods.factorized import LogCoxProcess
from ..utils.neural import gen_IRP




class FactorizedGPLVM(FilterGPLVM):
    """
    Factorization across time points allows one to rely on latent marginals
    """

    gp: SparseGP
    likelihood: FactorizedLikelihood

    def __init__(self, gp, likelihood, ssgp=None, spikefilter=None):
        # checks
        assert likelihood.array_type == gp.array_type
        assert likelihood.f_dims == gp.kernel.out_dims

        super().__init__(ssgp, spikefilter, gp.array_type)
        self.gp = gp
        self.likelihood = likelihood

    def _gp_sample(self, prng_state, x_eval, prior, jitter):
        """
        Obtain the log conditional intensity given input path along which to evaluate

        :param jnp.ndarray x_eval: evaluation locs (num_samps, out_dims, eval_locs, in_dims)
        """
        if prior:
            pre_rates = self.gp.sample_prior(
                prng_state, x_eval, jitter
            )  # (evals, samp, f_dim)

        else:
            pre_rates, _ = self.gp.sample_posterior(
                prng_state, x_eval, jitter, compute_KL=False
            )  # (evals, samp, f_dim)

        return pre_rates

    ### variational inference ###
    def ELBO(
        self,
        prng_state,
        num_samps,
    ):
        """
        Compute ELBO
        """
        x_mean, x_cov, KL_x = self._posterior_input_marginals(
            x, t, num_samps)  # (num_samps, obs_dims, ts, 1)

        f_mean, f_cov, KL_f, _ = self.gp.evaluate_posterior(
            prng_state, x_samples, jitter, compute_KL=True
        )  # (evals, samp, f_dim)

        y_filtered, KL_y = self._spiketrain_filter(prng_state, y)

        Ell = self.likelihood.variational_expectation(
            prng_state, jitter, y, f_mean, f_cov
        )

        ELBO = Ell - KL_x - KL_f - KL_y
        return ELBO
    
    ### evaluation ###
    def evaluate_conditional_rate(self, prng_state, x_eval):
        """
        rho(t) = r(t) p(u) / (1 - int p(u) du)
        """
        pre_rates = self._gp_sample(prng_state, x_eval, prior, jitter)
        rates = self.renewal.inverse_link(pre_rates)
        
        # rate rescaling, rescaled time since last spike
        _, tau_since_spike = self.rate_rescale(
            spikes, rates.T, compute_ll=False, return_tau=True)
        #taus = self.dt / self.shape_scale() * jnp.cumsum(rates, axis=2)
        
        renewal_density = jnp.exp(self.renewal.log_density(tau_since_spike))
        survival = 1.0 - self.renewal.cum_renewal_density(tau_since_spike)
        
        rho_t = rates * renewal_density / survival
        return rho_t  # (num_samps, out_dims, ts)
    
    def evaluate_metric(self):
        """
        predictive posterior log likelihood
        log posterior predictive likelihood
        """
        return

    ### sample ###
    def sample_prior(self, prng_state, num_samps, x_eval):
        """
        Sample from the generative model
        """
        x_sample = self._prior_input_samples()
        
        f_samples = self._gp_sample(self, prng_state, x_eval, prior, jitter)  # (evals, samp, f_dim)
        
        self.likelihood.sample_Y()
        return y_samples, f_samples, x_samples

    def sample_posterior(self, prng_state, num_samps, x_eval):
        """
        Sample from posterior predictive
        """
        x_sample = self._posterior_input_samples()
        
        qf_x, KL = self.gp.sample_posterior(
            prng_state, xx.repeat(num_samps, axis=0), jitter, compute_KL=True
        )  # (evals, samp, f_dim)

        self.likelihood.sample_Y()
        return y_samples, f_samples, x_samples


class RenewalGPLVM(FilterGPLVM):
    """
    Renewal likelihood GPLVM
    """

    gp: SparseGP
    renewal: RenewalLikelihood

    def __init__(self, gp, renewal, ssgp=None, spikefilter=None):
        # checks
        assert renewal.array_type == gp.array_type
        assert renewal.f_dims == gp.kernel.out_dims

        super().__init__(ssgp, spikefilter, gp.array_type)
        self.gp = gp
        self.renewal = renewal

    def _gp_sample(self, prng_state, x_eval, prior, jitter):
        """
        Obtain the log conditional intensity given input path along which to evaluate

        :param jnp.ndarray x_eval: evaluation locs (num_samps, out_dims, eval_locs, in_dims)
        """
        if prior:
            pre_rates = self.gp.sample_prior(
                prng_state, x_eval, jitter
            )  # (evals, samp, f_dim)

        else:
            pre_rates, _ = self.gp.sample_posterior(
                prng_state, x_eval, jitter, compute_KL=False
            )  # (evals, samp, f_dim)

        return pre_rates
    
    def _sample_spikes(self, prng_state, ini_spikes, pre_rates):
        """
        Sample the spike train autoregressively
        
        :param jnp.dnarray ini_spikes: initial spike train (num_samps, obs_dims, filter_length)
        """
        num_samps, obs_dims, ts = pre_rates.shape
        prng_states = jr.split(prng_state, ts)
        
        def true_func(tau, rate):
            return tau + rate * self.dt

        def false_func(tau, rate):
            return rate * self.dt

        def step(carry, inputs):
            prng_state, pre_rate = inputs
            tau_since, past_spikes = carry
            
            h = self._spikefilter(past_spikes)
            
            f = pre_rate + h
            rate = self.renewal.inverse_link(f)
            
            cond = (past_spikes[..., -1] > 0)
            tau_since = lax.cond(cond, true_func, false_func, tau_since, rate)
            
            renewal_density = jnp.exp(self.renewal.log_density(tau_since_spike))
            survival = 1.0 - self.renewal.cum_renewal_density(tau_since_spike)

            rho_t = rate * renewal_density / survival
            p_spike = jnp.maximum(rho_t * self.dt, 1.)  # approximate by discrete Bernoulli
            spikes = jr.bernoulli(prng_state, p_spike)
              
            past_spikes = jnp.concatenate((past_spikes[..., 1:], spikes[..., None]), axis=-1)
            
            return (tau_since, past_spikes), (spikes, rho_t, rate)

        init = (tau_start, ini_spikes)
        xs = (prng_states, pre_rates)
        _, (spikes, rho_ts, rates) = lax.scan(step, init, xs)
        return spikes, rho_ts, rates  # (num_samps, obs_dims, ts)

    ### inference ###
    def ELBO(self, prng_state, y, pre_rates, covariates, neuron, num_ISIs):
        """
        Compute the evidence lower bound
        """
        x_samples, KL_x = self._sample_input_trajectories(prng_state, x, t, num_samps)

        f_samples, KL_f = self.gp.sample_posterior(
            prng_state, x_samples, jitter, compute_KL=True
        )  # (evals, samp, f_dim)

        y_filtered, KL_y = self._spiketrain_filter(prng_state, y)
        pre_rates = f_samples + y_filtered
        
        Ell = self.renewal.variational_expectation(
            y, pre_rates, 
        )

        ELBO = Ell - KL_x - KL_f - KL_y
        return ELBO
    
    ### evaluation ###
    def evaluate_conditional_intensity(self, prng_state, x_eval, y):
        """
        rho(t) = r(t) p(u) / (1 - int p(u) du)
        """
        pre_rates = self._gp_sample(prng_state, x_eval, prior, jitter)
        rates = self.renewal.inverse_link(pre_rates)
        
        # rate rescaling, rescaled time since last spike
        spikes = (y.T > 0)
        _, tau_since_spike = self._rate_rescale(
            spikes, rates.T, compute_ll=False, return_tau=True)
        #taus = self.dt / self.shape_scale() * jnp.cumsum(rates, axis=2)
        
        renewal_density = jnp.exp(self.renewal.log_density(tau_since_spike))
        survival = 1.0 - self.renewal.cum_renewal_density(tau_since_spike)
        
        rho_t = rates * renewal_density / survival
        return rho_t  # (num_samps, out_dims, ts)

    def evaluate_metric(self):
        return

    ### sample ###
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
        tau_eval = (
            self.renewal.inverse_link(f_samples) * t_eval / self.renewal.shape_scale()[:, None]
        )
        
        log_dens = vmap(vmap(self.renewal.log_renewal_density), 2, 2)(tau_eval)
        return jnp.exp(log_dens)  # (num_samps, obs_dims, ts)
    
    def sample_prior(self, prng_state, num_samps):
        """
        Sample from the generative model
        Sample spike trains from the modulated renewal process.
        :return:
            pike train of shape (trials, neuron, timesteps)
        """
        x_samples, _ = self._sample_input_trajectories(
            prng_state, x, t, num_samps, True, compute_KL)
        
        f_samples = self._gp_sample(
            prng_state, x_cond[..., None, :], True, jitter
        )  # (num_samps, obs_dims, 1)

        y_samples = self._sample_spikes()
        return y_samples, f_samples, x_samples

    def sample_posterior(self, prng_state, num_samps):
        """
        Sample from posterior predictive
        """
        x_samples, _ = self._sample_input_trajectories(
            prng_state, x, t, num_samps, False, compute_KL)
        
        f_samples = self._gp_sample(
            prng_state, x_cond[..., None, :], False, jitter
        )  # (num_samps, obs_dims, 1)

        y_samples = self._sample_spikes()
        return y_samples, f_samples, x_samples

    


class NonparametricPPGPLVM(FilterGPLVM):
    """
    Bayesian nonparametric modulated point process likelihood
    """

    gp: Union[MultiOutputLTI, KroneckerLTI]
    pp: LogCoxProcess

    t0: jnp.ndarray
    refract_tau: jnp.ndarray
    mean_bias: jnp.ndarray

    def __init__(self, gp, t0, refract_tau, mean_bias, dt):
        """
        :param jnp.ndarray t0: time transform timescales of shape (out_dims,)
        :param jnp.ndarray tau: refractory mean timescales of shape (out_dims,)
        """
        super().__init__(t0, refract_tau, mean_bias, dt, gp.array_type)
        self.gp = gp
        self.pp = LogCoxProcess(gp.kernel.out_dims, dt, self.array_type)

        self.t0 = self._to_jax(t0)
        self.refract_tau = self._to_jax(refract_tau)
        self.mean_bias = self._to_jax(mean_bias)

        self.modulated = False if type(gp) == MultiOutputLTI else True

    ### functions ###
    def _log_time_transform(self, t, inverse=False):
        """
        Inverse transform is from tau [0, 1] to t in R
        """
        if inverse:
            t_ = -jnp.log(1 - t) * self.t0
            log_jac = jnp.log(self.t0) - jnp.log(1 - t)  # t0 / (1 - t)
        else:
            s = jnp.exp(-t / self.t0)
            t_ = 1 - s
            log_jac = -t / self.t0 - jnp.log(self.t0)  # s / t0
        return t_, log_jac

    def _refractory_mean(self, tau, neg=-12.0):
        """
        Refractory period implemented by GP mean function
        """
        return neg * jnp.exp(-tau / self.refract_tau) + self.mean_bias

    def _transform_gp_sample(self, prng_state, tau_eval, x_eval, prior):
        """
        Obtain the log conditional intensity given input path along which to evaluate
        """
        if self.modulated:
            return

        else:
            if prior:
                f_samples = self.gp.sample_prior(
                    prng_state, num_samps, tau_eval, jitter
                )

            else:
                f_samples, _ = self.gp.sample_posterior(
                    prng_state, num_samps, tau_eval, jitter, False
                )  # (tr, time, N, 1)

        m_eval = self._refractory_mean(tau_eval)
        log_rho_tau = f_samples[..., 0] + m_eval[None, :, None]
        return log_rho_tau

    ### inference ###
    def ELBO(self, prng_state, x, t, num_samps):
        self.gp.evaluate_posterior()

        self.likelihood.variational_expectation()

        return
    
    ### evaluation ###
    def evaluate_conditional_intensity(self, prng_state, t_eval, x_eval, y):
        """
        Evaluate the conditional intensity along an input path
        """
        tau_eval, log_dtau_dt_eval = self._log_time_transform(t_eval, inverse=False)
        
        log_rho_tau_eval = self._transform_gp_sample(prng_state, tau_eval, x_eval, prior)
        log_rho_t_eval = log_rho_tau_eval + log_dtau_dt_eval[None, :, None]
        return jnp.exp(log_rho_t_eval)

    def evaluate_metric(self):
        return

    ### sample ###
    def sample_instantaneous_renewal(
        self,
        prng_state,
        t_eval,
        x_cond,
        num_samps=20,
        int_eval_pts=1000,
        num_quad_pts=100,
        prior=True,
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

        vvinterp = vmap(vmap(jnp.interp, (None, None, 0), 0), (None, None, 2), 2)

        # compute rho(tau) for integral
        tau_pts = jnp.linspace(0.0, 1.0, int_eval_pts)
        t_pts, log_dt_dtau_pts = self._log_time_transform(tau_pts, inverse=True)

        log_rho_tau_pts = self._transform_gp_sample(prng_state, tau_pts, x_cond, prior)
        log_rho_t_pts = log_rho_tau_pts - log_dt_dtau_pts[None, :, None]

        # compute rho(tau) and rho(t)
        rho_tau_pts = jnp.exp(log_rho_tau_pts)
        rho_t_pts = jnp.exp(log_rho_t_pts)

        # evaluation locs
        tau_eval, _ = self._log_time_transform(t_eval, inverse=False)
        rho_t_eval = vvinterp(t_eval, t_pts, rho_t_pts)

        # compute cumulative intensity int rho(tau) dtau
        sigma_pts, weights = gauss_legendre(1, num_quad_pts)
        locs, ws = vmap(quad_integrate, (None, 0, None, None), (0, 0))(
            0.0, tau_eval, sigma_pts, weights
        )

        quad_rho_tau_eval = vmap(vvinterp, (1, None, None), 2)(
            locs[..., 0], tau_pts, rho_tau_pts
        )
        int_rho_tau_eval = (quad_rho_tau_eval * ws[None, ..., None]).sum(-2)

        # normalizer
        locs, ws = quad_integrate(0.0, 1.0, sigma_pts, weights)
        quad_rho = vvinterp(locs[:, 0], tau_pts, rho_tau_pts)
        int_rho = (quad_rho * ws[None, :, None]).sum(-2)
        normalizer = 1.0 - jnp.exp(-int_rho)  # (num_samps, out_dims)

        # compute renewal density
        exp_negintrho_t = jnp.exp(-int_rho_tau_eval)
        renewal_density = rho_t_eval * exp_negintrho_t / normalizer[:, None, :]
        return renewal_density
    
    def sample_prior(self, prng_state, num_samps):
        """
        Sample from the generative model
        Sample spike trains from the modulated renewal process.
        :return:
            pike train of shape (trials, neuron, timesteps)
        """
        x_samples, _ = self._sample_input_trajectories(
            prng_state, x, t, num_samps, True, compute_KL)
        
        f_samples = self._gp_sample(
            prng_state, x_cond[..., None, :], True, jitter
        )  # (num_samps, obs_dims, 1)

        y_samples = self._sample_spikes()
        return y_samples, f_samples, x_samples

    def sample_posterior(self, prng_state, num_samps):
        """
        Sample from posterior predictive
        """
        x_samples, _ = self._sample_input_trajectories(
            prng_state, x, t, num_samps, False, compute_KL)
        
        f_samples = self._gp_sample(
            prng_state, x_cond[..., None, :], False, jitter
        )  # (num_samps, obs_dims, 1)

        y_samples = self._sample_spikes()
        return y_samples, f_samples, x_samples

    
