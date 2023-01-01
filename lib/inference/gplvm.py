from typing import Union

import numpy as np

from functools import partial

import equinox as eqx

import jax
import jax.numpy as jnp
import jax.random as jr


from .base import FilterGPLVM
from ..likelihoods.base import FactorizedLikelihood, RenewalLikelihood
from ..likelihoods.factorized import LogCoxProcess
from ..GP.sparse import SparseGP
from ..GP.markovian import MultiOutputLTI
from ..GP.spatiotemporal import KroneckerLTI    


    
    
    
class FactorizedGPLVM(FilterGPLVM):
    """
    Factorization across time points allows one to rely on latent marginals
    """

    gp: SparseGP
    likelihood: FactorizedLikelihood
        
    def __init__(self, gp, ssgp = None, spikefilter = None):
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
            pre_rates = self.gp.sample_prior(prng_state, x_eval, jitter)  # (evals, samp, f_dim)

        else:
            pre_rates, _ = self.gp.sample_posterior(
                prng_state, x_eval, jitter, compute_KL=False) # (evals, samp, f_dim)

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
        #xx = np.linspace(-5., 5., 100)[None, None, :, None]
        x_mean, x_cov, KL_x = self._sample_input_marginals(x, t, num_samps)
        
        f_mean, f_cov, KL_f, _ = self.gp.evaluate_posterior(
            prng_state, x_samples, jitter, compute_KL=True) # (evals, samp, f_dim)
        
        y_filtered, KL_y = self._spiketrain_filter()
        
        Ell = self.likelihood.variational_expectation(
            prng_state, jitter, y, f_mean, f_cov)
        
        ELBO = Ell - KL_x - KL_f - KL_y
        return ELBO
    

    ### sample ###
    def sample_prior(self, prng_state, num_samps, x_obs=None, timedata=None):
        """
        Sample from the generative model
        """
        pf_x = self.gp.sample_prior(prng_state, xx.repeat(num_samps, axis=0), jitter)  # (evals, samp, f_dim)
        return y, q_vh, I, eps_samples

    def sample_posterior(self, prng_state, num_samps):
        """
        Sample from posterior predictive
        """
        qf_x, KL = self.gp.sample_posterior(
            prng_state, xx.repeat(num_samps, axis=0), jitter, compute_KL=True) # (evals, samp, f_dim)

        return y, q_vh, I, eps_samples

    ### evaluation ###
    def evaluate_metric(self):
        return
    
    
    
class RenewalGPLVM(FilterGPLVM):
    """
    Renewal likelihood GPLVM
    """
    
    gp: SparseGP
    renewal: RenewalLikelihood
    
    def __init__(self, gp, renewal, ssgp = None, spikefilter = None):
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
            pre_rates = self.gp.sample_prior(prng_state, x_eval, jitter)  # (evals, samp, f_dim)

        else:
            pre_rates, _ = self.gp.sample_posterior(
                prng_state, x_eval, jitter, compute_KL=False) # (evals, samp, f_dim)

        return pre_rates
        
    ### inference ###
    def ELBO(self, prng_state, spiketimes, pre_rates, covariates, neuron, num_ISIs):
        """
        Compute the evidence lower bound
        """
        x_samples, KL_x = self._sample_input_trajectories(x, t, num_samps)
        
        f_samples, KL_f = self.gp.sample_posterior(
            prng_state, x_samples, jitter, compute_KL=True) # (evals, samp, f_dim)
        
        y_filtered, KL_y = self._spiketrain_filter()
        
        Ell = self.renewal.variational_expectation(
            spiketimes, pre_rates, covariates, neuron, num_ISIs)
        
        ELBO = Ell - KL_x - KL_f - KL_y
        return ELBO  
    
    ### sample ###
    def sample_instantaneous_renewal(self, prng_state, t_eval, x_cond, num_samps = 20, prior = True):
        """
        :param jnp.ndarray t_eval: evaluation times since last spike, i.e. ISI
        """
        f_samp = self._gp_sample(prng_state, x_cond, prior, jitter)  # (num_samps, obs_dims)
        tau_eval = self.inverse_link(f_samp) * t_eval[:, None] / self.shape_scale()[None, :]
        log_dens = vmap(self.renewal.log_renewal_density)(tau_eval[..., None])
        return jnp.exp(log_dens)
    
    
    def sample_prior(self, prng_state, num_samps, x_obs=None, timedata=None):
        """
        Sample from the generative model
        Sample spike trains from the modulated renewal process.
        :return:
            pike train of shape (trials, neuron, timesteps)
        """
        pf_x = self.gp.sample_prior(prng_state, xx.repeat(num_samps, axis=0), jitter)  # (evals, samp, f_dim)
        
        timesteps = rate.shape[-1]
        neuron = self._validate_neuron(neuron)
        spiketimes = gen_IRP(
            self.sample_ISI, rate[:, neuron, :], self.dt.item()
        )

        if return_spiketrain:
            train = jnp.empty((rate.shape[0], -1, timesteps))
            for en, sp in enumerate(spiketimes):
                train[en, :, sp] += 1

            return train

        else:
            return spiketimes
        
        return y, q_vh, I, eps_samples

    def sample_posterior(self, prng_state, num_samps):
        """
        Sample from posterior predictive
        """
        qf_x, KL = self.gp.sample_posterior(
            prng_state, xx.repeat(num_samps, axis=0), jitter, compute_KL=True) # (evals, samp, f_dim)

        return y, q_vh, I, eps_samples

    
    ### evaluation ###
    def evaluate_conditional_intensity(self, prng_state, tx_eval):
        """
        rho(t) = r(t) p(u) / (1 - int p(u) du)
        """
        pre_rates = self._gp_sample(prng_state, x_eval, prior, jitter)
        
        rates = self.renewal.inverse_link(pre_rates)
        taus = self.dt * jnp.cumsum(rates, axis=2)
        
        # rate rescaling
        rISI = jnp.empty((mc, self.out_dims, num_ISIs))
        
        for en, spkinds in enumerate(spiketimes):
            isi_count = jnp.maximum(spkinds.shape[0] - 1, 0)
            
            def body(i, val):
                val[:, en, i] = taus[:, i]
                return val
            
            rISI[:, en, :] = lax.fori_loop(0, isi_count, body, rISI[:, en, :])
            
        tau_since_spike = rates  # rescaled time since last spike
        rhot = rates * renewal_density(tau_since_spike) / survival(tau_since_spike)
        return rhot  # (num_samps, out_dims, ts)
    
    
    def evaluate_metric(self):
        return

    
        
        
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
    def log_time_transform(self, t, inverse=False):
        """
        Inverse transform is from tau [0, 1] to t in R
        """
        if inverse:
            t_ = -jnp.log(1 - t) * self.t0
            log_jac = jnp.log(self.t0) - jnp.log(1 - t) # t0 / (1 - t)
        else:
            s = jnp.exp(-t / self.t0)
            t_ = 1 - s
            log_jac = -t / self.t0 - jnp.log(self.t0)  # s / t0
        return t_, log_jac


    def refractory_mean(self, tau, neg=-12.):
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
                    prng_state, num_samps, tau_eval, jitter)

            else:
                f_samples, _ = self.gp.sample_posterior(
                    prng_state, num_samps, tau_eval, jitter, False)  # (tr, time, N, 1)
            
        m_eval = self.refractory_mean(tau_eval)
        log_rhotau = f_samples[..., 0] + m_eval[None, :, None]
        return log_rhotau
    
    def _tau_since_spike(self, dt, tau_start, t_since_spike, rates):
        """
        rescale time since last spike
        
        :param jnp.ndarray t_since_spike: time (ts,)
        :param jnp.ndarray rates: (ts,)
        """
        def true_func(tau, rate):
            return tau + rate * dt

        def false_func(tau, rate):
            return rate * dt

        def step(carry, inputs):
            t_since, rate = inputs
            tau_since = carry

            cond = (t_since == 1.)
            tau_since = lax.cond(cond, true_func, false_func, tau_since, rate)
            return tau_since, tau_since

        init = tau_start
        xs = (t_since_spike, rates)
        _, tau_since_spike = lax.scan(step, init, xs)
        return tau_since_spike  # (time,)
        
        
    ### inference ###
    def ELBO(self, prng_state, x, t, num_samps):
        self.gp.evaluate_posterior()
        
        self.likelihood.variational_expectation()
        
        return
    
        
    ### sampling ###    
    def sample_instantaneous_renewal(self, prng_state, t_eval, x_cond, num_samps = 20, int_eval_pts = 1000, num_quad_pts = 100, prior = True):
        """
        Compute the instantaneous renewal density with rho(ISI;X) from model
        Uses linear interpolation with Gauss-Legendre quadrature for integrals
        
        :param jnp.ndarray t_eval: evaluation time points (eval_locs,)
        """
        def quad_integrate(a, b, sigma_pts, weights):
            # transform interval
            sigma_pts = 0.5*(sigma_pts + 1)*(b - a) + a
            weights *= jnp.prod( 0.5*(b - a) )
            return sigma_pts, weights

        vvinterp = vmap(vmap(jnp.interp, (None, None, 0), 0), (None, None, 2), 2)
        
        # compute rho(tau) for integral
        tau_pts = jnp.linspace(0.0, 1.0, int_eval_pts)
        t_pts, log_dt_dtau_pts = self.log_time_transform(tau_pts, inverse=True)

        log_rhotau_pts = self._transform_gp_sample(prng_state, tau_pts, x_cond, prior)
        log_rhot_pts = log_rhotau_pts - log_dt_dtau_pts[None, :, None]
        
        # compute rho(tau) and rho(t)
        rhotau_pts = jnp.exp(log_rhotau_pts)
        rhot_pts = jnp.exp(log_rhot_pts)
        
        # evaluation locs
        tau_eval, _ = self.log_time_transform(t_eval, inverse=False)
        rhot_eval = vvinterp(t_eval, t_pts, rhot_pts)

        # compute cumulative intensity int rho(tau) dtau
        sigma_pts, weights = gauss_legendre(1, num_quad_pts)
        locs, ws = vmap(quad_integrate, (None, 0, None, None), (0, 0))(
            0., tau_eval, sigma_pts, weights)

        quad_rhotau_eval = vmap(vvinterp, (1, None, None), 2)(locs[..., 0], tau_pts, rhotau_pts)
        int_rhotau_eval = (quad_rhotau_eval * ws[None, ..., None]).sum(-2)

        # normalizer
        locs, ws = quad_integrate(0., 1., sigma_pts, weights)
        quad_rho = vvinterp(locs[:, 0], tau_pts, rhotau_pts)
        int_rho = (quad_rho * ws[None, :, None]).sum(-2)
        normalizer = 1. - jnp.exp(-int_rho)  # (num_samps, out_dims)

        # compute renewal density
        exp_negintrhot = jnp.exp(-int_rhotau_eval)
        renewal_density = rhot_eval * exp_negintrhot / normalizer[:, None, :]
        return renewal_density
        
    ### sample ###
    def sample_prior(self, prng_state, num_samps, x_obs=None, timedata=None):
        """
        Sample from the generative model
        """
        pf_x = self.gp.sample_prior(prng_state, xx.repeat(num_samps, axis=0), jitter)  # (evals, samp, f_dim)
        return y, q_vh, I, eps_samples

    def sample_posterior(self, prng_state, num_samps):
        """
        Sample from posterior predictive
        """
        qf_x, KL = self.gp.sample_posterior(
            prng_state, xx.repeat(num_samps, axis=0), jitter, compute_KL=True) # (evals, samp, f_dim)

        return y, q_vh, I, eps_samples

    
    ### evaluation ###
    def evaluate_conditional_intensity(self, prng_state, t_eval, x_eval):
        """
        Evaluate the conditional intensity along an input path
        """
        tau_eval, log_dtau_dt_eval = self.log_time_transform(t_eval, inverse=False)

        f_samples, _ = self.gp.sample_posterior(
            prng_state, num_samps, tau_eval, jitter, False)  # (tr, time, N, 1)
        
        tau_since = self._tau_since_spike(self.renewal.dt, tau_start, t_since_spike, rates)
        log_rhotau_eval = self._transform_gp_sample(prng_state, tau_eval, x_eval, prior)
        log_rhot_eval = log_rhotau_eval + log_dtau_dt_eval[None, :, None]  
        return jnp.exp(log_rhot_eval)
    
    
    def evaluate_metric(self):
        return
    
