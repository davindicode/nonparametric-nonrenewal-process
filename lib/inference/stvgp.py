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
from ..utils.linalg import gauss_legendre
from ..utils.jax import safe_log, safe_sqrt



class NonparametricPointProcess(Observations):
    """
    Bayesian nonparametric modulated point process likelihood
    """

    gp: Union[IndependentLTI, KroneckerLTI]
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
        super().__init__(ArrayTypes_[gp.array_type])
        self.gp = gp
        self.pp = PointProcess(gp.kernel.out_dims, dt, "log", ArrayTypes_[gp.array_type])

        self.warp_tau = self._to_jax(warp_tau)
        self.refract_tau = self._to_jax(refract_tau)
        self.refract_neg = refract_neg
        self.mean_bias = self._to_jax(mean_bias)
        
        self.modulated = False if type(gp) == IndependentLTI else True

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
        The refractory negative value of the mean at tau = 0 is a fixed, chosen a priori
        
        :param jnp.ndarray tau: (out_dims,)
        """
        delta = self.refract_neg - self.mean_bias
        return delta * (tau + 1.) ** (-self.warp_tau / self.refract_tau) + self.mean_bias
    
    def _combine_input(self, isi_eval, x_eval):
        """
        :param jnp.ndarray isi_eval: (out_dims, ts, order)
        :param jnp.ndarray x_eval: (out_dims or 1, ts, x_dims)
        """
        out_dims = isi_eval.shape[0]
        
        cov_eval = []
        if isi_eval is not None:
            tau_isi_eval = vmap(vmap(self._log_time_transform, (1, None), 1), (1, None), 1)(
                isi_eval, False)  # (out_dims, ts, order)
            cov_eval.append(tau_isi_eval)

        if x_eval is not None:
            cov_eval.append(jnp.broadcast_to(x_eval, (out_dims, *x_eval.shape[-2:])))

        return jnp.concatenate(cov_eval, axis=-1)

    def _log_rho_from_gp_sample(self, prng_state, num_samps, tau_eval, isi_eval, x_eval, prior, jitter):
        """
        Obtain the log conditional intensity given input path along which to evaluate
        
        :param jnp.ndarray tau_eval: evaluation time locs (out_dims, locs)
        :param jnp.ndarray isi_eval: higher order ISIs (out_dims, locs, order)
        :param jnp.ndarray x_eval: external covariates (num_samps, out_dims or 1, locs, x_dims)
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
    
    def _log_rho_from_gp_post(self, prng_state, tau_eval, isi_eval, x_eval, mean_only, compute_KL, jitter):
        """
        Obtain the log conditional intensity given input path along which to evaluate
        
        :param jnp.ndarray tau_eval: evaluation time locs (out_dims, locs)
        :param jnp.ndarray isi_eval: higher order ISIs (out_dims, locs, order)
        :param jnp.ndarray x_eval: external covariates (out_dims or 1, locs, x_dims)
        :return:
            log intensity in rescaled time tau mean and cov (out_dims, locs), KL scalar
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
            
            # compute current t_since and tau
            t_since += self.pp.dt
                
            tau_since, log_dtau_dt = vmap(self._log_time_transform_jac, (0, None), (0, 0))(
                t_since, False)  # (num_samps, out_dims)

            # compute intensity
            log_rho_tau = vmap(self._log_rho_from_gp_sample, 
                               (0, None, 0, None if past_ISI is None else 0, None if x is None else 0, None, None), 0)(
                prng_gp, 1, tau_since[..., None], past_ISI, x, False, jitter)[..., 0]
            log_rho_t = log_rho_tau + log_dtau_dt  # (num_samps, out_dims)
            
            p_spike = jnp.minimum(jnp.exp(log_rho_t) * self.pp.dt, 1.)  # approximate by discrete Bernoulli
            spikes = jr.bernoulli(prng_state[-1], p_spike).astype(self.array_dtype())  # (num_samps, obs_dims)
            
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
    def VI(self, prng_state, num_samps, jitter, xs, deltas, ys, lik_int_method):
        tau_eval, isi_eval = deltas[..., 0], deltas[..., 1:]
        
        lrp = lambda xs: self._log_rho_from_gp_post(
            prng_state, tau_eval, isi_eval, xs, mean_only=False, compute_KL=True, jitter=jitter)
        log_rho_tau_mean, log_rho_tau_cov, KL = vmap(lrp)(xs)  # vmap over MC
        
        log_rho_tau_mean = log_rho_tau_mean.transpose(0, 2, 1)[..., None]  # (num_samps, ts, out_dims, 1)
        log_rho_tau_cov = log_rho_tau_cov.transpose(0, 2, 1)  # (num_samps, ts, out_dims, 1)
        log_rho_tau_cov = vmap(vmap(jnp.diag))(log_rho_tau_cov)  # (num_samps, ts, out_dims, out_dims)
        
        llf = lambda m, c: self.pp.variational_expectation(
            ys.T, m, c, prng_state, jitter, lik_int_method)
        Ell = vmap(vmap(llf))(log_rho_tau_mean, log_rho_tau_cov).mean()  # mean over mc and ts
        
        return Ell, KL.mean()
    
    ### evaluation ###
    def evaluate_log_conditional_intensity(self, prng_state, t_eval, isi_eval, x_eval, jitter):
        """
        Evaluate the conditional intensity along an input path
        """
        tau_eval, log_dtau_dt_eval = vmap(self._log_time_transform_jac, (1, None), (1, 1))(
            t_eval[None, :], False)  # (out_dims, ts)
        
        log_rho_tau_mean, log_rho_tau_cov, _ = self._log_rho_from_gp_post(
            prng_state, tau_eval, isi_eval, x_eval, False, False, jitter)
        log_rho_t_mean = log_rho_tau_mean + log_dtau_dt_eval  # (num_samps, out_dims, ts)
        log_rho_t_cov = log_rho_tau_cov  # additive transform in log space
        
        return log_rho_t_mean, log_rho_t_cov

    def evaluate_metric(self):
        return
    
    def evaluate_conditional_ISI_expectation(
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
    def sample_conditional_ISI(
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
            NonparametricPointProcess,
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
        if type(self.obs_model) == NonparametricPointProcess:
            xs, KL_x = self.inp_model.sample_marginal_posterior(
                prng_key_x, num_samps, ts, xs, jitter, compute_KL=True)
            xs = xs[:, None]  # add dummy out_dims
            
            Ell, KL_o = self.obs_model.VI(
                prng_key_o, num_samps, jitter, xs, deltas, ys, lik_int_method)
            
        Ell *= tot_ts  # rescale due to temporal batching
        return Ell# - KL_o - KL_x