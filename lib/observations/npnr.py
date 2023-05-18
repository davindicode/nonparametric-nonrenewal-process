from typing import Union

import equinox as eqx

import jax
import jax.numpy as jnp
import jax.random as jr

import numpy as np
from jax import lax, vmap

from ..base import ArrayTypes_, module

from ..GP.sparse import SparseGP
from ..likelihoods.base import FactorizedLikelihood, LinkTypes, RenewalLikelihood
from ..likelihoods.factorized import PointProcess
from ..utils.jax import safe_log, safe_sqrt
from ..utils.linalg import gauss_legendre, gauss_quad_integrate
from ..utils.spikes import time_rescale

from .base import Observations


class NonparametricPointProcess(Observations):
    """
    Bayesian nonparametric modulated point process likelihood
    """

    gp: SparseGP
    pp: PointProcess

    log_warp_tau: jnp.ndarray
    log_mean_tau: jnp.ndarray
    mean_amp: jnp.ndarray
    mean_bias: jnp.ndarray

    def __init__(self, gp, warp_tau, mean_tau, mean_amp, mean_bias, dt):
        """
        :param jnp.ndarray warp_tau: time transform timescales of shape (obs_dims,)
        :param jnp.ndarray mean_tau: mean decay timescales of shape (obs_dims,)
        :param jnp.ndarray mean_amp: mean amplitude of shape (obs_dims,)
        :param jnp.ndarray mean_bias: mean offset of shape (obs_dims,)
        :param float dt: time step size
        """
        super().__init__(ArrayTypes_[gp.array_type])
        self.gp = gp
        self.pp = PointProcess(
            gp.kernel.out_dims, dt, "log", ArrayTypes_[gp.array_type]
        )

        self.log_warp_tau = self._to_jax(np.log(warp_tau))
        self.log_mean_tau = self._to_jax(np.log(mean_tau))
        self.mean_amp = self._to_jax(mean_amp)
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

        :param jnp.ndarray t: time of shape (obs_dims,)
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

        :param jnp.ndarray t: time of shape (obs_dims,)
        """
        warp_tau = jnp.exp(self.log_warp_tau)
        t_ = self._log_time_transform(t, inverse)

        if inverse:
            log_jac = self.log_warp_tau - jnp.log(1 - t)  # warp_tau / (1 - t)
        else:
            log_jac = -t / warp_tau - self.log_warp_tau  # s / warp_tau

        return t_, log_jac

    def _combine_input(self, tau_tilde, isi, x):
        """
        :param jnp.dnarray tau_tilde: (obs_dims, ts)
        :param jnp.ndarray isi: (obs_dims, ts, order)
        :param jnp.ndarray x: (obs_dims or 1, ts, x_dims)
        """
        cov_eval = [tau_tilde[..., None]]
        if isi is not None:
            isi_tilde = vmap(
                vmap(self._log_time_transform, (1, None), 1), (1, None), 1
            )(
                isi, False
            )  # (obs_dims, ts, order)
            cov_eval.append(isi_tilde)

        if x is not None:
            cov_eval.append(
                jnp.broadcast_to(x, (self.gp.kernel.out_dims, *x.shape[-2:]))
            )

        return jnp.concatenate(cov_eval, axis=-1)

    def _mean(self, tau_tilde, sel_outdims):
        """
        Refractory period implemented by GP mean function if negative amplitude

        :param jnp.ndarray tau_tilde: (obs_dims,)
        """
        tau_tilde = tau_tilde[sel_outdims]
        mean_amp, mean_bias = self.mean_amp[sel_outdims], self.mean_bias[sel_outdims]
        div_taus = jnp.exp((self.log_warp_tau - self.log_mean_tau)[sel_outdims])
        return mean_amp * (1.0 - tau_tilde) ** (div_taus) + mean_bias

    def _log_lambda_tilde_sample(
        self, prng_state, num_samps, tau_tilde, isi, x, prior, jitter, sel_outdims
    ):
        """
        Obtain the log conditional intensity given input path along which to evaluate

        :param jnp.ndarray tau_tilde: evaluation time locs (num_samps, obs_dims, locs)
        :param jnp.ndarray isi: higher order ISIs (num_samps, obs_dims, locs, order)
        :param jnp.ndarray x: external covariates (num_samps, obs_dims or 1, locs, x_dims)
        :return:
            log intensity in rescaled time tau (num_samps, obs_dims, locs)
        """
        obs_dims = self.gp.kernel.out_dims

        tau_tilde = jnp.broadcast_to(
            tau_tilde, (num_samps, obs_dims, *tau_tilde.shape[2:])
        )
        isi = (
            jnp.broadcast_to(isi, (num_samps, obs_dims, *isi.shape[2:]))
            if isi is not None
            else None
        )
        x = (
            jnp.broadcast_to(x, (num_samps, obs_dims, *x.shape[2:]))
            if x is not None
            else None
        )

        covariates = vmap(
            self._combine_input,
            (0, 0 if isi is not None else None, 0 if x is not None else None),
        )(
            tau_tilde, isi, x
        )  # vmap over MC

        if prior:
            f_samples = self.gp.sample_prior(
                prng_state, covariates, jitter, sel_outdims
            )  # (samp, f_dim, evals)

        else:
            f_samples, _ = self.gp.sample_posterior(
                prng_state,
                covariates,
                compute_KL=True,
                jitter=jitter,
                sel_outdims=sel_outdims,
            )  # (samp, f_dim, evals)

        m_eval = vmap(vmap(self._mean, (0, None)), (2, None), 2)(
            tau_tilde, sel_outdims
        )  # (num_samps, obs_dims, locs)
        log_lambda_tilde = f_samples + m_eval
        return log_lambda_tilde

    def _log_lambda_tilde_post(
        self, tau_tilde, isi, x, mean_only, compute_KL, jitter, sel_outdims
    ):
        """
        Obtain the log conditional intensity given input path along which to evaluate

        :param jnp.ndarray tau_tilde: evaluation time locs (num_samps, obs_dims, locs)
        :param jnp.ndarray isi: higher order ISIs (num_samps, obs_dims, locs, order)
        :param jnp.ndarray x: external covariates (num_samps, obs_dims or 1, locs, x_dims)
        :return:
            log intensity in rescaled time tau mean and cov (num_samps, obs_dims, locs, 1), KL scalar
        """
        mc = x.shape[0]
        tau_tilde = jnp.broadcast_to(tau_tilde, (mc, *tau_tilde.shape[1:]))
        isi = jnp.broadcast_to(isi, (mc, *isi.shape[1:]))

        covariates = vmap(self._combine_input)(tau_tilde, isi, x)  # vmap over MC

        f_mean, f_var, KL, _ = self.gp.evaluate_posterior(
            covariates,
            mean_only,
            diag_cov=True,
            compute_KL=compute_KL,
            compute_aux=False,
            jitter=jitter,
            sel_outdims=sel_outdims,
        )  # (num_samps, obs_dims, time, 1)

        m_eval = vmap(vmap(self._mean, (0, None)), (2, None), 2)(
            tau_tilde, sel_outdims
        )  # (num_samps, obs_dims, locs)
        log_lambda_tilde_mean = f_mean + m_eval[..., None]
        return log_lambda_tilde_mean, f_var, KL

    def _sample_spikes(self, prng_state, timesteps, ini_tau, past_ISIs, x_eval, jitter):
        """
        Sample the spike train autoregressively

        :param jnp.ndarray ini_tau: initial tau values at start (num_samps, obs_dims)
        :param jnp.ndarray past_ISI: past ISIs (num_samps, obs_dims, order)
        :param jnp.ndarray x_eval: covariates (num_samps, obs_dims, ts, x_dims)
        """
        sel_outdims = jnp.arange(self.gp.kernel.out_dims)
        num_samps = ini_tau.shape[0]
        prng_gp, prng_state = jr.split(prng_state)
        prng_states = jr.split(prng_state, timesteps).reshape(timesteps, 1, -1)

        def step(carry, inputs):
            prng_state, x = inputs  # (num_samps, obs_dims, x_dims)
            tau, past_ISI, spikes = carry  # (num_samps, obs_dims)

            # compute current tau and tau_tilde
            tau += self.pp.dt
            tau_tilde, log_dtilde_dt = vmap(
                self._log_time_transform_jac, (0, None), (0, 0)
            )(
                tau, False
            )  # (num_samps, obs_dims)

            # compute intensity
            log_lambda_tilde = self._log_lambda_tilde_sample(
                prng_gp,
                num_samps,
                tau_tilde[..., None],
                past_ISI,
                x,
                False,
                jitter,
                sel_outdims,
            )[
                ..., 0
            ]  # add dummy time/locs dimension to tau_tilde
            log_lambda_t = log_lambda_tilde + log_dtilde_dt  # (num_samps, obs_dims)

            p_spike = jnp.minimum(
                jnp.exp(log_lambda_t) * self.pp.dt, 1.0
            )  # approximate by discrete Bernoulli
            spikes = jr.bernoulli(prng_state[-1], p_spike).astype(
                self.array_dtype()
            )  # (num_samps, obs_dims)

            # spike reset
            spike_cond = spikes > 0  # (num_samps, obs_dims)
            if past_ISI.shape[-1] > 0:
                shift_ISIs = jnp.concatenate(
                    (tau[..., None], past_ISI[..., 0, :-1]), axis=-1
                )
                past_ISI = jnp.where(
                    spike_cond[..., None], shift_ISIs, past_ISI[..., 0, :]
                )[..., None, :]
            tau = jnp.where(spike_cond, 0.0, tau)

            return (tau, past_ISI, spikes), (spikes, log_lambda_t)

        # add dummy time dimension
        if x_eval is not None:
            x_eval = x_eval.transpose(2, 0, 1, 3)[..., None, :]
        past_ISIs = past_ISIs[..., None, :]

        init = (ini_tau, past_ISIs, jnp.zeros_like(ini_tau))
        xs = (prng_states, x_eval)
        _, (spikes, log_lambda_ts) = lax.scan(step, init, xs)
        return spikes.transpose(1, 2, 0), log_lambda_ts.transpose(
            1, 2, 0
        )  # (num_samps, obs_dims, ts)

    ### inference ###
    def _get_log_lambda_post(
        self, xs, deltas, mean_only, compute_KL, jitter, sel_outdims
    ):
        """
        :param jnp.ndarray xs: inputs (num_samps, 1, ts, x_dims)
        :param jnp.ndarray deltas: lagged ISIs (num_samps, obs_dims, ts, orders)
        """
        tau, isi = (
            deltas[..., 0],
            deltas[..., 1:],
        )  # (num_samps, obs_dims, ts, (orders))
        tau_tilde, log_dtilde_dt = vmap(
            vmap(self._log_time_transform_jac, (0, None), (0, 0)),
            (2, None),
            (2, 2),
        )(
            tau, False
        )  # (num_samps, obs_dims, ts)

        log_lambda_tilde_mean, log_lambda_t_var, KL = self._log_lambda_tilde_post(
            tau_tilde,
            isi,
            xs,
            mean_only=mean_only,
            compute_KL=compute_KL,
            jitter=jitter,
            sel_outdims=sel_outdims,
        )

        log_lambda_t_mean = (
            log_lambda_tilde_mean + log_dtilde_dt[:, sel_outdims, :, None]
        )  # (num_samps, obs_dims, ts)
        return log_lambda_t_mean, log_lambda_t_var, KL

    def variational_expectation(
        self,
        prng_state,
        jitter,
        xs,
        deltas,
        ys,
        compute_KL,
        total_samples,
        lik_int_method,
        log_predictive=False,
    ):
        """
        :param jnp.ndarray xs: covariates of shape (num_samps, obs_dims or 1, ts, x_dims)
        :param jnp.ndarray deltas: covariates of shape (obs_dims, ts, orders)
        :param jnp.ndarray ys: covariates of shape (obs_dims, ts)
        """
        sel_outdims = jnp.arange(self.gp.kernel.out_dims)
        num_samps, ts = xs.shape[0], xs.shape[2]
        prng_state = jr.split(prng_state, num_samps * ts).reshape(num_samps, ts, -1)

        log_lambda_t_mean, log_lambda_t_var, KL = self._get_log_lambda_post(
            xs, deltas[None], False, compute_KL, jitter, sel_outdims
        )

        log_lambda_t_mean = log_lambda_t_mean.transpose(
            0, 2, 1, 3
        )  # (num_samps, ts, obs_dims, 1)
        log_lambda_t_var = log_lambda_t_var[..., 0].transpose(
            0, 2, 1
        )  # (num_samps, ts, obs_dims, 1)
        log_lambda_t_cov = vmap(vmap(jnp.diag))(
            log_lambda_t_var
        )  # (num_samps, ts, obs_dims, obs_dims)

        llf = lambda y, m, c, p: self.pp.variational_expectation(
            y, m, c, p, jitter, lik_int_method, log_predictive
        )
        Eq = vmap(vmap(llf), (None, 0, 0, 0))(
            ys.T, log_lambda_t_mean, log_lambda_t_cov, prng_state
        ).mean()  # mean over mc and ts

        return total_samples * Eq, KL

    ### evaluation ###
    def log_conditional_intensity(
        self, prng_state, xs, deltas, jitter, sel_outdims, sampling=False
    ):
        """
        Evaluate the conditional intensity along an input path

        :param jnp.ndarray x_eval: evaluation locations (num_samps, obs_dims, ts, x_dims) or None
        :param jnp.ndarray deltas: evaluation ISIs (num_samps, obs_dims, ts, orders)
        :param bool sampling: flag to use samples from the posterior, otherwise use the mean,
                              note that if the link function is not log then the the output is
                              not the mean value of the transformed quantity
        """
        num_samps = deltas.shape[0]
        if sel_outdims is None:
            obs_dims = len(self.log_warp_tau)
            sel_outdims = jnp.arange(obs_dims)

        tau, isi = deltas[..., 0], deltas[..., 1:]
        tau_tilde, log_dtilde_dt = vmap(
            vmap(self._log_time_transform_jac, (0, None), (0, 0)),
            (2, None),
            (2, 2),
        )(
            tau, False
        )  # (num_samps, obs_dims, ts)

        if sampling:
            log_lambda_tilde, _ = self._log_lambda_tilde_sample(
                prng_state,
                num_samps,
                tau_tilde,
                isi,
                xs,
                compute_KL=compute_KL,
                prior=False,
                jitter=jitter,
                sel_outdims=sel_outdims,
            )

        else:
            log_lambda_tilde, _, _ = self._log_lambda_tilde_post(
                tau_tilde,
                isi,
                xs,
                mean_only=True,
                compute_KL=False,
                jitter=jitter,
                sel_outdims=sel_outdims,
            )

        log_lambda_t = (
            log_lambda_tilde + log_dtilde_dt[:, sel_outdims, :, None]
        )  # (num_samps, obs_dims, ts, 1)
        return log_lambda_t

    def posterior_mean(self, xs, deltas, jitter, sel_outdims):
        """
        Use the posterior mean to perform the time rescaling

        :param jnp.ndarray xs: covariates of shape (ts, x_dims)
        :param jnp.ndarray deltas: covariates of shape (obs_dims, ts, orders)
        :param jnp.ndarray ys: covariates of shape (obs_dims, ts)
        """
        if sel_outdims is None:
            obs_dims = len(self.log_warp_tau)
            sel_outdims = jnp.arange(obs_dims)

        log_lambda_t_mean, log_lambda_t_var, _ = self._get_log_lambda_post(
            xs[None, None], deltas[None], False, False, jitter, sel_outdims
        )
        log_lambda_t_mean, log_lambda_t_var = (
            log_lambda_t_mean[0, ..., 0],
            log_lambda_t_var[0, ..., 0],
        )  # (obs_dims, ts)

        post_lambda_mean = jnp.exp(log_lambda_t_mean + log_lambda_t_var / 2.0)
        return post_lambda_mean

    ### sample ###
    def _sample_log_ISI_tilde(
        self,
        prng_state,
        num_samps,
        tau_tilde,
        isi_cond,
        x_cond,
        sigma_pts,
        weights,
        sel_outdims,
        int_eval_pts,
        prior,
        jitter,
    ):
        """
        Sample conditional log ISI distribution at tau_tilde

        :param jnp.ndarray tau_tilde: evaluation time points (obs_dims, locs)
        :param jnp.ndarray isi_cond: past ISI values to condition on (obs_dims, order) or None
        :param jnp.ndarray x_cond: covariate values to condition on (x_dims,) or None
        :return:
            log rho tilde, integral rho tilde, and log normalizer shapes (num_samps, obs_dims, taus or 1)
        """
        vvinterp = vmap(
            vmap(jnp.interp, (None, 0, 0), 0), (None, None, 0), 0
        )  # vmap mc, then obs_dims
        vvvinterp = vmap(
            vmap(vmap(jnp.interp, (0, None, None), 0)), (None, None, 0), 0
        )  # vmap mc, then obs_dims, then evals

        # evaluation locs for integration points
        vvquad_integrate = vmap(
            vmap(gauss_quad_integrate, (None, 0, None, None), (0, 0)),
            (None, 0, None, None),
            (0, 0),
        )  # vmap obs_dims and locs
        locs, ws = vvquad_integrate(
            0.0, tau_tilde, sigma_pts, weights
        )  # rescale quadrature points from standard [0, 1] interval (obs_dims, eval_locs, cub_pts, 1)

        # integral points for interpolation
        tau_tilde_pts = jnp.linspace(0.0, 1.0, int_eval_pts)[None, :].repeat(
            tau_tilde.shape[0], axis=0
        )  # (obs_dims, pts)

        # compute cumulative intensity int rho(tau_tilde) dtau_tilde
        tau_tilde_cat = jnp.concatenate([tau_tilde_pts, tau_tilde], axis=1)
        cat_pts = tau_tilde_cat.shape[1]
        isi_cond = (
            jnp.broadcast_to(
                isi_cond[None, :, None],
                (1, isi_cond.shape[0], cat_pts, isi_cond.shape[-1]),
            )
            if isi_cond is not None
            else None
        )  # (1, obs_dims, pts, orders)
        x_cond = (
            jnp.broadcast_to(
                x_cond[None, None, None],
                (1, 1, cat_pts, x_cond.shape[-1]),
            )
            if x_cond is not None
            else None
        )  # (1, obs_dims, pts, x_dims)

        log_lambda_tilde_cat = self._log_lambda_tilde_sample(
            prng_state,
            num_samps,
            tau_tilde_cat[None],
            isi_cond,
            x_cond,
            prior,
            jitter,
            sel_outdims,
        )
        log_lambda_tilde = log_lambda_tilde_cat[..., int_eval_pts:]
        lambda_tilde_pts = jnp.exp(log_lambda_tilde_cat[..., :int_eval_pts])

        # compute integral over rho
        quad_lambda_tau_tilde = vvvinterp(
            locs[sel_outdims, ..., 0], tau_tilde_pts[sel_outdims], lambda_tilde_pts
        )  # num_samps, obs_dims, taus, cub_pts
        int_lambda_tau_tilde = (quad_lambda_tau_tilde * ws[sel_outdims]).sum(-1)

        # normalizer
        locs, ws = gauss_quad_integrate(0.0, 1.0, sigma_pts, weights)
        quad_rho = vvinterp(locs[:, 0], tau_tilde_pts[sel_outdims], lambda_tilde_pts)
        int_rho = (quad_rho * ws).sum(-1)  # (num_samps, obs_dims, taus)
        log_normalizer = safe_log(1.0 - jnp.exp(-int_rho))[
            ..., None
        ]  # (num_samps, obs_dims, 1)

        return log_lambda_tilde, int_lambda_tau_tilde, log_normalizer

    def sample_conditional_ISI(
        self,
        prng_state,
        num_samps,
        tau_eval,
        isi_cond,
        x_cond,
        sel_outdims,
        int_eval_pts=1000,
        num_quad_pts=100,
        prior=True,
        jitter=1e-6,
    ):
        """
        Compute the instantaneous renewal density with rho(ISI;X) from model
        Uses linear interpolation with Gauss-Legendre quadrature for integrals

        :param jnp.ndarray t_eval: evaluation time points (eval_locs,)
        :param jnp.ndarray isi_cond: past ISI values to condition on (obs_dims, order)
        :param jnp.ndarray x_cond: covariate values to condition on (x_dims,)
        """
        if sel_outdims is None:
            obs_dims = self.gp.kernel.out_dims
            sel_outdims = jnp.arange(obs_dims)

        sigma_pts, weights = gauss_legendre(1, num_quad_pts)
        sigma_pts, weights = self._to_jax(sigma_pts), self._to_jax(weights)

        # evaluation locs
        tau_tilde, log_dtilde_dt = vmap(
            self._log_time_transform_jac, (1, None), (1, 1)
        )(
            tau_eval[None, :], False
        )  # (obs_dims, locs)

        (
            log_lambda_tilde,
            int_lambda_tau_tilde,
            log_normalizer,
        ) = self._sample_log_ISI_tilde(
            prng_state,
            num_samps,
            tau_tilde,
            isi_cond,
            x_cond,
            sigma_pts,
            weights,
            sel_outdims,
            int_eval_pts,
            prior,
            jitter,
        )

        log_ISI_tilde = log_lambda_tilde - int_lambda_tau_tilde - log_normalizer
        ISI_density = jnp.exp(log_ISI_tilde + log_dtilde_dt[sel_outdims])
        return ISI_density

    def sample_conditional_ISI_expectation(
        self,
        prng_state,
        num_samps,
        func_of_tau,
        isi_cond,
        x_cond,
        sel_outdims,
        int_eval_pts=1000,
        f_num_quad_pts=100,
        isi_num_quad_pts=100,
        prior=True,
        jitter=1e-6,
    ):
        """
        Compute expectations in warped time space

        :param jnp.ndarray isi_cond: past ISI values to condition on (obs_dims, order)
        :param jnp.ndarray x_cond: covariate values to condition on (x_dims,)
        """
        if sel_outdims is None:
            obs_dims = self.gp.kernel.out_dims
            sel_outdims = jnp.arange(obs_dims)

        sigma_pts, weights = gauss_legendre(1, f_num_quad_pts)
        sigma_pts, weights = self._to_jax(sigma_pts), self._to_jax(weights)

        tau_tilde_pts, ws = gauss_quad_integrate(
            0.0, 1.0, sigma_pts, weights
        )  # (cub_pts, 1), (cub_pts,)
        tau_tilde_pts = tau_tilde_pts.T.repeat(len(sel_outdims), axis=0)

        sigma_pts, weights = gauss_legendre(1, isi_num_quad_pts)
        sigma_pts, weights = self._to_jax(sigma_pts), self._to_jax(weights)

        tau_pts = vmap(self._log_time_transform, (1, None), 1)(
            tau_tilde_pts, True
        )  # (obs_dims, locs)

        (
            log_lambda_tilde,
            int_lambda_tau_tilde,
            log_normalizer,
        ) = self._sample_log_ISI_tilde(
            prng_state,
            num_samps,
            tau_tilde_pts,
            isi_cond,
            x_cond,
            sigma_pts,
            weights,
            sel_outdims,
            int_eval_pts,
            prior,
            jitter,
        )

        log_ISI_tilde_pts = log_lambda_tilde - int_lambda_tau_tilde - log_normalizer

        f_pts = func_of_tau(tau_pts[sel_outdims])
        return (f_pts * jnp.exp(log_ISI_tilde_pts) * ws).sum(-1)

    def sample_prior(
        self,
        prng_state,
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

        y_samples, log_lambda_ts = self._sample_spikes(
            prng_states[1], timesteps, ini_t_since, past_ISIs, x_samples, jitter
        )
        return y_samples, log_lambda_ts

    def sample_posterior(
        self,
        prng_state,
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

        y_samples, log_lambda_ts = self._sample_spikes(
            prng_states[1], timesteps, ini_t_since, past_ISIs, x_samples, jitter
        )
        return y_samples, log_lambda_ts
