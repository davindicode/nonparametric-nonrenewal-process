from typing import Union

import equinox as eqx

import jax
import jax.numpy as jnp
import jax.random as jr

import numpy as np
from jax import lax, vmap

from ..base import ArrayTypes_, module

from ..GP.sparse import SparseGP
from ..likelihoods.base import (
    FactorizedLikelihood,
    LinkTypes,
    LinkTypes_,
    RenewalLikelihood,
)
from ..likelihoods.factorized import PointProcess
from ..utils.jax import safe_log, safe_sqrt
from ..utils.spikes import time_rescale

from .base import FilterObservations


class SparseGPFilterObservations(FilterObservations):
    """
    Spiketrain filter + GP observation model

    """

    gp: SparseGP
    gp_mean: Union[
        jnp.ndarray, None
    ]  # constant mean (obs_dims,), i.e. bias if not None

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

    def gp_sample(self, prng_state, x_eval, prior, compute_KL, jitter, sel_outdims):
        """
        Sample from Gaussian process with mean function

        :param jnp.ndarray x_eval: evaluation locs (num_samps, out_dims, eval_locs, in_dims)
        :return:
            samples (num_samps, out_dims, eval_locs), KL scalar
        """
        if prior:
            f_samples = self.gp.sample_prior(
                prng_state, x_eval, jitter, sel_outdims
            )  # (num_samps, out_dims, time)
            KL = 0

        else:
            f_samples, KL = self.gp.sample_posterior(
                prng_state, x_eval, jitter, compute_KL, sel_outdims
            )  # (num_samps, out_dims, time)

        if self.gp_mean is not None:
            if sel_outdims is None:
                sel_outdims = jnp.arange(self.gp.kernel.out_dims)
            f_samples += self.gp_mean[None, sel_outdims, None]

        return f_samples, KL

    def gp_posterior(
        self, x_samples, mean_only, diag_cov, compute_KL, jitter, sel_outdims
    ):
        """
        Evaluate Gaussian process posterior with mean function

        :param jnp.ndarray x_samples: input (num_samps, obs_dims, ts, x_dims)
        :return:
            mean (num_samps, out_dims, time, 1)
            (co)variance (num_samps, out_dims, time, 1 or time)
            KL scalar
        """
        f_mean, f_cov, KL, _ = self.gp.evaluate_posterior(
            x_samples, mean_only, diag_cov, compute_KL, False, jitter, sel_outdims
        )  # (num_samps, out_dims, time, 1 or time)

        if self.gp_mean is not None:
            if sel_outdims is None:
                sel_outdims = jnp.arange(self.gp.kernel.out_dims)
            f_mean += self.gp_mean[None, sel_outdims, None, None]

        return f_mean, f_cov, KL

    def filtered_gp_sample(
        self, prng_state, x_eval, ys_filt, prior, compute_KL, jitter, sel_outdims
    ):
        """
        Include spike history filtering, samples from the joint posterior

        :param jnp.ndarray x_eval: input (num_samps, obs_dims, ts, x_dims)
        :return:
            samples (num_samps, out_dims, eval_locs), KL scalar
        """
        f_samples, KL = self.gp_sample(
            prng_state, x_eval, prior, compute_KL, jitter, sel_outdims
        )

        if self.spikefilter is not None:
            y_filtered, KL_y = self.spikefilter.apply_filter(
                prng_state, ys_filt, True, compute_KL, prior, sel_outdims, jitter
            )
            f_samples += y_filtered
            KL += KL_y

        return f_samples, KL

    def filtered_gp_posterior(
        self,
        prng_state,
        x_samples,
        ys_filt,
        mean_only,
        diag_cov,
        joint_filter_samples,
        compute_KL,
        jitter,
        sel_outdims,
    ):
        """
        Include spike history filtering
        Ignores covariance due to probabilistic filter in f_cov, but can draw from the filter
        posterior when setting joint_samples to True

        :param jnp.ndarray x_samples: input (num_samps, obs_dims, ts, x_dims)
        :param jnp.ndarray ys_filt: shape (num_samps, obs_dims, ts)
        :return:
            mean (num_samps, out_dims, time, 1)
            (co)variance (num_samps, out_dims, time, 1 or time)
            KL scalar
        """
        f_mean, f_cov, KL = self.gp_posterior(
            x_samples, mean_only, diag_cov, compute_KL, jitter, sel_outdims
        )

        if self.spikefilter is not None:
            y_filtered, KL_y = self.spikefilter.apply_filter(
                prng_state,
                ys_filt,
                joint_filter_samples,
                compute_KL,
                False,
                sel_outdims,
                jitter,
            )
            f_mean += y_filtered[..., None]
            KL += KL_y

        return f_mean, f_cov, KL

    def filtered_gp_marg_post_sample(
        self, prng_state, x_samples, ys_filt, compute_KL, jitter, sel_outdims
    ):
        """
        Include spike history filtering, samples from the marginal posterior

        :return:
            samples (num_samps, out_dims, eval_locs), KL scalar
        """
        f_mean, f_cov, KL = self.filtered_gp_posterior(
            prng_state,
            x_samples,
            ys_filt,
            False,
            True,
            True,
            compute_KL,
            jitter,
            sel_outdims,
        )

        prng_state, _ = jr.split(prng_state)
        f_std = safe_sqrt(f_cov)
        f_samples = f_mean + f_std * jr.normal(
            prng_state, shape=f_mean.shape
        )  # (num_samps, ts, f_dims, 1)

        return f_samples[..., 0], KL


class ModulatedFactorized(SparseGPFilterObservations):
    """
    Factorization across time points allows one to rely on latent marginals
    """

    likelihood: FactorizedLikelihood

    def __init__(self, gp, gp_mean, likelihood, spikefilter=None):
        # checks
        if likelihood.array_type != gp.array_type:
            raise ValueError("Likelihood and GP array types must match")
        if likelihood.f_dims != gp.kernel.out_dims:
            raise ValueError("Likelihood and GP output dimensions must match")

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

    def _sample_Y(self, prng_state, ini_Y, f_samples, jitter):
        """
        Sample the spike train autoregressively

        For heteroscedastic, couple filters to the rate parameter in each
        num_f_per_obs group, f_dims = obs_dims x num_f_per_obs

        :param jnp.dnarray ini_spikes: initial spike train (num_samps, obs_dims, filter_length)
        :param jnp.ndarray f_samples: f values of shape (num_samps, out_dims, ts)
        :return:
            spike train of shape ()
        """
        num_samps, obs_dims, ts = f_samples.shape
        prng_states = jr.split(prng_state, ts * num_samps).reshape(ts, num_samps, -1)
        f_samples = f_samples.transpose(2, 0, 1)  # (ts, num_samps, out_dims)

        if self.spikefilter is not None:
            prng_filter, _ = jr.split(prng_state)

            def step(carry, inputs):
                prng_keys, f = inputs
                past_Y = carry  # (mc, obs_dims, ts)

                h, _ = self.spikefilter.apply_filter(
                    prng_filter,  # same filter sample over time
                    past_Y,
                    joint_samples=True,
                    compute_KL=False,
                    prior=False,
                    sel_outdims=None,
                    jitter=jitter,
                )
                f = f.at[:, :: self.likelihood.num_f_per_obs].add(h[..., 0])

                y = vmap(self.likelihood.sample_Y)(prng_keys, f)  # vmap over MC
                past_Y = jnp.concatenate((past_Y[..., 1:], y[..., None]), axis=-1)

                return past_Y, (y, f)

            init = ini_Y
            xs = (prng_states, f_samples)
            _, (Y, f_samples) = lax.scan(step, init, xs)

        else:
            Y = vmap(vmap(self.likelihood.sample_Y))(prng_states, f_samples)

        return Y.transpose(1, 2, 0), f_samples.transpose(
            1, 2, 0
        )  # (num_samps, obs_dims, ts)

    ### variational inference ###
    def variational_expectation(
        self,
        prng_state,
        jitter,
        xs,
        ys,
        ys_filt,
        compute_KL,
        total_samples,
        lik_int_method,
        log_predictive=False,
    ):
        """
        Compute variational expectation of likelihood and KL divergence

        :param jnp.ndarray xs: inputs (num_samps, out_dims, ts, x_dims)
        :param jnp.ndarray ys: observations (obs_dims, ts)
        :param jnp.ndarray ys_filt: observations (obs_dims, ts)
        :param int total_samples: total number of data points in the dataset, for subsampling scaling
        :return:
            likelihood expectation (scaled to total_samples size), KL-divergence
        """
        num_samps, ts = xs.shape[0], xs.shape[2]

        f_mean, f_cov, KL = self.filtered_gp_posterior(
            prng_state,
            xs,
            ys_filt[None, ...] if ys_filt is not None else None,
            mean_only=False,
            diag_cov=True,
            joint_filter_samples=True,
            compute_KL=compute_KL,
            jitter=jitter,
            sel_outdims=None,
        )  # (num_samps, out_dims, ts, 1)

        prng_state = jr.split(prng_state, num_samps * ts).reshape(num_samps, ts, -1)

        f_mean = f_mean.transpose(0, 2, 1, 3)  # (num_samps, ts, out_dims, 1)
        f_cov = vmap(vmap(jnp.diag))(
            f_cov[..., 0].transpose(0, 2, 1)
        )  # (num_samps, ts, out_dims, out_dims)
        ### TODO: add possible block diagonal structure by linear mappings of f_cov
        llf = lambda y, m, c, p: self.likelihood.variational_expectation(
            y, m, c, p, jitter, lik_int_method
        )

        lls = vmap(vmap(llf), (None, 0, 0, 0))(
            ys.T, f_mean, f_cov, prng_state
        )  # vmap (mc, ts)
        if log_predictive:
            Eq = jax.nn.logsumexp(
                lls - jnp.log(num_samps),
                axis=0,  # take mean over num_samps inside log, ts outside log
            ).mean()
        else:
            Eq = lls.mean()  # take mean over num_samps and ts

        return total_samples * Eq, KL

    ### evaluation ###
    def log_conditional_intensity(
        self, prng_state, xs, ys_filt, jitter, sel_outdims, sampling=False
    ):
        """
        :param jnp.ndarray xs: evaluation locations (num_samps, obs_dims, ts, x_dims)
        :param bool sampling: flag to use samples from the posterior, otherwise use the mean,
                              note that if the link function is not log then the the output is
                              not the mean value of the transformed quantity
        """
        if sampling:
            pre_intens, _ = self.filtered_gp_sample(
                prng_state,
                xs,
                ys_filt,
                prior=False,
                compute_KL=False,
                jitter=jitter,
                sel_outdims=None,
            )

        else:
            pre_intens, _, _ = self.filtered_gp_posterior(
                prng_state,
                xs,
                ys_filt,
                mean_only=False,
                diag_cov=True,
                joint_filter_samples=False,
                compute_KL=False,
                jitter=jitter,
                sel_outdims=sel_outdims,
            )  # (num_samps, out_dims, ts, 1), TODO: account for probabilistic filter covariances
            pre_intens = pre_intens[..., 0]

        log_intens = (
            pre_intens
            if self.likelihood.link_type == LinkTypes["log"]
            else safe_log(self.likelihood.inverse_link(pre_intens).transpose(0, 2, 1))
        )
        return log_intens

    def posterior_mean(self, xs, ys_filt, jitter, sel_outdims, quadrature_pts=30):
        """
        Use the posterior mean to perform the time rescaling

        :param jnp.ndarray xs: covariates of shape (ts, x_dims)
        :param jnp.ndarray ys_filt: covariates of shape (obs_dims, ts)
        """
        f_mean, f_var, _ = self.filtered_gp_posterior(
            None,
            xs[None, None],
            ys_filt[None] if ys_filt is not None else None,
            mean_only=False,
            diag_cov=True,
            joint_filter_samples=False,
            compute_KL=False,
            jitter=jitter,
            sel_outdims=sel_outdims,
        )  # (num_samps, out_dims, ts, 1), TODO: account for probabilistic filter covariances
        f_mean, f_var = (
            f_mean[0, ..., 0],
            f_var[0, ..., 0],
        )  # (obs_dims, ts)

        if self.likelihood.link_type == LinkTypes["log"]:
            post_lambda_mean = jnp.exp(f_mean + f_var / 2.0)

        else:  # quadratures
            if sel_outdims is None:
                sel_outdims = jnp.arange(self.likelihood.obs_dims)

            f, w = gauss_hermite(1, quadrature_pts)
            f = jnp.broadcast_to(
                f.T, (len(sel_outdims), quadrature_pts)
            )  # copy over obs_dims, (obs_dims, approx_points)

            f_std = jnp.sqrt(f_var + jitter)  # safe sqrt
            f_points = (
                f_std[..., None] * f[:, None, :] + f_mean[..., None]
            )  # (obs_dims, ts, approx_points)
            integrand = self.likelihood.inverse_link(f_points)
            post_lambda_mean = (w * integrand).sum(-1)

        return post_lambda_mean

    ### sample ###
    def _sample_generative(self, prng_state, x_samples, prior, ini_Y, jitter):
        """
        Sample from the generative model
        """
        prng_states = jr.split(prng_state, 2)

        f_samples, _ = self.gp_sample(
            prng_states[0],
            x_samples,
            prior,
            False,
            jitter,
            sel_outdims=None,
        )  # (samp, evals, f_dim)
        y_samples, filtered_f = self._sample_Y(prng_states[1], ini_Y, f_samples, jitter)

        return y_samples, filtered_f

    def sample_prior(self, prng_state, x_samples, ini_Y, jitter):
        return self._sample_generative(prng_state, x_samples, True, ini_Y, jitter)

    def sample_posterior(self, prng_state, x_samples, ini_Y, jitter):
        """
        Sample from posterior predictive
        """
        return self._sample_generative(prng_state, x_samples, False, ini_Y, jitter)


class RateRescaledRenewal(SparseGPFilterObservations):
    """
    Renewal likelihood GPLVM
    """

    renewal: RenewalLikelihood

    def __init__(self, gp, gp_mean, renewal, spikefilter=None):
        # checks
        if renewal.array_type != gp.array_type:
            raise ValueError("Renewal and GP array types must match")
        if renewal.f_dims != gp.kernel.out_dims:
            raise ValueError("Renewal and GP output dimensions must match")

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

    def _sample_spikes(self, prng_state, ini_spikes, ini_t_tilde, f_samples, jitter):
        """
        Sample the spike train autoregressively

        Initial conditions are values before time points of f_samples

        :param jnp.dnarray ini_spikes: initial spike train for history filter (num_samps, obs_dims, filter_length)
                                       or when no history filter (num_samps, obs_dims, 1)
        :param jnp.ndarray t_tilde_start: initial t_tilde values at start (num_samps, obs_dims)
        """
        vsample_ISI = vmap(self.renewal.sample_ISI)

        num_samps, obs_dims, ts = f_samples.shape
        prng_state = jr.split(prng_state, num_samps)
        ini_rescaled_ISI = vsample_ISI(prng_state)

        prng_state = prng_state[0]
        if self.spikefilter is not None:  #
            prng_filter, prng_state = jr.split(prng_state)

        prng_states = jr.split(prng_state, ts * num_samps).reshape(ts, num_samps, -1)

        if ini_spikes is None:
            ini_spikes = jnp.zeros((num_samps, obs_dims, 1))

        def step(carry, inputs):
            prng_keys, f = inputs  # current time step
            tau_tilde, rescaled_t_spike, past_spikes = carry  # from previous time step

            # compute intensity
            if self.spikefilter is not None:
                h, _ = self.spikefilter.apply_filter(
                    prng_filter,  # same filter sample over time
                    past_spikes,
                    joint_samples=True,
                    compute_KL=False,
                    prior=False,
                    sel_outdims=None,
                    jitter=jitter,
                )
                f += h[..., 0]

            rate = self.renewal.inverse_link(f)  # (num_samps, obs_dims)

            # compute current t_tilde
            tau_tilde = jnp.where(
                (past_spikes[..., -1] > 0),  # previous spikes
                rate * self.renewal.dt,
                tau_tilde + rate * self.renewal.dt,
            )  # (num_samps, obs_dims)

            # log intensities
            log_rate = (
                f if self.renewal.link_type == LinkTypes["log"] else safe_log(rate)
            )
            log_hazard = vmap(self.renewal.log_hazard)(tau_tilde)
            log_lambda_t = log_rate + log_hazard

            # generate spikes
            spikes = jnp.where(
                tau_tilde >= rescaled_t_spike, 1.0, 0.0
            )  # (num_samps, obs_dims)
            rescaled_t_spike += jnp.where(
                tau_tilde >= rescaled_t_spike, vsample_ISI(prng_keys), 0
            )

            if self.spikefilter is not None:
                past_spikes = jnp.concatenate(
                    (past_spikes[..., 1:], spikes[..., None]), axis=-1
                )
            else:
                past_spikes = spikes[..., None]

            return (tau_tilde, rescaled_t_spike, past_spikes), (spikes, log_lambda_t)

        init = (ini_t_tilde, ini_rescaled_ISI, ini_spikes)
        xs = (prng_states, f_samples.transpose(2, 0, 1))
        _, (spikes, log_lambda_ts) = lax.scan(step, init, xs)
        return spikes.transpose(1, 2, 0), log_lambda_ts.transpose(
            1, 2, 0
        )  # (num_samps, obs_dims, ts)

    def _rate_rescale(
        self,
        spikes,
        pre_rates,
        ini_t_tilde,
        compute_ll,
        return_t_tildes,
        unroll,
        min_ISI=1e-6,
    ):
        """
        rate rescaling, computes the log density on the way

        :param jnp.ndarray spikes: (ts, obs_dims)
        :param jnp.ndarray pre_rates: (ts, obs_dims)
        :param jnp.ndarray ini_t_tilde: (obs_dims,), NaN indicates unknown previous spike time
        :param int unroll: number of steps to unroll in JIT compilation
        :param float min_ISI: minimum ISI value for renewal log density
        :return:
            log likelihood (obs_dims,)
            reciprocal of time samples used (obs_dims,)
            rescaled times (ts, obs_dims)
        """

        def step(carry, inputs):
            has_prev_spike, t_tilde, ll, valid_ts, inv_valid_ts = carry
            pre_rate, spike = inputs

            rate = self.renewal.inverse_link(pre_rate)
            t_tilde += self.renewal.dt * rate
            if compute_ll:
                ll += jnp.where(
                    spike,
                    (
                        pre_rate
                        if self.renewal.link_type == LinkTypes["log"]
                        else safe_log(rate)
                    )
                    + has_prev_spike * self.renewal.log_density(t_tilde + min_ISI),
                    0.0, 
                )  # avoid zero ISI for refractory densities numerically

                valid_ts = jnp.where(has_prev_spike, valid_ts + 1.0, valid_ts)
                inv_valid_ts = jnp.where(
                    spike, 1.0 / valid_ts, inv_valid_ts
                )  # only update at spikes

            has_prev_spike = jnp.where(spike, True, has_prev_spike)
            t_tilde_ = jnp.where(spike, 0.0, t_tilde)  # reset after spike
            return (
                has_prev_spike,
                t_tilde_,
                ll,
                valid_ts,
                inv_valid_ts,
            ), t_tilde if return_t_tildes else None
        
        init = (
            ~jnp.isnan(ini_t_tilde),
            jnp.nan_to_num(ini_t_tilde),
            jnp.zeros(self.renewal.obs_dims),
            jnp.ones(
                self.renewal.obs_dims
            ),  # one valid time step once a spike is encountered
            jnp.zeros(self.renewal.obs_dims),
        )
        (_, next_t_tilde, log_lik, _, inv_valid_ts), t_tildes = lax.scan(
            step, init=init, xs=(pre_rates, spikes), unroll=unroll
        )
        return log_lik, inv_valid_ts, next_t_tilde, t_tildes

    ### inference ###
    def variational_expectation(
        self,
        prng_state,
        jitter,
        xs,
        ys,
        ys_filt,
        metadata, 
        compute_KL,
        total_samples,
        lik_int_method,
        joint_samples,
        unroll=10,
        log_predictive=False,
    ):
        """
        Compute variational expectation of likelihood and KL divergence

        :param jnp.ndarray xs: inputs (num_samps, out_dims, ts, x_dims)
        :param jnp.ndarray ys: observations (obs_dims, ts)
        :param jnp.ndarray ys_filt: observations (obs_dims, ts)
        """
        if lik_int_method["type"] != "MC":
            raise ValueError("Rate rescaling must use MC estimation")
        num_samps = lik_int_method["approx_pts"]
        xs = jnp.broadcast_to(xs, (num_samps, *xs.shape[1:]))

        if joint_samples:
            pre_rates, KL = self.filtered_gp_sample(
                prng_state,
                xs,
                ys_filt[None, ...] if ys_filt is not None else None,
                prior=False,
                compute_KL=compute_KL,
                jitter=jitter,
                sel_outdims=None,
            )  # (num_samps, out_dims, time)
        else:
            pre_rates, KL = self.filtered_gp_marg_post_sample(
                prng_state,
                xs,
                ys_filt[None, ...] if ys_filt is not None else None,
                compute_KL=compute_KL,
                jitter=jitter,
                sel_outdims=None,
            )  # (num_samps, out_dims, time)

        spikes = ys.T > 0
        ini_t_tildes = metadata["ini_t_tildes"]  # (num_samps, out_dims)
        rrs = lambda f, ini_ts: self._rate_rescale(
            spikes,
            f.T,
            ini_ts,
            compute_ll=True,
            return_t_tildes=False,
            unroll=unroll,
        )[:3]
        ll, inv_valid_ts, next_t_tildes = vmap(rrs)(pre_rates, ini_t_tildes)
        ll = jnp.sum(
            ll * inv_valid_ts, axis=1
        )  # sum over obs_dims, inv_valid_ts is zero when no ISIs

        if log_predictive:
            Eq = jax.nn.logsumexp(ll - jnp.log(num_samps), axis=0)
        else:
            Eq = ll.mean()  # mean over num_samps

        metadata["ini_t_tildes"] = lax.stop_gradient(next_t_tildes)  # (num_samps, out_dims)
        return total_samples * Eq, KL, metadata

    ### evaluation ###
    def posterior_mean(self, xs, ys_filt, jitter, sel_outdims, quadrature_pts=30):
        """
        Use the posterior mean to perform the time rescaling

        :param jnp.ndarray xs: covariates of shape (ts, x_dims)
        :param jnp.ndarray ys_filt: spikes for filter of shape (obs_dims, ts)
        """
        f_mean, f_var, _ = self.filtered_gp_posterior(
            None,
            xs[None, None],
            ys_filt[None] if ys_filt is not None else None,
            mean_only=False,
            diag_cov=True,
            joint_filter_samples=False,
            compute_KL=False,
            jitter=jitter,
            sel_outdims=sel_outdims,
        )  # (num_samps, out_dims, ts, 1), TODO: account for probabilistic filter covariances
        f_mean, f_var = (
            f_mean[0, ..., 0],
            f_var[0, ..., 0],
        )  # (obs_dims, ts)

        if self.renewal.link_type == LinkTypes["log"]:
            post_lambda_mean = jnp.exp(f_mean + f_var / 2.0)

        else:  # quadratures
            if sel_outdims is None:
                sel_outdims = jnp.arange(self.renewal.obs_dims)

            f, w = gauss_hermite(1, quadrature_pts)
            f = jnp.broadcast_to(
                f.T, (len(sel_outdims), quadrature_pts)
            )  # copy over obs_dims, (obs_dims, approx_points)

            f_std = jnp.sqrt(f_var + jitter)  # safe sqrt
            f_points = (
                f_std[..., None] * f[:, None, :] + f_mean[..., None]
            )  # (obs_dims, ts, approx_points)
            integrand = self.renewal.inverse_link(f_points)
            post_lambda_mean = (w * integrand).sum(-1)

        return post_lambda_mean

    def log_conditional_intensity(
        self,
        prng_state,
        ini_t_tilde,
        xs,
        ys,
        ys_filt,
        jitter,
        sel_outdims,
        sampling=False,
        unroll=10,
    ):
        """
        rho(t) = r(t) p(u) / (1 - int p(u) du)

        :param jnp.ndarray x_eval: evaluation locations (num_samps, obs_dims, ts, x_dims)
        :param jnp.ndarray ys: spike train corresponding to segment (neurons, ts),
                               ys = ys_filt[..., filter_length:] + last time step
        :param bool sampling: flag to use samples from the posterior, otherwise use the mean,
                              note that if the link function is not log then the the output is
                              not the mean value of the transformed quantity
        """
        if sel_outdims is None:
            sel_outdims = jnp.arange(self.renewal.obs_dims)

        num_samps = xs.shape[0]
        ys = jnp.broadcast_to(ys, (num_samps,) + ys.shape[1:])

        if sampling:
            pre_rates, _ = self.filtered_gp_sample(
                prng_state,
                xs,
                ys_filt,
                prior=False,
                compute_KL=False,
                jitter=jitter,
                sel_outdims=None,
            )  # (num_samps, obs_dims, ts)

        else:
            pre_rates, _, _ = self.filtered_gp_posterior(
                prng_state,
                xs,
                ys_filt,
                mean_only=True,
                diag_cov=True,
                joint_filter_samples=False,
                compute_KL=False,
                jitter=jitter,
                sel_outdims=None,
            )  # (num_samps, out_dims, ts, 1), TODO: account for probabilistic filter covariances
            pre_rates = pre_rates[..., 0]

        rates = self.renewal.inverse_link(pre_rates).transpose(0, 2, 1)

        # rate rescaling, rescaled time since last spike
        rrs = lambda spikes, rates, ini_t_tildes: self._rate_rescale(
            spikes, rates, ini_t_tildes, False, True, unroll)[-1]
        
        spikes = (ys > 0).transpose(0, 2, 1)
        tau_tilde = vmap(rrs)(
            spikes, rates, ini_t_tilde, 
        )  # (num_samps, ts, obs_dims)

        log_hazard = vmap(vmap(self.renewal.log_hazard))(tau_tilde)
        log_rates = (
            pre_rates if self.renewal.link_type == LinkTypes["log"] else safe_log(rates)
        )
        log_lambda_t = log_rates + log_hazard.transpose(0, 2, 1)
        return log_lambda_t[:, sel_outdims, :]  # (num_samps, out_dims, ts)

    def sample_conditional_ISI(
        self, prng_state, t_eval, x_cond, jitter, sel_outdims, num_samps=20, prior=True
    ):
        """
        :param jnp.ndarray t_eval: evaluation times since last spike, i.e. ISI (ts,)
        :param jnp.ndarray x_cond: evaluation locations (num_samps, obs_dims, x_dims)
        """
        if sel_outdims is None:
            sel_outdims = jnp.arange(self.renewal.obs_dims)

        f_samples, _ = self.gp_sample(
            prng_state,
            x_cond[..., None, :],
            prior,
            False,
            jitter,
            sel_outdims=None,
        )  # (num_samps, obs_dims, 1)

        rate = self.renewal.inverse_link(f_samples)
        t_tilde_eval = rate * t_eval

        log_dens = vmap(vmap(self.renewal.log_density), 2, 2)(t_tilde_eval)
        return jnp.exp(log_dens)[:, sel_outdims, :]  # (num_samps, obs_dims, ts)

    ### sample ###
    def _sample_generative(
        self, prng_state, x_samples, prior, ini_spikes, ini_t_tilde, jitter
    ):
        """
        Sample from the generative model
        Sample spike trains from the modulated renewal process.
        :return:
            pike train of shape (trials, neuron, timesteps)
        """
        prng_states = jr.split(prng_state, 2)

        f_samples, _ = self.gp_sample(
            prng_states[0],
            x_samples,
            prior,
            False,
            jitter,
            sel_outdims=None,
        )  # (num_samps, obs_dims, 1)

        y_samples, log_lambda_ts = self._sample_spikes(
            prng_states[1], ini_spikes, ini_t_tilde, f_samples, jitter
        )
        return y_samples, log_lambda_ts

    def sample_prior(self, prng_state, x_samples, ini_spikes, ini_t_tilde, jitter):
        return self._sample_generative(
            prng_state, x_samples, True, ini_spikes, ini_t_tilde, jitter
        )

    def sample_posterior(self, prng_state, x_samples, ini_spikes, ini_t_tilde, jitter):
        """
        Sample from posterior predictive
        """
        return self._sample_generative(
            prng_state, x_samples, False, ini_spikes, ini_t_tilde, jitter
        )


class ModulatedRenewal(SparseGPFilterObservations):
    """
    Renewal likelihood, interaction with GP modulator
    """

    renewal: RenewalLikelihood
    pp: PointProcess

    log_scale_tau: jnp.ndarray

    def __init__(self, gp, gp_mean, scale_tau, renewal, spikefilter=None):
        # checks
        if renewal.array_type != gp.array_type:
            raise ValueError("Renewal and GP array types must match")
        if renewal.f_dims != gp.kernel.out_dims:
            raise ValueError("Renewal and GP output dimensions must match")

        super().__init__(gp, gp_mean, spikefilter)
        self.renewal = renewal
        self.pp = PointProcess(
            gp.kernel.out_dims,
            renewal.dt,
            LinkTypes_[renewal.link_type],
            ArrayTypes_[gp.array_type],
        )
        self.log_scale_tau = self._to_jax(np.log(scale_tau))

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
        invscale_tau = jnp.exp(-self.log_scale_tau)

        if self.spikefilter is not None:
            # assert ini_spikes.shape[-1] == self.spikefilter.filter_length
            prng_states = jr.split(prng_state, ts * 2).reshape(ts, 2, -1)
        else:
            prng_states = jr.split(prng_state, ts).reshape(ts, 1, -1)
            ini_spikes = jnp.zeros_like(ini_tau[..., None])

        def step(carry, inputs):
            prng_state, f = inputs  # (num_samps, obs_dims)
            (
                tau,
                past_spikes,
            ) = carry  # (num_samps, obs_dims), (num_samps, obs_dims, hist)

            # compute current tau
            tau = jnp.where(
                (past_spikes[..., -1] > 0),  # spikes
                self.renewal.dt,
                tau + self.renewal.dt,
            )  # (num_samps, obs_dims)

            # compute intensity
            if self.spikefilter is not None:
                h, _ = self.spikefilter.apply_filter(
                    prng_state[0], past_spikes, False, sel_outdims=None
                )
                f += h[..., 0]

            log_modulator = (
                f
                if self.renewal.link_type == LinkTypes["log"]
                else safe_log(self.renewal.inverse_link(f))
            )
            log_hazard = self.renewal.log_hazard(invscale_tau * tau)
            log_lambda_t = log_modulator + log_hazard

            # generate spikes
            p_spike = jnp.minimum(
                jnp.exp(log_lambda_t) * self.renewal.dt, 1.0
            )  # approximate by discrete Bernoulli
            spikes = jr.bernoulli(prng_state[-1], p_spike).astype(
                self.array_dtype()
            )  # (num_samps, obs_dims)

            if self.spikefilter is not None:
                past_spikes = jnp.concatenate(
                    (past_spikes[..., 1:], spikes[..., None]), axis=-1
                )
            else:
                past_spikes = spikes[..., None]

            return (tau, past_spikes), (spikes, log_lambda_t)

        init = (ini_tau, ini_spikes)
        xs = (prng_states, f_samples.transpose(2, 0, 1))
        _, (spikes, log_lambda_ts) = lax.scan(step, init, xs)
        return spikes.transpose(1, 2, 0), log_lambda_ts.transpose(
            1, 2, 0
        )  # (num_samps, obs_dims, ts)

    def _log_hazard(
        self,
        spikes,
        ini_tau,
    ):
        """
        Time since spike component

        ini_tau zero means spike occured right before start

        :param jnp.ndarray spikes: binary spike train (obs_dims, ts)
        :param jnp.ndarray ini_tau: initial times since (obs_dims,)
        """
        spikes = y.T > 0
        invscale_tau = jnp.exp(-self.log_scale_tau)

        def step(carry, inputs):
            spikes = inputs  # (obs_dims,)
            tau = carry  # (obs_dims,)

            # compute current tau
            tau = jnp.where(
                (spikes > 0),  # spikes
                self.renewal.dt,
                tau + self.renewal.dt,
            )  # (num_samps, obs_dims)

            log_hazard = self.renewal.log_hazard(invscale_tau * tau)
            return tau, log_hazard

        init = ini_tau
        _, log_hazards = lax.scan(step, init=init, xs=spikes)
        return log_hazards

    ### inference ###
    def variational_expectation(
        self,
        prng_state,
        jitter,
        xs,
        taus,
        ys,
        ys_filt,
        compute_KL,
        total_samples,
        lik_int_method,
        log_predictive=False,
    ):
        """
        Compute variational expectation of likelihood and KL divergence

        :param jnp.ndarray xs: time since last spikes (num_samps, obs_dims, ts, x_dims)
        :param jnp.ndarray taus: time since last spikes (obs_dims, ts)
        :param jnp.ndarray ys: observations (obs_dims, ts)
        :param jnp.ndarray ys_filt: observations (obs_dims, ts)
        """
        num_samps, ts = xs.shape[0], xs.shape[2]
        invscale_tau = jnp.exp(-self.log_scale_tau)

        f_mean, f_cov, KL = self.filtered_gp_posterior(
            prng_state,
            xs,
            ys_filt[None, ...] if ys_filt is not None else None,
            mean_only=False,
            diag_cov=True,
            joint_filter_samples=True,
            compute_KL=compute_KL,
            jitter=jitter,
            sel_outdims=None,
        )  # (num_samps, out_dims, ts, 1)

        prng_state = jr.split(prng_state, num_samps * ts).reshape(num_samps, ts, -1)

        f_mean = f_mean.transpose(0, 2, 1, 3)  # (num_samps, ts, out_dims, 1)
        f_cov = vmap(vmap(jnp.diag))(
            f_cov[..., 0].transpose(0, 2, 1)
        )  # (num_samps, ts, out_dims, out_dims)

        log_modulator = (
            f_mean
            if self.renewal.link_type == LinkTypes["log"]
            else safe_log(self.renewal.inverse_link(f_mean))
        )
        log_hazard = vmap(self.renewal.log_hazard)(invscale_tau * taus.T)
        log_lambda_t = log_modulator + log_hazard[None, ..., None]

        pre_lambda_t = (
            log_lambda_t
            if self.renewal.link_type == LinkTypes["log"]
            else self.renewal.link(jnp.exp(log_lambda_t))
        )

        llf = lambda y, m, c, p: self.pp.variational_expectation(
            y, m, c, p, jitter, lik_int_method
        )

        if log_predictive:
            Eq = jax.nn.logsumexp(
                vmap(vmap(llf), (None, 0, 0, 0))(ys.T, pre_lambda_t, f_cov, prng_state)
                - jnp.log(num_samps),
                axis=0,  # take mean over num_samps inside log, ts outside log
            ).mean()

        else:
            Eq = vmap(vmap(llf), (None, 0, 0, 0))(
                ys.T, pre_lambda_t, f_cov, prng_state
            ).mean()  # vmap and take mean over num_samps and ts

        return total_samples * Eq, KL

    ### evaluation ###
    def log_conditional_intensity(self, prng_state, ini_tau, xs, ys, ys_filt, jitter):
        """
        lambda(t) = h(tau) * rho(x)

        :param jnp.ndarray x_eval: evaluation locations (num_samps, obs_dims, ts, x_dims)
        :param jnp.ndarray y: spike train corresponding to segment (neurons, ts),
                              y = y_filt[..., filter_length:] + last time step
        """
        num_samps = xs.shape[0]
        ys = jnp.broadcast_to(ys, (num_samps,) + ys.shape[1:])

        pre_modulator, _ = self.filtered_gp_sample(
            prng_state,
            xs,
            ys_filt,
            prior=False,
            compute_KL=False,
            jitter=jitter,
            sel_outdims=None,
        )  # (num_samps, obs_dims, ts)

        log_modulator = (
            pre_modulator
            if self.renewal.link_type == LinkTypes["log"]
            else safe_log(self.renewal.inverse_link(pre_modulator))
        )

        # hazard
        spikes = (y > 0).transpose(0, 2, 1)
        log_hazards = vmap(self._log_hazard, (0, 0))(
            spikes, ini_tau
        )  # (num_samps, ts, obs_dims)

        log_lambda_t = log_hazards + log_modulator
        return log_lambda_t  # (num_samps, out_dims, ts)

    def posterior_mean(
        self, prng_state, xs, taus, ys_filt, jitter, sel_outdims, quadrature_pts=30
    ):
        """
        Use the posterior mean to perform the time rescaling

        :param jnp.ndarray xs: covariates of shape (ts, x_dims)
        :param jnp.ndarray taus: covariates of shape (obs_dims, ts)
        :param jnp.ndarray ys: covariates of shape (obs_dims, ts)
        """
        if sel_outdims is None:
            sel_outdims = jnp.arange(self.renewal.obs_dims)
        invscale_tau = jnp.exp(-self.log_scale_tau)
        f_mean, f_var, _ = self.filtered_gp_posterior(
            prng_state,
            xs[None, None],
            ys_filt,
            mean_only=False,
            diag_cov=True,
            joint_filter_samples=False,
            compute_KL=False,
            jitter=jitter,
            sel_outdims=sel_outdims,
        )  # (num_samps, out_dims, ts, 1), TODO: ignore filter contribution to f_var
        f_mean, f_var = (
            f_mean[0, ..., 0],
            f_var[0, ..., 0],
        )  # (obs_dims, ts)
        log_hazard = vmap(self.renewal.log_hazard)(invscale_tau * taus.T).T[sel_outdims]

        if self.renewal.link_type == LinkTypes["log"]:
            log_lambda_t_mean = f_mean + log_hazard
            post_lambda_mean = jnp.exp(log_lambda_t_mean + f_var / 2.0)

        else:  # quadratues
            f, w = gauss_hermite(1, quadrature_pts)
            f = jnp.broadcast_to(
                f.T, (len(sel_outdims), quadrature_pts)
            )  # copy over obs_dims, (obs_dims, approx_points)

            f_std = jnp.sqrt(f_var + jitter)  # safe sqrt
            f_points = (
                f_std[..., None] * f[:, None, :] + f_mean[..., None]
            )  # (obs_dims, ts, approx_points)
            integrand = self.renewal.inverse_link(f_points) * jnp.exp(log_hazard)
            post_lambda_mean = (w * integrand).sum(-1)

        return post_lambda_mean

    ### sample ###
    def sample_conditional_ISI(
        self,
        prng_state,
        t_eval,
        x_cond,
        jitter,
        num_samps,
        prior,
    ):
        """
        :param jnp.ndarray t_eval: evaluation times since last spike, i.e. ISI (ts,)
        :param jnp.ndarray x_cond: evaluation locations (num_samps, obs_dims, x_dims)
        """
        f_samples, _ = self.gp_sample(
            prng_state,
            x_cond[..., None, :],
            prior,
            False,
            jitter,
            sel_outdims=None,
        )  # (num_samps, obs_dims, 1)

        modulator = self.renewal.inverse_link(f_samples)
        tau_eval = modulator * t_eval

        log_dens = vmap(vmap(self.renewal.log_density), 2, 2)(tau_eval)
        return jnp.exp(log_dens)  # (num_samps, obs_dims, ts)

    def _sample_generative(
        self, prng_state, x_samples, prior, ini_spikes, ini_tau, jitter
    ):
        """
        Sample from the generative model
        Sample spike trains from the modulated renewal process.
        :return:
            pike train of shape (trials, neuron, timesteps)
        """
        prng_states = jr.split(prng_state, 2)

        f_samples, _ = self.gp_sample(
            prng_states[0],
            x_samples,
            prior,
            False,
            jitter,
            sel_outdims=None,
        )  # (num_samps, obs_dims, 1)

        y_samples, log_lambda_ts = self._sample_spikes(
            prng_states[1], ini_spikes, ini_tau, f_samples
        )
        return y_samples, log_lambda_ts

    def sample_prior(self, prng_state, x_samples, ini_spikes, ini_tau, jitter):
        return self._sample_generative(
            prng_state, x_samples, True, ini_spikes, ini_tau, jitter
        )

    def sample_posterior(self, prng_state, x_samples, ini_spikes, ini_tau, jitter):
        """
        Sample from posterior predictive
        """
        return self._sample_generative(
            prng_state, x_samples, False, ini_spikes, ini_tau, jitter
        )
