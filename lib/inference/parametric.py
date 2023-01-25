import math

from functools import partial
from typing import Union

import equinox as eqx

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from ..GP.markovian import GaussianLTI
from ..likelihoods.base import FactorizedLikelihood

from .base import FilterObservations

_log_twopi = math.log(2 * math.pi)


class IdentityFactorized(FilterObservations):
    """
    Factorization across time points allows one to rely on latent marginals
    """

    likelihood: FactorizedLikelihood

    def __init__(self, likelihood, spikefilter=None):
        super().__init__(spikefilter, likelihood.array_type)
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
            prng_states = jr.split(prng_state, ts * (num_samps + 1)).reshape(
                ts, num_samps + 1, -1
            )

            def step(carry, inputs):
                prng_keys, f = inputs
                past_Y = carry  # (mc, obs_dims, ts)

                h, _ = self.spikefilter.apply_filter(
                    prng_keys[-1], past_Y, compute_KL=False
                )
                f = f.at[:, :: self.likelihood.num_f_per_obs].add(h[..., 0])

                y = vmap(self.likelihood.sample_Y)(prng_keys[:-1], f)  # vmap over MC
                past_Y = jnp.concatenate((past_Y[..., 1:], y[..., None]), axis=-1)

                return past_Y, (y, f)

            init = ini_Y
            xs = (prng_states, f_samples.transpose(2, 0, 1))
            _, (Y, f_filtered) = lax.scan(step, init, xs)
            return Y.transpose(1, 2, 0), f_filtered.transpose(
                1, 2, 0
            )  # (num_samps, obs_dims, ts)

        else:
            prng_states = jr.split(prng_state, ts * num_samps).reshape(
                num_samps, ts, -1
            )
            Y = vmap(vmap(self.likelihood.sample_Y), (1, 2), 2)(prng_states, f_samples)
            return Y, f_samples

    ### variational inference ###
    def ELBO(
        self,
        prng_state,
        num_samps,
        x,
        y,
    ):
        """
        Compute ELBO

        :param jnp.ndarray x: inputs (num_samps, out_dims, ts, x_dims)
        """
        f_mean, f_cov, KL_f, _ = self.gp.evaluate_posterior(
            prng_state, x_samples, jitter, compute_KL=True
        )  # (num_samps, out_dims, ts, 1)

        if self.spikefilter is not None:
            y_filtered, KL_y = self.spikefilter.apply_filter(
                prng_state, y, compute_KL=True
            )
            prng_state, _ = jr.split(prng_state)
            f_mu = f_mean + y_filtered

        Ell = self.likelihood.variational_expectation(
            prng_state, jitter, y, f_mu, f_cov
        )

        ELBO = Ell - KL_x - KL_f - KL_y
        return ELBO

    ### evaluation ###
    def evaluate_pre_conditional_rate(self, prng_state, x_eval, y, jitter):
        """
        evaluate posterior rate
        """
        pre_rates_mean, pre_rates_cov, _, _ = self.gp.evaluate_posterior(
            x_eval,
            mean_only=False,
            diag_cov=False,
            compute_KL=False,
            compute_aux=False,
            jitter=jitter,
        )
        pre_rates_mean = pre_rates_mean[..., 0]

        if self.spikefilter is not None:
            y_filtered, _ = self.spikefilter.apply_filter(
                prng_state, y[..., :-1], compute_KL=False
            )  # leave out last time step (causality)
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

        f_samples = self._gp_sample(
            prng_states[1], x_samples, True, jitter
        )  # (samp, evals, f_dim)
        y_samples, filtered_f = self._sample_Y(prng_states[2], ini_Y, f_samples)

        return y_samples, filtered_f

    def sample_posterior(self, prng_state, num_samps, x_samples, ini_Y, jitter):
        """
        Sample from posterior predictive
        """
        prng_states = jr.split(prng_state, 3)

        f_samples = self._gp_sample(
            prng_states[1], x_samples, False, jitter
        )  # (samp, evals, f_dim)
        y_samples, filtered_f = self._sample_Y(prng_states[2], ini_Y, f_samples)

        return y_samples, filtered_f


class LinearFactorized(FilterObservations):
    """
    Factorization across time points allows one to rely on latent marginals
    """

    A: jnp.ndarray
    b: jnp.ndarray
    likelihood: FactorizedLikelihood

    def __init__(self, gp, likelihood, spikefilter=None):
        # checks
        assert likelihood.array_type == gp.array_type
        assert likelihood.f_dims == gp.kernel.out_dims

        super().__init__(spikefilter, gp.array_type)
        self.gp = gp
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

    def _sample_Y(self, prng_state, ini_Y, f_samples):
        """
        Sample the spike train autoregressively

        For heteroscedastic, couple filters to the rate parameter in each
        num_f_per_obs group, f_dims = obs_dims x num_f_per_obs

        :param jnp.dnarray ini_spikes: initial spike train (num_samps, obs_dims, filter_length)
        """
        num_samps, obs_dims, ts = f_samples.shape
        if self.spikefilter is not None:
            prng_states = jr.split(prng_state, ts * (num_samps + 1)).reshape(
                ts, num_samps + 1, -1
            )

            def step(carry, inputs):
                prng_keys, f = inputs
                past_Y = carry  # (mc, obs_dims, ts)

                h, _ = self.spikefilter.apply_filter(
                    prng_keys[-1], past_Y, compute_KL=False
                )
                f = f.at[:, :: self.likelihood.num_f_per_obs].add(h[..., 0])

                y = vmap(self.likelihood.sample_Y)(prng_keys[:-1], f)  # vmap over MC
                past_Y = jnp.concatenate((past_Y[..., 1:], y[..., None]), axis=-1)

                return past_Y, (y, f)

            init = ini_Y
            xs = (prng_states, f_samples.transpose(2, 0, 1))
            _, (Y, f_filtered) = lax.scan(step, init, xs)
            return Y.transpose(1, 2, 0), f_filtered.transpose(
                1, 2, 0
            )  # (num_samps, obs_dims, ts)

        else:
            prng_states = jr.split(prng_state, ts * num_samps).reshape(
                num_samps, ts, -1
            )
            Y = vmap(vmap(self.likelihood.sample_Y), (1, 2), 2)(prng_states, f_samples)
            return Y, f_samples

    ### variational inference ###
    def ELBO(
        self,
        prng_state,
        num_samps,
        x,
        y,
    ):
        """
        Compute ELBO

        :param jnp.ndarray x: inputs (num_samps, out_dims, ts, x_dims)
        """
        f_mean, f_cov, KL_f, _ = self.gp.evaluate_posterior(
            prng_state, x_samples, jitter, compute_KL=True
        )  # (evals, samp, f_dim)

        if self.spikefilter is not None:
            y_filtered, KL_y = self.spikefilter.apply_filter(
                prng_state, y, compute_KL=True
            )
            prng_state, _ = jr.split(prng_state)
            f_mu = f_mean + y_filtered

        Ell = self.likelihood.variational_expectation(
            prng_state, jitter, y, f_mu, f_cov
        )

        ELBO = Ell - KL_x - KL_f - KL_y
        return ELBO

    ### evaluation ###
    def evaluate_pre_conditional_rate(self, prng_state, x_eval, y, jitter):
        """
        evaluate posterior rate
        """
        pre_rates_mean, pre_rates_cov, _, _ = self.gp.evaluate_posterior(
            x_eval,
            mean_only=False,
            diag_cov=False,
            compute_KL=False,
            compute_aux=False,
            jitter=jitter,
        )
        pre_rates_mean = pre_rates_mean[..., 0]

        if self.spikefilter is not None:
            y_filtered, _ = self.spikefilter.apply_filter(
                prng_state, y[..., :-1], compute_KL=False
            )  # leave out last time step (causality)
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

        f_samples = self._gp_sample(
            prng_states[1], x_samples, True, jitter
        )  # (samp, evals, f_dim)
        y_samples, filtered_f = self._sample_Y(prng_states[2], ini_Y, f_samples)

        return y_samples, filtered_f

    def sample_posterior(self, prng_state, num_samps, x_samples, ini_Y, jitter):
        """
        Sample from posterior predictive
        """
        prng_states = jr.split(prng_state, 3)

        f_samples = self._gp_sample(
            prng_states[1], x_samples, False, jitter
        )  # (samp, evals, f_dim)
        y_samples, filtered_f = self._sample_Y(prng_states[2], ini_Y, f_samples)

        return y_samples, filtered_f
