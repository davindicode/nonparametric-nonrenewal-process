from typing import Union

import equinox as eqx

import jax
import jax.numpy as jnp
import jax.random as jr

import numpy as np

from ..base import ArrayTypes, ArrayTypes_, module
from ..filters.base import Filter
from ..GP.sparse import SparseGP
from ..utils.jax import safe_sqrt


class Observations(module):
    """
    GP observation model
    """

    def ELBO(self, prng_state, x, t, num_samps):
        raise NotImplementedError


class FilterObservations(Observations):
    """
    Spiketrain filter + GP observation model
    """

    spikefilter: Union[Filter, None]

    def __init__(self, spikefilter, array_type):
        if spikefilter is not None:  # checks
            assert spikefilter.array_type == ArrayTypes[array_type]
        super().__init__(array_type)
        self.spikefilter = spikefilter

    def apply_constraints(self):
        """
        Constrain parameters in optimization
        """
        model = jax.tree_map(lambda p: p, self)  # copy
        if model.spikefilter is not None:
            model = eqx.tree_at(
                lambda tree: tree.spikefilter,
                model,
                replace_fn=lambda obj: obj.apply_constraints(),
            )

        return model


class SparseGPFilterObservations(FilterObservations):
    """
    Uses SVGP to model modulation

    Option to learn mean
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

    def _pre_rho_sample(self, prng_state, x_eval, y_filt, prior, compute_KL, jitter):
        """
        Include spike history filtering
        """
        f_samples, KL = self._gp_sample(prng_state, x_eval, prior, compute_KL, jitter)

        if self.spikefilter is not None:
            y_filtered, KL_y = self.spikefilter.apply_filter(
                prng_state, y_filt, compute_KL=compute_KL
            )
            f_samples += y_filtered[..., None]
            KL += KL_y

        return f_samples, KL

    def _pre_rho_posterior(
        self, prng_state, x_samples, ys_filt, mean_only, diag_cov, compute_KL, jitter
    ):
        """
        Include spike history filtering
        """
        f_mean, f_cov, KL = self._gp_posterior(
            x_samples, mean_only, diag_cov, compute_KL, jitter
        )

        if self.spikefilter is not None:
            y_filtered, KL_y = self.spikefilter.apply_filter(
                prng_state, ys_filt[None, ...], compute_KL=compute_KL
            )
            f_mean += y_filtered[..., None]
            KL += KL_y

        return f_mean, f_cov, KL

    def _pre_rho_marg_sample(self, prng_state, x_samples, ys_filt, compute_KL, jitter):
        """
        Include spike history filtering
        """
        f_mean, f_cov, KL = self._pre_rho_posterior(
            prng_state, x_samples, ys_filt, False, True, compute_KL, jitter
        )

        prng_state, _ = jr.split(prng_state)
        f_std = safe_sqrt(f_cov)
        f_samples = f_mean + f_std * jr.normal(
            prng_state, shape=f_mean.shape
        )  # (num_samps, ts, f_dims, 1)

        return f_samples[..., 0], KL
