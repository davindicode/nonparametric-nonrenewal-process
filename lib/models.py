import equinox as eqx

import jax
import jax.numpy as jnp
import jax.random as jr

import numpy as np
from jax import lax, vmap

from .base import ArrayTypes_, module
from .inputs.gaussian import GaussianLatentObservedSeries
from .observations.base import Observations
from .observations.npnr import NonparametricPointProcess
from .observations.svgp import (
    ModulatedFactorized,
    ModulatedRenewal,
    RateRescaledRenewal,
)


class GaussianTwoLayer(module):
    """
    Models with Gaussian input layer and an observation layer
    Includes GPFA and GPLVM models
    """

    inp_model: GaussianLatentObservedSeries
    obs_model: Observations

    def __init__(self, inp_model, obs_model):
        """
        Add latent-observed inputs and GP observation model
        """
        assert type(obs_model) in [
            NonparametricPointProcess,
            ModulatedFactorized,
            ModulatedRenewal,
            RateRescaledRenewal,
        ]
        assert inp_model.array_type == obs_model.array_type
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

    def ELBO(
        self,
        prng_state,
        num_samps,
        jitter,
        tot_ts,
        data,
        metadata,
        lik_int_method,
        joint_samples=False,
        unroll=10,
    ):
        ts, xs, deltas, ys, ys_filt = data
        prng_key_x, prng_key_o = jr.split(prng_state)

        if type(self.obs_model) == ModulatedFactorized:
            xs, KL_x = self.inp_model.sample_marginal_posterior(
                prng_key_x, num_samps, ts, xs, jitter, compute_KL=True
            )
            xs = xs[:, None]  # add dummy out_dims

            Ell, KL_o = self.obs_model.variational_expectation(
                prng_key_o, jitter, xs, ys, ys_filt, True, tot_ts, lik_int_method
            )

        elif type(self.obs_model) == ModulatedRenewal:
            xs, KL_x = self.inp_model.sample_posterior(
                prng_key_x, num_samps, ts, xs, jitter, compute_KL=True
            )
            xs = xs[:, None]  # add dummy out_dims

            taus = deltas[..., 0]
            Ell, KL_o = self.obs_model.variational_expectation(
                prng_key_o, jitter, xs, taus, ys, ys_filt, True, tot_ts, lik_int_method
            )

        elif type(self.obs_model) == RateRescaledRenewal:
            xs, KL_x = self.inp_model.sample_posterior(
                prng_key_x, num_samps, ts, xs, jitter, compute_KL=True
            )
            xs = xs[:, None]  # add dummy out_dims

            Ell, KL_o, metadata = self.obs_model.variational_expectation(
                prng_key_o,
                jitter,
                xs,
                ys,
                ys_filt,
                metadata,
                True,
                tot_ts,
                lik_int_method,
                joint_samples,
                unroll,
            )

        elif type(self.obs_model) == NonparametricPointProcess:
            xs, KL_x = self.inp_model.sample_marginal_posterior(
                prng_key_x, num_samps, ts, xs, jitter, compute_KL=True
            )
            xs = xs[:, None]  # add dummy out_dims

            Ell, KL_o = self.obs_model.variational_expectation(
                prng_key_o, jitter, xs, deltas, ys, True, tot_ts, lik_int_method
            )

        return Ell - KL_o - KL_x, metadata
