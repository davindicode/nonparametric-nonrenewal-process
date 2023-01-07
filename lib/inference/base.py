from typing import Union

import equinox as eqx

import jax
import jax.numpy as jnp
import jax.random as jr

import numpy as np

from ..base import module
from ..filters.base import Filter
from ..GP.markovian import GaussianLTI
from ..GP.switching import SwitchingLTI
from ..GP.gpssm import DTGPSSM


class FilterModule(module):
    """
    Spiketrain filter + GP with optional SSGP latent states
    """

    spikefilter: Union[Filter, None]

    def __init__(self, spikefilter, array_type):
        if spikefilter is not None:  # checks
            assert spikefilter.array_type == array_type
        super().__init__(array_type)
        self.spikefilter = spikefilter

    def apply_constraints(self):
        """
        Constrain parameters in optimization
        """
        model = jax.tree_map(lambda p: p, self)  # copy
        model = eqx.tree_at(
            lambda tree: tree.spikefilter,
            model,
            replace_fn=lambda obj: obj.apply_constraints(),
        )

        return model

    def _spiketrain_filter(self, prng_state, spktrain):
        """
        Apply the spike train filter
        """
        if self.spikefilter is not None:
            filtered, KL = self.spikefilter.apply_filter(spktrain)
            return filtered, KL
        
        return 0., 0.

    def ELBO(self, prng_state, x, t, num_samps):
        raise NotImplementedError



class FilterGPLVM(FilterModule):
    """
    Spiketrain filter + GP with optional SSGP latent states
    """

    ssgp: Union[GaussianLTI, None]
    switchgp: Union[SwitchingLTI, None]
    gpssm: Union[DTGPSSM, None]

    def __init__(self, ssgp, switchgp, gpssm, spikefilter, array_type):
        if ssgp is not None:  # checks
            assert ssgp.array_type == array_type
            
        if switchgp is not None:  # checks
            assert switchgp.array_type == array_type
            
        if gpssm is not None:  # checks
            assert gpssm.array_type == array_type

        super().__init__(spikefilter, array_type)
        self.ssgp = ssgp
        self.switchgp = switchgp
        self.gpssm = gpssm

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
    
    def _prior_input_samples(self, prng_state, num_samps, t_eval, jitter):
        if self.ssgp is not None:
            prior_samples = self.ssgp.sample_prior(
                prng_state, num_samps, t_eval, jitter)
            
        return prior_samples

    def _posterior_input_samples(self, prng_state, num_samps, t_eval, jitter, compute_KL):
        """
        Combines observed inputs with latent trajectories
        """
        if self.ssgp is not None:
            x_samples, KL = self.ssgp.sample_posterior(
                prng_state, num_samps, t_eval, jitter, compute_KL)  # (tr, time, N, 1)

            return x_samples, KL
        
        return None, 0.

    def _posterior_input_marginals(self, prng_state, num_samps, t_eval, jitter, compute_KL):
        """
        Combines observed inputs with latent marginal samples
        """
        if self.ssgp is not None:  # filtering-smoothing
            post_mean, post_cov, _ = self.ssgp.evaluate_posterior(
                t_eval, False, compute_KL, jitter)
            post_mean = post_mean[..., 0]  # (time, tr, x_dims, 1)

            return post_mean, post_cov, KL
        
        return None, None, 0.
