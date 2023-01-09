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
    
    def _prior_input_samples(self, prng_state, num_samps, x_eval, t_eval, jitter):
        """
        Combines observed inputs with latent trajectories
        """
        x = []
        
        if x_eval is not None:
            x.append(x_eval)
            
        if self.ssgp is not None:
            x.append(self.ssgp.sample_prior(prng_state, num_samps, t_eval, jitter))
        
        if len(x) > 0:
            x = jnp.concatenate(x, axis=-1)
        else:
            x = None
        return x

    def _posterior_input_samples(self, prng_state, num_samps, x_eval, t_eval, jitter, compute_KL):
        """
        Combines observed inputs with latent trajectories
        """
        x, KL = [], 0.
        
        if x_eval is not None:
            x.append(x_eval)
            
        if self.ssgp is not None:
            x_samples, KL_x = self.ssgp.sample_posterior(
                prng_state, num_samps, t_eval, jitter, compute_KL)  # (tr, time, N, 1)

            x.append(x_samples)
            KL += KL_x
            
        if len(x) > 0:
            x = jnp.concatenate(x, axis=-1)
        else:
            x = None
        return x, KL

    def _posterior_input_marginals(self, prng_state, num_samps, x_eval, t_eval, jitter, compute_KL):
        """
        Combines observed inputs with latent marginal samples
        """
        x, x_cov, KL = [], [], 0.
        
        if x_eval is not None:
            x.append(x_eval)
            x_cov.append(jnp.zeros_like(x_eval))
            
        if self.ssgp is not None:  # filtering-smoothing
            post_mean, post_cov, KL_x = self.ssgp.evaluate_posterior(
                t_eval, False, compute_KL, jitter)
            post_mean = post_mean[..., 0]  # (time, tr, x_dims, 1)

            x.append(post_mean)
            x_cov.append(post_cov)
            KL += KL_x
        
        if len(x) > 0:
            x, x_cov = jnp.concatenate(x, axis=-1), jnp.concatenate(x_cov, axis=-1)
        else:
            x, x_cov = None, None
        return x, x_cov, KL
