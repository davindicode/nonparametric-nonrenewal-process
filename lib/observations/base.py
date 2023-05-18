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
    Spiketrain filter observation model
    """

    spikefilter: Union[Filter, None]

    def __init__(self, spikefilter, array_type):
        if spikefilter is not None:  # checks
            if spikefilter.array_type != ArrayTypes[array_type]:
                raise ValueError("Spike filter array type inconsistent with model")
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
