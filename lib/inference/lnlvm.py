from typing import Union

import math
import numpy as np

from functools import partial

import equinox as eqx

import jax
import jax.numpy as jnp
import jax.random as jr


from .base import FilterModule
from ..GP.markovian import GaussianLTI

_log_twopi = math.log(2 * math.pi)




class FilterGLM(FilterModule):
    """
    The input-ouput (IO) mapping is deterministic or stochastic
    
    Sample from prior, then use cubature for E_q(f|x) log p(y|f)
    
    Examples: GPFA, GLM (deterministic), GPLVM (stochastic)
    
    Spiketrain filter + GLM with optional SSGP latent states
    """
    
    ssgp: Union[GaussianLTI, None]
        
    def __init__(self, ssgp, spikefilter):
        # checks
        super().__init__(spikefilter)
        self.ssgp = ssgp
        self.spikefilter = spikefilter