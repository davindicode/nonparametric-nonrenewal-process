from typing import Any, List

import jax.numpy as jnp

import equinox as eqx



class module(eqx.Module):
    array_type: Any
        
    def __init__(self, array_type):
        self.array_type = array_type
    
    def _to_jax(self, _array_like):
        return jnp.array(_array_like, dtype=self.array_type)
    
    def apply_constraints(self):
        return self
    
    
    
class TimeSeriesMapping(module):
    """
    Maps multiple inputs to outputs (MIMO) system
    Designed for VI framework, returns KL divergences
    If mapping is deterministic, we have no posterior covariance and KL is regularizer
    
    Input is a time series X
    Ouput is a time series F
    """
    lat_inputs: List[module]
    
    def __init__(self, obs_inputs, lat_inputs, dtype=jnp.float32):
        super().__init__()
        self.lat_inputs = lat_inputs
        
    def apply_constraints(self):
        def update(lat_inputs):
            for en, latent in enumerate(lat_inputs):
                lat_inputs[en] = latent.apply_constraints()
            return lat_inputs

        model = jax.tree_map(lambda p: p, self)
        model = eqx.tree_at(
            lambda tree: tree.lat_inputs,
            model,
            replace_fn=update,
        )

        return model
    
    
    def marginal_posterior_gaussian_moments(self, inputs):
        """
        :param jnp.ndarray inputs: input array of shape (mc, out_dims, ts, in_dims)
        """
        return mean, cov, KL
    
    def joint_posterior_sample(self, inputs):
        """
        :param jnp.ndarray inputs: input array of shape (mc, out_dims, ts, in_dims)
        """
        return samples, KL

    def set_data(self, t, x_obs, y, mask=None):
        """
        :param t: training inputs
        :param y: training data / observations

        :param y: observed data [N, obs_dim]
        :param dt: step sizes Δtₙ = tₙ - tₙ₋₁ [N, 1]
        """
        self.t, self.dt, self.x_obs, self.y = process_inputs(t, x_obs, y, self.dtype)
        self.mask = mask
