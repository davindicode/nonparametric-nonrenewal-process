from typing import Any, List, Union

import equinox as eqx

import jax.numpy as jnp


# cannot store strings or callables in equinox modules
ArrayTypes = {
    "float32": 0,
    "float64": 1,
}
ArrayTypes_ = {v: k for k, v in ArrayTypes.items()}


class module(eqx.Module):
    """
    Base equinox module with array type. Note strings are not supported
    for filtered gradient operations in equinox modules, hence using ints
    to enumerate array types. Similar structure in other modules.
    """

    array_type: int

    def __init__(self, array_type):
        """
        :param str array_type: array type string
        """
        self.array_type = ArrayTypes[array_type]

    def array_dtype(self):
        return jnp.dtype(ArrayTypes_[self.array_type])

    def _to_jax(self, _array_like):
        return jnp.array(_array_like, dtype=self.array_dtype())

    def apply_constraints(self):
        return self


class SSM(module):
    """
    State space model with Gaussian pseudo-observations
    """

    site_locs: Union[jnp.ndarray, None]
    site_obs: jnp.ndarray
    site_Lcov: jnp.ndarray

    def __init__(self, site_locs, site_obs, site_Lcov, array_type):
        super().__init__(array_type)
        self.site_locs = self._to_jax(site_locs) if site_locs is not None else None
        self.site_obs = self._to_jax(site_obs)
        self.site_Lcov = self._to_jax(site_Lcov)
