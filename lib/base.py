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
