from typing import Any, List, Union

import equinox as eqx

import jax.numpy as jnp


ArrayTypes = {
    "float32": 0,
    "float64": 1, 
}
ArrayTypes_ = {v: k for k, v in ArrayTypes.items()}


class module(eqx.Module):
    
    array_type: int

    def __init__(self, array_type):
        self.array_type = ArrayTypes[array_type]
        
    def array_dtype(self):
        return jnp.dtype(ArrayTypes_[self.array_type])

    def _to_jax(self, _array_like):
        return jnp.array(_array_like, dtype=self.array_dtype())

    def apply_constraints(self):
        return self
