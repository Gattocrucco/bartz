"""
Script to check if jax parallelizes independent branches of the computation.

Run on GPU, timing f(1, 10_000), f(2, 10_000), ..., f(<big number>, 10_000). If
jax is parallelizing, the times stay be the same instead of increasing linearly.
"""

import functools

import jax
from jax import numpy as jnp
from jax import lax

def silly_scan(x, n):
    def loop(x, _):
        return jnp.sin(x + 1), None
    x, _ =  lax.scan(loop, x, None, n)
    return x

@functools.partial(jax.jit, static_argnums=(0, 1))
def f(branches, loops):
    results = []
    x = jnp.ones(())
    for _ in range(branches):
        results.append(silly_scan(x, loops))
    return tuple(results)
