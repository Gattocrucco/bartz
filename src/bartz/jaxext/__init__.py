# bartz/src/bartz/jaxext/__init__.py
#
# Copyright (c) 2024-2025, Giacomo Petrillo
#
# This file is part of bartz.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Additions to jax."""

import functools
import math
from collections.abc import Sequence

import jax
from jax import numpy as jnp
from jax import random
from jax.lax import scan
from jax.scipy.special import ndtr
from jaxtyping import Array, Bool, Float32, Key, Scalar, Shaped

from bartz.jaxext._autobatch import autobatch  # noqa: F401
from bartz.jaxext.scipy.special import ndtri


def vmap_nodoc(fun, *args, **kw):
    """
    Acts like `jax.vmap` but preserves the docstring of the function unchanged.

    This is useful if the docstring already takes into account that the
    arguments have additional axes due to vmap.
    """
    doc = fun.__doc__
    fun = jax.vmap(fun, *args, **kw)
    fun.__doc__ = doc
    return fun


def minimal_unsigned_dtype(value):
    """Return the smallest unsigned integer dtype that can represent `value`."""
    if value < 2**8:
        return jnp.uint8
    if value < 2**16:
        return jnp.uint16
    if value < 2**32:
        return jnp.uint32
    return jnp.uint64


@functools.partial(jax.jit, static_argnums=(1,))
def unique(
    x: Shaped[Array, ' _'], size: int, fill_value: Scalar
) -> tuple[Shaped[Array, ' {size}'], int]:
    """
    Restricted version of `jax.numpy.unique` that uses less memory.

    Parameters
    ----------
    x
        The input array.
    size
        The length of the output.
    fill_value
        The value to fill the output with if `size` is greater than the number
        of unique values in `x`.

    Returns
    -------
    out : Shaped[Array, '{size}']
        The unique values in `x`, sorted, and right-padded with `fill_value`.
    actual_length : int
        The number of used values in `out`.
    """
    if x.size == 0:
        return jnp.full(size, fill_value, x.dtype), 0
    if size == 0:
        return jnp.empty(0, x.dtype), 0
    x = jnp.sort(x)

    def loop(carry, x):
        i_out, last, out = carry
        i_out = jnp.where(x == last, i_out, i_out + 1)
        out = out.at[i_out].set(x)
        return (i_out, x, out), None

    carry = 0, x[0], jnp.full(size, fill_value, x.dtype)
    (actual_length, _, out), _ = scan(loop, carry, x[:size])
    return out, actual_length + 1


class split:
    """
    Split a key into `num` keys.

    Parameters
    ----------
    key
        The key to split.
    num
        The number of keys to split into.
    """

    def __init__(self, key: Key[Array, ''], num: int = 2):
        self._keys = random.split(key, num)

    def __len__(self):
        return self._keys.size

    def pop(self, shape: int | tuple[int, ...] | None = None) -> Key[Array, '*']:
        """
        Pop one or more keys from the list.

        Parameters
        ----------
        shape
            The shape of the keys to pop. If `None`, a single key is popped.
            If an integer, that many keys are popped. If a tuple, the keys are
            reshaped to that shape.

        Returns
        -------
        The popped keys as a jax array with the requested shape.

        Raises
        ------
        IndexError
            If `shape` is larger than the number of keys left in the list.

        Notes
        -----
        The keys are popped from the beginning of the list, so for example
        ``list(keys.pop(2))`` is equivalent to ``[keys.pop(), keys.pop()]``.
        """
        if shape is None:
            shape = ()
        elif not isinstance(shape, tuple):
            shape = (shape,)
        size_to_pop = math.prod(shape)
        if size_to_pop > self._keys.size:
            msg = f'Cannot pop {size_to_pop} keys from {self._keys.size} keys'
            raise IndexError(msg)
        popped_keys = self._keys[:size_to_pop]
        self._keys = self._keys[size_to_pop:]
        return popped_keys.reshape(shape)


def truncated_normal_onesided(
    key: Key[Array, ''],
    shape: Sequence[int],
    upper: Bool[Array, '*'],
    bound: Float32[Array, '*'],
) -> Float32[Array, '*']:
    """
    Sample from a one-sided truncated standard normal distribution.

    Parameters
    ----------
    key
        JAX random key.
    shape
        Shape of output array, broadcasted with other inputs.
    upper
        True for (-∞, bound], False for [bound, ∞).
    bound
        The truncation boundary.

    Returns
    -------
    Array of samples from the truncated normal distribution.
    """
    # Pseudocode:
    # | if upper:
    # |     if bound < 0:
    # |         ndtri(uniform(0, ndtr(bound))) =
    # |         ndtri(ndtr(bound) * u)
    # |     if bound > 0:
    # |         -ndtri(uniform(ndtr(-bound), 1)) =
    # |         -ndtri(ndtr(-bound) + ndtr(bound) * (1 - u))
    # | if not upper:
    # |     if bound < 0:
    # |         ndtri(uniform(ndtr(bound), 1)) =
    # |         ndtri(ndtr(bound) + ndtr(-bound) * (1 - u))
    # |     if bound > 0:
    # |         -ndtri(uniform(0, ndtr(-bound))) =
    # |         -ndtri(ndtr(-bound) * u)
    shape = jnp.broadcast_shapes(shape, upper.shape, bound.shape)
    bound_pos = bound > 0
    ndtr_bound = ndtr(bound)
    ndtr_neg_bound = ndtr(-bound)
    scale = jnp.where(upper, ndtr_bound, ndtr_neg_bound)
    shift = jnp.where(upper, ndtr_neg_bound, ndtr_bound)
    u = random.uniform(key, shape)
    left_u = scale * (1 - u)  # ~ uniform in (0, ndtr(±bound)]
    right_u = shift + scale * u  # ~ uniform in [ndtr(∓bound), 1)
    truncated_u = jnp.where(upper ^ bound_pos, left_u, right_u)
    truncated_norm = ndtri(truncated_u)
    return jnp.where(bound_pos, -truncated_norm, truncated_norm)
