# bartz/src/bartz/prepcovars.py
#
# Copyright (c) 2024, Giacomo Petrillo
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

import functools

import jax
from jax import numpy as jnp

from . import jaxext
from . import grove

@functools.partial(jax.jit, static_argnums=(1,))
def quantilized_splits_from_matrix(X, max_bins):
    """
    Determine bins that make the distribution of each predictor uniform.

    Parameters
    ----------
    X : array (p, n)
        A matrix with `p` predictors and `n` observations.
    max_bins : int
        The maximum number of bins to produce.

    Returns
    -------
    splits : array (p, m)
        A matrix containing, for each predictor, the boundaries between bins.
        `m` is ``min(max_bins, n) - 1``, which is an upper bound on the number
        of splits. Each predictor may have a different number of splits; unused
        values at the end of each row are filled with the maximum value
        representable in the type of `X`.
    max_split : array (p,)
        The number of actually used values in each row of `splits`.
    """
    out_length = min(max_bins, X.shape[1]) - 1
    # return _quantilized_splits_from_matrix(X, out_length)
    @functools.partial(jaxext.autobatch, max_io_nbytes=500_000_000)
    def func(X):
        return _quantilized_splits_from_matrix(X, out_length)
    return func(X)

@functools.partial(jax.vmap, in_axes=(0, None))
def _quantilized_splits_from_matrix(x, out_length):
    huge = jaxext.huge_value(x)
    u, actual_length = jaxext.unique(x, size=x.size, fill_value=huge)
    actual_length -= 1
    if jnp.issubdtype(x.dtype, jnp.integer):
        midpoints = u[:-1] + jaxext.ensure_unsigned(u[1:] - u[:-1]) // 2
        indices = jnp.arange(midpoints.size, dtype=jaxext.minimal_unsigned_dtype(midpoints.size - 1))
        midpoints = jnp.where(indices < actual_length, midpoints, huge)
    else:
        midpoints = (u[1:] + u[:-1]) / 2
    indices = jnp.linspace(-1, actual_length, out_length + 2)[1:-1]
    indices = jnp.around(indices).astype(jaxext.minimal_unsigned_dtype(midpoints.size - 1))
        # indices calculation with float rather than int to avoid potential
        # overflow with int32, and to round to nearest instead of rounding down
    decimated_midpoints = midpoints[indices]
    truncated_midpoints = midpoints[:out_length]
    splits = jnp.where(actual_length > out_length, decimated_midpoints, truncated_midpoints)
    max_split = jnp.minimum(actual_length, out_length)
    max_split = max_split.astype(jaxext.minimal_unsigned_dtype(out_length))
    return splits, max_split

@jax.jit
def bin_predictors(X, splits):
    """
    Bin the predictors according to the given splits.

    A value ``x`` is mapped to bin ``i`` iff ``splits[i - 1] < x <= splits[i]``.

    Parameters
    ----------
    X : array (p, n)
        A matrix with `p` predictors and `n` observations.
    splits : array (p, m)
        A matrix containing, for each predictor, the boundaries between bins.
        `m` is the maximum number of splits; each row may have shorter
        actual length, marked by padding unused locations at the end of the
        row with the maximum value allowed by the type.

    Returns
    -------
    X_binned : int array (p, n)
        A matrix with `p` predictors and `n` observations, where each predictor
        has been replaced by the index of the bin it falls into.
    """
    return _bin_predictors(X, splits)

@jax.vmap
def _bin_predictors(x, splits):
    dtype = jaxext.minimal_unsigned_dtype(splits.size)
    return jnp.searchsorted(splits, x).astype(dtype)
