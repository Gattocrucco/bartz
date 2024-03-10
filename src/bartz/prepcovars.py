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

def quantilized_splits_from_matrix(X, max_bins):
    out_length = min(max_bins, X.shape[1]) - 1
    return quantilized_splits_from_matrix_impl(X, out_length)

@functools.partial(jax.vmap, in_axes=(0, None))
def quantilized_splits_from_matrix_impl(x, out_length):
    if jnp.issubdtype(x.dtype, jnp.integer):
        huge_value = jnp.iinfo(x.dtype).max
    else:
        huge_value = jnp.inf
    u = jnp.unique(x, size=x.size, fill_value=huge_value)
    actual_length = jnp.count_nonzero(u < huge_value) - 1
    midpoints = (u[1:] + u[:-1]) / 2
    indices = jnp.arange(out_length) * (actual_length - 1) // (out_length - 1) # <-- potential integer overflow with int32!
    decimated_midpoints = midpoints[indices]
    truncated_midpoints = midpoints[:out_length]
    return jnp.where(actual_length > out_length, decimated_midpoints, truncated_midpoints)
