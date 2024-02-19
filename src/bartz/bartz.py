# bartz/src/bartz/bartz.py
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

from jax import numpy as jnp
from jax import lax

def make_tree(depth, dtype):
    return jnp.zeros(2 ** depth, dtype)

@functools.partial(jnp.vectorize, excluded=(1,), signature='(),(p),(),(n,t),(n,t),(n,t)->()')
def evaluate_forest(X, depth, leaf_trees, var_trees, split_trees, out_dtype):
    
    carry = (
        jnp.zeros(n, bool),
        jnp.zeros((), out_dtype),
        jnp.zeros(n, int),
    )

    def loop(carry, _)
        leaf_found, out, index = carry

        is_leaf = split_trees[:, index] == 0
        leaf_sum = is_leaf @ leaf_trees[:, index]
        out += leaf_sum
        leaf_found |= is_leaf
        
        split = split_trees[:, index]
        var = var_trees[:, index]
        x = X[var]
        
        index <<= 1
        index += x >= split
        index = jnp.where(leaf_found, 0, index)

        carry = leaf_found, out, index
        return carry, _

    leaf_found, out, index = lax.scan(loop, init, None, depth)
    return out
