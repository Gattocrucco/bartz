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
import math

import jax
from jax import random
from jax import numpy as jnp
from jax import lax

def make_tree(depth, dtype):
    return jnp.zeros(2 ** depth, dtype)

def tree_depth(tree):
    return int(math.log2(tree.shape[-1]))

@functools.partial(jnp.vectorize, excluded=(1, 5), signature='(p),(n,t),(n,t),(n,t)->()')
def evaluate_forest(*, X, depth, leaf_trees, var_trees, split_trees, out_dtype):
    
    n, _ = leaf_trees.shape
    carry = (
        jnp.zeros(n, bool),
        jnp.zeros((), out_dtype),
        jnp.ones(n, int),
    )

    tree_index = jnp.arange(n)

    def loop(carry, _)
        leaf_found, out, node_index = carry

        is_leaf = split_trees[tree_index, node_index] == 0
        leaf_value = leaf_trees[tree_index, node_index]
        leaf_sum = jnp.dot(is_leaf, leaf_value) # TODO how should I set preferred_element_dtype?
        out += leaf_sum
        leaf_found |= is_leaf
        
        split = split_trees[tree_index, node_index]
        var = var_trees[tree_index, node_index]
        x = X[var]
        
        node_index <<= 1
        node_index += x >= split
        node_index = jnp.where(leaf_found, 0, node_index)

        carry = leaf_found, out, node_index
        return carry, _

    leaf_found, out, node_index = lax.scan(loop, init, None, depth)
    return out

def minimal_unsigned_dtype(max_value):
    if max_value < 2 ** 8:
        return jnp.uint8
    if max_value < 2 ** 16:
        return jnp.uint16
    if max_value < 2 ** 32:
        return jnp.uint32
    return jnp.uint64

def make_bart_mcmc(*,
    X: 'Array (n,p) int',
    y: 'Array (n,) float',
    max_split: 'Array (p,) int',
    num_trees: int,
    max_depth: int,
    sigma2_alpha: float,
    sigma2_beta: float,
    small_float_dtype: 'dtype',
    large_float_dtype: 'dtype',
    ):

    @functools.partial(jax.vmap, in_axes=None, out_axes=0, axis_size=num_trees)
    def make_forest(dtype):
        return make_tree(max_depth, dtype)

    return dict(
        state=dict(
            leaf_trees=make_forest(small_float_dtype),
            var_trees=make_forest(minimal_unsigned_dtype(X.shape[1] - 1)),
            split_trees=make_forest(max_split.dtype),
            sigma2=jnp.ones((), large_float_dtype),
            p_accept_grow=jnp.zeros((), long_float_dtype),
            p_accept_prune=jnp.zeros((), long_float_dtype),
        ),
        conf=dict(
            sigma2_alpha=jnp.asarray(sigma2_alpha, large_float_dtype),
            sigma2_beta=jnp.asarray(sigma2_beta, large_float_dtype),
            max_split=max_split,
            y=jnp.asarray(y, small_float_dtype),
            X=X,
        ),
    )

def mcmc_step(bart, key):
    bart = mcmc_sample_forest(bart, key)
    bart = mcmc_sample_sigma(bart, key)
    return bart

def recursive_dict_copy(d):
    if isinstance(d, dict):
        return {k: recursive_dict_copy(v) for k, v in d.items()}
    return d

def mcmc_sample_sigma(bart, key):
    bart = recursive_dict_copy(bart)

    conf = bart['conf']
    state = bart['state']
    
    y = conf['y']
    n = y.size
    alpha = conf['sigma2_alpha'] + n / 2
    mean = evaluate_forest(
        X=conf['X'],
        depth=tree_depth(state['leaf_trees']),
        leaf_trees=state['leaf_trees'],
        var_trees=state['var_trees'],
        split_trees=state['split_trees'],
        out_dtype=y.dtype,
    )
    resid = y - mean
    norm = jnp.dot(resid, resid, preferred_element_type=conf['sigma2_beta'].dtype)
    beta = conf['sigma2_beta'] + norm / 2

    sample = random.gamma(key, alpha)
    state['sigma2'] = beta / sample

    return bart
