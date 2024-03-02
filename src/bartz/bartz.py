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
    return int(round(math.log2(tree.shape[-1])))

def evaluate_tree(X, depth, leaf_trees, var_trees, split_trees, out_dtype):

    is_forest = leaf_trees.ndim == 2
    
    if is_forest:
        n, _ = leaf_trees.shape
        forest_shape = n,
        tree_index = jnp.arange(n),
    else:
        forest_shape = ()
        tree_index = ()

    carry = (
        jnp.zeros(forest_shape, bool),
        jnp.zeros((), out_dtype),
        jnp.ones(forest_shape, int),
    )

    def loop(carry, _)
        leaf_found, out, node_index = carry

        is_leaf = split_trees[tree_index + (node_index,)] == 0
        leaf_value = leaf_trees[tree_index + (node_index,)]
        if is_forest:
            leaf_sum = jnp.dot(is_leaf, leaf_value) # TODO how should I set preferred_element_dtype?
        else:
            leaf_sum = jnp.where(is_leaf, leaf_value, 0)
        out += leaf_sum
        leaf_found |= is_leaf
        
        split = split_trees[tree_index + (node_index,)]
        var = var_trees[tree_index + (node_index,)]
        x = X[var]
        
        node_index <<= 1
        node_index += x >= split
        node_index = jnp.where(leaf_found, 0, node_index)

        carry = leaf_found, out, node_index
        return carry, _

    (_, out, _), _ = lax.scan(loop, carry, None, depth)
    return out

def minimal_unsigned_dtype(max_value):
    if max_value < 2 ** 8:
        return jnp.uint8
    if max_value < 2 ** 16:
        return jnp.uint16
    if max_value < 2 ** 32:
        return jnp.uint32
    return jnp.uint64

def make_bart(*,
    X: 'Array (p, n) int',
    y: 'Array (n,) float',
    max_split: 'Array (p,) int',
    num_trees: int,
    p_nonterminal: 'Array (d - 1,) float',
    sigma2_alpha: float,
    sigma2_beta: float,
    small_float_dtype: 'dtype',
    large_float_dtype: 'dtype',
    ):

    p_nonterminal = jnp.asarray(p_nonterminal, large_float_dtype)
    max_depth = p_nonterminal.size + 1

    @functools.partial(jax.vmap, in_axes=None, out_axes=0, axis_size=num_trees)
    def make_forest(dtype):
        return make_tree(max_depth, dtype)

    return dict(
        leaf_trees=make_forest(small_float_dtype),
        var_trees=make_forest(minimal_unsigned_dtype(X.shape[1] - 1)),
        split_trees=make_forest(max_split.dtype),
        y_trees=jnp.zeros(y.size, large_float_dtype), # large float to avoid roundoff
        sigma2=jnp.ones((), large_float_dtype),
        grow_prop_count=jnp.zeros((), int),
        grow_acc_count=jnp.zeros((), int),
        prune_prop_count=jnp.zeros((), int),
        prune_acc_count=jnp.zeros((), int),
        p_nonterminal=p_nonterminal,
        sigma2_alpha=jnp.asarray(sigma2_alpha, large_float_dtype),
        sigma2_beta=jnp.asarray(sigma2_beta, large_float_dtype),
        max_split=max_split,
        y=jnp.asarray(y, small_float_dtype),
        X=X,
    )

def mcmc_step(bart, key):
    key, subkey = random.split(key)
    bart = mcmc_sample_trees(bart, subkey)
    key, subkey = random.split(key)
    bart = mcmc_sample_sigma(bart, subkey)
    return bart

def mcmc_sample_trees(bart, key):
    bart = bart.copy()
    for count_var in ['grow_prop_count', 'grow_acc_count', 'prune_prop_count', 'prune_acc_count']:
        bart[count_var] = bart[count_var].at[:].set(0)
    
    carry = 0, bart, key
    def loop(carry, _):
        i, bart, key = carry
        key, subkey = random.split(key)
        bart = mcmc_sample_tree(bart, subkey, i)
        return (i + 1, bart, key), None
    
    (_, bart, _), _ = lax.scan(loop, carry, None, len(bart['leaf_trees']))
    return bart

@functools.partial(jax.vmap, in_axes=(1, None, None, None, None), out_axes=0)
def evaluate_tree_vmap_x(X, leaf_tree, var_tree, split_tree, out_dtype):
    depth = tree_depth(leaf_trees)
    return evaluate_tree(X, depth, leaf_tree, var_tree, split_tree, out_dtype)

def mcmc_sample_tree(bart, key, i_tree):
    bart = bart.copy()
    
    y_tree = evaluate_tree_vmap_x(
        bart['X'],
        bart['leaf_trees'][i_tree],
        bart['var_trees'][i_tree],
        bart['split_trees'][i_tree],
        bart['y_trees'].dtype,
    )
    bart['y_trees'] -= y_tree
    
    key, subkey = random.split(key)
    bart = mcmc_sample_tree_structure(bart, subkey, i_tree)
    key, subkey = random.split(key)
    bart = mcmc_sample_tree_leaves(bart, subkey, i_tree)
    
    y_tree = evaluate_tree_vmap_x(
        bart['X'],
        bart['leaf_trees'][i_tree],
        bart['var_trees'][i_tree],
        bart['split_trees'][i_tree],
        bart['y_trees'].dtype,
    )
    bart['y_trees'] += y_tree
    
    return bart

def mcmc_sample_tree_structure(bart, key, i_tree):
    bart = bart.copy()
    
    var_tree = bart['var_trees'][i_tree]
    split_tree = bart['split_trees'][i_tree]
    resid = bart['y'] - bart['y_trees']
    
    key, subkey = random.split(key)
    args = [
        bart['X'],
        var_tree,
        split_tree,
        bart['max_split'],
        bart['p_nonterminal'],
        bart['sigma2'],
        resid,
        subkey,
    ]
    grow_var_tree, grow_split_tree, grow_allowed, grow_ratio = grow_move(*args)

    key, args[-1] = random.split(key)
    prune_var_tree, prune_split_tree, prune_allowed, prune_ratio = prune_move(*args)

    key, subkey = random.split(key)
    u0, u1 = random.uniform(subkey, 2)

    p_grow = jnp.where(grow_allowed & prune_allowed, 0.5, grow_allowed)
    try_grow = u0 <= p_grow
    try_prune = prune_allowed & ~do_grow

    do_grow = try_grow & (u1 <= grow_ratio)
    do_prune = try_prune & (u1 <= prune_ratio)

    var_tree = jnp.where(do_grow, grow_var_tree, var_tree)
    split_tree = jnp.where(do_grow, grow_split_tree, split_tree)
    var_tree = jnp.where(do_prune, prune_var_tree, var_tree)
    split_tree = jnp.where(do_prune, prune_split_tree, split_tree)

    bart['var_trees'] = bart['var_trees'].at[i_tree].set(var_tree)
    bart['split_trees'] = bart['split_trees'].at[i_tree].set(split_tree)

    bart['grow_prop_count'] += try_grow
    bart['grow_acc_count'] += do_grow
    bart['prune_prop_count'] += try_prune
    bart['prune_acc_count'] += do_prune

    return bart

def grow_move(X, var_tree, split_tree, max_split, p_nonterminal, sigma2, resid, key):
    is_growable = is_actual_leaf(split_tree[:split_tree.size // 2])
    key, subkey = random.split(key)
    leaf_to_grow = randint_masked(subkey, is_growable)
    allowed = leaf_to_grow < is_growable.size

def is_actual_leaf(split_tree):
    index = jnp.arange(split_tree.size, dtype=minimal_unsigned_dtype(split_tree.size - 1))
    parent_index = index >> 1
    parent_nonleaf = split_tree[parent_index].astype(bool)
    parent_nonleaf = parent_nonleaf.at[1].set(True)
    return (split_tree == 0) & parent_nonleaf

def randint_masked(key, mask):
    ecdf = jnp.cumsum(mask)
    u = random.randint(key, (), 0, ecdf[-1])
    return jnp.searchsorted(ecdf, u, 'right')

def mcmc_sample_tree_leaves(bart, key, i_tree):
    bart = bart.copy()

    resid = bart['y'] - bart['y_trees']
    resid_tree, count_tree = agg_resid(
        bart['X'],
        bart['var_trees'][i_tree],
        bart['split_trees'][i_tree],
        resid,
    )

    mean_lk = resid_tree / count_tree
    prec_lk = count_tree / bart['sigma2']
    prec_prior = len(bart['leaf_trees'])
    var_post = 1 / (prec_lk + prec_prior)
    mean_post = mean_lk * prec_lk * var_post

    key, subkey = random.split(key)
    z = random.normal(subkey, mean_post.size, mean_post.dtype)
    leaf_tree = mean_post + z * jnp.sqrt(var_post)
    leaf_tree = leaf_trees.at[0].set(0)
    bart['leaf_trees'] = bart['leaf_trees'].at(i_tree).set(leaf_tree)

    return bart

def agg_resid(X: 'array (p, n)', var_tree: 'array 2^d', split_tree: 'array 2^d', resid: 'array n'):

    depth = tree_depth(var_trees)
    carry = (
        jnp.zeros(resid.size, bool),
        jnp.ones(resid.size, minimal_unsigned_dtype(var_tree.size - 1))
        make_tree(depth, resid.dtype),
        make_tree(depth, minimal_unsigned_dtype(resid.size - 1)),
    )
    unit_index = jnp.arange(resid.size, minimal_unsigned_dtype(resid.size - 1))

    def loop(carry, _)
        leaf_found, node_index, resid_tree, count_tree = carry

        is_leaf = split_tree[node_index] == 0
        leaf_count = is_leaf & ~leaf_found
        leaf_resid = jnp.where(leaf_count, resid, 0)
        resid_tree = resid_tree.at[node_index].add(leaf_resid)
        count_tree = count_tree.at[node_index].add(leaf_count)
        leaf_found |= is_leaf
        
        split = split_tree[node_index]
        var = var_tree[node_index]
        x = X[var, unit_index]
        
        node_index <<= 1
        node_index += x >= split
        node_index = jnp.where(leaf_found, 0, node_index)

        carry = leaf_found, node_index, resid_tree, count_tree
        return carry, None

    (_, _, resid_tree, count_tree), _ = lax.scan(loop, carry, None, depth)
    return resid_tree, count_tree

def mcmc_sample_sigma(bart, key):
    bart = bart.copy()

    alpha = bart['sigma2_alpha'] + bart['y'].size / 2
    resid = bart['y'] - bart['y_trees']
    norm = jnp.dot(resid, resid, preferred_element_type=bart['sigma2_beta'].dtype)
    beta = bart['sigma2_beta'] + norm / 2

    key, subkey = random.split(key)
    sample = random.gamma(subkey, alpha)
    bart['sigma2'] = beta / sample

    return bart
