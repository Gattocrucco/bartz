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
    key1, key2 = random.split(key, 2)
    bart = mcmc_sample_trees(bart, key1)
    bart = mcmc_sample_sigma(bart, key2)
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
    
    key1, key2 = random.split(key, 2)
    bart = mcmc_sample_tree_structure(bart, key1, i_tree)
    bart = mcmc_sample_tree_leaves(bart, key2, i_tree)
    
    y_tree = evaluate_tree_vmap_x(
        bart['X'],
        bart['leaf_trees'][i_tree],
        bart['var_trees'][i_tree],
        bart['split_trees'][i_tree],
        bart['y_trees'].dtype,
    )
    bart['y_trees'] += y_tree
    
    return bart

@functools.partial(jax.vmap, in_axes=(1, None, None, None, None), out_axes=0)
def evaluate_tree_vmap_x(X, leaf_tree, var_tree, split_tree, out_dtype):
    depth = tree_depth(leaf_trees)
    return evaluate_tree(X, depth, leaf_tree, var_tree, split_tree, out_dtype)

def mcmc_sample_tree_structure(bart, key, i_tree):
    bart = bart.copy()
    
    var_tree = bart['var_trees'][i_tree]
    split_tree = bart['split_trees'][i_tree]
    resid = bart['y'] - bart['y_trees']
    
    key1, key2, key3 = random.split(key, 3)
    args = [
        bart['X'],
        var_tree,
        split_tree,
        bart['max_split'],
        bart['p_nonterminal'],
        bart['sigma2'],
        resid,
        len(bart['var_trees']),
        key1,
    ]
    grow_var_tree, grow_split_tree, grow_allowed, grow_ratio = grow_move(*args)

    args[-1] = key2
    prune_var_tree, prune_split_tree, prune_allowed, prune_ratio = prune_move(*args)

    u0, u1 = random.uniform(key3, 2)

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

def grow_move(X, var_tree, split_tree, max_split, p_nonterminal, sigma2, resid, n_tree, key):
    key1, key2, key3 = random.split(key, 3)
    
    leaf_to_grow, num_growable, num_prunable = choose_leaf(split_tree, key1)
    
    var, num_available_var = choose_variable(var_tree, max_split, leaf_to_grow, key2)
    var_tree = var_tree.at[leaf_to_grow].set(var)
    
    split, num_available_split = choose_split(var_tree, split_tree, max_split, leaf_to_grow, key3)
    split_tree = split_tree.at[leaf_to_grow].set(split)

    allowed = num_growable > 0
    # can_grow_again = num_growable + jnp.where(leaf_to_grow < split_tree.size // 4, 1, -1) > 0

    trans_ratio = compute_trans_ratio(num_growable, num_prunable, num_available_var, num_available_split, split_tree.size)
    log_likelihood_ratio = compute_likelihood_ratio(X, var_tree, split_tree, resid, sigma2, leaf_to_grow, n_tree)
    tree_ratio = compute_tree_ratio() # <----- continue here

def choose_leaf(split_tree, key):
    is_growable = is_actual_leaf(split_tree[:split_tree.size // 2])
    leaf_to_grow = randint_masked(key, is_growable)
    num_growable = jnp.count_nonzero(is_growable)
    is_parent = is_leaf_parent(split_tree[:split_tree.size // 2].at[leaf_to_grow].set(1))
    num_prunable = jnp.count_nonzero(is_parent)
    return leaf_to_grow, num_growable, num_prunable

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

def is_leaf_parent(split_tree):
    index = jnp.arange(split_tree.size, dtype=minimal_unsigned_dtype(2 * (split_tree.size - 1)))
    child_index = index << 1 # left child
    child_leaf = split_tree.at[child_index].get(mode='fill', fill_value=0) == 0
    return split_tree.astype(bool) & child_leaf
        # the 0-th item has split == 0, so it's not counted

def choose_variable(var_tree, max_split, leaf_index, key):
    var_to_ignore = fully_used_variables(var_tree, max_split, leaf_index)
    return randint_exclude(key, max_split.size, var_to_ignore)

def fully_used_variables(var_tree, max_split, leaf_index):
    novar_fill = max_split.size
    nparents = tree_depth(var_tree) - 1
    var = jnp.full(nparents, novar_fill, minimal_unsigned_dtype(novar_fill))
    count = jnp.zeros(nparents, minimal_unsigned_dtype(nparents))

    carry = leaf_index, var, count
    def loop(carry, _):
        index, var, count = carry
        index >>= 1
        parent_var = var_tree[index]
        notfound = var.size
        var_index, = jnp.nonzero(var == parent_var, size=1, fill_value=notfound)
        vacant_index, = jnp.nonzero(var == novar_fill, size=1, fill_value=notfound)
        var_index = jnp.where(var_index == notfound, vacant_index, var_index)
        var = var.at[var_index].set(parent_var, mode='drop')
        count = count.at[var_index].add(index.astype(bool), mode='drop')
        return (index, var, count), None
    (_, var, count), _ = lax.scan(loop, carry, None, nparents)

    max_count = max_split.at[var].get(mode='fill', fill_value=1)
    still_usable = count < max_count
    var = jnp.where(still_usable, novar_fill, var)

    return var

def randint_exclude(key, sup, exclude):
    exclude = jnp.sort(exclude)
    actual_sup = sup - jnp.count_nonzero(exclude < sup)
    u = random.randint(key, (), 0, actual_sup)
    def loop(u, i):
        return jnp.where(i <= u, u + 1, u), None
    u, _ = lax.scan(loop, u, exclude)
    return u, actual_sup

def choose_split(var_tree, split_tree, max_split, leaf_index, key):
    nparents = tree_depth(var_tree) - 1
    var = var_tree[leaf_index]
    carry = 0, max_split[var].astype(jnp.int32) + 1, leaf_index
    def loop(carry, _):
        l, r, index = carry
        right_child = (index & 1).astype(bool)
        index >>= 1
        split = split_tree[index]
        cond = (var_tree[index] == var) & index.astype(bool)
        l = jnp.where(cond & right_child, jnp.maximum(l, split), l)
        r = jnp.where(cond & ~right_child, jnp.minimum(r, split), r)
        return (l, r, index), None
    (l, r, _), _ = lax.scan(loop, carry, None, nparents)
    
    split = random.uniform(key, (), l, r)
    num_available_split = r - l
    return split, num_available_split

def compute_trans_ratio(num_growable, num_prunable, num_available_var, num_available_split, tree_size):
    p_grow = jnp.where(num_growable > 1, 0.5, 1)
        # if num_growable == 1, then the starting tree is a root, and can not be pruned
    p_prune = jnp.where(num_prunable < tree_size // 4, 0.5, 1)
        # if num_prunable == 2^(depth - 2), then all leaf parents are at level depth - 2, which means that all leaves are at level depth - 1, which means the new tree can't be grown again, which means the probability of trying to prune it is 1
    return p_grow / p_prune * num_growable / num_prunable * num_available_var * num_available_split

def compute_likelihood_ratio(X, var_tree, split_tree, resid, sigma2, leaf_to_grow, n_tree):
    resid2_tree, count_tree = agg_resid(
        X,
        var_tree,
        split_tree,
        resid ** 2,
        sigma2.dtype,
    )

    left_child = leaf_to_grow << 1
    right_child = left_child + 1

    left_resid2 = resid2_tree[left_child]
    right_resid2 = resid2_tree[right_child]
    total_resid2 = left_resid2 + right_resid2

    left_count = count_tree[left_child]
    right_count = count_tree[right_child]
    total_count = left_count + right_count

    sigma_mu2 = 1 / n_tree
    sigma2_left = sigma2 + left_count * sigma_mu2
    sigma2_right = sigma2 + right_count * sigma_mu2
    sigma2_total = sigma2 + total_count * sigma_mu2

    sqrt_term = sigma2 * sigma2_total / (sigma2_left * sigma2_right)

    exp_term = sigma_mu2 / (2 * sigma2) * (
        left_resid2 / sigma2_left +
        right_resid2 / sigma2_right -
        total_resid2 / sigma2_total
    )

    return jnp.sqrt(sqrt_term) * jnp.exp(exp_term)

def mcmc_sample_tree_leaves(bart, key, i_tree):
    bart = bart.copy()

    resid = bart['y'] - bart['y_trees']
    resid_tree, count_tree = agg_resid(
        bart['X'],
        bart['var_trees'][i_tree],
        bart['split_trees'][i_tree],
        resid,
        bart['sigma2'].dtype,
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

def agg_resid(X: 'array (p, n)', var_tree: 'array 2^d', split_tree: 'array 2^d', resid: 'array n', long_float_dtype):

    depth = tree_depth(var_trees)
    carry = (
        jnp.zeros(resid.size, bool),
        jnp.ones(resid.size, minimal_unsigned_dtype(var_tree.size - 1))
        make_tree(depth, long_float_dtype),
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
