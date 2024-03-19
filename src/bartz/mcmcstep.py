# bartz/src/bartz/mcmcstep.py
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

"""
Functions that implement the BART posterior MCMC initialization and update step.

Functions that do MCMC steps operate by taking as input a bart state, and
outputting a new dictionary with the new state. The input dict/arrays are not
modified.

In general, integer types are chosen to be the minimal types that contain the
range of possible values.
"""

import functools
import math

import jax
from jax import random
from jax import numpy as jnp
from jax import lax

from . import grove

def make_bart(*,
    X,
    y,
    max_split,
    num_trees,
    p_nonterminal,
    sigma2_alpha,
    sigma2_beta,
    small_float_dtype=jnp.float32,
    large_float_dtype=jnp.float32,
    min_points_per_leaf=None,
    ):
    """
    Make a BART posterior sampling MCMC initial state.

    Parameters
    ----------
    X : int array (p, n)
        The predictors. Note this is trasposed compared to the usual convention.
    y : float array (n,)
        The response.
    max_split : int array (p,)
        The maximum split index for each variable. All split ranges start at 1.
    num_trees : int
        The number of trees in the forest.
    p_nonterminal : float array (d - 1,)
        The probability of a nonterminal node at each depth. The maximum depth
        of trees is fixed by the length of this array.
    sigma2_alpha : float
        The shape parameter of the inverse gamma prior on the noise variance.
    sigma2_beta : float
        The scale parameter of the inverse gamma prior on the noise variance.
    small_float_dtype : dtype, default float32
        The dtype for large arrays used in the algorithm.
    large_float_dtype : dtype, default float32
        The dtype for scalars, small arrays, and arrays which require accuracy.
    min_points_per_leaf : int, optional
        The minimum number of data points in a leaf node. 0 if not specified.

    Returns
    -------
    bart : dict
        A dictionary with array values, representing a BART mcmc state. The
        keys are:

        'leaf_trees' : int array (num_trees, 2 ** d)
            The leaf values of the trees.
        'var_trees' : int array (num_trees, 2 ** (d - 1))
            The variable indices of the trees. The bottom level is missing since
            it can only contain leaves.
        'split_trees' : int array (num_trees, 2 ** (d - 1))
            The splitting points.
        'resid' : large_float_dtype array (n,)
            The residuals (data minus forest value). Large float to avoid
            roundoff.
        'sigma2' : large_float_dtype
            The noise variance.
        'grow_prop_count', 'prune_prop_count' : int
            The number of grow/prune proposals made during one full MCMC cycle.
        'grow_acc_count', 'prune_acc_count' : int
            The number of grow/prune moves accepted during one full MCMC cycle.
        'p_nonterminal' : large_float_dtype array (d - 1,)
            The probability of a nonterminal node at each depth.
        'sigma2_alpha' : large_float_dtype
            The shape parameter of the inverse gamma prior on the noise variance.
        'sigma2_beta' : large_float_dtype
            The scale parameter of the inverse gamma prior on the noise variance.
        'max_split' : int array (p,)
            The maximum split index for each variable.
        'y' : small_float_dtype array (n,)
            The response.
        'X' : int array (p, n)
            The predictors.
        'min_points_per_leaf' : int or None
            The minimum number of data points in a leaf node.
        'affluence_trees' : bool array (num_trees, 2 ** (d - 1)) or None
            Whether a non-bottom leaf nodes contains twice `min_points_per_leaf`
            datapoints. If `min_points_per_leaf` is not specified, this is None.
    """

    p_nonterminal = jnp.asarray(p_nonterminal, large_float_dtype)
    max_depth = p_nonterminal.size + 1

    @functools.partial(jax.vmap, in_axes=None, out_axes=0, axis_size=num_trees)
    def make_forest(max_depth, dtype):
        return grove.make_tree(max_depth, dtype)

    bart = dict(
        leaf_trees=make_forest(max_depth, small_float_dtype),
        var_trees=make_forest(max_depth - 1, grove.minimal_unsigned_dtype(X.shape[0] - 1)),
        split_trees=make_forest(max_depth - 1, max_split.dtype),
        resid=jnp.asarray(y, large_float_dtype),
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
        min_points_per_leaf=(
            None if min_points_per_leaf is None else
            jnp.asarray(min_points_per_leaf)
        ),
        affluence_trees=(
            None if min_points_per_leaf is None else
            make_forest(max_depth - 1, bool).at[:, 1].set(y.size >= 2 * min_points_per_leaf)
        ),
    )

    return bart

def mcmc_step(bart, key):
    """
    Perform one full MCMC step on a BART state.

    Parameters
    ----------
    bart : dict
        A BART mcmc state, as created by `make_bart`.
    key : jax.dtypes.prng_key array
        A jax random key.

    Returns
    -------
    bart : dict
        The new BART mcmc state.
    """
    key1, key2 = random.split(key, 2)
    bart = mcmc_sample_trees(bart, key1)
    bart = mcmc_sample_sigma(bart, key2)
    return bart

def mcmc_sample_trees(bart, key):
    """
    Forest sampling step of BART MCMC.

    Parameters
    ----------
    bart : dict
        A BART mcmc state, as created by `make_bart`.
    key : jax.dtypes.prng_key array
        A jax random key.

    Returns
    -------
    bart : dict
        The new BART mcmc state.

    Notes
    -----
    This function zeroes the proposal counters.
    """
    bart = bart.copy()
    for count_var in ['grow_prop_count', 'grow_acc_count', 'prune_prop_count', 'prune_acc_count']:
        bart[count_var] = jnp.zeros_like(bart[count_var])
    
    carry = 0, bart, key
    def loop(carry, _):
        i, bart, key = carry
        key, subkey = random.split(key)
        bart = mcmc_sample_tree(bart, subkey, i)
        return (i + 1, bart, key), None
    
    (_, bart, _), _ = lax.scan(loop, carry, None, len(bart['leaf_trees']))
    return bart

def mcmc_sample_tree(bart, key, i_tree):
    """
    Single tree sampling step of BART MCMC.

    Parameters
    ----------
    bart : dict
        A BART mcmc state, as created by `make_bart`.
    key : jax.dtypes.prng_key array
        A jax random key.
    i_tree : int
        The index of the tree to sample.

    Returns
    -------
    bart : dict
        The new BART mcmc state.
    """
    bart = bart.copy()
    
    y_tree = grove.evaluate_tree_vmap_x(
        bart['X'],
        bart['leaf_trees'][i_tree],
        bart['var_trees'][i_tree],
        bart['split_trees'][i_tree],
        bart['resid'].dtype,
    )
    bart['resid'] += y_tree
    
    key1, key2 = random.split(key, 2)
    bart = mcmc_sample_tree_structure(bart, key1, i_tree)
    bart = mcmc_sample_tree_leaves(bart, key2, i_tree)
    
    y_tree = grove.evaluate_tree_vmap_x(
        bart['X'],
        bart['leaf_trees'][i_tree],
        bart['var_trees'][i_tree],
        bart['split_trees'][i_tree],
        bart['resid'].dtype,
    )
    bart['resid'] -= y_tree
    
    return bart

def mcmc_sample_tree_structure(bart, key, i_tree):
    """
    Single tree structure sampling step of BART MCMC.

    Parameters
    ----------
    bart : dict
        A BART mcmc state, as created by `make_bart`. The ``'resid'`` field
        shall contain only the residuals w.r.t. the other trees.
    key : jax.dtypes.prng_key array
        A jax random key.
    i_tree : int
        The index of the tree to sample.

    Returns
    -------
    bart : dict
        The new BART mcmc state.
    """
    bart = bart.copy()
    
    var_tree = bart['var_trees'][i_tree]
    split_tree = bart['split_trees'][i_tree]
    affluence_tree = (
        None if bart['affluence_trees'] is None else
        bart['affluence_trees'][i_tree]
    )
    
    key1, key2, key3 = random.split(key, 3)
    args = [
        bart['X'],
        var_tree,
        split_tree,
        affluence_tree,
        bart['max_split'],
        bart['p_nonterminal'],
        bart['sigma2'],
        bart['resid'],
        len(bart['var_trees']),
        bart['min_points_per_leaf'],
        key1,
    ]
    grow_var_tree, grow_split_tree, grow_affluence_tree, grow_allowed, grow_ratio = grow_move(*args)

    args[-1] = key2
    prune_var_tree, prune_split_tree, prune_affluence_tree, prune_allowed, prune_ratio = prune_move(*args)

    u0, u1 = random.uniform(key3, (2,))

    p_grow = jnp.where(grow_allowed & prune_allowed, 0.5, grow_allowed)
    try_grow = u0 < p_grow
    try_prune = prune_allowed & ~try_grow

    do_grow = try_grow & (u1 < grow_ratio)
    do_prune = try_prune & (u1 < prune_ratio)

    var_tree = jnp.where(do_grow, grow_var_tree, var_tree)
    split_tree = jnp.where(do_grow, grow_split_tree, split_tree)
    var_tree = jnp.where(do_prune, prune_var_tree, var_tree)
    split_tree = jnp.where(do_prune, prune_split_tree, split_tree)

    bart['var_trees'] = bart['var_trees'].at[i_tree].set(var_tree)
    bart['split_trees'] = bart['split_trees'].at[i_tree].set(split_tree)

    if bart['min_points_per_leaf'] is not None:
        affluence_tree = jnp.where(do_grow, grow_affluence_tree, affluence_tree)
        affluence_tree = jnp.where(do_prune, prune_affluence_tree, affluence_tree)
        bart['affluence_trees'] = bart['affluence_trees'].at[i_tree].set(affluence_tree)

    bart['grow_prop_count'] += try_grow
    bart['grow_acc_count'] += do_grow
    bart['prune_prop_count'] += try_prune
    bart['prune_acc_count'] += do_prune

    return bart

def grow_move(X, var_tree, split_tree, affluence_tree, max_split, p_nonterminal, sigma2, resid, n_tree, min_points_per_leaf, key):
    """
    Tree structure grow move proposal of BART MCMC.

    Parameters
    ----------
    X : array (p, n)
        The predictors.
    var_tree : array (2 ** (d - 1),)
        The variable indices of the tree.
    split_tree : array (2 ** (d - 1),)
        The splitting points of the tree.
    affluence_tree : bool array (2 ** (d - 1),) or None
        Whether a leaf has enough points to be grown.
    max_split : array (p,)
        The maximum split index for each variable.
    p_nonterminal : array (d - 1,)
        The probability of a nonterminal node at each depth.
    sigma2 : float
        The noise variance.
    resid : array (n,)
        The residuals (data minus forest value), computed using all trees but
        the tree under consideration.
    n_tree : int
        The number of trees in the forest.
    min_points_per_leaf : int
        The minimum number of data points in a leaf node.
    key : jax.dtypes.prng_key array
        A jax random key.

    Returns
    -------
    var_tree : array (2 ** (d - 1),)
        The new variable indices of the tree.
    split_tree : array (2 ** (d - 1),)
        The new splitting points of the tree.
    affluence_tree : bool array (2 ** (d - 1),) or None
        The new indicator whether a leaf has enough points to be grown.
    allowed : bool
        Whether the move is allowed.
    ratio : float
        The Metropolis-Hastings ratio.

    Notes
    -----
    This moves picks a leaf node and converts it to a non-terminal node with
    two leaf children. The move is not possible if all the leaves are already at
    maximum depth.
    """

    key1, key2, key3 = random.split(key, 3)
    
    leaf_to_grow, num_growable, num_prunable, allowed = choose_leaf(split_tree, affluence_tree, key1)
    
    var = choose_variable(var_tree, split_tree, max_split, leaf_to_grow, key2)
    var_tree = var_tree.at[leaf_to_grow].set(var.astype(var_tree.dtype))
    
    split = choose_split(var_tree, split_tree, max_split, leaf_to_grow, key3)
    new_split_tree = split_tree.at[leaf_to_grow].set(split.astype(split_tree.dtype))

    likelihood_ratio, new_affluence_tree = compute_likelihood_ratio(X, var_tree, new_split_tree, resid, sigma2, leaf_to_grow, n_tree, min_points_per_leaf)
    
    trans_tree_ratio = compute_trans_tree_ratio(num_growable, num_prunable, split_tree.size, p_nonterminal, leaf_to_grow, split_tree, new_split_tree, affluence_tree, new_affluence_tree)

    ratio = trans_tree_ratio * likelihood_ratio

    return var_tree, new_split_tree, new_affluence_tree, allowed, ratio

def growable_leaves(split_tree, affluence_tree):
    """
    Return a mask indicating the leaf nodes that can be proposed for growth.

    Parameters
    ----------
    split_tree : array (2 ** (d - 1),)
        The splitting points of the tree.
    affluence_tree : bool array (2 ** (d - 1),) or None
        Whether a leaf has enough points to be grown.

    Returns
    -------
    is_growable : bool array (2 ** (d - 1),)
        The mask indicating the leaf nodes that can be proposed to grow, i.e.,
        that are not at the bottom level and have at least two times the number
        of minimum points per leaf.
    allowed : bool
        Whether the grow move is allowed, i.e., there are growable leaves.
    """
    is_growable = grove.is_actual_leaf(split_tree)
    if affluence_tree is not None:
        is_growable &= affluence_tree
    return is_growable, jnp.any(is_growable)

def choose_leaf(split_tree, affluence_tree, key):
    """
    Choose a leaf node to grow in a tree.

    Parameters
    ----------
    split_tree : array (2 ** (d - 1),)
        The splitting points of the tree.
    affluence_tree : bool array (2 ** (d - 1),) or None
        Whether a leaf has enough points to be grown.
    key : jax.dtypes.prng_key array
        A jax random key.

    Returns
    -------
    leaf_to_grow : int
        The index of the leaf to grow. If ``num_growable == 0``, return
        ``split_tree.size``.
    num_growable : int
        The number of leaf nodes that can be grown.
    num_prunable : int
        The number of leaf parents that could be pruned, after converting the
        selected leaf to a non-terminal node.
    allowed : bool
        Whether the grow move is allowed.
    """
    is_growable, allowed = growable_leaves(split_tree, affluence_tree)
    leaf_to_grow = randint_masked(key, is_growable)
    num_growable = jnp.count_nonzero(is_growable)
    is_parent = grove.is_leaves_parent(split_tree.at[leaf_to_grow].set(1))
    num_prunable = jnp.count_nonzero(is_parent)
    return leaf_to_grow, num_growable, num_prunable, allowed

def randint_masked(key, mask):
    """
    Return a random integer in a range, including only some values.

    Parameters
    ----------
    key : jax.dtypes.prng_key array
        A jax random key.
    mask : bool array (n,)
        The mask indicating the allowed values.

    Returns
    -------
    u : int
        A random integer in the range ``[0, n)``, and which satisfies
        ``mask[u] == True``. If all values in the mask are `False`, return `n`.
    """
    ecdf = jnp.cumsum(mask)
    u = random.randint(key, (), 0, ecdf[-1])
    return jnp.searchsorted(ecdf, u, 'right')

def choose_variable(var_tree, split_tree, max_split, leaf_index, key):
    """
    Choose a variable to split on for a new non-terminal node.

    Parameters
    ----------
    var_tree : int array (2 ** (d - 1),)
        The variable indices of the tree.
    split_tree : int array (2 ** (d - 1),)
        The splitting points of the tree.
    max_split : int array (p,)
        The maximum split index for each variable.
    leaf_index : int
        The index of the leaf to grow.
    key : jax.dtypes.prng_key array
        A jax random key.

    Returns
    -------
    var : int
        The index of the variable to split on.

    Notes
    -----
    The variable is chosen among the variables that have a non-empty range of
    allowed splits. If no variable has a non-empty range, return `p`.
    """
    var_to_ignore = fully_used_variables(var_tree, split_tree, max_split, leaf_index)
    return randint_exclude(key, max_split.size, var_to_ignore)

def fully_used_variables(var_tree, split_tree, max_split, leaf_index):
    """
    Return a list of variables that have an empty split range at a given node.

    Parameters
    ----------
    var_tree : int array (2 ** (d - 1),)
        The variable indices of the tree.
    split_tree : int array (2 ** (d - 1),)
        The splitting points of the tree.
    max_split : int array (p,)
        The maximum split index for each variable.
    leaf_index : int
        The index of the node, assumed to be valid for `var_tree`.

    Returns
    -------
    var_to_ignore : int array (d - 2,)
        The indices of the variables that have an empty split range. Since the
        number of such variables is not fixed, unused values in the array are
        filled with `p`. The fill values are not guaranteed to be placed in any
        particular order. Variables may appear more than once.
    """
    
    var_to_ignore = ancestor_variables(var_tree, max_split, leaf_index)
    split_range_vec = jax.vmap(split_range, in_axes=(None, None, None, None, 0))
    l, r = split_range_vec(var_tree, split_tree, max_split, leaf_index, var_to_ignore)
    num_split = r - l
    return jnp.where(num_split == 0, var_to_ignore, max_split.size)

def ancestor_variables(var_tree, max_split, node_index):
    """
    Return the list of variables in the ancestors of a node.

    Parameters
    ----------
    var_tree : int array (2 ** (d - 1),)
        The variable indices of the tree.
    max_split : int array (p,)
        The maximum split index for each variable. Used only to get `p`.
    node_index : int
        The index of the node, assumed to be valid for `var_tree`.

    Returns
    -------
    ancestor_vars : int array (d - 2,)
        The variable indices of the ancestors of the node, from the root to
        the parent. Unused spots are filled with `p`.
    """
    max_num_ancestors = grove.tree_depth(var_tree) - 1
    ancestor_vars = jnp.zeros(max_num_ancestors, grove.minimal_unsigned_dtype(max_split.size))
    carry = ancestor_vars.size - 1, node_index, ancestor_vars
    def loop(carry, _):
        i, index, ancestor_vars = carry
        index >>= 1
        var = var_tree[index]
        var = jnp.where(index, var, max_split.size)
        ancestor_vars = ancestor_vars.at[i].set(var)
        return (i - 1, index, ancestor_vars), None
    (_, _, ancestor_vars), _ = lax.scan(loop, carry, None, ancestor_vars.size)
    return ancestor_vars

def split_range(var_tree, split_tree, max_split, node_index, ref_var):
    """
    Return the range of allowed splits for a variable at a given node.

    Parameters
    ----------
    var_tree : int array (2 ** (d - 1),)
        The variable indices of the tree.
    split_tree : int array (2 ** (d - 1),)
        The splitting points of the tree.
    max_split : int array (p,)
        The maximum split index for each variable.
    node_index : int
        The index of the node, assumed to be valid for `var_tree`.
    ref_var : int
        The variable for which to measure the split range.

    Returns
    -------
    l, r : int
        The range of allowed splits is [l, r).
    """
    max_num_ancestors = grove.tree_depth(var_tree) - 1
    initial_r = 1 + max_split.at[ref_var].get(mode='fill', fill_value=0).astype(jnp.int32)
    carry = 0, initial_r, node_index
    def loop(carry, _):
        l, r, index = carry
        right_child = (index & 1).astype(bool)
        index >>= 1
        split = split_tree[index]
        cond = (var_tree[index] == ref_var) & index.astype(bool)
        l = jnp.where(cond & right_child, jnp.maximum(l, split), l)
        r = jnp.where(cond & ~right_child, jnp.minimum(r, split), r)
        return (l, r, index), None
    (l, r, _), _ = lax.scan(loop, carry, None, max_num_ancestors)
    return l + 1, r

def randint_exclude(key, sup, exclude):
    """
    Return a random integer in a range, excluding some values.

    Parameters
    ----------
    key : jax.dtypes.prng_key array
        A jax random key.
    sup : int
        The exclusive upper bound of the range.
    exclude : int array (n,)
        The values to exclude from the range. Values greater than or equal to
        `sup` are ignored. Values can appear more than once.

    Returns
    -------
    u : int
        A random integer in the range ``[0, sup)``, and which satisfies
        ``u not in exclude``. If all values in the range are excluded, return
        `sup`.
    """
    exclude = jnp.unique(exclude, size=exclude.size, fill_value=sup)
    num_allowed = sup - jnp.count_nonzero(exclude < sup)
    u = random.randint(key, (), 0, num_allowed)
    def loop(u, i):
        return jnp.where(i <= u, u + 1, u), None
    u, _ = lax.scan(loop, u, exclude)
    return u

def choose_split(var_tree, split_tree, max_split, leaf_index, key):
    """
    Choose a split point for a new non-terminal node.

    Parameters
    ----------
    var_tree : int array (2 ** (d - 1),)
        The variable indices of the tree.
    split_tree : int array (2 ** (d - 1),)
        The splitting points of the tree.
    max_split : int array (p,)
        The maximum split index for each variable.
    leaf_index : int
        The index of the leaf to grow. It is assumed that `var_tree` already
        contains the target variable at this index.
    key : jax.dtypes.prng_key array
        A jax random key.

    Returns
    -------
    split : int
        The split point.
    """
    var = var_tree[leaf_index]
    l, r = split_range(var_tree, split_tree, max_split, leaf_index, var)
    return random.randint(key, (), l, r)

def compute_trans_tree_ratio(num_growable, num_prunable, tree_halfsize, p_nonterminal, leaf_to_grow, initial_split_tree, new_split_tree, initial_affluence_tree, new_affluence_tree):
    """
    Compute the product of the transition and prior ratios of a grow move.

    Parameters
    ----------
    num_growable : int
        The number of leaf nodes that can be grown.
    num_prunable : int
        The number of leaf parents that could be pruned, after converting the
        leaf to be grown to a non-terminal node.
    tree_halfsize : int
        Half the length of the tree array, i.e., 2 ** (d - 1).
    p_nonterminal : array (d - 1,)
        The probability of a nonterminal node at each depth.
    leaf_to_grow : int
        The index of the leaf to grow.
    initial_split_tree : array (2 ** (d - 1),)
        The splitting points of the tree, before the leaf is grown.
    new_split_tree : array (2 ** (d - 1),)
        The splitting points of the tree, after the leaf is grown.
    initial_affluence_tree : bool array (2 ** (d - 1),) or None
        Whether a leaf has enough points to be grown, before the leaf is grown.
    new_affluence_tree : bool array (2 ** (d - 1),) or None
        Whether a leaf has enough points to be grown, after the leaf is grown.

    Returns
    -------
    ratio : float
        The transition ratio P(new tree -> old tree) / P(old tree -> new tree)
        times the prior ratio P(new tree) / P(old tree).
    """

    # the two ratios also contain factors num_available_split *
    # num_available_var, but they cancel out

    prune_was_allowed = prune_allowed(initial_split_tree)
    p_grow = jnp.where(prune_was_allowed, 0.5, 1)

    _, grow_again_allowed = growable_leaves(new_split_tree, new_affluence_tree)
    p_prune = jnp.where(grow_again_allowed, 0.5, 1)

    trans_ratio = p_prune * num_growable / (p_grow * num_prunable)

    depth = grove.index_depth(leaf_to_grow, tree_halfsize)
    p_parent = p_nonterminal[depth]
    cp_children = 1 - p_nonterminal.at[depth + 1].get(mode='fill', fill_value=0)
    tree_ratio = cp_children * cp_children * p_parent / (1 - p_parent)

    return trans_ratio * tree_ratio

def compute_likelihood_ratio(X, var_tree, split_tree, resid, sigma2, new_node, n_tree, min_points_per_leaf):
    """
    Compute the likelihood ratio of a grow move.

    Parameters
    ----------
    X : array (p, n)
        The predictors.
    var_tree : array (2 ** (d - 1),)
        The variable indices of the tree, after the grow move.
    split_tree : array (2 ** (d - 1),)
        The splitting points of the tree, after the grow move.
    resid : array (n,)
        The residuals (data minus forest value), for all trees but the one
        under consideration.
    sigma2 : float
        The noise variance.
    new_node : int
        The index of the leaf that has been grown.
    n_tree : int
        The number of trees in the forest.
    min_points_per_leaf : int or None
        The minimum number of data points in a leaf node.

    Returns
    -------
    ratio : float
        The likelihood ratio P(data | new tree) / P(data | old tree).
    affluence_tree : bool array (2 ** (d - 1),) or None
        Whether a leaf has enough points to be grown, after the grow move.
    """

    resid_tree, count_tree = agg_values(
        X,
        var_tree,
        split_tree,
        resid,
        sigma2.dtype,
    )

    left_child = new_node << 1
    right_child = left_child + 1

    left_resid = resid_tree[left_child]
    right_resid = resid_tree[right_child]
    total_resid = left_resid + right_resid

    left_count = count_tree[left_child]
    right_count = count_tree[right_child]
    total_count = left_count + right_count

    sigma_mu2 = 1 / n_tree
    sigma2_left = sigma2 + left_count * sigma_mu2
    sigma2_right = sigma2 + right_count * sigma_mu2
    sigma2_total = sigma2 + total_count * sigma_mu2

    sqrt_term = sigma2 * sigma2_total / (sigma2_left * sigma2_right)

    exp_term = sigma_mu2 / (2 * sigma2) * (
        left_resid * left_resid / sigma2_left +
        right_resid * right_resid / sigma2_right -
        total_resid * total_resid / sigma2_total
    )

    ratio = jnp.sqrt(sqrt_term) * jnp.exp(exp_term)

    if min_points_per_leaf is not None:
        ratio = jnp.where(right_count >= min_points_per_leaf, ratio, 0)
        ratio = jnp.where(left_count >= min_points_per_leaf, ratio, 0)
        affluence_tree = count_tree[:count_tree.size // 2] >= 2 * min_points_per_leaf
    else:
        affluence_tree = None

    return ratio, affluence_tree

def prune_move(X, var_tree, split_tree, affluence_tree, max_split, p_nonterminal, sigma2, resid, n_tree, min_points_per_leaf, key):
    """
    Tree structure prune move proposal of BART MCMC.

    Parameters
    ----------
    X : array (p, n)
        The predictors.
    var_tree : array (2 ** (d - 1),)
        The variable indices of the tree.
    split_tree : array (2 ** (d - 1),)
        The splitting points of the tree.
    affluence_tree : bool array (2 ** (d - 1),) or None
        Whether a leaf has enough points to be grown.
    max_split : array (p,)
        The maximum split index for each variable.
    p_nonterminal : array (d - 1,)
        The probability of a nonterminal node at each depth.
    sigma2 : float
        The noise variance.
    resid : array (n,)
        The residuals (data minus forest value), computed using all trees but
        the tree under consideration.
    n_tree : int
        The number of trees in the forest.
    min_points_per_leaf : int
        The minimum number of data points in a leaf node.
    key : jax.dtypes.prng_key array
        A jax random key.

    Returns
    -------
    var_tree : array (2 ** (d - 1),)
        The new variable indices of the tree.
    split_tree : array (2 ** (d - 1),)
        The new splitting points of the tree.
    affluence_tree : bool array (2 ** (d - 1),) or None
        The new indicator whether a leaf has enough points to be grown.
    allowed : bool
        Whether the move is allowed.
    ratio : float
        The Metropolis-Hastings ratio.
    """
    node_to_prune, num_prunable, num_growable = choose_leaf_parent(split_tree, affluence_tree, key)
    allowed = prune_allowed(split_tree)

    new_split_tree = split_tree.at[node_to_prune].set(0)
    # should I clean up var_tree as well? just for debugging. it hasn't given me problems though

    likelihood_ratio, _ = compute_likelihood_ratio(X, var_tree, split_tree, resid, sigma2, node_to_prune, n_tree, min_points_per_leaf)
    new_affluence_tree = (
        None if affluence_tree is None else
        affluence_tree.at[node_to_prune].set(True)
    )
    trans_tree_ratio = compute_trans_tree_ratio(num_growable, num_prunable, split_tree.size, p_nonterminal, node_to_prune, new_split_tree, split_tree, new_affluence_tree, affluence_tree)

    ratio = trans_tree_ratio * likelihood_ratio
    ratio = 1 / ratio # Question: should I use lax.reciprocal for this?

    return var_tree, new_split_tree, new_affluence_tree, allowed, ratio

def choose_leaf_parent(split_tree, affluence_tree, key):
    """
    Pick a non-terminal node with leaf children to prune in a tree.

    Parameters
    ----------
    split_tree : array (2 ** (d - 1),)
        The splitting points of the tree.
    affluence_tree : bool array (2 ** (d - 1),) or None
        Whether a leaf has enough points to be grown.
    key : jax.dtypes.prng_key array
        A jax random key.

    Returns
    -------
    node_to_prune : int
        The index of the node to prune. If ``num_prunable == 0``, return
        ``split_tree.size``.
    num_prunable : int
        The number of leaf parents that could be pruned.
    num_growable : int
        The number of leaf nodes that can be grown, after pruning the chosen
        node.
    """
    is_prunable = grove.is_leaves_parent(split_tree)
    node_to_prune = randint_masked(key, is_prunable)
    num_prunable = jnp.count_nonzero(is_prunable)

    pruned_split_tree = split_tree.at[node_to_prune].set(0)
    pruned_affluence_tree = (
        None if affluence_tree is None else
        affluence_tree.at[node_to_prune].set(True)
    )
    is_growable_leaf, _ = growable_leaves(pruned_split_tree, pruned_affluence_tree)
    num_growable = jnp.count_nonzero(is_growable_leaf)

    return node_to_prune, num_prunable, num_growable

def prune_allowed(split_tree):
    """
    Return whether a prune move is allowed.

    Parameters
    ----------
    split_tree : array (2 ** (d - 1),)
        The splitting points of the tree.

    Returns
    -------
    allowed : bool
        Whether a prune move is allowed.
    """
    return split_tree.at[1].get(mode='fill', fill_value=0).astype(bool)

def mcmc_sample_tree_leaves(bart, key, i_tree):
    """
    Single tree leaves sampling step of BART MCMC.

    Parameters
    ----------
    bart : dict
        A BART mcmc state, as created by `make_bart`. The ``'resid'`` field
        shall contain the residuals only w.r.t. the other trees.
    key : jax.dtypes.prng_key array
        A jax random key.
    i_tree : int
        The index of the tree to sample.

    Returns
    -------
    bart : dict
        The new BART mcmc state.
    """
    bart = bart.copy()

    resid_tree, count_tree = agg_values(
        bart['X'],
        bart['var_trees'][i_tree],
        bart['split_trees'][i_tree],
        bart['resid'],
        bart['sigma2'].dtype,
    )

    prec_lk = count_tree / bart['sigma2']
    prec_prior = len(bart['leaf_trees'])
    var_post = 1 / (prec_lk + prec_prior) # lax.reciprocal?
    mean_post = resid_tree / bart['sigma2'] * var_post # = mean_lk * prec_lk * var_post

    z = random.normal(key, mean_post.shape, mean_post.dtype)
        # TODO maybe use long float here, I guess this part is not a bottleneck
    leaf_tree = mean_post + z * jnp.sqrt(var_post)
    leaf_tree = leaf_tree.at[0].set(0) # this 0 is used by evaluate_tree
    bart['leaf_trees'] = bart['leaf_trees'].at[i_tree].set(leaf_tree)

    return bart

def agg_values(X, var_tree, split_tree, values, acc_dtype):
    """
    Aggregate values at the leaves of a tree.

    Parameters
    ----------
    X : array (p, n)
        The predictors.
    var_tree : array (2 ** (d - 1),)
        The variable indices of the tree.
    split_tree : array (2 ** (d - 1),)
        The splitting points of the tree.
    values : array (n,)
        The values to aggregate.
    acc_dtype : dtype
        The dtype of the output.

    Returns
    -------
    acc_tree : acc_dtype array (2 ** d,)
        Tree leaves for the tree structure indicated by the arguments, where
        each leaf contains the sum of the `values` whose corresponding `X` fall
        into the leaf.
    count_tree : int array (2 ** d,)
        Tree leaves containing the count of such values.
    """

    depth = grove.tree_depth(var_tree) + 1
    carry = (
        jnp.zeros(values.size, bool),
        jnp.ones(values.size, grove.minimal_unsigned_dtype(2 * var_tree.size - 1)),
        grove.make_tree(depth, acc_dtype),
        grove.make_tree(depth, grove.minimal_unsigned_dtype(values.size - 1)),
    )
    unit_index = jnp.arange(values.size, dtype=grove.minimal_unsigned_dtype(values.size - 1))

    def loop(carry, _):
        leaf_found, node_index, acc_tree, count_tree = carry

        is_leaf = split_tree.at[node_index].get(mode='fill', fill_value=0) == 0
        leaf_count = is_leaf & ~leaf_found
        leaf_values = jnp.where(leaf_count, values, jnp.array(0, values.dtype))
        acc_tree = acc_tree.at[node_index].add(leaf_values)
        count_tree = count_tree.at[node_index].add(leaf_count)
        leaf_found |= is_leaf
        
        split = split_tree[node_index]
        var = var_tree.at[node_index].get(mode='fill', fill_value=0)
        x = X[var, unit_index]
        
        node_index <<= 1
        node_index += x >= split
        node_index = jnp.where(leaf_found, 0, node_index)

        carry = leaf_found, node_index, acc_tree, count_tree
        return carry, None

    (_, _, acc_tree, count_tree), _ = lax.scan(loop, carry, None, depth)
    return acc_tree, count_tree

def mcmc_sample_sigma(bart, key):
    """
    Noise variance sampling step of BART MCMC.

    Parameters
    ----------
    bart : dict
        A BART mcmc state, as created by `make_bart`.
    key : jax.dtypes.prng_key array
        A jax random key.

    Returns
    -------
    bart : dict
        The new BART mcmc state.
    """
    bart = bart.copy()

    resid = bart['resid']
    alpha = bart['sigma2_alpha'] + resid.size / 2
    norm = jnp.dot(resid, resid, preferred_element_type=bart['sigma2_beta'].dtype)
    beta = bart['sigma2_beta'] + norm / 2

    sample = random.gamma(key, alpha)
    bart['sigma2'] = beta / sample

    return bart
