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

from . import jaxext
from . import grove

def init(*,
    X,
    y,
    max_split,
    num_trees,
    p_nonterminal,
    sigma2_alpha,
    sigma2_beta,
    small_float=jnp.float32,
    large_float=jnp.float32,
    min_points_per_leaf=None,
    resid_batch_size='auto',
    count_batch_size='auto',
    save_ratios=False,
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
    small_float : dtype, default float32
        The dtype for large arrays used in the algorithm.
    large_float : dtype, default float32
        The dtype for scalars, small arrays, and arrays which require accuracy.
    min_points_per_leaf : int, optional
        The minimum number of data points in a leaf node. 0 if not specified.
    resid_batch_size, count_batch_sizes : int, None, str, default 'auto'
        The batch sizes, along datapoints, for summing the residuals and
        counting the number of datapoints in each leaf. `None` for no batching.
        If 'auto', pick a value based on the device of `y`, or the default
        device.
    save_ratios : bool, default False
        Whether to save the Metropolis-Hastings ratios.

    Returns
    -------
    bart : dict
        A dictionary with array values, representing a BART mcmc state. The
        keys are:

        'leaf_trees' : small_float array (num_trees, 2 ** d)
            The leaf values.
        'var_trees' : int array (num_trees, 2 ** (d - 1))
            The decision axes.
        'split_trees' : int array (num_trees, 2 ** (d - 1))
            The decision boundaries.
        'resid' : large_float array (n,)
            The residuals (data minus forest value). Large float to avoid
            roundoff.
        'sigma2' : large_float
            The noise variance.
        'grow_prop_count', 'prune_prop_count' : int
            The number of grow/prune proposals made during one full MCMC cycle.
        'grow_acc_count', 'prune_acc_count' : int
            The number of grow/prune moves accepted during one full MCMC cycle.
        'p_nonterminal' : large_float array (d,)
            The probability of a nonterminal node at each depth, padded with a
            zero.
        'p_propose_grow' : large_float array (2 ** (d - 1),)
            The unnormalized probability of picking a leaf for a grow proposal.
        'sigma2_alpha' : large_float
            The shape parameter of the inverse gamma prior on the noise variance.
        'sigma2_beta' : large_float
            The scale parameter of the inverse gamma prior on the noise variance.
        'max_split' : int array (p,)
            The maximum split index for each variable.
        'y' : small_float array (n,)
            The response.
        'X' : int array (p, n)
            The predictors.
        'leaf_indices' : int array (num_trees, n)
            The index of the leaf each datapoints falls into, for each tree.
        'min_points_per_leaf' : int or None
            The minimum number of data points in a leaf node.
        'affluence_trees' : bool array (num_trees, 2 ** (d - 1)) or None
            Whether a non-bottom leaf nodes contains twice `min_points_per_leaf`
            datapoints. If `min_points_per_leaf` is not specified, this is None.
        'opt' : LeafDict
            A dictionary with config values:

            'small_float' : dtype
                The dtype for large arrays used in the algorithm.
            'large_float' : dtype
                The dtype for scalars, small arrays, and arrays which require
                accuracy.
            'require_min_points' : bool
                Whether the `min_points_per_leaf` parameter is specified.
            'resid_batch_size', 'count_batch_size' : int or None
                The data batch sizes for computing the sufficient statistics.
    """

    p_nonterminal = jnp.asarray(p_nonterminal, large_float)
    p_nonterminal = jnp.pad(p_nonterminal, (0, 1))
    max_depth = p_nonterminal.size

    @functools.partial(jax.vmap, in_axes=None, out_axes=0, axis_size=num_trees)
    def make_forest(max_depth, dtype):
        return grove.make_tree(max_depth, dtype)

    small_float = jnp.dtype(small_float)
    large_float = jnp.dtype(large_float)
    y = jnp.asarray(y, small_float)
    resid_batch_size, count_batch_size = _choose_suffstat_batch_size(resid_batch_size, count_batch_size, y)
    sigma2 = jnp.array(sigma2_beta / sigma2_alpha, large_float)
    sigma2 = jnp.where(jnp.isfinite(sigma2) & (sigma2 > 0), sigma2, 1)

    bart = dict(
        leaf_trees=make_forest(max_depth, small_float),
        var_trees=make_forest(max_depth - 1, jaxext.minimal_unsigned_dtype(X.shape[0] - 1)),
        split_trees=make_forest(max_depth - 1, max_split.dtype),
        resid=jnp.asarray(y, large_float),
        sigma2=sigma2,
        grow_prop_count=jnp.zeros((), int),
        grow_acc_count=jnp.zeros((), int),
        prune_prop_count=jnp.zeros((), int),
        prune_acc_count=jnp.zeros((), int),
        p_nonterminal=p_nonterminal,
        p_propose_grow=p_nonterminal[grove.tree_depths(2 ** (max_depth - 1))],
        sigma2_alpha=jnp.asarray(sigma2_alpha, large_float),
        sigma2_beta=jnp.asarray(sigma2_beta, large_float),
        max_split=jnp.asarray(max_split),
        y=y,
        X=jnp.asarray(X),
        leaf_indices=jnp.ones((num_trees, y.size), jaxext.minimal_unsigned_dtype(2 ** max_depth - 1)),
        min_points_per_leaf=(
            None if min_points_per_leaf is None else
            jnp.asarray(min_points_per_leaf)
        ),
        affluence_trees=(
            None if min_points_per_leaf is None else
            make_forest(max_depth - 1, bool).at[:, 1].set(y.size >= 2 * min_points_per_leaf)
        ),
        opt=jaxext.LeafDict(
            small_float=small_float,
            large_float=large_float,
            require_min_points=min_points_per_leaf is not None,
            resid_batch_size=resid_batch_size,
            count_batch_size=count_batch_size,
        ),
    )

    if save_ratios:
        bart['ratios'] = dict(
            grow=dict(
                trans_prior=jnp.full(num_trees, jnp.nan),
                likelihood=jnp.full(num_trees, jnp.nan),
            ),
            prune=dict(
                trans_prior=jnp.full(num_trees, jnp.nan),
                likelihood=jnp.full(num_trees, jnp.nan),
            ),
        )

    return bart

def _choose_suffstat_batch_size(resid_batch_size, count_batch_size, y):

    @functools.cache
    def get_platform():
        try:
            device = y.devices().pop()
        except jax.errors.ConcretizationTypeError:
            device = jax.devices()[0]
        platform = device.platform
        if platform not in ('cpu', 'gpu'):
            raise KeyError(f'Unknown platform: {platform}')
        return platform

    if resid_batch_size == 'auto':
        platform = get_platform()
        n = max(1, y.size)
        if platform == 'cpu':
            resid_batch_size = 2 ** int(round(math.log2(n / 6))) # n/6
        elif platform == 'gpu':
            resid_batch_size = 2 ** int(round((1 + math.log2(n)) / 3)) # n^1/3
        resid_batch_size = max(1, resid_batch_size)

    if count_batch_size == 'auto':
        platform = get_platform()
        if platform == 'cpu':
            count_batch_size = None
        elif platform == 'gpu':
            n = max(1, y.size)
            count_batch_size = 2 ** int(round(math.log2(n) / 2 - 2)) # n^1/2
                # /4 is good on V100, /2 on L4/T4, still haven't tried A100
            count_batch_size = max(1, count_batch_size)

    return resid_batch_size, count_batch_size

def step(bart, key):
    """
    Perform one full MCMC step on a BART state.

    Parameters
    ----------
    bart : dict
        A BART mcmc state, as created by `init`.
    key : jax.dtypes.prng_key array
        A jax random key.

    Returns
    -------
    bart : dict
        The new BART mcmc state.
    """
    key, subkey = random.split(key)
    bart = sample_trees(bart, subkey)
    return sample_sigma(bart, key)

def sample_trees(bart, key):
    """
    Forest sampling step of BART MCMC.

    Parameters
    ----------
    bart : dict
        A BART mcmc state, as created by `init`.
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
    key, subkey = random.split(key)
    grow_moves, prune_moves = sample_moves(bart, subkey)
    return accept_moves_and_sample_leaves(bart, grow_moves, prune_moves, key)

def sample_moves(bart, key):
    """
    Propose moves for all the trees.

    Parameters
    ----------
    bart : dict
        BART mcmc state.
    key : jax.dtypes.prng_key array
        A jax random key.

    Returns
    -------
    grow_moves, prune_moves : dict
        The proposals for grow and prune moves. See `grow_move` and `prune_move`.
    """
    key = random.split(key, bart['var_trees'].shape[0])
    return _sample_moves_vmap_trees(bart['var_trees'], bart['split_trees'], bart['affluence_trees'], bart['max_split'], bart['p_nonterminal'], bart['p_propose_grow'], key)

@functools.partial(jaxext.vmap_nodoc, in_axes=(0, 0, 0, None, None, None, 0))
def _sample_moves_vmap_trees(*args):
    args, key = args[:-1], args[-1]
    key, key1 = random.split(key)
    grow = grow_move(*args, key)
    prune = prune_move(*args, key1)
    return grow, prune

def grow_move(var_tree, split_tree, affluence_tree, max_split, p_nonterminal, p_propose_grow, key):
    """
    Tree structure grow move proposal of BART MCMC.

    This moves picks a leaf node and converts it to a non-terminal node with
    two leaf children. The move is not possible if all the leaves are already at
    maximum depth.

    Parameters
    ----------
    var_tree : array (2 ** (d - 1),)
        The variable indices of the tree.
    split_tree : array (2 ** (d - 1),)
        The splitting points of the tree.
    affluence_tree : bool array (2 ** (d - 1),) or None
        Whether a leaf has enough points to be grown.
    max_split : array (p,)
        The maximum split index for each variable.
    p_nonterminal : array (d,)
        The probability of a nonterminal node at each depth.
    p_propose_grow : array (2 ** (d - 1),)
        The unnormalized probability of choosing a leaf to grow.
    key : jax.dtypes.prng_key array
        A jax random key.

    Returns
    -------
    grow_move : dict
        A dictionary with fields:

        'num_growable' : int
            The number of growable leaves.
        'node' : int
            The index of the leaf to grow. ``2 ** d`` if there are no growable
            leaves.
        'left', 'right' : int
            The indices of the children of 'node'.
        'var', 'split' : int
            The decision axis and boundary of the new rule.
        'partial_ratio' : float
            A factor of the Metropolis-Hastings ratio of the move. It lacks
            the likelihood ratio and the probability of proposing the prune
            move.
        'var_tree', 'split_tree' : array (2 ** (d - 1),)
            The updated decision axes and boundaries of the tree.
    """

    key, key1, key2 = random.split(key, 3)

    leaf_to_grow, num_growable, prob_choose, num_prunable = choose_leaf(split_tree, affluence_tree, p_propose_grow, key)

    var = choose_variable(var_tree, split_tree, max_split, leaf_to_grow, key1)
    var_tree = var_tree.at[leaf_to_grow].set(var.astype(var_tree.dtype))

    split = choose_split(var_tree, split_tree, max_split, leaf_to_grow, key2)
    split_tree = split_tree.at[leaf_to_grow].set(split.astype(split_tree.dtype))

    ratio = compute_partial_ratio(prob_choose, num_prunable, p_nonterminal, leaf_to_grow, split_tree)

    left = leaf_to_grow << 1
    return dict(
        num_growable=num_growable,
        node=leaf_to_grow,
        left=left,
        right=left + 1,
        var=var,
        split=split,
        partial_ratio=ratio,
        var_tree=var_tree,
        split_tree=split_tree,
    )

def choose_leaf(split_tree, affluence_tree, p_propose_grow, key):
    """
    Choose a leaf node to grow in a tree.

    Parameters
    ----------
    split_tree : array (2 ** (d - 1),)
        The splitting points of the tree.
    affluence_tree : bool array (2 ** (d - 1),) or None
        Whether a leaf has enough points to be grown.
    p_propose_grow : array (2 ** (d - 1),)
        The unnormalized probability of choosing a leaf to grow.
    key : jax.dtypes.prng_key array
        A jax random key.

    Returns
    -------
    leaf_to_grow : int
        The index of the leaf to grow. If ``num_growable == 0``, return
        ``2 ** d``.
    num_growable : int
        The number of leaf nodes that can be grown.
    prob_choose : float
        The normalized probability of choosing the selected leaf.
    num_prunable : int
        The number of leaf parents that could be pruned, after converting the
        selected leaf to a non-terminal node.
    """
    is_growable = growable_leaves(split_tree, affluence_tree)
    num_growable = jnp.count_nonzero(is_growable)
    distr = jnp.where(is_growable, p_propose_grow, 0)
    leaf_to_grow, distr_norm = categorical(key, distr)
    leaf_to_grow = jnp.where(num_growable, leaf_to_grow, 2 * split_tree.size)
    prob_choose = distr[leaf_to_grow] / distr_norm
    is_parent = grove.is_leaves_parent(split_tree.at[leaf_to_grow].set(1))
    num_prunable = jnp.count_nonzero(is_parent)
    return leaf_to_grow, num_growable, prob_choose, num_prunable

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
    """
    is_growable = grove.is_actual_leaf(split_tree)
    if affluence_tree is not None:
        is_growable &= affluence_tree
    return is_growable

def categorical(key, distr):
    """
    Return a random integer from an arbitrary distribution.

    Parameters
    ----------
    key : jax.dtypes.prng_key array
        A jax random key.
    distr : float array (n,)
        An unnormalized probability distribution.

    Returns
    -------
    u : int
        A random integer in the range ``[0, n)``. If all probabilities are zero,
        return ``n``.
    """
    ecdf = jnp.cumsum(distr)
    u = random.uniform(key, (), ecdf.dtype, 0, ecdf[-1])
    return jnp.searchsorted(ecdf, u, 'right'), ecdf[-1]

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
    ancestor_vars = jnp.zeros(max_num_ancestors, jaxext.minimal_unsigned_dtype(max_split.size))
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

def compute_partial_ratio(prob_choose, num_prunable, p_nonterminal, leaf_to_grow, new_split_tree):
    """
    Compute the product of the transition and prior ratios of a grow move.

    Parameters
    ----------
    num_growable : int
        The number of leaf nodes that can be grown.
    num_prunable : int
        The number of leaf parents that could be pruned, after converting the
        leaf to be grown to a non-terminal node.
    p_nonterminal : array (d,)
        The probability of a nonterminal node at each depth.
    leaf_to_grow : int
        The index of the leaf to grow.
    new_split_tree : array (2 ** (d - 1),)
        The splitting points of the tree, after the leaf is grown.

    Returns
    -------
    ratio : float
        The transition ratio P(new tree -> old tree) / P(old tree -> new tree)
        times the prior ratio P(new tree) / P(old tree), but the transition
        ratio is missing the factor P(propose prune) in the numerator.
    """

    # the two ratios also contain factors num_available_split *
    # num_available_var, but they cancel out

    # p_prune can't be computed here because it needs the count trees, which are
    # computed in the acceptance phase

    prune_allowed = leaf_to_grow != 1
        # prune allowed  <--->  the initial tree is not a root
        # leaf to grow is root  -->  the tree can only be a root
        # tree is a root  -->  the only leaf I can grow is root

    p_grow = jnp.where(prune_allowed, 0.5, 1)

    inv_trans_ratio = p_grow * prob_choose * num_prunable

    depth = grove.tree_depths(new_split_tree.size)[leaf_to_grow]
    p_parent = p_nonterminal[depth]
    cp_children = 1 - p_nonterminal[depth + 1]
    tree_ratio = cp_children * cp_children * p_parent / (1 - p_parent)

    return tree_ratio / inv_trans_ratio

def prune_move(var_tree, split_tree, affluence_tree, max_split, p_nonterminal, p_propose_grow, key):
    """
    Tree structure prune move proposal of BART MCMC.

    Parameters
    ----------
    var_tree : int array (2 ** (d - 1),)
        The variable indices of the tree.
    split_tree : int array (2 ** (d - 1),)
        The splitting points of the tree.
    affluence_tree : bool array (2 ** (d - 1),) or None
        Whether a leaf has enough points to be grown.
    max_split : int array (p,)
        The maximum split index for each variable.
    p_nonterminal : float array (d,)
        The probability of a nonterminal node at each depth.
    p_propose_grow : float array (2 ** (d - 1),)
        The unnormalized probability of choosing a leaf to grow.
    key : jax.dtypes.prng_key array
        A jax random key.

    Returns
    -------
    prune_move : dict
        A dictionary with fields:

        'allowed' : bool
            Whether the move is possible.
        'node' : int
            The index of the node to prune. ``2 ** d`` if no node can be pruned.
        'left', 'right' : int
            The indices of the children of 'node'.
        'partial_ratio' : float
            A factor of the Metropolis-Hastings ratio of the move. It lacks
            the likelihood ratio and the probability of proposing the prune
            move. This ratio is inverted.
    """
    node_to_prune, num_prunable, prob_choose = choose_leaf_parent(split_tree, affluence_tree, p_propose_grow, key)
    allowed = split_tree[1].astype(bool) # allowed iff the tree is not a root

    ratio = compute_partial_ratio(prob_choose, num_prunable, p_nonterminal, node_to_prune, split_tree)

    left = node_to_prune << 1
    return dict(
        allowed=allowed,
        node=node_to_prune,
        left=left,
        right=left + 1,
        partial_ratio=ratio, # it is inverted in accept_move_and_sample_leaves
    )

def choose_leaf_parent(split_tree, affluence_tree, p_propose_grow, key):
    """
    Pick a non-terminal node with leaf children to prune in a tree.

    Parameters
    ----------
    split_tree : array (2 ** (d - 1),)
        The splitting points of the tree.
    affluence_tree : bool array (2 ** (d - 1),) or None
        Whether a leaf has enough points to be grown.
    p_propose_grow : array (2 ** (d - 1),)
        The unnormalized probability of choosing a leaf to grow.
    key : jax.dtypes.prng_key array
        A jax random key.

    Returns
    -------
    node_to_prune : int
        The index of the node to prune. If ``num_prunable == 0``, return
        ``2 ** d``.
    num_prunable : int
        The number of leaf parents that could be pruned.
    prob_choose : float
        The normalized probability of choosing the node to prune for growth.
    """
    is_prunable = grove.is_leaves_parent(split_tree)
    num_prunable = jnp.count_nonzero(is_prunable)
    node_to_prune = randint_masked(key, is_prunable)
    node_to_prune = jnp.where(num_prunable, node_to_prune, 2 * split_tree.size)

    split_tree = split_tree.at[node_to_prune].set(0)
    affluence_tree = (
        None if affluence_tree is None else
        affluence_tree.at[node_to_prune].set(True)
    )
    is_growable_leaf = growable_leaves(split_tree, affluence_tree)
    prob_choose = p_propose_grow[node_to_prune]
    prob_choose /= jnp.sum(p_propose_grow, where=is_growable_leaf)

    return node_to_prune, num_prunable, prob_choose

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

def accept_moves_and_sample_leaves(bart, grow_moves, prune_moves, key):
    """
    Accept or reject the proposed moves and sample the new leaf values.

    Parameters
    ----------
    bart : dict
        A BART mcmc state.
    grow_moves : dict
        The proposals for grow moves, batched over the first axis. See
        `grow_move`.
    prune_moves : dict
        The proposals for prune moves, batched over the first axis. See
        `prune_move`.
    key : jax.dtypes.prng_key array
        A jax random key.

    Returns
    -------
    bart : dict
        The new BART mcmc state.
    """
    bart, grow_moves, prune_moves, count_trees, move_counts, u, z = accept_moves_parallel_stage(bart, grow_moves, prune_moves, key)
    bart, counts = accept_moves_sequential_stage(bart, count_trees, grow_moves, prune_moves, move_counts, u, z)
    return accept_moves_final_stage(bart, counts, grow_moves, prune_moves)

def accept_moves_parallel_stage(bart, grow_moves, prune_moves, key):
    """
    Pre-computes quantities used to accept moves, in parallel across trees.

    Parameters
    ----------
    bart : dict
        A BART mcmc state.
    grow_moves, prune_moves : dict
        The proposals for the moves, batched over the first axis. See
        `grow_move` and `prune_move`.
    key : jax.dtypes.prng_key array
        A jax random key.

    Returns
    -------
    bart : dict
        A partially updated BART mcmc state.
    grow_moves, prune_moves : dict
        The proposals for the moves, with the field 'partial_ratio' replaced
        by 'trans_prior_ratio'.
    count_trees : array (num_trees, 2 ** (d - 1))
        The number of points in each potential or actual leaf node.
    move_counts : dict
        The counts of the number of points in the the nodes modified by the
        moves.
    u : float array (num_trees, 2)
        Random uniform values used to accept the moves.
    z : float array (num_trees, 2 ** d)
        Random standard normal values used to sample the new leaf values.
    """
    bart = bart.copy()

    bart['var_trees'] = grow_moves['var_tree']
        # Since var_tree can contain garbage, I can set the var of leaf to be
        # grown irrespectively of what move I'm gonna accept in the end.

    bart['leaf_indices'] = apply_grow_to_indices(grow_moves, bart['leaf_indices'], bart['X'])

    count_trees, move_counts = compute_count_trees(bart['leaf_indices'], grow_moves, prune_moves, bart['opt']['count_batch_size'])

    grow_moves, prune_moves = complete_ratio(grow_moves, prune_moves, move_counts, bart['min_points_per_leaf'])

    if bart['opt']['require_min_points']:
        count_half_trees = count_trees[:, :grow_moves['split_tree'].shape[1]]
        bart['affluence_trees'] = count_half_trees >= 2 * bart['min_points_per_leaf']

    bart['leaf_trees'] = adapt_leaf_trees_to_grow_indices(bart['leaf_trees'], grow_moves)

    key, subkey = random.split(key)
    u = random.uniform(subkey, (len(bart['leaf_trees']), 2), bart['opt']['large_float'])
    z = random.normal(key, bart['leaf_trees'].shape, bart['opt']['large_float'])

    return bart, grow_moves, prune_moves, count_trees, move_counts, u, z

def apply_grow_to_indices(grow_moves, leaf_indices, X):
    """
    Update the leaf indices to apply a grow move.

    Parameters
    ----------
    grow_moves : dict
        The proposals for grow moves. See `grow_move`.
    leaf_indices : array (num_trees, n)
        The index of the leaf each datapoint falls into.
    X : array (p, n)
        The predictors matrix.

    Returns
    -------
    grow_leaf_indices : array (num_trees, n)
        The updated leaf indices.
    """
    left_child = grow_moves['node'].astype(leaf_indices.dtype) << 1
    go_right = X[grow_moves['var'], :] >= grow_moves['split'][:, None]
    tree_size = jnp.array(2 * grow_moves['split_tree'].shape[1])
    node_to_update = jnp.where(grow_moves['num_growable'], grow_moves['node'], tree_size)
    return jnp.where(
        leaf_indices == node_to_update[:, None],
        left_child[:, None] + go_right,
        leaf_indices,
    )

def compute_count_trees(grow_leaf_indices, grow_moves, prune_moves, batch_size):
    """
    Count the number of datapoints in each leaf.

    Parameters
    ----------
    grow_leaf_indices : int array (num_trees, n)
        The index of the leaf each datapoint falls into, if the grow move is
        accepted.
    grow_moves, prune_moves : dict
        The proposals for the moves. See `grow_move` and `prune_move`.
    batch_size : int or None
        The data batch size to use for the summation.

    Returns
    -------
    count_trees : int array (num_trees, 2 ** (d - 1))
        The number of points in each potential or actual leaf node.
    counts : dict
        The counts of the number of points in the the nodes modified by the
        moves, organized as two dictionaries 'grow' and 'prune', with subfields
        'left', 'right', and 'total'.
    """

    ntree, tree_size = grow_moves['split_tree'].shape
    tree_size *= 2
    counts = dict(grow=dict(), prune=dict())
    tree_indices = jnp.arange(ntree)

    count_trees = count_datapoints_per_leaf(grow_leaf_indices, tree_size, batch_size)

    # count datapoints in leaf to grow
    counts['grow']['left'] = count_trees[tree_indices, grow_moves['left']]
    counts['grow']['right'] = count_trees[tree_indices, grow_moves['right']]
    counts['grow']['total'] = counts['grow']['left'] + counts['grow']['right']
    count_trees = count_trees.at[tree_indices, grow_moves['node']].set(counts['grow']['total'])

    # count datapoints in node to prune
    counts['prune']['left'] = count_trees[tree_indices, prune_moves['left']]
    counts['prune']['right'] = count_trees[tree_indices, prune_moves['right']]
    counts['prune']['total'] = counts['prune']['left'] + counts['prune']['right']
    count_trees = count_trees.at[tree_indices, prune_moves['node']].set(counts['prune']['total'])

    return count_trees, counts

def count_datapoints_per_leaf(leaf_indices, tree_size, batch_size):
    """
    Count the number of datapoints in each leaf.

    Parameters
    ----------
    leaf_indices : int array (num_trees, n)
        The index of the leaf each datapoint falls into.
    tree_size : int
        The size of the leaf tree array (2 ** d).
    batch_size : int or None
        The data batch size to use for the summation.

    Returns
    -------
    count_trees : int array (num_trees, 2 ** (d - 1))
        The number of points in each leaf node.
    """
    if batch_size is None:
        return _count_scan(leaf_indices, tree_size)
    else:
        return _count_vec(leaf_indices, tree_size, batch_size)

def _count_scan(leaf_indices, tree_size):
    def loop(_, leaf_indices):
        return None, _aggregate_scatter(1, leaf_indices, tree_size, jnp.uint32)
    _, count_trees = lax.scan(loop, None, leaf_indices)
    return count_trees

def _aggregate_scatter(values, indices, size, dtype):
    return (jnp
        .zeros(size, dtype)
        .at[indices]
        .add(values)
    )

def _count_vec(leaf_indices, tree_size, batch_size):
    return _aggregate_batched_alltrees(1, leaf_indices, tree_size, jnp.uint32, batch_size)
        # uint16 is super-slow on gpu, don't use it even if n < 2^16

def _aggregate_batched_alltrees(values, indices, size, dtype, batch_size):
    ntree, n = indices.shape
    tree_indices = jnp.arange(ntree)
    nbatches = n // batch_size + bool(n % batch_size)
    batch_indices = jnp.arange(n) % nbatches
    return (jnp
        .zeros((ntree, size, nbatches), dtype)
        .at[tree_indices[:, None], indices, batch_indices]
        .add(values)
        .sum(axis=2)
    )

def complete_ratio(grow_moves, prune_moves, move_counts, min_points_per_leaf):
    """
    Complete non-likelihood MH ratio calculation.

    This functions adds the probability of choosing the prune move.

    Parameters
    ----------
    grow_moves, prune_moves : dict
        The proposals for the moves. See `grow_move` and `prune_move`.
    move_counts : dict
        The counts of the number of points in the the nodes modified by the
        moves.
    min_points_per_leaf : int or None
        The minimum number of data points in a leaf node.

    Returns
    -------
    grow_moves, prune_moves : dict
        The proposals for the moves, with the field 'partial_ratio' replaced
        by 'trans_prior_ratio'.
    """
    grow_moves = grow_moves.copy()
    prune_moves = prune_moves.copy()
    compute_p_prune_vec = jax.vmap(compute_p_prune, in_axes=(0, 0, 0, None))
    grow_p_prune, prune_p_prune = compute_p_prune_vec(grow_moves, move_counts['grow']['left'], move_counts['grow']['right'], min_points_per_leaf)
    grow_moves['trans_prior_ratio'] = grow_moves.pop('partial_ratio') * grow_p_prune
    prune_moves['trans_prior_ratio'] = prune_moves.pop('partial_ratio') * prune_p_prune
    return grow_moves, prune_moves

def compute_p_prune(grow_move, grow_left_count, grow_right_count, min_points_per_leaf):
    """
    Compute the probability of proposing a prune move.

    Parameters
    ----------
    grow_move : dict
        The proposal for the grow move, see `grow_move`.
    grow_left_count, grow_right_count : int
        The number of datapoints in the proposed children of the leaf to grow.
    min_points_per_leaf : int or None
        The minimum number of data points in a leaf node.

    Returns
    -------
    grow_p_prune : float
        The probability of proposing a prune move, after accepting the grow
        move.
    prune_p_prune : float
        The probability of proposing the prune move.
    """
    other_growable_leaves = grow_move['num_growable'] >= 2
    new_leaves_growable = grow_move['node'] < grow_move['split_tree'].size // 2
    if min_points_per_leaf is not None:
        any_above_threshold = grow_left_count >= 2 * min_points_per_leaf
        any_above_threshold |= grow_right_count >= 2 * min_points_per_leaf
        new_leaves_growable &= any_above_threshold
    grow_again_allowed = other_growable_leaves | new_leaves_growable
    grow_p_prune = jnp.where(grow_again_allowed, 0.5, 1)
    prune_p_prune = jnp.where(grow_move['num_growable'], 0.5, 1)
    return grow_p_prune, prune_p_prune

def adapt_leaf_trees_to_grow_indices(leaf_trees, grow_moves):
    """
    Modify leaf values such that the indices of the grow move work on the
    original tree.

    Parameters
    ----------
    leaf_trees : float array (num_trees, 2 ** d)
        The leaf values.
    grow_moves : dict
        The proposals for grow moves. See `grow_move`.

    Returns
    -------
    leaf_trees : float array (num_trees, 2 ** d)
        The modified leaf values. The value of the leaf to grow is copied to
        what would be its children if the grow move was accepted.
    """
    ntree, _ = leaf_trees.shape
    tree_indices = jnp.arange(ntree)
    values_at_node = leaf_trees[tree_indices, grow_moves['node']]
    return (leaf_trees
        .at[tree_indices, grow_moves['left']]
        .set(values_at_node)
        .at[tree_indices, grow_moves['right']]
        .set(values_at_node)
    )

def accept_moves_sequential_stage(bart, count_trees, grow_moves, prune_moves, move_counts, u, z):
    """
    The part of accepting the moves that has to be done one tree at a time.

    Parameters
    ----------
    bart : dict
        A partially updated BART mcmc state.
    count_trees : array (num_trees, 2 ** (d - 1))
        The number of points in each potential or actual leaf node.
    grow_moves, prune_moves : dict
        The proposals for the moves, with completed ratios. See `grow_move` and
        `prune_move`.
    move_counts : dict
        The counts of the number of points in the the nodes modified by the
        moves.
    u : float array (num_trees, 2)
        Random uniform values used to for proposal and accept decisions.
    z : float array (num_trees, 2 ** d)
        Random standard normal values used to sample the new leaf values.

    Returns
    -------
    bart : dict
        A partially updated BART mcmc state.
    counts : dict
        The indicators of proposals and acceptances for grow and prune moves.
    """
    bart = bart.copy()

    def loop(resid, item):
        resid, leaf_tree, split_tree, counts, ratios = accept_move_and_sample_leaves(
            bart['X'],
            len(bart['leaf_trees']),
            bart['opt']['resid_batch_size'],
            resid,
            bart['sigma2'],
            bart['min_points_per_leaf'],
            'ratios' in bart,
            *item,
        )
        return resid, (leaf_tree, split_tree, counts, ratios)

    items = (
        bart['leaf_trees'], count_trees,
        grow_moves, prune_moves, move_counts,
        bart['leaf_indices'],
        u, z,
    )
    resid, (leaf_trees, split_trees, counts, ratios) = lax.scan(loop, bart['resid'], items)

    bart['resid'] = resid
    bart['leaf_trees'] = leaf_trees
    bart['split_trees'] = split_trees
    bart.get('ratios', {}).update(ratios)

    return bart, counts

def accept_move_and_sample_leaves(X, ntree, resid_batch_size, resid, sigma2, min_points_per_leaf, save_ratios, leaf_tree, count_tree, grow_move, prune_move, move_counts, grow_leaf_indices, u, z):
    """
    Accept or reject a proposed move and sample the new leaf values.

    Parameters
    ----------
    X : int array (p, n)
        The predictors.
    ntree : int
        The number of trees in the forest.
    resid_batch_size : int, None
        The batch size for computing the sum of residuals in each leaf.
    resid : float array (n,)
        The residuals (data minus forest value).
    sigma2 : float
        The noise variance.
    min_points_per_leaf : int or None
        The minimum number of data points in a leaf node.
    save_ratios : bool
        Whether to save the acceptance ratios.
    leaf_tree : float array (2 ** d,)
        The leaf values of the tree.
    count_tree : int array (2 ** d,)
        The number of datapoints in each leaf.
    grow_move, prune_move : dict
        The proposals for the moves, with completed ratios. See `grow_move` and
        `prune_move`.
    grow_leaf_indices : int array (n,)
        The leaf indices of the tree proposed by the grow move.
    u : float array (2,)
        Two uniform random values in [0, 1).
    z : float array (2 ** d,)
        Standard normal random values.

    Returns
    -------
    resid : float array (n,)
        The updated residuals (data minus forest value).
    leaf_tree : float array (2 ** d,)
        The new leaf values of the tree.
    split_tree : int array (2 ** (d - 1),)
        The updated decision boundaries of the tree.
    counts : dict
        The indicators of proposals and acceptances for grow and prune moves.
    ratios : dict
        The acceptance ratios for the moves. Empty if not to be saved.
    """

    # sum residuals and count units per leaf, in tree proposed by grow move
    resid_tree = sum_resid(resid, grow_leaf_indices, leaf_tree.size, resid_batch_size)

    # subtract starting tree from function
    resid_tree += count_tree * leaf_tree

    # get indices of grow move
    grow_node = grow_move['node']
    assert grow_node.dtype == jnp.int32
    grow_left = grow_move['left']
    grow_right = grow_move['right']

    # sum residuals in leaf to grow
    grow_resid_left = resid_tree[grow_left]
    grow_resid_right = resid_tree[grow_right]
    grow_resid_total = grow_resid_left + grow_resid_right
    resid_tree = resid_tree.at[grow_node].set(grow_resid_total)

    # get indices of prune move
    prune_node = prune_move['node']
    assert prune_node.dtype == jnp.int32
    prune_left = prune_move['left']
    prune_right = prune_move['right']

    # sum residuals in node to prune
    prune_resid_left = resid_tree[prune_left]
    prune_resid_right = resid_tree[prune_right]
    prune_resid_total = prune_resid_left + prune_resid_right
    resid_tree = resid_tree.at[prune_node].set(prune_resid_total)

    # Now resid_tree and count_tree contain correct values whatever move is
    # accepted.

    # compute likelihood ratios
    grow_lk_ratio = compute_likelihood_ratio(grow_resid_total, grow_resid_left, grow_resid_right, move_counts['grow']['total'], move_counts['grow']['left'], move_counts['grow']['right'], sigma2, ntree)
    prune_lk_ratio = compute_likelihood_ratio(prune_resid_total, prune_resid_left, prune_resid_right, move_counts['prune']['total'], move_counts['prune']['left'], move_counts['prune']['right'], sigma2, ntree)

    # compute acceptance ratios
    grow_ratio = grow_move['trans_prior_ratio'] * grow_lk_ratio
    if min_points_per_leaf is not None:
        grow_ratio = jnp.where(move_counts['grow']['left'] >= min_points_per_leaf, grow_ratio, 0)
        grow_ratio = jnp.where(move_counts['grow']['right'] >= min_points_per_leaf, grow_ratio, 0)
    prune_ratio = prune_move['trans_prior_ratio'] * prune_lk_ratio
    prune_ratio = lax.reciprocal(prune_ratio)

    # save acceptance ratios
    ratios = {}
    if save_ratios:
        ratios.update(
            grow=dict(
                trans_prior=grow_move['trans_prior_ratio'],
                likelihood=grow_lk_ratio,
            ),
            prune=dict(
                trans_prior=lax.reciprocal(prune_move['trans_prior_ratio']),
                likelihood=lax.reciprocal(prune_lk_ratio),
            ),
        )

    # determine what move to propose (not proposing anything is an option)
    grow_allowed = grow_move['num_growable'].astype(bool)
    p_grow = jnp.where(grow_allowed & prune_move['allowed'], 0.5, grow_allowed)
    try_grow = u[0] < p_grow # use < instead of <= because coins are in [0, 1)
    try_prune = prune_move['allowed'] & ~try_grow

    # determine whether to accept the move
    do_grow = try_grow & (u[1] < grow_ratio)
    do_prune = try_prune & (u[1] < prune_ratio)

    # pick split tree for chosen move
    split_tree = grow_move['split_tree']
    split_tree = split_tree.at[jnp.where(do_grow, split_tree.size, grow_node)].set(0)
    split_tree = split_tree.at[jnp.where(do_prune, prune_node, split_tree.size)].set(0)
    # I can leave garbage in var_tree, resid_tree, count_tree

    # compute leaves posterior and sample leaves
    inv_sigma2 = lax.reciprocal(sigma2)
    prec_lk = count_tree * inv_sigma2
    var_post = lax.reciprocal(prec_lk + ntree) # = 1 / (prec_lk + prec_prior)
    mean_post = resid_tree * inv_sigma2 * var_post # = mean_lk * prec_lk * var_post
    initial_leaf_tree = leaf_tree
    leaf_tree = mean_post + z * jnp.sqrt(var_post)

    # copy leaves around such that the grow leaf indices select the right leaf
    leaf_tree = (leaf_tree
        .at[jnp.where(do_prune, prune_left, leaf_tree.size)]
        .set(leaf_tree[prune_node])
        .at[jnp.where(do_prune, prune_right, leaf_tree.size)]
        .set(leaf_tree[prune_node])
    )
    leaf_tree = (leaf_tree
        .at[jnp.where(do_grow, leaf_tree.size, grow_left)]
        .set(leaf_tree[grow_node])
        .at[jnp.where(do_grow, leaf_tree.size, grow_right)]
        .set(leaf_tree[grow_node])
    )

    # replace old tree with new tree in function values
    resid += (initial_leaf_tree - leaf_tree)[grow_leaf_indices]

    # pack proposal and acceptance indicators
    counts = dict(
        grow_prop_count=try_grow,
        grow_acc_count=do_grow,
        prune_prop_count=try_prune,
        prune_acc_count=do_prune,
    )

    return resid, leaf_tree, split_tree, counts, ratios

def sum_resid(resid, leaf_indices, tree_size, batch_size):
    """
    Sum the residuals in each leaf.

    Parameters
    ----------
    resid : float array (n,)
        The residuals (data minus forest value).
    leaf_indices : int array (n,)
        The leaf indices of the tree (in which leaf each data point falls into).
    tree_size : int
        The size of the tree array (2 ** d).
    batch_size : int, None
        The data batch size for the aggregation. Batching increases numerical
        accuracy and parallelism.

    Returns
    -------
    resid_tree : float array (2 ** d,)
        The sum of the residuals at data points in each leaf.
    """
    if batch_size is None:
        aggr_func = _aggregate_scatter
    else:
        aggr_func = functools.partial(_aggregate_batched_onetree, batch_size=batch_size)
    return aggr_func(resid, leaf_indices, tree_size, jnp.float32)

def _aggregate_batched_onetree(values, indices, size, dtype, batch_size):
    n, = indices.shape
    nbatches = n // batch_size + bool(n % batch_size)
    batch_indices = jnp.arange(n) % nbatches
    return (jnp
        .zeros((size, nbatches), dtype)
        .at[indices, batch_indices]
        .add(values)
        .sum(axis=1)
    )

def compute_likelihood_ratio(total_resid, left_resid, right_resid, total_count, left_count, right_count, sigma2, n_tree):
    """
    Compute the likelihood ratio of a grow move.

    Parameters
    ----------
    total_resid : float
        The sum of the residuals in the leaf to grow.
    left_resid, right_resid : float
        The sum of the residuals in the left/right child of the leaf to grow.
    total_count : int
        The number of datapoints in the leaf to grow.
    left_count, right_count : int
        The number of datapoints in the left/right child of the leaf to grow.
    sigma2 : float
        The noise variance.
    n_tree : int
        The number of trees in the forest.

    Returns
    -------
    ratio : float
        The likelihood ratio P(data | new tree) / P(data | old tree).
    """

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

    return jnp.sqrt(sqrt_term) * jnp.exp(exp_term)

def accept_moves_final_stage(bart, counts, grow_moves, prune_moves):
    """
    The final part of accepting the moves, in parallel across trees.

    Parameters
    ----------
    bart : dict
        A partially updated BART mcmc state.
    counts : dict
        The indicators of proposals and acceptances for grow and prune moves.
    grow_moves, prune_moves : dict
        The proposals for the moves. See `grow_move` and `prune_move`.

    Returns
    -------
    bart : dict
        The fully updated BART mcmc state.
    """
    bart = bart.copy()

    for k, v in counts.items():
        bart[k] = jnp.sum(v, axis=0)

    bart['leaf_indices'] = apply_moves_to_indices(bart['leaf_indices'], counts, grow_moves, prune_moves)

    return bart

def apply_moves_to_indices(leaf_indices, counts, grow_moves, prune_moves):
    """
    Update the leaf indices to match the accepted move.

    Parameters
    ----------
    leaf_indices : int array (num_trees, n)
        The index of the leaf each datapoint falls into, if the grow move was
        accepted.
    counts : dict
        The indicators of proposals and acceptances for grow and prune moves.
    grow_moves, prune_moves : dict
        The proposals for the moves. See `grow_move` and `prune_move`.

    Returns
    -------
    leaf_indices : int array (num_trees, n)
        The updated leaf indices.
    """
    mask = ~jnp.array(1, leaf_indices.dtype) # ...1111111110
    cond = (leaf_indices & mask) == grow_moves['left'][:, None]
    leaf_indices = jnp.where(
        cond & ~counts['grow_acc_count'][:, None],
        grow_moves['node'][:, None].astype(leaf_indices.dtype),
        leaf_indices,
    )
    cond = (leaf_indices & mask) == prune_moves['left'][:, None]
    return jnp.where(
        cond & counts['prune_acc_count'][:, None],
        prune_moves['node'][:, None].astype(leaf_indices.dtype),
        leaf_indices,
    )

def sample_sigma(bart, key):
    """
    Noise variance sampling step of BART MCMC.

    Parameters
    ----------
    bart : dict
        A BART mcmc state, as created by `init`.
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
    norm2 = jnp.dot(resid, resid, preferred_element_type=bart['opt']['large_float'])
    beta = bart['sigma2_beta'] + norm2 / 2

    sample = random.gamma(key, alpha)
    bart['sigma2'] = beta / sample

    return bart
