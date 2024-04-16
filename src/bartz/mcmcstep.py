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
        'ratios' : dict, optional
            If `save_ratios` is True, this field is present. It has the fields:

            'log_trans_prior' : large_float array (num_trees,)
                The log transition and prior Metropolis-Hastings ratio for the
                proposed move on each tree.
            'log_likelihood' : large_float array (num_trees,)
                The log likelihood ratio.
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
    resid_batch_size, count_batch_size = _choose_suffstat_batch_size(resid_batch_size, count_batch_size, y, 2 ** max_depth * num_trees)
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
            log_trans_prior=jnp.full(num_trees, jnp.nan),
            log_likelihood=jnp.full(num_trees, jnp.nan),
        )

    return bart

def _choose_suffstat_batch_size(resid_batch_size, count_batch_size, y, forest_size):

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
            max_memory = 2 ** 29
            itemsize = 4
            min_batch_size = int(math.ceil(forest_size * itemsize * n / max_memory))
            count_batch_size = max(count_batch_size, min_batch_size)
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
    moves = sample_moves(bart, subkey)
    return accept_moves_and_sample_leaves(bart, moves, key)

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
    moves : dict
        A dictionary with fields:

        'allowed' : bool array (num_trees,)
            Whether the move is possible.
        'grow' : bool array (num_trees,)
            Whether the move is a grow move or a prune move.
        'num_growable' : int array (num_trees,)
            The number of growable leaves in the original tree.
        'node' : int array (num_trees,)
            The index of the leaf to grow or node to prune.
        'left', 'right' : int array (num_trees,)
            The indices of the children of 'node'.
        'partial_ratio' : float array (num_trees,)
            A factor of the Metropolis-Hastings ratio of the move. It lacks
            the likelihood ratio and the probability of proposing the prune
            move. If the move is Prune, the ratio is inverted.
        'grow_var' : int array (num_trees,)
            The decision axes of the new rules.
        'grow_split' : int array (num_trees,)
            The decision boundaries of the new rules.
        'var_trees' : int array (num_trees, 2 ** (d - 1))
            The updated decision axes of the trees, valid whatever move.
        'logu' : float array (num_trees,)
            The logarithm of a uniform (0, 1] random variable to be used to
            accept the move. It's in (-oo, 0].
    """
    ntree = bart['leaf_trees'].shape[0]
    key = random.split(key, 1 + ntree)
    key, subkey = key[0], key[1:]

    # compute moves
    grow_moves, prune_moves = _sample_moves_vmap_trees(bart['var_trees'], bart['split_trees'], bart['affluence_trees'], bart['max_split'], bart['p_nonterminal'], bart['p_propose_grow'], subkey)

    u, logu = random.uniform(key, (2, ntree), bart['opt']['large_float'])

    # choose between grow or prune
    grow_allowed = grow_moves['num_growable'].astype(bool)
    p_grow = jnp.where(grow_allowed & prune_moves['allowed'], 0.5, grow_allowed)
    grow = u < p_grow # use < instead of <= because u is in [0, 1)

    # compute children indices
    node = jnp.where(grow, grow_moves['node'], prune_moves['node'])
    left = node << 1
    right = left + 1

    return dict(
        allowed=grow | prune_moves['allowed'],
        grow=grow,
        num_growable=grow_moves['num_growable'],
        node=node,
        left=left,
        right=right,
        partial_ratio=jnp.where(grow, grow_moves['partial_ratio'], prune_moves['partial_ratio']),
        grow_var=grow_moves['var'],
        grow_split=grow_moves['split'],
        var_trees=grow_moves['var_tree'],
        logu=jnp.log1p(-logu),
    )

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
        'var', 'split' : int
            The decision axis and boundary of the new rule.
        'partial_ratio' : float
            A factor of the Metropolis-Hastings ratio of the move. It lacks
            the likelihood ratio and the probability of proposing the prune
            move.
        'var_tree' : array (2 ** (d - 1),)
            The updated decision axes of the tree.
    """

    key, key1, key2 = random.split(key, 3)

    leaf_to_grow, num_growable, prob_choose, num_prunable = choose_leaf(split_tree, affluence_tree, p_propose_grow, key)

    var = choose_variable(var_tree, split_tree, max_split, leaf_to_grow, key1)
    var_tree = var_tree.at[leaf_to_grow].set(var.astype(var_tree.dtype))

    split = choose_split(var_tree, split_tree, max_split, leaf_to_grow, key2)

    ratio = compute_partial_ratio(prob_choose, num_prunable, p_nonterminal, leaf_to_grow)

    return dict(
        num_growable=num_growable,
        node=leaf_to_grow,
        var=var,
        split=split,
        partial_ratio=ratio,
        var_tree=var_tree,
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

def compute_partial_ratio(prob_choose, num_prunable, p_nonterminal, leaf_to_grow):
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

    depth = grove.tree_depths(2 ** (p_nonterminal.size - 1))[leaf_to_grow]
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
        'partial_ratio' : float
            A factor of the Metropolis-Hastings ratio of the move. It lacks
            the likelihood ratio and the probability of proposing the prune
            move. This ratio is inverted.
    """
    node_to_prune, num_prunable, prob_choose = choose_leaf_parent(split_tree, affluence_tree, p_propose_grow, key)
    allowed = split_tree[1].astype(bool) # allowed iff the tree is not a root

    ratio = compute_partial_ratio(prob_choose, num_prunable, p_nonterminal, node_to_prune)

    return dict(
        allowed=allowed,
        node=node_to_prune,
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

def accept_moves_and_sample_leaves(bart, moves, key):
    """
    Accept or reject the proposed moves and sample the new leaf values.

    Parameters
    ----------
    bart : dict
        A BART mcmc state.
    moves : dict
        The proposed moves, see `sample_moves`.
    key : jax.dtypes.prng_key array
        A jax random key.

    Returns
    -------
    bart : dict
        The new BART mcmc state.
    """
    bart, moves, count_trees, move_counts, prelkv, prelk, prelf = accept_moves_parallel_stage(bart, moves, key)
    bart, moves = accept_moves_sequential_stage(bart, count_trees, moves, move_counts, prelkv, prelk, prelf)
    return accept_moves_final_stage(bart, moves)

def accept_moves_parallel_stage(bart, moves, key):
    """
    Pre-computes quantities used to accept moves, in parallel across trees.

    Parameters
    ----------
    bart : dict
        A BART mcmc state.
    moves : dict
        The proposed moves, see `sample_moves`.
    key : jax.dtypes.prng_key array
        A jax random key.

    Returns
    -------
    bart : dict
        A partially updated BART mcmc state.
    moves : dict
        The proposed moves, with the field 'partial_ratio' replaced
        by 'log_trans_prior_ratio'.
    count_trees : array (num_trees, 2 ** d)
        The number of points in each potential or actual leaf node.
    move_counts : dict
        The counts of the number of points in the the nodes modified by the
        moves.
    prelkv, prelk, prelf : dict
        Dictionary with pre-computed terms of the likelihood ratios and leaf
        samples.
    """
    bart = bart.copy()

    # where the move is grow, modify the state like the move was accepted
    bart['var_trees'] = moves['var_trees']
    bart['leaf_indices'] = apply_grow_to_indices(moves, bart['leaf_indices'], bart['X'])
    bart['leaf_trees'] = adapt_leaf_trees_to_grow_indices(bart['leaf_trees'], moves)

    # count number of datapoints per leaf
    count_trees, move_counts = compute_count_trees(bart['leaf_indices'], moves, bart['opt']['count_batch_size'])
    if bart['opt']['require_min_points']:
        count_half_trees = count_trees[:, :bart['var_trees'].shape[1]]
        bart['affluence_trees'] = count_half_trees >= 2 * bart['min_points_per_leaf']

    # compute some missing information about moves
    moves = complete_ratio(moves, move_counts, bart['min_points_per_leaf'])
    bart['grow_prop_count'] = jnp.sum(moves['grow'])
    bart['prune_prop_count'] = jnp.sum(moves['allowed'] & ~moves['grow'])

    prelkv, prelk = precompute_likelihood_terms(count_trees, bart['sigma2'], move_counts)
    prelf = precompute_leaf_terms(count_trees, bart['sigma2'], key)

    return bart, moves, count_trees, move_counts, prelkv, prelk, prelf

@functools.partial(jaxext.vmap_nodoc, in_axes=(0, 0, None))
def apply_grow_to_indices(moves, leaf_indices, X):
    """
    Update the leaf indices to apply a grow move.

    Parameters
    ----------
    moves : dict
        The proposed moves, see `sample_moves`.
    leaf_indices : array (num_trees, n)
        The index of the leaf each datapoint falls into.
    X : array (p, n)
        The predictors matrix.

    Returns
    -------
    grow_leaf_indices : array (num_trees, n)
        The updated leaf indices.
    """
    left_child = moves['node'].astype(leaf_indices.dtype) << 1
    go_right = X[moves['grow_var'], :] >= moves['grow_split']
    tree_size = jnp.array(2 * moves['var_trees'].size)
    node_to_update = jnp.where(moves['grow'], moves['node'], tree_size)
    return jnp.where(
        leaf_indices == node_to_update,
        left_child + go_right,
        leaf_indices,
    )

def compute_count_trees(leaf_indices, moves, batch_size):
    """
    Count the number of datapoints in each leaf.

    Parameters
    ----------
    grow_leaf_indices : int array (num_trees, n)
        The index of the leaf each datapoint falls into, if the grow move is
        accepted.
    moves : dict
        The proposed moves, see `sample_moves`.
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

    ntree, tree_size = moves['var_trees'].shape
    tree_size *= 2
    tree_indices = jnp.arange(ntree)

    count_trees = count_datapoints_per_leaf(leaf_indices, tree_size, batch_size)

    # count datapoints in nodes modified by move
    counts = dict()
    counts['left'] = count_trees[tree_indices, moves['left']]
    counts['right'] = count_trees[tree_indices, moves['right']]
    counts['total'] = counts['left'] + counts['right']

    # write count into non-leaf node
    count_trees = count_trees.at[tree_indices, moves['node']].set(counts['total'])

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

def complete_ratio(moves, move_counts, min_points_per_leaf):
    """
    Complete non-likelihood MH ratio calculation.

    This functions adds the probability of choosing the prune move.

    Parameters
    ----------
    moves : dict
        The proposed moves, see `sample_moves`.
    move_counts : dict
        The counts of the number of points in the the nodes modified by the
        moves.
    min_points_per_leaf : int or None
        The minimum number of data points in a leaf node.

    Returns
    -------
    moves : dict
        The updated moves, with the field 'partial_ratio' replaced by
        'log_trans_prior_ratio'.
    """
    moves = moves.copy()
    p_prune = compute_p_prune(moves, move_counts['left'], move_counts['right'], min_points_per_leaf)
    moves['log_trans_prior_ratio'] = jnp.log(moves.pop('partial_ratio') * p_prune)
    return moves

def compute_p_prune(moves, left_count, right_count, min_points_per_leaf):
    """
    Compute the probability of proposing a prune move.

    Parameters
    ----------
    moves : dict
        The proposed moves, see `sample_moves`.
    left_count, right_count : int
        The number of datapoints in the proposed children of the leaf to grow.
    min_points_per_leaf : int or None
        The minimum number of data points in a leaf node.

    Returns
    -------
    p_prune : float
        The probability of proposing a prune move. If grow: after accepting the
        grow move, if prune: right away.
    """

    # calculation in case the move is grow
    other_growable_leaves = moves['num_growable'] >= 2
    new_leaves_growable = moves['node'] < moves['var_trees'].shape[1] // 2
    if min_points_per_leaf is not None:
        any_above_threshold = left_count >= 2 * min_points_per_leaf
        any_above_threshold |= right_count >= 2 * min_points_per_leaf
        new_leaves_growable &= any_above_threshold
    grow_again_allowed = other_growable_leaves | new_leaves_growable
    grow_p_prune = jnp.where(grow_again_allowed, 0.5, 1)

    # calculation in case the move is prune
    prune_p_prune = jnp.where(moves['num_growable'], 0.5, 1)

    return jnp.where(moves['grow'], grow_p_prune, prune_p_prune)

@jaxext.vmap_nodoc
def adapt_leaf_trees_to_grow_indices(leaf_trees, moves):
    """
    Modify leaf values such that the indices of the grow moves work on the
    original tree.

    Parameters
    ----------
    leaf_trees : float array (num_trees, 2 ** d)
        The leaf values.
    moves : dict
        The proposed moves, see `sample_moves`.

    Returns
    -------
    leaf_trees : float array (num_trees, 2 ** d)
        The modified leaf values. The value of the leaf to grow is copied to
        what would be its children if the grow move was accepted.
    """
    values_at_node = leaf_trees[moves['node']]
    return (leaf_trees
        .at[jnp.where(moves['grow'], moves['left'], leaf_trees.size)]
        .set(values_at_node)
        .at[jnp.where(moves['grow'], moves['right'], leaf_trees.size)]
        .set(values_at_node)
    )

def precompute_likelihood_terms(count_trees, sigma2, move_counts):
    """
    Pre-compute terms used in the likelihood ratio of the acceptance step.

    Parameters
    ----------
    count_trees : array (num_trees, 2 ** d)
        The number of points in each potential or actual leaf node.
    sigma2 : float
        The noise variance.
    move_counts : dict
        The counts of the number of points in the the nodes modified by the
        moves.

    Returns
    -------
    prelkv : dict
        Dictionary with pre-computed terms of the likelihood ratio, one per
        tree.
    prelk : dict
        Dictionary with pre-computed terms of the likelihood ratio, shared by
        all trees.
    """
    ntree = len(count_trees)
    sigma_mu2 = 1 / ntree
    prelkv = dict()
    prelkv['sigma2_left'] = sigma2 + move_counts['left'] * sigma_mu2
    prelkv['sigma2_right'] = sigma2 + move_counts['right'] * sigma_mu2
    prelkv['sigma2_total'] = sigma2 + move_counts['total'] * sigma_mu2
    prelkv['sqrt_term'] = jnp.log(
        sigma2 * prelkv['sigma2_total'] /
        (prelkv['sigma2_left'] * prelkv['sigma2_right'])
    ) / 2
    return prelkv, dict(
        exp_factor=sigma_mu2 / (2 * sigma2),
    )

def precompute_leaf_terms(count_trees, sigma2, key):
    """
    Pre-compute terms used to sample leaves from their posterior.

    Parameters
    ----------
    count_trees : array (num_trees, 2 ** d)
        The number of points in each potential or actual leaf node.
    sigma2 : float
        The noise variance.
    key : jax.dtypes.prng_key array
        A jax random key.

    Returns
    -------
    prelf : dict
        Dictionary with pre-computed terms of the leaf sampling, with fields:

        'mean_factor' : float array (num_trees, 2 ** d)
            The factor to be multiplied by the sum of residuals to obtain the
            posterior mean.
        'centered_leaves' : float array (num_trees, 2 ** d)
            The mean-zero normal values to be added to the posterior mean to
            obtain the posterior leaf samples.
    """
    ntree = len(count_trees)
    prec_lk = count_trees / sigma2
    var_post = lax.reciprocal(prec_lk + ntree) # = 1 / (prec_lk + prec_prior)
    z = random.normal(key, count_trees.shape, sigma2.dtype)
    return dict(
        mean_factor=var_post / sigma2, # = mean_lk * prec_lk * var_post / resid_tree
        centered_leaves=z * jnp.sqrt(var_post),
    )

def accept_moves_sequential_stage(bart, count_trees, moves, move_counts, prelkv, prelk, prelf):
    """
    The part of accepting the moves that has to be done one tree at a time.

    Parameters
    ----------
    bart : dict
        A partially updated BART mcmc state.
    count_trees : array (num_trees, 2 ** d)
        The number of points in each potential or actual leaf node.
    moves : dict
        The proposed moves, see `sample_moves`.
    move_counts : dict
        The counts of the number of points in the the nodes modified by the
        moves.
    prelkv, prelk, prelf : dict
        Dictionaries with pre-computed terms of the likelihood ratios and leaf
        samples.

    Returns
    -------
    bart : dict
        A partially updated BART mcmc state.
    moves : dict
        The proposed moves, with these additional fields:

        'acc' : bool array (num_trees,)
            Whether the move was accepted.
        'to_prune' : bool array (num_trees,)
            Whether, to reflect the acceptance status of the move, the state
            should be updated by pruning the leaves involved in the move.
    """
    bart = bart.copy()
    moves = moves.copy()

    def loop(resid, item):
        resid, leaf_tree, acc, to_prune, ratios = accept_move_and_sample_leaves(
            bart['X'],
            len(bart['leaf_trees']),
            bart['opt']['resid_batch_size'],
            resid,
            bart['min_points_per_leaf'],
            'ratios' in bart,
            prelk,
            *item,
        )
        return resid, (leaf_tree, acc, to_prune, ratios)

    items = (
        bart['leaf_trees'], count_trees,
        moves, move_counts,
        bart['leaf_indices'],
        prelkv, prelf,
    )
    resid, (leaf_trees, acc, to_prune, ratios) = lax.scan(loop, bart['resid'], items)

    bart['resid'] = resid
    bart['leaf_trees'] = leaf_trees
    bart.get('ratios', {}).update(ratios)
    moves['acc'] = acc
    moves['to_prune'] = to_prune

    return bart, moves

def accept_move_and_sample_leaves(X, ntree, resid_batch_size, resid, min_points_per_leaf, save_ratios, prelk, leaf_tree, count_tree, move, move_counts, leaf_indices, prelkv, prelf):
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
    min_points_per_leaf : int or None
        The minimum number of data points in a leaf node.
    save_ratios : bool
        Whether to save the acceptance ratios.
    prelk : dict
        The pre-computed terms of the likelihood ratio which are shared across
        trees.
    leaf_tree : float array (2 ** d,)
        The leaf values of the tree.
    count_tree : int array (2 ** d,)
        The number of datapoints in each leaf.
    move : dict
        The proposed move, see `sample_moves`.
    leaf_indices : int array (n,)
        The leaf indices for the largest version of the tree compatible with
        the move.
    prelkv, prelf : dict
        The pre-computed terms of the likelihood ratio and leaf sampling which
        are specific to the tree.

    Returns
    -------
    resid : float array (n,)
        The updated residuals (data minus forest value).
    leaf_tree : float array (2 ** d,)
        The new leaf values of the tree.
    acc : bool
        Whether the move was accepted.
    to_prune : bool
        Whether, to reflect the acceptance status of the move, the state should
        be updated by pruning the leaves involved in the move.
    ratios : dict
        The acceptance ratios for the moves. Empty if not to be saved.
    """

    # sum residuals and count units per leaf, in tree proposed by grow move
    resid_tree = sum_resid(resid, leaf_indices, leaf_tree.size, resid_batch_size)

    # subtract starting tree from function
    resid_tree += count_tree * leaf_tree

    # get indices of move
    node = move['node']
    assert node.dtype == jnp.int32
    left = move['left']
    right = move['right']

    # sum residuals in parent node modified by move
    resid_left = resid_tree[left]
    resid_right = resid_tree[right]
    resid_total = resid_left + resid_right
    resid_tree = resid_tree.at[node].set(resid_total)

    # compute acceptance ratio
    log_lk_ratio = compute_likelihood_ratio(resid_total, resid_left, resid_right, prelkv, prelk)
    log_ratio = move['log_trans_prior_ratio'] + log_lk_ratio
    log_ratio = jnp.where(move['grow'], log_ratio, -log_ratio)
    ratios = {}
    if save_ratios:
        ratios.update(
            log_trans_prior=move['log_trans_prior_ratio'],
            log_likelihood=log_lk_ratio,
        )

    # determine whether to accept the move
    acc = move['allowed'] & (move['logu'] <= log_ratio)
    if min_points_per_leaf is not None:
        acc &= move_counts['left'] >= min_points_per_leaf
        acc &= move_counts['right'] >= min_points_per_leaf

    # compute leaves posterior and sample leaves
    initial_leaf_tree = leaf_tree
    mean_post = resid_tree * prelf['mean_factor']
    leaf_tree = mean_post + prelf['centered_leaves']

    # copy leaves around such that the leaf indices select the right leaf
    to_prune = acc ^ move['grow']
    leaf_tree = (leaf_tree
        .at[jnp.where(to_prune, left, leaf_tree.size)]
        .set(leaf_tree[node])
        .at[jnp.where(to_prune, right, leaf_tree.size)]
        .set(leaf_tree[node])
    )

    # replace old tree with new tree in function values
    resid += (initial_leaf_tree - leaf_tree)[leaf_indices]

    return resid, leaf_tree, acc, to_prune, ratios

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

def compute_likelihood_ratio(total_resid, left_resid, right_resid, prelkv, prelk):
    """
    Compute the likelihood ratio of a grow move.

    Parameters
    ----------
    total_resid : float
        The sum of the residuals in the leaf to grow.
    left_resid, right_resid : float
        The sum of the residuals in the left/right child of the leaf to grow.
    prelkv, prelk : dict
        The pre-computed terms of the likelihood ratio, see
        `precompute_likelihood_terms`.

    Returns
    -------
    ratio : float
        The likelihood ratio P(data | new tree) / P(data | old tree).
    """
    exp_term = prelk['exp_factor'] * (
        left_resid * left_resid / prelkv['sigma2_left'] +
        right_resid * right_resid / prelkv['sigma2_right'] -
        total_resid * total_resid / prelkv['sigma2_total']
    )
    return prelkv['sqrt_term'] + exp_term

def accept_moves_final_stage(bart, moves):
    """
    The final part of accepting the moves, in parallel across trees.

    Parameters
    ----------
    bart : dict
        A partially updated BART mcmc state.
    counts : dict
        The indicators of proposals and acceptances for grow and prune moves.
    moves : dict
        The proposed moves (see `sample_moves`) as updated by
        `accept_moves_sequential_stage`.

    Returns
    -------
    bart : dict
        The fully updated BART mcmc state.
    """
    bart = bart.copy()
    bart['grow_acc_count'] = jnp.sum(moves['acc'] & moves['grow'])
    bart['prune_acc_count'] = jnp.sum(moves['acc'] & ~moves['grow'])
    bart['leaf_indices'] = apply_moves_to_leaf_indices(bart['leaf_indices'], moves)
    bart['split_trees'] = apply_moves_to_split_trees(bart['split_trees'], moves)
    return bart

@jax.vmap
def apply_moves_to_leaf_indices(leaf_indices, moves):
    """
    Update the leaf indices to match the accepted move.

    Parameters
    ----------
    leaf_indices : int array (num_trees, n)
        The index of the leaf each datapoint falls into, if the grow move was
        accepted.
    moves : dict
        The proposed moves (see `sample_moves`), as updated by
        `accept_moves_sequential_stage`.

    Returns
    -------
    leaf_indices : int array (num_trees, n)
        The updated leaf indices.
    """
    mask = ~jnp.array(1, leaf_indices.dtype) # ...1111111110
    is_child = (leaf_indices & mask) == moves['left']
    return jnp.where(
        is_child & moves['to_prune'],
        moves['node'].astype(leaf_indices.dtype),
        leaf_indices,
    )

@jax.vmap
def apply_moves_to_split_trees(split_trees, moves):
    """
    Update the split trees to match the accepted move.

    Parameters
    ----------
    split_trees : int array (num_trees, 2 ** (d - 1))
        The cutpoints of the decision nodes in the initial trees.
    moves : dict
        The proposed moves (see `sample_moves`), as updated by
        `accept_moves_sequential_stage`.

    Returns
    -------
    split_trees : int array (num_trees, 2 ** (d - 1))
        The updated split trees.
    """
    return (split_trees
        .at[jnp.where(
            moves['grow'],
            moves['node'],
            split_trees.size,
        )]
        .set(moves['grow_split'].astype(split_trees.dtype))
        .at[jnp.where(
            moves['to_prune'],
            moves['node'],
            split_trees.size,
        )]
        .set(0)
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
