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
    suffstat_batch_size='auto',
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
    suffstat_batch_size : int, None, str, default 'auto'
        The batch size for computing sufficient statistics. `None` for no
        batching. If 'auto', pick a value based on the device of `y`, or the
        default device.

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
        'min_points_per_leaf' : int or None
            The minimum number of data points in a leaf node.
        'affluence_trees' : bool array (num_trees, 2 ** (d - 1)) or None
            Whether a non-bottom leaf nodes contains twice `min_points_per_leaf`
            datapoints. If `min_points_per_leaf` is not specified, this is None.
        'opt' : LeafDict
            A dictionary with config values:

            'suffstat_batch_size' : int or None
                The batch size for computing sufficient statistics.
            'small_float' : dtype
                The dtype for large arrays used in the algorithm.
            'large_float' : dtype
                The dtype for scalars, small arrays, and arrays which require
                accuracy.
            'require_min_points' : bool
                Whether the `min_points_per_leaf` parameter is specified.
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
    suffstat_batch_size = _choose_suffstat_batch_size(suffstat_batch_size, y)

    bart = dict(
        leaf_trees=make_forest(max_depth, small_float),
        var_trees=make_forest(max_depth - 1, jaxext.minimal_unsigned_dtype(X.shape[0] - 1)),
        split_trees=make_forest(max_depth - 1, max_split.dtype),
        resid=jnp.asarray(y, large_float),
        sigma2=jnp.ones((), large_float),
        grow_prop_count=jnp.zeros((), int),
        grow_acc_count=jnp.zeros((), int),
        prune_prop_count=jnp.zeros((), int),
        prune_acc_count=jnp.zeros((), int),
        p_nonterminal=p_nonterminal,
        sigma2_alpha=jnp.asarray(sigma2_alpha, large_float),
        sigma2_beta=jnp.asarray(sigma2_beta, large_float),
        max_split=jnp.asarray(max_split),
        y=y,
        X=jnp.asarray(X),
        min_points_per_leaf=(
            None if min_points_per_leaf is None else
            jnp.asarray(min_points_per_leaf)
        ),
        affluence_trees=(
            None if min_points_per_leaf is None else
            make_forest(max_depth - 1, bool).at[:, 1].set(y.size >= 2 * min_points_per_leaf)
        ),
        opt=jaxext.LeafDict(
            suffstat_batch_size=suffstat_batch_size,
            small_float=small_float,
            large_float=large_float,
            require_min_points=min_points_per_leaf is not None,
        ),
    )

    return bart

def _choose_suffstat_batch_size(size, y):
    if size == 'auto':
        try:
            device = y.devices().pop()
        except jax.errors.ConcretizationTypeError:
            device = jax.devices()[0]
        platform = device.platform

        if platform == 'cpu':
            return None
                # maybe I should batch residuals (not counts) for numerical
                # accuracy, even if it's slower
        elif platform == 'gpu':
            return 128 # 128 is good on A100, and V100 at high n
                       # 512 is good on T4, and V100 at low n
        else:
            raise KeyError(f'Unknown platform: {platform}')
    
    elif size is not None:
        return int(size)
    
    return size

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
    This function zeroes the proposal counters before using them.
    """
    bart = bart.copy()
    key, subkey = random.split(key)
    grow_moves, prune_moves = sample_moves(bart, subkey)
    bart['var_trees'] = grow_moves['var_tree']
    grow_leaf_indices = grove.traverse_forest(bart['X'], grow_moves['var_tree'], grow_moves['split_tree'])
    return accept_moves_and_sample_leaves(bart, grow_moves, prune_moves, grow_leaf_indices, key)

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
    return sample_moves_vmap_trees(bart['var_trees'], bart['split_trees'], bart['affluence_trees'], bart['max_split'], bart['p_nonterminal'], key)

@functools.partial(jaxext.vmap_nodoc, in_axes=(0, 0, 0, None, None, 0))
def sample_moves_vmap_trees(var_tree, split_tree, affluence_tree, max_split, p_nonterminal, key):
    key, key1 = random.split(key)
    args = var_tree, split_tree, affluence_tree, max_split, p_nonterminal
    grow = grow_move(*args, key)
    prune = prune_move(*args, key1)
    return grow, prune

def grow_move(var_tree, split_tree, affluence_tree, max_split, p_nonterminal, key):
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
    key : jax.dtypes.prng_key array
        A jax random key.

    Returns
    -------
    grow_move : dict
        A dictionary with fields:

        'allowed' : bool
            Whether the move is possible.
        'node' : int
            The index of the leaf to grow.
        'var_tree' : array (2 ** (d - 1),)
            The new decision axes of the tree.
        'split_tree' : array (2 ** (d - 1),)
            The new decision boundaries of the tree.
        'partial_ratio' : float
            A factor of the Metropolis-Hastings ratio of the move. It lacks
            the likelihood ratio and the probability of proposing the prune
            move.
    """

    key, key1, key2 = random.split(key, 3)
    
    leaf_to_grow, num_growable, num_prunable, allowed = choose_leaf(split_tree, affluence_tree, key)

    var = choose_variable(var_tree, split_tree, max_split, leaf_to_grow, key1)
    var_tree = var_tree.at[leaf_to_grow].set(var.astype(var_tree.dtype))
    
    split = choose_split(var_tree, split_tree, max_split, leaf_to_grow, key2)
    split_tree = split_tree.at[leaf_to_grow].set(split.astype(split_tree.dtype))

    ratio = compute_partial_ratio(num_growable, num_prunable, p_nonterminal, leaf_to_grow, split_tree)

    return dict(
        allowed=allowed,
        node=leaf_to_grow,
        partial_ratio=ratio,
        var_tree=var_tree,
        split_tree=split_tree,
    )

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
        ``2 ** d``.
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
    leaf_to_grow = jnp.where(allowed, leaf_to_grow, 2 * split_tree.size)
    num_growable = jnp.count_nonzero(is_growable)
    is_parent = grove.is_leaves_parent(split_tree.at[leaf_to_grow].set(1))
    num_prunable = jnp.count_nonzero(is_parent)
    return leaf_to_grow, num_growable, num_prunable, allowed

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

def compute_partial_ratio(num_growable, num_prunable, p_nonterminal, leaf_to_grow, new_split_tree):
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

    prune_allowed = leaf_to_grow != 1
        # prune allowed  <--->  the initial tree is not a root
        # leaf to grow is root  -->  the tree can only be a root
        # tree is a root  -->  the only leaf I can grow is root

    p_grow = jnp.where(prune_allowed, 0.5, 1)

    trans_ratio = num_growable / (p_grow * num_prunable)

    depth = grove.tree_depths(new_split_tree.size)[leaf_to_grow]
    p_parent = p_nonterminal[depth]
    cp_children = 1 - p_nonterminal[depth + 1]
    tree_ratio = cp_children * cp_children * p_parent / (1 - p_parent)

    return trans_ratio * tree_ratio

def prune_move(var_tree, split_tree, affluence_tree, max_split, p_nonterminal, key):
    """
    Tree structure prune move proposal of BART MCMC.

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
    key : jax.dtypes.prng_key array
        A jax random key.

    Returns
    -------
    prune_move : dict
        A dictionary with fields:

        'allowed' : bool
            Whether the move is possible.
        'node' : int
            The index of the node to prune.
        'partial_ratio' : float
            A factor of the Metropolis-Hastings ratio of the move. It lacks
            the likelihood ratio and the probability of proposing the prune
            move. This ratio is inverted.
    """
    node_to_prune, num_prunable, num_growable = choose_leaf_parent(split_tree, affluence_tree, key)
    allowed = split_tree[1].astype(bool) # allowed iff the tree is not a root

    ratio = compute_partial_ratio(num_growable, num_prunable, p_nonterminal, node_to_prune, split_tree)

    return dict(
        allowed=allowed,
        node=node_to_prune,
        partial_ratio=ratio, # it is inverted in accept_move_and_sample_leaves
    )

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

def accept_moves_and_sample_leaves(bart, grow_moves, prune_moves, grow_leaf_indices, key):
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
    grow_leaf_indices : int array (num_trees, n)
        The leaf indices of the trees proposed by the grow move.
    key : jax.dtypes.prng_key array
        A jax random key.

    Returns
    -------
    bart : dict
        The new BART mcmc state.
    """
    bart = bart.copy()
    def loop(carry, item):
        resid = carry.pop('resid')
        resid, carry, trees = accept_move_and_sample_leaves(
            bart['X'],
            len(bart['leaf_trees']),
            bart['opt']['suffstat_batch_size'],
            resid,
            bart['sigma2'],
            bart['min_points_per_leaf'],
            carry,
            *item,
        )
        carry['resid'] = resid
        return carry, trees
    carry = {
        k: jnp.zeros_like(bart[k]) for k in
        ['grow_prop_count', 'prune_prop_count', 'grow_acc_count', 'prune_acc_count']
    }
    carry['resid'] = bart['resid']
    items = (
        bart['leaf_trees'],
        bart['split_trees'],
        bart['affluence_trees'],
        grow_moves,
        prune_moves,
        grow_leaf_indices,
        random.split(key, len(bart['leaf_trees'])),
    )
    carry, trees = lax.scan(loop, carry, items)
    bart.update(carry)
    bart.update(trees)
    return bart

def accept_move_and_sample_leaves(X, ntree, suffstat_batch_size, resid, sigma2, min_points_per_leaf, counts, leaf_tree, split_tree, affluence_tree, grow_move, prune_move, grow_leaf_indices, key):
    """
    Accept or reject a proposed move and sample the new leaf values.

    Parameters
    ----------
    X : int array (p, n)
        The predictors.
    ntree : int
        The number of trees in the forest.
    suffstat_batch_size : int, None
        The batch size for computing sufficient statistics.
    resid : float array (n,)
        The residuals (data minus forest value).
    sigma2 : float
        The noise variance.
    min_points_per_leaf : int or None
        The minimum number of data points in a leaf node.
    counts : dict
        The acceptance counts from the mcmc state dict.
    leaf_tree : float array (2 ** d,)
        The leaf values of the tree.
    split_tree : int array (2 ** (d - 1),)
        The decision boundaries of the tree.
    affluence_tree : bool array (2 ** (d - 1),) or None
        Whether a leaf has enough points to be grown.
    grow_move : dict
        The proposal for the grow move. See `grow_move`.
    prune_move : dict
        The proposal for the prune move. See `prune_move`.
    grow_leaf_indices : int array (n,)
        The leaf indices of the tree proposed by the grow move.
    key : jax.dtypes.prng_key array
        A jax random key.

    Returns
    -------
    resid : float array (n,)
        The updated residuals (data minus forest value).
    counts : dict
        The updated acceptance counts.
    trees : dict
        The updated tree arrays.
    """
    
    # compute leaf indices in starting tree
    grow_node = grow_move['node']
    grow_left = grow_node << 1
    grow_right = grow_left + 1
    leaf_indices = jnp.where(
        (grow_leaf_indices == grow_left) | (grow_leaf_indices == grow_right),
        grow_node,
        grow_leaf_indices,
    )

    # compute leaf indices in prune tree
    prune_node = prune_move['node']
    prune_left = prune_node << 1
    prune_right = prune_left + 1
    prune_leaf_indices = jnp.where(
        (leaf_indices == prune_left) | (leaf_indices == prune_right),
        prune_node,
        leaf_indices,
    )

    # subtract starting tree from function
    resid += leaf_tree[leaf_indices]

    # aggregate residuals and count units per leaf
    grow_resid_tree, grow_count_tree = sufficient_stat(resid, grow_leaf_indices, leaf_tree.size, suffstat_batch_size)

    # compute aggregations in starting tree
    # I do not zero the children because garbage there does not matter
    resid_tree = (grow_resid_tree.at[grow_node]
        .set(grow_resid_tree[grow_left] + grow_resid_tree[grow_right]))
    count_tree = (grow_count_tree.at[grow_node]
        .set(grow_count_tree[grow_left] + grow_count_tree[grow_right]))

    # compute aggregations in prune tree
    prune_resid_tree = (resid_tree.at[prune_node]
        .set(resid_tree[prune_left] + resid_tree[prune_right]))
    prune_count_tree = (count_tree.at[prune_node]
        .set(count_tree[prune_left] + count_tree[prune_right]))

    # compute affluence trees
    if min_points_per_leaf is not None:
        grow_affluence_tree = grow_count_tree[:grow_count_tree.size // 2] >= 2 * min_points_per_leaf
        prune_affluence_tree = affluence_tree.at[prune_node].set(True)

    # compute probability of proposing prune
    grow_p_prune = compute_p_prune_back(grow_move['split_tree'], grow_affluence_tree)
    prune_p_prune = compute_p_prune_back(split_tree, affluence_tree)

    # compute likelihood ratios
    grow_lk_ratio = compute_likelihood_ratio(grow_resid_tree, grow_count_tree, sigma2, grow_node, ntree, min_points_per_leaf)
    prune_lk_ratio = compute_likelihood_ratio(resid_tree, count_tree, sigma2, prune_node, ntree, min_points_per_leaf)

    # compute acceptance ratios
    grow_ratio = grow_p_prune * grow_move['partial_ratio'] * grow_lk_ratio
    prune_ratio = prune_p_prune * prune_move['partial_ratio'] * prune_lk_ratio
    prune_ratio = lax.reciprocal(prune_ratio)

    # random coins in [0, 1) for proposal and acceptance
    key, subkey = random.split(key)
    u0, u1 = random.uniform(subkey, (2,))

    # determine what move to propose (not proposing anything is an option)
    p_grow = jnp.where(grow_move['allowed'] & prune_move['allowed'], 0.5, grow_move['allowed'])
    try_grow = u0 < p_grow
    try_prune = prune_move['allowed'] & ~try_grow

    # determine whether to accept the move
    do_grow = try_grow & (u1 < grow_ratio)
    do_prune = try_prune & (u1 < prune_ratio)

    # pick trees for chosen move
    trees = {}
    split_tree = jnp.where(do_grow, grow_move['split_tree'], split_tree)
    # the prune var tree is equal to the initial one, because I leave garbage values behind
    split_tree = split_tree.at[prune_node].set(
        jnp.where(do_prune, 0, split_tree[prune_node]))
    if min_points_per_leaf is not None:
        affluence_tree = jnp.where(do_grow, grow_affluence_tree, affluence_tree)
        affluence_tree = jnp.where(do_prune, prune_affluence_tree, affluence_tree)
    resid_tree = jnp.where(do_grow, grow_resid_tree, resid_tree)
    count_tree = jnp.where(do_grow, grow_count_tree, count_tree)
    resid_tree = jnp.where(do_prune, prune_resid_tree, resid_tree)
    count_tree = jnp.where(do_prune, prune_count_tree, count_tree)

    # update acceptance counts
    counts = counts.copy()
    counts['grow_prop_count'] += try_grow
    counts['grow_acc_count'] += do_grow
    counts['prune_prop_count'] += try_prune
    counts['prune_acc_count'] += do_prune

    # compute leaves posterior
    prec_lk = count_tree / sigma2
    var_post = lax.reciprocal(prec_lk + ntree) # = 1 / (prec_lk + prec_prior)
    mean_post = resid_tree / sigma2 * var_post # = mean_lk * prec_lk * var_post

    # sample leaves
    z = random.normal(key, mean_post.shape, mean_post.dtype)
    leaf_tree = mean_post + z * jnp.sqrt(var_post)

    # add new tree to function
    leaf_indices = jnp.where(do_grow, grow_leaf_indices, leaf_indices)
    leaf_indices = jnp.where(do_prune, prune_leaf_indices, leaf_indices)
    resid -= leaf_tree[leaf_indices]

    # pack trees
    trees = {
        'leaf_trees': leaf_tree,
        'split_trees': split_tree,
        'affluence_trees': affluence_tree,
    }

    return resid, counts, trees

def sufficient_stat(resid, leaf_indices, tree_size, batch_size):
    """
    Compute the sufficient statistics for the likelihood ratio of a tree move.

    Parameters
    ----------
    resid : float array (n,)
        The residuals (data minus forest value).
    leaf_indices : int array (n,)
        The leaf indices of the tree (in which leaf each data point falls into).
    tree_size : int
        The size of the tree array (2 ** d).
    batch_size : int, None
        The batch size for the aggregation. Batching increases numerical
        accuracy and parallelism.

    Returns
    -------
    resid_tree : float array (2 ** d,)
        The sum of the residuals at data points in each leaf.
    count_tree : int array (2 ** d,)
        The number of data points in each leaf.
    """
    if batch_size is None:
        aggr_func = _aggregate_scatter
    else:
        aggr_func = functools.partial(_aggregate_batched, batch_size=batch_size)
    resid_tree = aggr_func(resid, leaf_indices, tree_size, jnp.float32)
    count_tree = aggr_func(1, leaf_indices, tree_size, jnp.uint32)
    return resid_tree, count_tree

def _aggregate_scatter(values, indices, size, dtype):
    return (jnp
        .zeros(size, dtype)
        .at[indices]
        .add(values)
    )

def _aggregate_batched(values, indices, size, dtype, batch_size):
    nbatches = indices.size // batch_size + bool(indices.size % batch_size)
    batch_indices = jnp.arange(indices.size) // batch_size
    return (jnp
        .zeros((nbatches, size), dtype)
        .at[batch_indices, indices]
        .add(values)
        .sum(axis=0)
    )

def compute_p_prune_back(new_split_tree, new_affluence_tree):
    """
    Compute the probability of proposing a prune move after doing a grow move.

    Parameters
    ----------
    new_split_tree : int array (2 ** (d - 1),)
        The decision boundaries of the tree, after the grow move.
    new_affluence_tree : bool array (2 ** (d - 1),)
        Which leaves have enough points to be grown, after the grow move.

    Returns
    -------
    p_prune : float
        The probability of proposing a prune move after the grow move. This is
        0.5 if grow is possible again, and 1 if it isn't. It can't be 0 because
        at least the node just grown can be pruned.
    """
    _, grow_again_allowed = growable_leaves(new_split_tree, new_affluence_tree)
    return jnp.where(grow_again_allowed, 0.5, 1)

def compute_likelihood_ratio(resid_tree, count_tree, sigma2, node, n_tree, min_points_per_leaf):
    """
    Compute the likelihood ratio of a grow move.

    Parameters
    ----------
    resid_tree : float array (2 ** d,)
        The sum of the residuals at data points in each leaf.
    count_tree : int array (2 ** d,)
        The number of data points in each leaf.
    sigma2 : float
        The noise variance.
    node : int
        The index of the leaf that has been grown.
    n_tree : int
        The number of trees in the forest.
    min_points_per_leaf : int or None
        The minimum number of data points in a leaf node.

    Returns
    -------
    ratio : float
        The likelihood ratio P(data | new tree) / P(data | old tree).

    Notes
    -----
    The ratio is set to 0 if the grow move would create leaves with not enough
    datapoints per leaf, although this is part of the prior rather than the
    likelihood.
    """

    left_child = node << 1
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

    return ratio

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
    norm2 = jnp.dot(resid, resid, preferred_element_type=bart['sigma2_beta'].dtype)
    beta = bart['sigma2_beta'] + norm2 / 2

    sample = random.gamma(key, alpha)
    bart['sigma2'] = beta / sample

    return bart
