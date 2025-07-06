# bartz/src/bartz/mcmcstep.py
#
# Copyright (c) 2024-2025, Giacomo Petrillo
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
outputting a new state. The inputs are not modified.

The entry points are:

  - `State`: The dataclass that represents a BART MCMC state.
  - `init`: Creates an initial `State` from data and configurations.
  - `step`: Performs one full MCMC step on a `State`, returning a new `State`.
  - `step_sparse`: Performs the MCMC update for variable selection, which is skipped in `step`.
"""

import math
from dataclasses import replace
from functools import cache, partial
from typing import Any, Literal

import jax
from equinox import Module, field, tree_at
from jax import lax, random
from jax import numpy as jnp
from jax.scipy.special import gammaln, logsumexp
from jaxtyping import Array, Bool, Float32, Int32, Integer, Key, Shaped, UInt

from bartz import grove
from bartz.jaxext import (
    minimal_unsigned_dtype,
    split,
    truncated_normal_onesided,
    vmap_nodoc,
)


class Forest(Module):
    """
    Represents the MCMC state of a sum of trees.

    Parameters
    ----------
    leaf_tree
        The leaf values.
    var_tree
        The decision axes.
    split_tree
        The decision boundaries.
    affluence_tree
        Marks leaves that can be grown.
    max_split
        The maximum split index for each predictor.
    blocked_vars
        Indices of variables that are not used. This shall include at least
        the `i` such that ``max_split[i] == 0``, otherwise behavior is
        undefined.
    p_nonterminal
        The prior probability of each node being nonterminal, conditional on
        its ancestors. Includes the nodes at maximum depth which should be set
        to 0.
    p_propose_grow
        The unnormalized probability of picking a leaf for a grow proposal.
    leaf_indices
        The index of the leaf each datapoints falls into, for each tree.
    min_points_per_decision_node
        The minimum number of data points in a decision node.
    min_points_per_leaf
        The minimum number of data points in a leaf node.
    resid_batch_size
    count_batch_size
        The data batch sizes for computing the sufficient statistics. If `None`,
        they are computed with no batching.
    log_trans_prior
        The log transition and prior Metropolis-Hastings ratio for the
        proposed move on each tree.
    log_likelihood
        The log likelihood ratio.
    grow_prop_count
    prune_prop_count
        The number of grow/prune proposals made during one full MCMC cycle.
    grow_acc_count
    prune_acc_count
        The number of grow/prune moves accepted during one full MCMC cycle.
    sigma_mu2
        The prior variance of a leaf, conditional on the tree structure.
    log_s
        The logarithm of the prior probability for choosing a variable to split
        along in a decision rule, conditional on the ancestors. Not normalized.
        If `None`, use a uniform distribution.
    theta
        The concentration parameter for the Dirichlet prior on the variable
        distribution `s`. Required only to update `s`.
    a
    b
    rho
        Parameters of the prior on `theta`. Required only to sample `theta`.
        See `step_theta`.
    """

    leaf_tree: Float32[Array, 'num_trees 2**d']
    var_tree: UInt[Array, 'num_trees 2**(d-1)']
    split_tree: UInt[Array, 'num_trees 2**(d-1)']
    affluence_tree: Bool[Array, 'num_trees 2**(d-1)']
    max_split: UInt[Array, ' p']
    blocked_vars: UInt[Array, ' k'] | None
    p_nonterminal: Float32[Array, ' 2**d']
    p_propose_grow: Float32[Array, ' 2**(d-1)']
    leaf_indices: UInt[Array, 'num_trees n']
    min_points_per_decision_node: Int32[Array, ''] | None
    min_points_per_leaf: Int32[Array, ''] | None
    resid_batch_size: int | None = field(static=True)
    count_batch_size: int | None = field(static=True)
    log_trans_prior: Float32[Array, ' num_trees'] | None
    log_likelihood: Float32[Array, ' num_trees'] | None
    grow_prop_count: Int32[Array, '']
    prune_prop_count: Int32[Array, '']
    grow_acc_count: Int32[Array, '']
    prune_acc_count: Int32[Array, '']
    sigma_mu2: Float32[Array, '']
    log_s: Float32[Array, ' p'] | None
    theta: Float32[Array, ''] | None
    a: Float32[Array, ''] | None
    b: Float32[Array, ''] | None
    rho: Float32[Array, ''] | None


class State(Module):
    """
    Represents the MCMC state of BART.

    Parameters
    ----------
    X
        The predictors.
    y
        The response. If the data type is `bool`, the model is binary regression.
    resid
        The residuals (`y` or `z` minus sum of trees).
    z
        The latent variable for binary regression. `None` in continuous
        regression.
    offset
        Constant shift added to the sum of trees.
    sigma2
        The error variance. `None` in binary regression.
    prec_scale
        The scale on the error precision, i.e., ``1 / error_scale ** 2``.
        `None` in binary regression.
    sigma2_alpha
    sigma2_beta
        The shape and scale parameters of the inverse gamma prior on the noise
        variance. `None` in binary regression.
    forest
        The sum of trees model.
    """

    X: UInt[Array, 'p n']
    y: Float32[Array, ' n'] | Bool[Array, ' n']
    z: None | Float32[Array, ' n']
    offset: Float32[Array, '']
    resid: Float32[Array, ' n']
    sigma2: Float32[Array, ''] | None
    prec_scale: Float32[Array, ' n'] | None
    sigma2_alpha: Float32[Array, ''] | None
    sigma2_beta: Float32[Array, ''] | None
    forest: Forest


def init(
    *,
    X: UInt[Any, 'p n'],
    y: Float32[Any, ' n'] | Bool[Any, ' n'],
    offset: float | Float32[Any, ''] = 0.0,
    max_split: UInt[Any, ' p'],
    num_trees: int,
    p_nonterminal: Float32[Any, ' d-1'],
    sigma_mu2: float | Float32[Any, ''],
    sigma2_alpha: float | Float32[Any, ''] | None = None,
    sigma2_beta: float | Float32[Any, ''] | None = None,
    error_scale: Float32[Any, ' n'] | None = None,
    min_points_per_decision_node: int | Integer[Any, ''] | None = None,
    resid_batch_size: int | None | Literal['auto'] = 'auto',
    count_batch_size: int | None | Literal['auto'] = 'auto',
    save_ratios: bool = False,
    filter_splitless_vars: bool = True,
    min_points_per_leaf: int | Integer[Any, ''] | None = None,
    log_s: Float32[Any, ' p'] | None = None,
    theta: float | Float32[Any, ''] | None = None,
    a: float | Float32[Any, ''] | None = None,
    b: float | Float32[Any, ''] | None = None,
    rho: float | Float32[Any, ''] | None = None,
) -> State:
    """
    Make a BART posterior sampling MCMC initial state.

    Parameters
    ----------
    X
        The predictors. Note this is trasposed compared to the usual convention.
    y
        The response. If the data type is `bool`, the regression model is binary
        regression with probit.
    offset
        Constant shift added to the sum of trees. 0 if not specified.
    max_split
        The maximum split index for each variable. All split ranges start at 1.
    num_trees
        The number of trees in the forest.
    p_nonterminal
        The probability of a nonterminal node at each depth. The maximum depth
        of trees is fixed by the length of this array.
    sigma_mu2
        The prior variance of a leaf, conditional on the tree structure. The
        prior variance of the sum of trees is ``num_trees * sigma_mu2``. The
        prior mean of leaves is always zero.
    sigma2_alpha
    sigma2_beta
        The shape and scale parameters of the inverse gamma prior on the error
        variance. Leave unspecified for binary regression.
    error_scale
        Each error is scaled by the corresponding factor in `error_scale`, so
        the error variance for ``y[i]`` is ``sigma2 * error_scale[i] ** 2``.
        Not supported for binary regression. If not specified, defaults to 1 for
        all points, but potentially skipping calculations.
    min_points_per_decision_node
        The minimum number of data points in a decision node. 0 if not
        specified.
    resid_batch_size
    count_batch_size
        The batch sizes, along datapoints, for summing the residuals and
        counting the number of datapoints in each leaf. `None` for no batching.
        If 'auto', pick a value based on the device of `y`, or the default
        device.
    save_ratios
        Whether to save the Metropolis-Hastings ratios.
    filter_splitless_vars
        Whether to check `max_split` for variables without available cutpoints.
        If any are found, they are put into a list of variables to exclude from
        the MCMC. If `False`, no check is performed, but the results may be
        wrong if any variable is blocked. The function is jax-traceable only
        if this is set to `False`.
    min_points_per_leaf
        The minimum number of datapoints in a leaf node. 0 if not specified.
        Unlike `min_points_per_decision_node`, this constraint is not taken into
        account in the Metropolis-Hastings ratio because it would be expensive
        to compute. Grow moves that would violate this constraint are vetoed.
        This parameter is independent of `min_points_per_decision_node` and
        there is no check that they are coherent. It makes sense to set
        ``min_points_per_decision_node >= 2 * min_points_per_leaf``.
    log_s
        The logarithm of the prior probability for choosing a variable to split
        along in a decision rule, conditional on the ancestors. Not normalized.
        If not specified, use a uniform distribution. If not specified and
        `theta` or `rho`, `a`, `b` are, it's initialized automatically.
    theta
        The concentration parameter for the Dirichlet prior on `s`. Required
        only to update `log_s`. If not specified, and `rho`, `a`, `b` are
        specified, it's initialized automatically.
    a
    b
    rho
        Parameters of the prior on `theta`. Required only to sample `theta`.

    Returns
    -------
    An initialized BART MCMC state.

    Raises
    ------
    ValueError
        If `y` is boolean and arguments unused in binary regression are set.

    Notes
    -----
    In decision nodes, the values in ``X[i, :]`` are compared to a cutpoint out
    of the range ``[1, 2, ..., max_split[i]]``. A point belongs to the left
    child iff ``X[i, j] < cutpoint``. Thus it makes sense for ``X[i, :]`` to be
    integers in the range ``[0, 1, ..., max_split[i]]``.
    """
    p_nonterminal = jnp.asarray(p_nonterminal)
    p_nonterminal = jnp.pad(p_nonterminal, (0, 1))
    max_depth = p_nonterminal.size

    @partial(jax.vmap, in_axes=None, out_axes=0, axis_size=num_trees)
    def make_forest(max_depth, dtype):
        return grove.make_tree(max_depth, dtype)

    y = jnp.asarray(y)
    offset = jnp.asarray(offset)

    resid_batch_size, count_batch_size = _choose_suffstat_batch_size(
        resid_batch_size, count_batch_size, y, 2**max_depth * num_trees
    )

    is_binary = y.dtype == bool
    if is_binary:
        if (error_scale, sigma2_alpha, sigma2_beta) != 3 * (None,):
            msg = (
                'error_scale, sigma2_alpha, and sigma2_beta must be set '
                ' to `None` for binary regression.'
            )
            raise ValueError(msg)
        sigma2 = None
    else:
        sigma2_alpha = jnp.asarray(sigma2_alpha)
        sigma2_beta = jnp.asarray(sigma2_beta)
        sigma2 = sigma2_beta / sigma2_alpha

    max_split = jnp.asarray(max_split)

    if filter_splitless_vars:
        (blocked_vars,) = jnp.nonzero(max_split == 0)
        blocked_vars = blocked_vars.astype(minimal_unsigned_dtype(max_split.size))
        # see `fully_used_variables` for the type cast
    else:
        blocked_vars = None

    # check and initialize sparsity parameters
    if not _all_none_or_not_none(rho, a, b):
        msg = 'rho, a, b are not either all `None` or all set'
        raise ValueError(msg)
    if theta is None and rho is not None:
        theta = rho
    if log_s is None and theta is not None:
        log_s = jnp.zeros(max_split.size)

    return State(
        X=jnp.asarray(X),
        y=y,
        z=jnp.full(y.shape, offset) if is_binary else None,
        offset=offset,
        resid=jnp.zeros(y.shape) if is_binary else y - offset,
        sigma2=sigma2,
        prec_scale=(
            None if error_scale is None else lax.reciprocal(jnp.square(error_scale))
        ),
        sigma2_alpha=sigma2_alpha,
        sigma2_beta=sigma2_beta,
        forest=Forest(
            leaf_tree=make_forest(max_depth, jnp.float32),
            var_tree=make_forest(max_depth - 1, minimal_unsigned_dtype(X.shape[0] - 1)),
            split_tree=make_forest(max_depth - 1, max_split.dtype),
            affluence_tree=(
                make_forest(max_depth - 1, bool)
                .at[:, 1]
                .set(
                    True
                    if min_points_per_decision_node is None
                    else y.size >= min_points_per_decision_node
                )
            ),
            blocked_vars=blocked_vars,
            max_split=max_split,
            grow_prop_count=jnp.zeros((), int),
            grow_acc_count=jnp.zeros((), int),
            prune_prop_count=jnp.zeros((), int),
            prune_acc_count=jnp.zeros((), int),
            p_nonterminal=p_nonterminal[grove.tree_depths(2**max_depth)],
            p_propose_grow=p_nonterminal[grove.tree_depths(2 ** (max_depth - 1))],
            leaf_indices=jnp.ones(
                (num_trees, y.size), minimal_unsigned_dtype(2**max_depth - 1)
            ),
            min_points_per_decision_node=_asarray_or_none(min_points_per_decision_node),
            min_points_per_leaf=_asarray_or_none(min_points_per_leaf),
            resid_batch_size=resid_batch_size,
            count_batch_size=count_batch_size,
            log_trans_prior=jnp.zeros(num_trees) if save_ratios else None,
            log_likelihood=jnp.zeros(num_trees) if save_ratios else None,
            sigma_mu2=jnp.asarray(sigma_mu2),
            log_s=_asarray_or_none(log_s),
            theta=_asarray_or_none(theta),
            rho=_asarray_or_none(rho),
            a=_asarray_or_none(a),
            b=_asarray_or_none(b),
        ),
    )


def _all_none_or_not_none(*args):
    is_none = [x is None for x in args]
    return all(is_none) or not any(is_none)


def _asarray_or_none(x):
    if x is None:
        return None
    return jnp.asarray(x)


def _choose_suffstat_batch_size(
    resid_batch_size, count_batch_size, y, forest_size
) -> tuple[int | None, ...]:
    @cache
    def get_platform():
        try:
            device = y.devices().pop()
        except jax.errors.ConcretizationTypeError:
            device = jax.devices()[0]
        platform = device.platform
        if platform not in ('cpu', 'gpu'):
            msg = f'Unknown platform: {platform}'
            raise KeyError(msg)
        return platform

    if resid_batch_size == 'auto':
        platform = get_platform()
        n = max(1, y.size)
        if platform == 'cpu':
            resid_batch_size = 2 ** round(math.log2(n / 6))  # n/6
        elif platform == 'gpu':
            resid_batch_size = 2 ** round((1 + math.log2(n)) / 3)  # n^1/3
        resid_batch_size = max(1, resid_batch_size)

    if count_batch_size == 'auto':
        platform = get_platform()
        if platform == 'cpu':
            count_batch_size = None
        elif platform == 'gpu':
            n = max(1, y.size)
            count_batch_size = 2 ** round(math.log2(n) / 2 - 2)  # n^1/2
            # /4 is good on V100, /2 on L4/T4, still haven't tried A100
            max_memory = 2**29
            itemsize = 4
            min_batch_size = math.ceil(forest_size * itemsize * n / max_memory)
            count_batch_size = max(count_batch_size, min_batch_size)
            count_batch_size = max(1, count_batch_size)

    return resid_batch_size, count_batch_size


@jax.jit
def step(key: Key[Array, ''], bart: State) -> State:
    """
    Do one MCMC step.

    Parameters
    ----------
    key
        A jax random key.
    bart
        A BART mcmc state, as created by `init`.

    Returns
    -------
    The new BART mcmc state.
    """
    keys = split(key)

    if bart.y.dtype == bool:  # binary regression
        bart = replace(bart, sigma2=jnp.float32(1))
        bart = step_trees(keys.pop(), bart)
        bart = replace(bart, sigma2=None)
        return step_z(keys.pop(), bart)

    else:  # continuous regression
        bart = step_trees(keys.pop(), bart)
        return step_sigma(keys.pop(), bart)


def step_trees(key: Key[Array, ''], bart: State) -> State:
    """
    Forest sampling step of BART MCMC.

    Parameters
    ----------
    key
        A jax random key.
    bart
        A BART mcmc state, as created by `init`.

    Returns
    -------
    The new BART mcmc state.

    Notes
    -----
    This function zeroes the proposal counters.
    """
    keys = split(key)
    moves = propose_moves(keys.pop(), bart.forest)
    return accept_moves_and_sample_leaves(keys.pop(), bart, moves)


class Moves(Module):
    """
    Moves proposed to modify each tree.

    Parameters
    ----------
    allowed
        Whether there is a possible move. If `False`, the other values may not
        make sense. The only case in which a move is marked as allowed but is
        then vetoed is if it does not satisfy `min_points_per_leaf`, which for
        efficiency is implemented post-hoc without changing the rest of the
        MCMC logic.
    grow
        Whether the move is a grow move or a prune move.
    num_growable
        The number of growable leaves in the original tree.
    node
        The index of the leaf to grow or node to prune.
    left
    right
        The indices of the children of 'node'.
    partial_ratio
        A factor of the Metropolis-Hastings ratio of the move. It lacks the
        likelihood ratio, the probability of proposing the prune move, and the
        probability that the children of the modified node are terminal. If the
        move is PRUNE, the ratio is inverted. `None` once
        `log_trans_prior_ratio` has been computed.
    log_trans_prior_ratio
        The logarithm of the product of the transition and prior terms of the
        Metropolis-Hastings ratio for the acceptance of the proposed move.
        `None` if not yet computed. If PRUNE, the log-ratio is negated.
    grow_var
        The decision axes of the new rules.
    grow_split
        The decision boundaries of the new rules.
    var_tree
        The updated decision axes of the trees, valid whatever move.
    affluence_tree
        A partially updated `affluence_tree`, marking non-leaf nodes that would
        become leaves if the move was accepted. This mark initially (out of
        `propose_moves`) takes into account if there would be available decision
        rules to grow the leaf, and whether there are enough datapoints in the
        node is marked in `accept_moves_parallel_stage`.
    logu
        The logarithm of a uniform (0, 1] random variable to be used to
        accept the move. It's in (-oo, 0].
    acc
        Whether the move was accepted. `None` if not yet computed.
    to_prune
        Whether the final operation to apply the move is pruning. This indicates
        an accepted prune move or a rejected grow move. `None` if not yet
        computed.
    """

    allowed: Bool[Array, ' num_trees']
    grow: Bool[Array, ' num_trees']
    num_growable: UInt[Array, ' num_trees']
    node: UInt[Array, ' num_trees']
    left: UInt[Array, ' num_trees']
    right: UInt[Array, ' num_trees']
    partial_ratio: Float32[Array, ' num_trees'] | None
    log_trans_prior_ratio: None | Float32[Array, ' num_trees']
    grow_var: UInt[Array, ' num_trees']
    grow_split: UInt[Array, ' num_trees']
    var_tree: UInt[Array, 'num_trees 2**(d-1)']
    affluence_tree: Bool[Array, 'num_trees 2**(d-1)']
    logu: Float32[Array, ' num_trees']
    acc: None | Bool[Array, ' num_trees']
    to_prune: None | Bool[Array, ' num_trees']


def propose_moves(key: Key[Array, ''], forest: Forest) -> Moves:
    """
    Propose moves for all the trees.

    There are two types of moves: GROW (convert a leaf to a decision node and
    add two leaves beneath it) and PRUNE (convert the parent of two leaves to a
    leaf, deleting its children).

    Parameters
    ----------
    key
        A jax random key.
    forest
        The `forest` field of a BART MCMC state.

    Returns
    -------
    The proposed move for each tree.
    """
    num_trees, _ = forest.leaf_tree.shape
    keys = split(key, 1 + 2 * num_trees)

    # compute moves
    grow_moves = propose_grow_moves(
        keys.pop(num_trees),
        forest.var_tree,
        forest.split_tree,
        forest.affluence_tree,
        forest.max_split,
        forest.blocked_vars,
        forest.p_nonterminal,
        forest.p_propose_grow,
        forest.log_s,
    )
    prune_moves = propose_prune_moves(
        keys.pop(num_trees),
        forest.split_tree,
        grow_moves.affluence_tree,
        forest.p_nonterminal,
        forest.p_propose_grow,
    )

    u, exp1mlogu = random.uniform(keys.pop(), (2, num_trees))

    # choose between grow or prune
    p_grow = jnp.where(
        grow_moves.allowed & prune_moves.allowed, 0.5, grow_moves.allowed
    )
    grow = u < p_grow  # use < instead of <= because u is in [0, 1)

    # compute children indices
    node = jnp.where(grow, grow_moves.node, prune_moves.node)
    left = node << 1
    right = left + 1

    return Moves(
        allowed=grow_moves.allowed | prune_moves.allowed,
        grow=grow,
        num_growable=grow_moves.num_growable,
        node=node,
        left=left,
        right=right,
        partial_ratio=jnp.where(
            grow, grow_moves.partial_ratio, prune_moves.partial_ratio
        ),
        log_trans_prior_ratio=None,  # will be set in complete_ratio
        grow_var=grow_moves.var,
        grow_split=grow_moves.split,
        # var_tree does not need to be updated if prune
        var_tree=grow_moves.var_tree,
        # affluence_tree is updated for both moves unconditionally, prune last
        affluence_tree=prune_moves.affluence_tree,
        logu=jnp.log1p(-exp1mlogu),
        acc=None,  # will be set in accept_moves_sequential_stage
        to_prune=None,  # will be set in accept_moves_sequential_stage
    )


class GrowMoves(Module):
    """
    Represent a proposed grow move for each tree.

    Parameters
    ----------
    allowed
        Whether the move is allowed for proposal.
    num_growable
        The number of leaves that can be proposed for grow.
    node
        The index of the leaf to grow. ``2 ** d`` if there are no growable
        leaves.
    var
    split
        The decision axis and boundary of the new rule.
    partial_ratio
        A factor of the Metropolis-Hastings ratio of the move. It lacks
        the likelihood ratio and the probability of proposing the prune
        move.
    var_tree
        The updated decision axes of the tree.
    affluence_tree
        A partially updated `affluence_tree` that marks each new leaf that
        would be produced as `True` if it would have available decision rules.
    """

    allowed: Bool[Array, ' num_trees']
    num_growable: UInt[Array, ' num_trees']
    node: UInt[Array, ' num_trees']
    var: UInt[Array, ' num_trees']
    split: UInt[Array, ' num_trees']
    partial_ratio: Float32[Array, ' num_trees']
    var_tree: UInt[Array, 'num_trees 2**(d-1)']
    affluence_tree: Bool[Array, 'num_trees 2**(d-1)']


@partial(vmap_nodoc, in_axes=(0, 0, 0, 0, None, None, None, None, None))
def propose_grow_moves(
    key: Key[Array, ' num_trees'],
    var_tree: UInt[Array, 'num_trees 2**(d-1)'],
    split_tree: UInt[Array, 'num_trees 2**(d-1)'],
    affluence_tree: Bool[Array, 'num_trees 2**(d-1)'],
    max_split: UInt[Array, ' p'],
    blocked_vars: Int32[Array, ' k'] | None,
    p_nonterminal: Float32[Array, ' 2**d'],
    p_propose_grow: Float32[Array, ' 2**(d-1)'],
    log_s: Float32[Array, ' p'] | None,
) -> GrowMoves:
    """
    Propose a GROW move for each tree.

    A GROW move picks a leaf node and converts it to a non-terminal node with
    two leaf children.

    Parameters
    ----------
    key
        A jax random key.
    var_tree
        The splitting axes of the tree.
    split_tree
        The splitting points of the tree.
    affluence_tree
        Whether each leaf has enough points to be grown.
    max_split
        The maximum split index for each variable.
    blocked_vars
        The indices of the variables that have no available cutpoints.
    p_nonterminal
        The a priori probability of a node to be nonterminal conditional on the
        ancestors, including at the maximum depth where it should be zero.
    p_propose_grow
        The unnormalized probability of choosing a leaf to grow.
    log_s
        Unnormalized log-probability used to choose a variable to split on
        amongst the available ones.

    Returns
    -------
    An object representing the proposed move.

    Notes
    -----
    The move is not proposed if each leaf is already at maximum depth, or has
    less datapoints than the requested threshold `min_points_per_decision_node`,
    or it does not have any available decision rules given its ancestors. This
    is marked by setting `allowed` to `False` and `num_growable` to 0.
    """
    keys = split(key, 3)

    leaf_to_grow, num_growable, prob_choose, num_prunable = choose_leaf(
        keys.pop(), split_tree, affluence_tree, p_propose_grow
    )

    # sample a decision rule
    var, num_available_var = choose_variable(
        keys.pop(), var_tree, split_tree, max_split, leaf_to_grow, blocked_vars, log_s
    )
    split_idx, l, r = choose_split(
        keys.pop(), var, var_tree, split_tree, max_split, leaf_to_grow
    )

    # determine if the new leaves would have available decision rules; if the
    # move is blocked, these values may not make sense
    left_growable = right_growable = num_available_var > 1
    left_growable |= l < split_idx
    right_growable |= split_idx + 1 < r
    left = leaf_to_grow << 1
    right = left + 1
    affluence_tree = affluence_tree.at[left].set(left_growable)
    affluence_tree = affluence_tree.at[right].set(right_growable)

    ratio = compute_partial_ratio(
        prob_choose, num_prunable, p_nonterminal, leaf_to_grow
    )

    return GrowMoves(
        allowed=num_growable > 0,
        num_growable=num_growable,
        node=leaf_to_grow,
        var=var,
        split=split_idx,
        partial_ratio=ratio,
        var_tree=var_tree.at[leaf_to_grow].set(var.astype(var_tree.dtype)),
        affluence_tree=affluence_tree,
    )


def choose_leaf(
    key: Key[Array, ''],
    split_tree: UInt[Array, ' 2**(d-1)'],
    affluence_tree: Bool[Array, ' 2**(d-1)'],
    p_propose_grow: Float32[Array, ' 2**(d-1)'],
) -> tuple[Int32[Array, ''], Int32[Array, ''], Float32[Array, ''], Int32[Array, '']]:
    """
    Choose a leaf node to grow in a tree.

    Parameters
    ----------
    key
        A jax random key.
    split_tree
        The splitting points of the tree.
    affluence_tree
        Whether a leaf has enough points that it could be split into two leaves
        satisfying the `min_points_per_leaf` requirement.
    p_propose_grow
        The unnormalized probability of choosing a leaf to grow.

    Returns
    -------
    leaf_to_grow : Int32[Array, '']
        The index of the leaf to grow. If ``num_growable == 0``, return
        ``2 ** d``.
    num_growable : Int32[Array, '']
        The number of leaf nodes that can be grown, i.e., are nonterminal
        and have at least twice `min_points_per_leaf`.
    prob_choose : Float32[Array, '']
        The (normalized) probability that this function had to choose that
        specific leaf, given the arguments.
    num_prunable : Int32[Array, '']
        The number of leaf parents that could be pruned, after converting the
        selected leaf to a non-terminal node.
    """
    is_growable = growable_leaves(split_tree, affluence_tree)
    num_growable = jnp.count_nonzero(is_growable)
    distr = jnp.where(is_growable, p_propose_grow, 0)
    leaf_to_grow, distr_norm = categorical(key, distr)
    leaf_to_grow = jnp.where(num_growable, leaf_to_grow, 2 * split_tree.size)
    prob_choose = distr[leaf_to_grow] / jnp.where(distr_norm, distr_norm, 1)
    is_parent = grove.is_leaves_parent(split_tree.at[leaf_to_grow].set(1))
    num_prunable = jnp.count_nonzero(is_parent)
    return leaf_to_grow, num_growable, prob_choose, num_prunable


def growable_leaves(
    split_tree: UInt[Array, ' 2**(d-1)'], affluence_tree: Bool[Array, ' 2**(d-1)']
) -> Bool[Array, ' 2**(d-1)']:
    """
    Return a mask indicating the leaf nodes that can be proposed for growth.

    The condition is that a leaf is not at the bottom level, has available
    decision rules given its ancestors, and has at least
    `min_points_per_decision_node` points.

    Parameters
    ----------
    split_tree
        The splitting points of the tree.
    affluence_tree
        Marks leaves that can be grown.

    Returns
    -------
    The mask indicating the leaf nodes that can be proposed to grow.

    Notes
    -----
    This function needs `split_tree` and not just `affluence_tree` because
    `affluence_tree` can be "dirty", i.e., mark unused nodes as `True`.
    """
    return grove.is_actual_leaf(split_tree) & affluence_tree


def categorical(
    key: Key[Array, ''], distr: Float32[Array, ' n']
) -> tuple[Int32[Array, ''], Float32[Array, '']]:
    """
    Return a random integer from an arbitrary distribution.

    Parameters
    ----------
    key
        A jax random key.
    distr
        An unnormalized probability distribution.

    Returns
    -------
    u : Int32[Array, '']
        A random integer in the range ``[0, n)``. If all probabilities are zero,
        return ``n``.
    norm : Float32[Array, '']
        The sum of `distr`.

    Notes
    -----
    This function uses a cumsum instead of the Gumbel trick, so it's ok only
    for small ranges with probabilities well greater than 0.
    """
    ecdf = jnp.cumsum(distr)
    u = random.uniform(key, (), ecdf.dtype, 0, ecdf[-1])
    return jnp.searchsorted(ecdf, u, 'right'), ecdf[-1]


def choose_variable(
    key: Key[Array, ''],
    var_tree: UInt[Array, ' 2**(d-1)'],
    split_tree: UInt[Array, ' 2**(d-1)'],
    max_split: UInt[Array, ' p'],
    leaf_index: Int32[Array, ''],
    blocked_vars: Int32[Array, ' k'] | None,
    log_s: Float32[Array, ' p'] | None,
) -> tuple[Int32[Array, ''], Int32[Array, '']]:
    """
    Choose a variable to split on for a new non-terminal node.

    Parameters
    ----------
    key
        A jax random key.
    var_tree
        The variable indices of the tree.
    split_tree
        The splitting points of the tree.
    max_split
        The maximum split index for each variable.
    leaf_index
        The index of the leaf to grow.
    blocked_vars
        The indices of the variables that have no available cutpoints. If
        `None`, all variables are assumed unblocked.
    log_s
        The logarithm of the prior probability for choosing a variable. If
        `None`, use a uniform distribution.

    Returns
    -------
    var : Int32[Array, '']
        The index of the variable to split on.
    num_available_var : Int32[Array, '']
        The number of variables with available decision rules `var` was chosen
        from.
    """
    var_to_ignore = fully_used_variables(var_tree, split_tree, max_split, leaf_index)
    if blocked_vars is not None:
        var_to_ignore = jnp.concatenate([var_to_ignore, blocked_vars])

    if log_s is None:
        return randint_exclude(key, max_split.size, var_to_ignore)
    else:
        return categorical_exclude(key, log_s, var_to_ignore)


def fully_used_variables(
    var_tree: UInt[Array, ' 2**(d-1)'],
    split_tree: UInt[Array, ' 2**(d-1)'],
    max_split: UInt[Array, ' p'],
    leaf_index: Int32[Array, ''],
) -> UInt[Array, ' d-2']:
    """
    Find variables in the ancestors of a node that have an empty split range.

    Parameters
    ----------
    var_tree
        The variable indices of the tree.
    split_tree
        The splitting points of the tree.
    max_split
        The maximum split index for each variable.
    leaf_index
        The index of the node, assumed to be valid for `var_tree`.

    Returns
    -------
    The indices of the variables that have an empty split range.

    Notes
    -----
    The number of unused variables is not known in advance. Unused values in the
    array are filled with `p`. The fill values are not guaranteed to be placed
    in any particular order, and variables may appear more than once.
    """
    var_to_ignore = ancestor_variables(var_tree, max_split, leaf_index)
    split_range_vec = jax.vmap(split_range, in_axes=(None, None, None, None, 0))
    l, r = split_range_vec(var_tree, split_tree, max_split, leaf_index, var_to_ignore)
    num_split = r - l
    return jnp.where(num_split == 0, var_to_ignore, max_split.size)
    # the type of var_to_ignore is already sufficient to hold max_split.size,
    # see ancestor_variables()


def ancestor_variables(
    var_tree: UInt[Array, ' 2**(d-1)'],
    max_split: UInt[Array, ' p'],
    node_index: Int32[Array, ''],
) -> UInt[Array, ' d-2']:
    """
    Return the list of variables in the ancestors of a node.

    Parameters
    ----------
    var_tree
        The variable indices of the tree.
    max_split
        The maximum split index for each variable. Used only to get `p`.
    node_index
        The index of the node, assumed to be valid for `var_tree`.

    Returns
    -------
    The variable indices of the ancestors of the node.

    Notes
    -----
    The ancestors are the nodes going from the root to the parent of the node.
    The number of ancestors is not known at tracing time; unused spots in the
    output array are filled with `p`.
    """
    max_num_ancestors = grove.tree_depth(var_tree) - 1
    ancestor_vars = jnp.zeros(max_num_ancestors, minimal_unsigned_dtype(max_split.size))
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


def split_range(
    var_tree: UInt[Array, ' 2**(d-1)'],
    split_tree: UInt[Array, ' 2**(d-1)'],
    max_split: UInt[Array, ' p'],
    node_index: Int32[Array, ''],
    ref_var: Int32[Array, ''],
) -> tuple[Int32[Array, ''], Int32[Array, '']]:
    """
    Return the range of allowed splits for a variable at a given node.

    Parameters
    ----------
    var_tree
        The variable indices of the tree.
    split_tree
        The splitting points of the tree.
    max_split
        The maximum split index for each variable.
    node_index
        The index of the node, assumed to be valid for `var_tree`.
    ref_var
        The variable for which to measure the split range.

    Returns
    -------
    The range of allowed splits as [l, r). If `ref_var` is out of bounds, l=r=1.
    """
    max_num_ancestors = grove.tree_depth(var_tree) - 1
    initial_r = 1 + max_split.at[ref_var].get(mode='fill', fill_value=0).astype(
        jnp.int32
    )
    carry = jnp.int32(0), initial_r, node_index

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


def randint_exclude(
    key: Key[Array, ''], sup: int | Integer[Array, ''], exclude: Integer[Array, ' n']
) -> tuple[Int32[Array, ''], Int32[Array, '']]:
    """
    Return a random integer in a range, excluding some values.

    Parameters
    ----------
    key
        A jax random key.
    sup
        The exclusive upper bound of the range.
    exclude
        The values to exclude from the range. Values greater than or equal to
        `sup` are ignored. Values can appear more than once.

    Returns
    -------
    u : Int32[Array, '']
        A random integer `u` in the range ``[0, sup)`` such that ``u not in
        exclude``.
    num_allowed : Int32[Array, '']
        The number of integers in the range that were not excluded.

    Notes
    -----
    If all values in the range are excluded, return `sup`.
    """
    exclude, num_allowed = _process_exclude(sup, exclude)
    u = random.randint(key, (), 0, num_allowed)

    def loop(u, i_excluded):
        return jnp.where(i_excluded <= u, u + 1, u), None

    u, _ = lax.scan(loop, u, exclude)
    return u, num_allowed


def _process_exclude(sup, exclude):
    exclude = jnp.unique(exclude, size=exclude.size, fill_value=sup)
    num_allowed = sup - jnp.count_nonzero(exclude < sup)
    return exclude, num_allowed


def categorical_exclude(
    key: Key[Array, ''], logits: Float32[Array, ' k'], exclude: Integer[Array, ' n']
) -> tuple[Int32[Array, ''], Int32[Array, '']]:
    """
    Draw from a categorical distribution, excluding a set of values.

    Parameters
    ----------
    key
        A jax random key.
    logits
        The unnormalized log-probabilities of each category.
    exclude
        The values to exclude from the range [0, k). Values greater than or
        equal to `logits.size` are ignored. Values can appear more than once.

    Returns
    -------
    u : Int32[Array, '']
        A random integer in the range ``[0, k)`` such that ``u not in exclude``.
    num_allowed : Int32[Array, '']
        The number of integers in the range that were not excluded.

    Notes
    -----
    If all values in the range are excluded, the result is unspecified.
    """
    exclude, num_allowed = _process_exclude(logits.size, exclude)
    kinda_neg_inf = jnp.finfo(logits.dtype).min
    logits = logits.at[exclude].set(kinda_neg_inf)
    u = random.categorical(key, logits)
    return u, num_allowed


def choose_split(
    key: Key[Array, ''],
    var: Int32[Array, ''],
    var_tree: UInt[Array, ' 2**(d-1)'],
    split_tree: UInt[Array, ' 2**(d-1)'],
    max_split: UInt[Array, ' p'],
    leaf_index: Int32[Array, ''],
) -> tuple[Int32[Array, ''], Int32[Array, ''], Int32[Array, '']]:
    """
    Choose a split point for a new non-terminal node.

    Parameters
    ----------
    key
        A jax random key.
    var
        The variable to split on.
    var_tree
        The splitting axes of the tree. Does not need to already contain `var`
        at `leaf_index`.
    split_tree
        The splitting points of the tree.
    max_split
        The maximum split index for each variable.
    leaf_index
        The index of the leaf to grow.

    Returns
    -------
    split : Int32[Array, '']
        The cutpoint.
    l : Int32[Array, '']
    r : Int32[Array, '']
        The integer range `split` was drawn from is [l, r).

    Notes
    -----
    If `var` is out of bounds, or if the available split range on that variable
    is empty, return 0.
    """
    l, r = split_range(var_tree, split_tree, max_split, leaf_index, var)
    return jnp.where(l < r, random.randint(key, (), l, r), 0), l, r


def compute_partial_ratio(
    prob_choose: Float32[Array, ''],
    num_prunable: Int32[Array, ''],
    p_nonterminal: Float32[Array, ' 2**d'],
    leaf_to_grow: Int32[Array, ''],
) -> Float32[Array, '']:
    """
    Compute the product of the transition and prior ratios of a grow move.

    Parameters
    ----------
    prob_choose
        The probability that the leaf had to be chosen amongst the growable
        leaves.
    num_prunable
        The number of leaf parents that could be pruned, after converting the
        leaf to be grown to a non-terminal node.
    p_nonterminal
        The a priori probability of each node being nonterminal conditional on
        its ancestors.
    leaf_to_grow
        The index of the leaf to grow.

    Returns
    -------
    The partial transition ratio times the prior ratio.

    Notes
    -----
    The transition ratio is P(new tree => old tree) / P(old tree => new tree).
    The "partial" transition ratio returned is missing the factor P(propose
    prune) in the numerator. The prior ratio is P(new tree) / P(old tree). The
    "partial" prior ratio is missing the factor P(children are leaves).
    """
    # the two ratios also contain factors num_available_split *
    # num_available_var * s[var], but they cancel out

    # p_prune and 1 - p_nonterminal[child] * I(is the child growable) can't be
    # computed here because they need the count trees, which are computed in the
    # acceptance phase

    prune_allowed = leaf_to_grow != 1
    # prune allowed  <--->  the initial tree is not a root
    # leaf to grow is root  -->  the tree can only be a root
    # tree is a root  -->  the only leaf I can grow is root
    p_grow = jnp.where(prune_allowed, 0.5, 1)
    inv_trans_ratio = p_grow * prob_choose * num_prunable

    # .at.get because if leaf_to_grow is out of bounds (move not allowed), this
    # would produce a 0 and then an inf when `complete_ratio` takes the log
    pnt = p_nonterminal.at[leaf_to_grow].get(mode='fill', fill_value=0.5)
    tree_ratio = pnt / (1 - pnt)

    return tree_ratio / jnp.where(inv_trans_ratio, inv_trans_ratio, 1)


class PruneMoves(Module):
    """
    Represent a proposed prune move for each tree.

    Parameters
    ----------
    allowed
        Whether the move is possible.
    node
        The index of the node to prune. ``2 ** d`` if no node can be pruned.
    partial_ratio
        A factor of the Metropolis-Hastings ratio of the move. It lacks the
        likelihood ratio, the probability of proposing the prune move, and the
        prior probability that the children of the node to prune are leaves.
        This ratio is inverted, and is meant to be inverted back in
        `accept_move_and_sample_leaves`.
    """

    allowed: Bool[Array, ' num_trees']
    node: UInt[Array, ' num_trees']
    partial_ratio: Float32[Array, ' num_trees']
    affluence_tree: Bool[Array, 'num_trees 2**(d-1)']


@partial(vmap_nodoc, in_axes=(0, 0, 0, None, None))
def propose_prune_moves(
    key: Key[Array, ''],
    split_tree: UInt[Array, ' 2**(d-1)'],
    affluence_tree: Bool[Array, ' 2**(d-1)'],
    p_nonterminal: Float32[Array, ' 2**d'],
    p_propose_grow: Float32[Array, ' 2**(d-1)'],
) -> PruneMoves:
    """
    Tree structure prune move proposal of BART MCMC.

    Parameters
    ----------
    key
        A jax random key.
    split_tree
        The splitting points of the tree.
    affluence_tree
        Whether each leaf can be grown.
    p_nonterminal
        The a priori probability of a node to be nonterminal conditional on
        the ancestors, including at the maximum depth where it should be zero.
    p_propose_grow
        The unnormalized probability of choosing a leaf to grow.

    Returns
    -------
    An object representing the proposed moves.
    """
    node_to_prune, num_prunable, prob_choose, affluence_tree = choose_leaf_parent(
        key, split_tree, affluence_tree, p_propose_grow
    )

    ratio = compute_partial_ratio(
        prob_choose, num_prunable, p_nonterminal, node_to_prune
    )

    return PruneMoves(
        allowed=split_tree[1].astype(bool),  # allowed iff the tree is not a root
        node=node_to_prune,
        partial_ratio=ratio,
        affluence_tree=affluence_tree,
    )


def choose_leaf_parent(
    key: Key[Array, ''],
    split_tree: UInt[Array, ' 2**(d-1)'],
    affluence_tree: Bool[Array, ' 2**(d-1)'],
    p_propose_grow: Float32[Array, ' 2**(d-1)'],
) -> tuple[
    Int32[Array, ''],
    Int32[Array, ''],
    Float32[Array, ''],
    Bool[Array, 'num_trees 2**(d-1)'],
]:
    """
    Pick a non-terminal node with leaf children to prune in a tree.

    Parameters
    ----------
    key
        A jax random key.
    split_tree
        The splitting points of the tree.
    affluence_tree
        Whether a leaf has enough points to be grown.
    p_propose_grow
        The unnormalized probability of choosing a leaf to grow.

    Returns
    -------
    node_to_prune : Int32[Array, '']
        The index of the node to prune. If ``num_prunable == 0``, return
        ``2 ** d``.
    num_prunable : Int32[Array, '']
        The number of leaf parents that could be pruned.
    prob_choose : Float32[Array, '']
        The (normalized) probability that `choose_leaf` would chose
        `node_to_prune` as leaf to grow, if passed the tree where
        `node_to_prune` had been pruned.
    affluence_tree : Bool[Array, 'num_trees 2**(d-1)']
        A partially updated `affluence_tree`, marking the node to prune as
        growable.
    """
    # sample a node to prune
    is_prunable = grove.is_leaves_parent(split_tree)
    num_prunable = jnp.count_nonzero(is_prunable)
    node_to_prune = randint_masked(key, is_prunable)
    node_to_prune = jnp.where(num_prunable, node_to_prune, 2 * split_tree.size)

    # compute stuff for reverse move
    split_tree = split_tree.at[node_to_prune].set(0)
    affluence_tree = affluence_tree.at[node_to_prune].set(True)
    is_growable_leaf = growable_leaves(split_tree, affluence_tree)
    distr_norm = jnp.sum(p_propose_grow, where=is_growable_leaf)
    prob_choose = p_propose_grow.at[node_to_prune].get(mode='fill', fill_value=0)
    prob_choose = prob_choose / jnp.where(distr_norm, distr_norm, 1)

    return node_to_prune, num_prunable, prob_choose, affluence_tree


def randint_masked(key: Key[Array, ''], mask: Bool[Array, ' n']) -> Int32[Array, '']:
    """
    Return a random integer in a range, including only some values.

    Parameters
    ----------
    key
        A jax random key.
    mask
        The mask indicating the allowed values.

    Returns
    -------
    A random integer in the range ``[0, n)`` such that ``mask[u] == True``.

    Notes
    -----
    If all values in the mask are `False`, return `n`.
    """
    ecdf = jnp.cumsum(mask)
    u = random.randint(key, (), 0, ecdf[-1])
    return jnp.searchsorted(ecdf, u, 'right')


def accept_moves_and_sample_leaves(
    key: Key[Array, ''], bart: State, moves: Moves
) -> State:
    """
    Accept or reject the proposed moves and sample the new leaf values.

    Parameters
    ----------
    key
        A jax random key.
    bart
        A valid BART mcmc state.
    moves
        The proposed moves, see `propose_moves`.

    Returns
    -------
    A new (valid) BART mcmc state.
    """
    pso = accept_moves_parallel_stage(key, bart, moves)
    bart, moves = accept_moves_sequential_stage(pso)
    return accept_moves_final_stage(bart, moves)


class Counts(Module):
    """
    Number of datapoints in the nodes involved in proposed moves for each tree.

    Parameters
    ----------
    left
        Number of datapoints in the left child.
    right
        Number of datapoints in the right child.
    total
        Number of datapoints in the parent (``= left + right``).
    """

    left: UInt[Array, ' num_trees']
    right: UInt[Array, ' num_trees']
    total: UInt[Array, ' num_trees']


class Precs(Module):
    """
    Likelihood precision scale in the nodes involved in proposed moves for each tree.

    The "likelihood precision scale" of a tree node is the sum of the inverse
    squared error scales of the datapoints selected by the node.

    Parameters
    ----------
    left
        Likelihood precision scale in the left child.
    right
        Likelihood precision scale in the right child.
    total
        Likelihood precision scale in the parent (``= left + right``).
    """

    left: Float32[Array, ' num_trees']
    right: Float32[Array, ' num_trees']
    total: Float32[Array, ' num_trees']


class PreLkV(Module):
    """
    Non-sequential terms of the likelihood ratio for each tree.

    These terms can be computed in parallel across trees.

    Parameters
    ----------
    sigma2_left
        The noise variance in the left child of the leaves grown or pruned by
        the moves.
    sigma2_right
        The noise variance in the right child of the leaves grown or pruned by
        the moves.
    sigma2_total
        The noise variance in the total of the leaves grown or pruned by the
        moves.
    sqrt_term
        The **logarithm** of the square root term of the likelihood ratio.
    """

    sigma2_left: Float32[Array, ' num_trees']
    sigma2_right: Float32[Array, ' num_trees']
    sigma2_total: Float32[Array, ' num_trees']
    sqrt_term: Float32[Array, ' num_trees']


class PreLk(Module):
    """
    Non-sequential terms of the likelihood ratio shared by all trees.

    Parameters
    ----------
    exp_factor
        The factor to multiply the likelihood ratio by, shared by all trees.
    """

    exp_factor: Float32[Array, '']


class PreLf(Module):
    """
    Pre-computed terms used to sample leaves from their posterior.

    These terms can be computed in parallel across trees.

    Parameters
    ----------
    mean_factor
        The factor to be multiplied by the sum of the scaled residuals to
        obtain the posterior mean.
    centered_leaves
        The mean-zero normal values to be added to the posterior mean to
        obtain the posterior leaf samples.
    """

    mean_factor: Float32[Array, 'num_trees 2**d']
    centered_leaves: Float32[Array, 'num_trees 2**d']


class ParallelStageOut(Module):
    """
    The output of `accept_moves_parallel_stage`.

    Parameters
    ----------
    bart
        A partially updated BART mcmc state.
    moves
        The proposed moves, with `partial_ratio` set to `None` and
        `log_trans_prior_ratio` set to its final value.
    prec_trees
        The likelihood precision scale in each potential or actual leaf node. If
        there is no precision scale, this is the number of points in each leaf.
    move_counts
        The counts of the number of points in the the nodes modified by the
        moves. If `bart.min_points_per_leaf` is not set and
        `bart.prec_scale` is set, they are not computed.
    move_precs
        The likelihood precision scale in each node modified by the moves. If
        `bart.prec_scale` is not set, this is set to `move_counts`.
    prelkv
    prelk
    prelf
        Objects with pre-computed terms of the likelihood ratios and leaf
        samples.
    """

    bart: State
    moves: Moves
    prec_trees: Float32[Array, 'num_trees 2**d'] | Int32[Array, 'num_trees 2**d']
    move_precs: Precs | Counts
    prelkv: PreLkV
    prelk: PreLk
    prelf: PreLf


def accept_moves_parallel_stage(
    key: Key[Array, ''], bart: State, moves: Moves
) -> ParallelStageOut:
    """
    Pre-compute quantities used to accept moves, in parallel across trees.

    Parameters
    ----------
    key : jax.dtypes.prng_key array
        A jax random key.
    bart : dict
        A BART mcmc state.
    moves : dict
        The proposed moves, see `propose_moves`.

    Returns
    -------
    An object with all that could be done in parallel.
    """
    # where the move is grow, modify the state like the move was accepted
    bart = replace(
        bart,
        forest=replace(
            bart.forest,
            var_tree=moves.var_tree,
            leaf_indices=apply_grow_to_indices(moves, bart.forest.leaf_indices, bart.X),
            leaf_tree=adapt_leaf_trees_to_grow_indices(bart.forest.leaf_tree, moves),
        ),
    )

    # count number of datapoints per leaf
    if (
        bart.forest.min_points_per_decision_node is not None
        or bart.forest.min_points_per_leaf is not None
        or bart.prec_scale is None
    ):
        count_trees, move_counts = compute_count_trees(
            bart.forest.leaf_indices, moves, bart.forest.count_batch_size
        )

    # mark which leaves & potential leaves have enough points to be grown
    if bart.forest.min_points_per_decision_node is not None:
        count_half_trees = count_trees[:, : bart.forest.var_tree.shape[1]]
        moves = replace(
            moves,
            affluence_tree=moves.affluence_tree
            & (count_half_trees >= bart.forest.min_points_per_decision_node),
        )

    # copy updated affluence_tree to state
    bart = tree_at(lambda bart: bart.forest.affluence_tree, bart, moves.affluence_tree)

    # veto grove move if new leaves don't have enough datapoints
    if bart.forest.min_points_per_leaf is not None:
        moves = replace(
            moves,
            allowed=moves.allowed
            & (move_counts.left >= bart.forest.min_points_per_leaf)
            & (move_counts.right >= bart.forest.min_points_per_leaf),
        )

    # count number of datapoints per leaf, weighted by error precision scale
    if bart.prec_scale is None:
        prec_trees = count_trees
        move_precs = move_counts
    else:
        prec_trees, move_precs = compute_prec_trees(
            bart.prec_scale,
            bart.forest.leaf_indices,
            moves,
            bart.forest.count_batch_size,
        )
    assert move_precs is not None

    # compute some missing information about moves
    moves = complete_ratio(moves, bart.forest.p_nonterminal)
    save_ratios = bart.forest.log_likelihood is not None
    bart = replace(
        bart,
        forest=replace(
            bart.forest,
            grow_prop_count=jnp.sum(moves.grow),
            prune_prop_count=jnp.sum(moves.allowed & ~moves.grow),
            log_trans_prior=moves.log_trans_prior_ratio if save_ratios else None,
        ),
    )

    # pre-compute some likelihood ratio & posterior terms
    assert bart.sigma2 is not None  # `step` shall temporarily set it to 1
    prelkv, prelk = precompute_likelihood_terms(
        bart.sigma2, bart.forest.sigma_mu2, move_precs
    )
    prelf = precompute_leaf_terms(key, prec_trees, bart.sigma2, bart.forest.sigma_mu2)

    return ParallelStageOut(
        bart=bart,
        moves=moves,
        prec_trees=prec_trees,
        move_precs=move_precs,
        prelkv=prelkv,
        prelk=prelk,
        prelf=prelf,
    )


@partial(vmap_nodoc, in_axes=(0, 0, None))
def apply_grow_to_indices(
    moves: Moves, leaf_indices: UInt[Array, 'num_trees n'], X: UInt[Array, 'p n']
) -> UInt[Array, 'num_trees n']:
    """
    Update the leaf indices to apply a grow move.

    Parameters
    ----------
    moves
        The proposed moves, see `propose_moves`.
    leaf_indices
        The index of the leaf each datapoint falls into.
    X
        The predictors matrix.

    Returns
    -------
    The updated leaf indices.
    """
    left_child = moves.node.astype(leaf_indices.dtype) << 1
    go_right = X[moves.grow_var, :] >= moves.grow_split
    tree_size = jnp.array(2 * moves.var_tree.size)
    node_to_update = jnp.where(moves.grow, moves.node, tree_size)
    return jnp.where(
        leaf_indices == node_to_update, left_child + go_right, leaf_indices
    )


def compute_count_trees(
    leaf_indices: UInt[Array, 'num_trees n'], moves: Moves, batch_size: int | None
) -> tuple[Int32[Array, 'num_trees 2**d'], Counts]:
    """
    Count the number of datapoints in each leaf.

    Parameters
    ----------
    leaf_indices
        The index of the leaf each datapoint falls into, with the deeper version
        of the tree (post-GROW, pre-PRUNE).
    moves
        The proposed moves, see `propose_moves`.
    batch_size
        The data batch size to use for the summation.

    Returns
    -------
    count_trees : Int32[Array, 'num_trees 2**d']
        The number of points in each potential or actual leaf node.
    counts : Counts
        The counts of the number of points in the leaves grown or pruned by the
        moves.
    """
    num_trees, tree_size = moves.var_tree.shape
    tree_size *= 2
    tree_indices = jnp.arange(num_trees)

    count_trees = count_datapoints_per_leaf(leaf_indices, tree_size, batch_size)

    # count datapoints in nodes modified by move
    left = count_trees[tree_indices, moves.left]
    right = count_trees[tree_indices, moves.right]
    counts = Counts(left=left, right=right, total=left + right)

    # write count into non-leaf node
    count_trees = count_trees.at[tree_indices, moves.node].set(counts.total)

    return count_trees, counts


def count_datapoints_per_leaf(
    leaf_indices: UInt[Array, 'num_trees n'], tree_size: int, batch_size: int | None
) -> Int32[Array, 'num_trees 2**(d-1)']:
    """
    Count the number of datapoints in each leaf.

    Parameters
    ----------
    leaf_indices
        The index of the leaf each datapoint falls into.
    tree_size
        The size of the leaf tree array (2 ** d).
    batch_size
        The data batch size to use for the summation.

    Returns
    -------
    The number of points in each leaf node.
    """
    if batch_size is None:
        return _count_scan(leaf_indices, tree_size)
    else:
        return _count_vec(leaf_indices, tree_size, batch_size)


def _count_scan(
    leaf_indices: UInt[Array, 'num_trees n'], tree_size: int
) -> Int32[Array, 'num_trees {tree_size}']:
    def loop(_, leaf_indices):
        return None, _aggregate_scatter(1, leaf_indices, tree_size, jnp.uint32)

    _, count_trees = lax.scan(loop, None, leaf_indices)
    return count_trees


def _aggregate_scatter(
    values: Shaped[Array, '*'],
    indices: Integer[Array, '*'],
    size: int,
    dtype: jnp.dtype,
) -> Shaped[Array, ' {size}']:
    return jnp.zeros(size, dtype).at[indices].add(values)


def _count_vec(
    leaf_indices: UInt[Array, 'num_trees n'], tree_size: int, batch_size: int
) -> Int32[Array, 'num_trees 2**(d-1)']:
    return _aggregate_batched_alltrees(
        1, leaf_indices, tree_size, jnp.uint32, batch_size
    )
    # uint16 is super-slow on gpu, don't use it even if n < 2^16


def _aggregate_batched_alltrees(
    values: Shaped[Array, '*'],
    indices: UInt[Array, 'num_trees n'],
    size: int,
    dtype: jnp.dtype,
    batch_size: int,
) -> Shaped[Array, 'num_trees {size}']:
    num_trees, n = indices.shape
    tree_indices = jnp.arange(num_trees)
    nbatches = n // batch_size + bool(n % batch_size)
    batch_indices = jnp.arange(n) % nbatches
    return (
        jnp.zeros((num_trees, size, nbatches), dtype)
        .at[tree_indices[:, None], indices, batch_indices]
        .add(values)
        .sum(axis=2)
    )


def compute_prec_trees(
    prec_scale: Float32[Array, ' n'],
    leaf_indices: UInt[Array, 'num_trees n'],
    moves: Moves,
    batch_size: int | None,
) -> tuple[Float32[Array, 'num_trees 2**d'], Precs]:
    """
    Compute the likelihood precision scale in each leaf.

    Parameters
    ----------
    prec_scale
        The scale of the precision of the error on each datapoint.
    leaf_indices
        The index of the leaf each datapoint falls into, with the deeper version
        of the tree (post-GROW, pre-PRUNE).
    moves
        The proposed moves, see `propose_moves`.
    batch_size
        The data batch size to use for the summation.

    Returns
    -------
    prec_trees : Float32[Array, 'num_trees 2**d']
        The likelihood precision scale in each potential or actual leaf node.
    precs : Precs
        The likelihood precision scale in the nodes involved in the moves.
    """
    num_trees, tree_size = moves.var_tree.shape
    tree_size *= 2
    tree_indices = jnp.arange(num_trees)

    prec_trees = prec_per_leaf(prec_scale, leaf_indices, tree_size, batch_size)

    # prec datapoints in nodes modified by move
    left = prec_trees[tree_indices, moves.left]
    right = prec_trees[tree_indices, moves.right]
    precs = Precs(left=left, right=right, total=left + right)

    # write prec into non-leaf node
    prec_trees = prec_trees.at[tree_indices, moves.node].set(precs.total)

    return prec_trees, precs


def prec_per_leaf(
    prec_scale: Float32[Array, ' n'],
    leaf_indices: UInt[Array, 'num_trees n'],
    tree_size: int,
    batch_size: int | None,
) -> Float32[Array, 'num_trees {tree_size}']:
    """
    Compute the likelihood precision scale in each leaf.

    Parameters
    ----------
    prec_scale
        The scale of the precision of the error on each datapoint.
    leaf_indices
        The index of the leaf each datapoint falls into.
    tree_size
        The size of the leaf tree array (2 ** d).
    batch_size
        The data batch size to use for the summation.

    Returns
    -------
    The likelihood precision scale in each leaf node.
    """
    if batch_size is None:
        return _prec_scan(prec_scale, leaf_indices, tree_size)
    else:
        return _prec_vec(prec_scale, leaf_indices, tree_size, batch_size)


def _prec_scan(
    prec_scale: Float32[Array, ' n'],
    leaf_indices: UInt[Array, 'num_trees n'],
    tree_size: int,
) -> Float32[Array, 'num_trees {tree_size}']:
    def loop(_, leaf_indices):
        return None, _aggregate_scatter(
            prec_scale, leaf_indices, tree_size, jnp.float32
        )

    _, prec_trees = lax.scan(loop, None, leaf_indices)
    return prec_trees


def _prec_vec(
    prec_scale: Float32[Array, ' n'],
    leaf_indices: UInt[Array, 'num_trees n'],
    tree_size: int,
    batch_size: int,
) -> Float32[Array, 'num_trees {tree_size}']:
    return _aggregate_batched_alltrees(
        prec_scale, leaf_indices, tree_size, jnp.float32, batch_size
    )


def complete_ratio(moves: Moves, p_nonterminal: Float32[Array, ' 2**d']) -> Moves:
    """
    Complete non-likelihood MH ratio calculation.

    This function adds the probability of choosing a prune move over the grow
    move in the inverse transition, and the a priori probability that the
    children nodes are leaves.

    Parameters
    ----------
    moves
        The proposed moves. Must have already been updated to keep into account
        the thresholds on the number of datapoints per node, this happens in
        `accept_moves_parallel_stage`.
    p_nonterminal
        The a priori probability of each node being nonterminal conditional on
        its ancestors, including at the maximum depth where it should be zero.

    Returns
    -------
    The updated moves, with `partial_ratio=None` and `log_trans_prior_ratio` set.
    """
    # can the leaves can be grown?
    num_trees, _ = moves.affluence_tree.shape
    tree_indices = jnp.arange(num_trees)
    left_growable = moves.affluence_tree.at[tree_indices, moves.left].get(
        mode='fill', fill_value=False
    )
    right_growable = moves.affluence_tree.at[tree_indices, moves.right].get(
        mode='fill', fill_value=False
    )

    # p_prune if grow
    other_growable_leaves = moves.num_growable >= 2
    grow_again_allowed = other_growable_leaves | left_growable | right_growable
    grow_p_prune = jnp.where(grow_again_allowed, 0.5, 1)

    # p_prune if prune
    prune_p_prune = jnp.where(moves.num_growable, 0.5, 1)

    # select p_prune
    p_prune = jnp.where(moves.grow, grow_p_prune, prune_p_prune)

    # prior probability of both children being terminal
    pt_left = 1 - p_nonterminal[moves.left] * left_growable
    pt_right = 1 - p_nonterminal[moves.right] * right_growable
    pt_children = pt_left * pt_right

    return replace(
        moves,
        log_trans_prior_ratio=jnp.log(moves.partial_ratio * pt_children * p_prune),
        partial_ratio=None,
    )


@vmap_nodoc
def adapt_leaf_trees_to_grow_indices(
    leaf_trees: Float32[Array, 'num_trees 2**d'], moves: Moves
) -> Float32[Array, 'num_trees 2**d']:
    """
    Modify leaves such that post-grow indices work on the original tree.

    The value of the leaf to grow is copied to what would be its children if the
    grow move was accepted.

    Parameters
    ----------
    leaf_trees
        The leaf values.
    moves
        The proposed moves, see `propose_moves`.

    Returns
    -------
    The modified leaf values.
    """
    values_at_node = leaf_trees[moves.node]
    return (
        leaf_trees.at[jnp.where(moves.grow, moves.left, leaf_trees.size)]
        .set(values_at_node)
        .at[jnp.where(moves.grow, moves.right, leaf_trees.size)]
        .set(values_at_node)
    )


def precompute_likelihood_terms(
    sigma2: Float32[Array, ''],
    sigma_mu2: Float32[Array, ''],
    move_precs: Precs | Counts,
) -> tuple[PreLkV, PreLk]:
    """
    Pre-compute terms used in the likelihood ratio of the acceptance step.

    Parameters
    ----------
    sigma2
        The error variance, or the global error variance factor is `prec_scale`
        is set.
    sigma_mu2
        The prior variance of each leaf.
    move_precs
        The likelihood precision scale in the leaves grown or pruned by the
        moves, under keys 'left', 'right', and 'total' (left + right).

    Returns
    -------
    prelkv : PreLkV
        Dictionary with pre-computed terms of the likelihood ratio, one per
        tree.
    prelk : PreLk
        Dictionary with pre-computed terms of the likelihood ratio, shared by
        all trees.
    """
    sigma2_left = sigma2 + move_precs.left * sigma_mu2
    sigma2_right = sigma2 + move_precs.right * sigma_mu2
    sigma2_total = sigma2 + move_precs.total * sigma_mu2
    prelkv = PreLkV(
        sigma2_left=sigma2_left,
        sigma2_right=sigma2_right,
        sigma2_total=sigma2_total,
        sqrt_term=jnp.log(sigma2 * sigma2_total / (sigma2_left * sigma2_right)) / 2,
    )
    return prelkv, PreLk(exp_factor=sigma_mu2 / (2 * sigma2))


def precompute_leaf_terms(
    key: Key[Array, ''],
    prec_trees: Float32[Array, 'num_trees 2**d'],
    sigma2: Float32[Array, ''],
    sigma_mu2: Float32[Array, ''],
) -> PreLf:
    """
    Pre-compute terms used to sample leaves from their posterior.

    Parameters
    ----------
    key
        A jax random key.
    prec_trees
        The likelihood precision scale in each potential or actual leaf node.
    sigma2
        The error variance, or the global error variance factor if `prec_scale`
        is set.
    sigma_mu2
        The prior variance of each leaf.

    Returns
    -------
    Pre-computed terms for leaf sampling.
    """
    prec_lk = prec_trees / sigma2
    prec_prior = lax.reciprocal(sigma_mu2)
    var_post = lax.reciprocal(prec_lk + prec_prior)
    z = random.normal(key, prec_trees.shape, sigma2.dtype)
    return PreLf(
        mean_factor=var_post / sigma2,
        # | mean = mean_lk * prec_lk * var_post
        # | resid_tree = mean_lk * prec_tree  -->
        # |    -->  mean_lk = resid_tree / prec_tree  (kind of)
        # | mean_factor =
        # |    = mean / resid_tree =
        # |    = resid_tree / prec_tree * prec_lk * var_post / resid_tree =
        # |    = 1 / prec_tree * prec_tree / sigma2 * var_post =
        # |    = var_post / sigma2
        centered_leaves=z * jnp.sqrt(var_post),
    )


def accept_moves_sequential_stage(pso: ParallelStageOut) -> tuple[State, Moves]:
    """
    Accept/reject the moves one tree at a time.

    This is the most performance-sensitive function because it contains all and
    only the parts of the algorithm that can not be parallelized across trees.

    Parameters
    ----------
    pso
        The output of `accept_moves_parallel_stage`.

    Returns
    -------
    bart : State
        A partially updated BART mcmc state.
    moves : Moves
        The accepted/rejected moves, with `acc` and `to_prune` set.
    """

    def loop(resid, pt):
        resid, leaf_tree, acc, to_prune, lkratio = accept_move_and_sample_leaves(
            resid,
            SeqStageInAllTrees(
                pso.bart.X,
                pso.bart.forest.resid_batch_size,
                pso.bart.prec_scale,
                pso.bart.forest.log_likelihood is not None,
                pso.prelk,
            ),
            pt,
        )
        return resid, (leaf_tree, acc, to_prune, lkratio)

    pts = SeqStageInPerTree(
        pso.bart.forest.leaf_tree,
        pso.prec_trees,
        pso.moves,
        pso.move_precs,
        pso.bart.forest.leaf_indices,
        pso.prelkv,
        pso.prelf,
    )
    resid, (leaf_trees, acc, to_prune, lkratio) = lax.scan(loop, pso.bart.resid, pts)

    bart = replace(
        pso.bart,
        resid=resid,
        forest=replace(pso.bart.forest, leaf_tree=leaf_trees, log_likelihood=lkratio),
    )
    moves = replace(pso.moves, acc=acc, to_prune=to_prune)

    return bart, moves


class SeqStageInAllTrees(Module):
    """
    The inputs to `accept_move_and_sample_leaves` that are shared by all trees.

    Parameters
    ----------
    X
        The predictors.
    resid_batch_size
        The batch size for computing the sum of residuals in each leaf.
    prec_scale
        The scale of the precision of the error on each datapoint. If None, it
        is assumed to be 1.
    save_ratios
        Whether to save the acceptance ratios.
    prelk
        The pre-computed terms of the likelihood ratio which are shared across
        trees.
    """

    X: UInt[Array, 'p n']
    resid_batch_size: int | None = field(static=True)
    prec_scale: Float32[Array, ' n'] | None
    save_ratios: bool = field(static=True)
    prelk: PreLk


class SeqStageInPerTree(Module):
    """
    The inputs to `accept_move_and_sample_leaves` that are separate for each tree.

    Parameters
    ----------
    leaf_tree
        The leaf values of the tree.
    prec_tree
        The likelihood precision scale in each potential or actual leaf node.
    move
        The proposed move, see `propose_moves`.
    move_precs
        The likelihood precision scale in each node modified by the moves.
    leaf_indices
        The leaf indices for the largest version of the tree compatible with
        the move.
    prelkv
    prelf
        The pre-computed terms of the likelihood ratio and leaf sampling which
        are specific to the tree.
    """

    leaf_tree: Float32[Array, ' 2**d']
    prec_tree: Float32[Array, ' 2**d']
    move: Moves
    move_precs: Precs | Counts
    leaf_indices: UInt[Array, ' n']
    prelkv: PreLkV
    prelf: PreLf


def accept_move_and_sample_leaves(
    resid: Float32[Array, ' n'], at: SeqStageInAllTrees, pt: SeqStageInPerTree
) -> tuple[
    Float32[Array, ' n'],
    Float32[Array, ' 2**d'],
    Bool[Array, ''],
    Bool[Array, ''],
    Float32[Array, ''] | None,
]:
    """
    Accept or reject a proposed move and sample the new leaf values.

    Parameters
    ----------
    resid
        The residuals (data minus forest value).
    at
        The inputs that are the same for all trees.
    pt
        The inputs that are separate for each tree.

    Returns
    -------
    resid : Float32[Array, 'n']
        The updated residuals (data minus forest value).
    leaf_tree : Float32[Array, '2**d']
        The new leaf values of the tree.
    acc : Bool[Array, '']
        Whether the move was accepted.
    to_prune : Bool[Array, '']
        Whether, to reflect the acceptance status of the move, the state should
        be updated by pruning the leaves involved in the move.
    log_lk_ratio : Float32[Array, ''] | None
        The logarithm of the likelihood ratio for the move. `None` if not to be
        saved.
    """
    # sum residuals in each leaf, in tree proposed by grow move
    if at.prec_scale is None:
        scaled_resid = resid
    else:
        scaled_resid = resid * at.prec_scale
    resid_tree = sum_resid(
        scaled_resid, pt.leaf_indices, pt.leaf_tree.size, at.resid_batch_size
    )

    # subtract starting tree from function
    resid_tree += pt.prec_tree * pt.leaf_tree

    # sum residuals in parent node modified by move
    resid_left = resid_tree[pt.move.left]
    resid_right = resid_tree[pt.move.right]
    resid_total = resid_left + resid_right
    assert pt.move.node.dtype == jnp.int32
    resid_tree = resid_tree.at[pt.move.node].set(resid_total)

    # compute acceptance ratio
    log_lk_ratio = compute_likelihood_ratio(
        resid_total, resid_left, resid_right, pt.prelkv, at.prelk
    )
    log_ratio = pt.move.log_trans_prior_ratio + log_lk_ratio
    log_ratio = jnp.where(pt.move.grow, log_ratio, -log_ratio)
    if not at.save_ratios:
        log_lk_ratio = None

    # determine whether to accept the move
    acc = pt.move.allowed & (pt.move.logu <= log_ratio)

    # compute leaves posterior and sample leaves
    mean_post = resid_tree * pt.prelf.mean_factor
    leaf_tree = mean_post + pt.prelf.centered_leaves

    # copy leaves around such that the leaf indices point to the correct leaf
    to_prune = acc ^ pt.move.grow
    leaf_tree = (
        leaf_tree.at[jnp.where(to_prune, pt.move.left, leaf_tree.size)]
        .set(leaf_tree[pt.move.node])
        .at[jnp.where(to_prune, pt.move.right, leaf_tree.size)]
        .set(leaf_tree[pt.move.node])
    )

    # replace old tree with new tree in function values
    resid += (pt.leaf_tree - leaf_tree)[pt.leaf_indices]

    return resid, leaf_tree, acc, to_prune, log_lk_ratio


def sum_resid(
    scaled_resid: Float32[Array, ' n'],
    leaf_indices: UInt[Array, ' n'],
    tree_size: int,
    batch_size: int | None,
) -> Float32[Array, ' {tree_size}']:
    """
    Sum the residuals in each leaf.

    Parameters
    ----------
    scaled_resid
        The residuals (data minus forest value) multiplied by the error
        precision scale.
    leaf_indices
        The leaf indices of the tree (in which leaf each data point falls into).
    tree_size
        The size of the tree array (2 ** d).
    batch_size
        The data batch size for the aggregation. Batching increases numerical
        accuracy and parallelism.

    Returns
    -------
    The sum of the residuals at data points in each leaf.
    """
    if batch_size is None:
        aggr_func = _aggregate_scatter
    else:
        aggr_func = partial(_aggregate_batched_onetree, batch_size=batch_size)
    return aggr_func(scaled_resid, leaf_indices, tree_size, jnp.float32)


def _aggregate_batched_onetree(
    values: Shaped[Array, '*'],
    indices: Integer[Array, '*'],
    size: int,
    dtype: jnp.dtype,
    batch_size: int,
) -> Float32[Array, ' {size}']:
    (n,) = indices.shape
    nbatches = n // batch_size + bool(n % batch_size)
    batch_indices = jnp.arange(n) % nbatches
    return (
        jnp.zeros((size, nbatches), dtype)
        .at[indices, batch_indices]
        .add(values)
        .sum(axis=1)
    )


def compute_likelihood_ratio(
    total_resid: Float32[Array, ''],
    left_resid: Float32[Array, ''],
    right_resid: Float32[Array, ''],
    prelkv: PreLkV,
    prelk: PreLk,
) -> Float32[Array, '']:
    """
    Compute the likelihood ratio of a grow move.

    Parameters
    ----------
    total_resid
    left_resid
    right_resid
        The sum of the residuals (scaled by error precision scale) of the
        datapoints falling in the nodes involved in the moves.
    prelkv
    prelk
        The pre-computed terms of the likelihood ratio, see
        `precompute_likelihood_terms`.

    Returns
    -------
    The likelihood ratio P(data | new tree) / P(data | old tree).
    """
    exp_term = prelk.exp_factor * (
        left_resid * left_resid / prelkv.sigma2_left
        + right_resid * right_resid / prelkv.sigma2_right
        - total_resid * total_resid / prelkv.sigma2_total
    )
    return prelkv.sqrt_term + exp_term


def accept_moves_final_stage(bart: State, moves: Moves) -> State:
    """
    Post-process the mcmc state after accepting/rejecting the moves.

    This function is separate from `accept_moves_sequential_stage` to signal it
    can work in parallel across trees.

    Parameters
    ----------
    bart
        A partially updated BART mcmc state.
    moves
        The proposed moves (see `propose_moves`) as updated by
        `accept_moves_sequential_stage`.

    Returns
    -------
    The fully updated BART mcmc state.
    """
    return replace(
        bart,
        forest=replace(
            bart.forest,
            grow_acc_count=jnp.sum(moves.acc & moves.grow),
            prune_acc_count=jnp.sum(moves.acc & ~moves.grow),
            leaf_indices=apply_moves_to_leaf_indices(bart.forest.leaf_indices, moves),
            split_tree=apply_moves_to_split_trees(bart.forest.split_tree, moves),
        ),
    )


@vmap_nodoc
def apply_moves_to_leaf_indices(
    leaf_indices: UInt[Array, 'num_trees n'], moves: Moves
) -> UInt[Array, 'num_trees n']:
    """
    Update the leaf indices to match the accepted move.

    Parameters
    ----------
    leaf_indices
        The index of the leaf each datapoint falls into, if the grow move was
        accepted.
    moves
        The proposed moves (see `propose_moves`), as updated by
        `accept_moves_sequential_stage`.

    Returns
    -------
    The updated leaf indices.
    """
    mask = ~jnp.array(1, leaf_indices.dtype)  # ...1111111110
    is_child = (leaf_indices & mask) == moves.left
    return jnp.where(
        is_child & moves.to_prune, moves.node.astype(leaf_indices.dtype), leaf_indices
    )


@vmap_nodoc
def apply_moves_to_split_trees(
    split_tree: UInt[Array, 'num_trees 2**(d-1)'], moves: Moves
) -> UInt[Array, 'num_trees 2**(d-1)']:
    """
    Update the split trees to match the accepted move.

    Parameters
    ----------
    split_tree
        The cutpoints of the decision nodes in the initial trees.
    moves
        The proposed moves (see `propose_moves`), as updated by
        `accept_moves_sequential_stage`.

    Returns
    -------
    The updated split trees.
    """
    assert moves.to_prune is not None
    return (
        split_tree.at[jnp.where(moves.grow, moves.node, split_tree.size)]
        .set(moves.grow_split.astype(split_tree.dtype))
        .at[jnp.where(moves.to_prune, moves.node, split_tree.size)]
        .set(0)
    )


def step_sigma(key: Key[Array, ''], bart: State) -> State:
    """
    MCMC-update the error variance (factor).

    Parameters
    ----------
    key
        A jax random key.
    bart
        A BART mcmc state.

    Returns
    -------
    The new BART mcmc state, with an updated `sigma2`.
    """
    resid = bart.resid
    alpha = bart.sigma2_alpha + resid.size / 2
    if bart.prec_scale is None:
        scaled_resid = resid
    else:
        scaled_resid = resid * bart.prec_scale
    norm2 = resid @ scaled_resid
    beta = bart.sigma2_beta + norm2 / 2

    sample = random.gamma(key, alpha)
    # random.gamma seems to be slow at compiling, maybe cdf inversion would
    # be better, but it's not implemented in jax
    return replace(bart, sigma2=beta / sample)


def step_z(key: Key[Array, ''], bart: State) -> State:
    """
    MCMC-update the latent variable for binary regression.

    Parameters
    ----------
    key
        A jax random key.
    bart
        A BART MCMC state.

    Returns
    -------
    The updated BART MCMC state.
    """
    trees_plus_offset = bart.z - bart.resid
    assert bart.y.dtype == bool
    resid = truncated_normal_onesided(key, (), ~bart.y, -trees_plus_offset)
    z = trees_plus_offset + resid
    return replace(bart, z=z, resid=resid)


def step_s(key: Key[Array, ''], bart: State) -> State:
    """
    Update `log_s` using Dirichlet sampling.

    The prior is s ~ Dirichlet(theta/p, ..., theta/p), and the posterior
    is s ~ Dirichlet(theta/p + varcount, ..., theta/p + varcount), where
    varcount is the count of how many times each variable is used in the
    current forest.

    Parameters
    ----------
    key
        Random key for sampling.
    bart
        The current BART state.

    Returns
    -------
    Updated BART state with re-sampled `log_s`.

    """
    assert bart.forest.theta is not None

    # histogram current variable usage
    p = bart.forest.max_split.size
    varcount = grove.var_histogram(p, bart.forest.var_tree, bart.forest.split_tree)

    # sample from Dirichlet posterior
    alpha = bart.forest.theta / p + varcount
    log_s = random.loggamma(key, alpha)

    # update forest with new s
    return replace(bart, forest=replace(bart.forest, log_s=log_s))


def step_theta(key: Key[Array, ''], bart: State, *, num_grid: int = 1000) -> State:
    """
    Update `theta`.

    The prior is theta / (theta + rho) ~ Beta(a, b).

    Parameters
    ----------
    key
        Random key for sampling.
    bart
        The current BART state.
    num_grid
        The number of points in the evenly-spaced grid used to sample
        theta / (theta + rho).

    Returns
    -------
    Updated BART state with re-sampled `theta`.
    """
    assert bart.forest.log_s is not None
    assert bart.forest.rho is not None
    assert bart.forest.a is not None
    assert bart.forest.b is not None

    # the grid points are the midpoints of num_grid bins in (0, 1)
    padding = 1 / (2 * num_grid)
    lamda_grid = jnp.linspace(padding, 1 - padding, num_grid)

    # normalize s
    log_s = bart.forest.log_s - logsumexp(bart.forest.log_s)

    # sample lambda
    logp, theta_grid = _log_p_lamda(
        lamda_grid, log_s, bart.forest.rho, bart.forest.a, bart.forest.b
    )
    i = random.categorical(key, logp)
    theta = theta_grid[i]

    return replace(bart, forest=replace(bart.forest, theta=theta))


def _log_p_lamda(
    lamda: Float32[Array, ' num_grid'],
    log_s: Float32[Array, ' p'],
    rho: Float32[Array, ''],
    a: Float32[Array, ''],
    b: Float32[Array, ''],
) -> tuple[Float32[Array, ' num_grid'], Float32[Array, ' num_grid']]:
    # in the following I use lamda[::-1] == 1 - lamda
    theta = rho * lamda / lamda[::-1]
    p = log_s.size
    return (
        (a - 1) * jnp.log1p(-lamda[::-1])  # log(lambda)
        + (b - 1) * jnp.log1p(-lamda)  # log(1 - lambda)
        + gammaln(theta)
        - p * gammaln(theta / p)
        + theta / p * jnp.sum(log_s)
    ), theta


def step_sparse(key: Key[Array, ''], bart: State) -> State:
    """
    Update the sparsity parameters.

    This invokes `step_s`, and then `step_theta` only if the parameters of
    the theta prior are defined.

    Parameters
    ----------
    key
        Random key for sampling.
    bart
        The current BART state.

    Returns
    -------
    Updated BART state with re-sampled `log_s` and `theta`.
    """
    keys = split(key)
    bart = step_s(keys.pop(), bart)
    if bart.forest.rho is not None:
        bart = step_theta(keys.pop(), bart)
    return bart
