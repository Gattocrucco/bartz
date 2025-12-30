# bartz/src/bartz/mcmcstep/_state.py
#
# Copyright (c) 2024-2025, The Bartz Contributors
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

"""Module defining the BART MCMC state and initialization."""

import math
from enum import Enum
from functools import cache, partial
from typing import Any, Literal

import jax
from equinox import Module, field
from jax import lax
from jax import numpy as jnp
from jax.scipy.linalg import solve_triangular
from jaxtyping import Array, Bool, Float32, Int32, Integer, UInt

from bartz import grove
from bartz.jaxext import get_default_device, minimal_unsigned_dtype


class Kind(str, Enum):
    """Indicator of regression type."""

    binary = 'binary'
    uv = 'uv'
    mv = 'mv'


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
    leaf_prior_cov_inv
        The prior precision matrix of a leaf, conditional on the tree structure.
        For the univariate case (k=1), this is a scalar (the inverse variance).
        The prior covariance of the sum of trees is
        ``num_trees * leaf_prior_cov_inv^-1``.
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

    leaf_tree: Float32[Array, 'num_trees 2**d'] | Float32[Array, 'num_trees k 2**d']
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
    leaf_prior_cov_inv: Float32[Array, ''] | Float32[Array, 'k k'] | None
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
    error_cov_inv
        The inverse error covariance (scalar for univariate, matrix for multivariate).
        `None` in binary regression.
    prec_scale
        The scale on the error precision, i.e., ``1 / error_scale ** 2``.
        `None` in binary regression.
    error_cov_df
    error_cov_scale
        The df and scale parameters of the inverse Wishart prior on the noise
        covariance. For the univariate case, the relationship to the inverse
        gamma prior parameters is ``alpha = df / 2``, ``beta = scale / 2``.
        `None` in binary regression.
    kind
        Inidicator of regression type.
    forest
        The sum of trees model.
    """

    X: UInt[Array, 'p n']
    y: Float32[Array, ' n'] | Float32[Array, ' k n'] | Bool[Array, ' n']
    z: None | Float32[Array, ' n']
    offset: Float32[Array, ''] | Float32[Array, ' k']
    resid: Float32[Array, ' n'] | Float32[Array, ' k n']
    error_cov_inv: Float32[Array, ''] | Float32[Array, 'k k'] | None
    prec_scale: Float32[Array, ' n'] | None
    error_cov_df: Float32[Array, ''] | None
    error_cov_scale: Float32[Array, ''] | Float32[Array, 'k k'] | None
    kind: Kind = field(static=True)
    forest: Forest


def _init_kind_parameters(
    kind: Kind | str | None,
    y: Float32[Any, ' n'] | Float32[Any, 'k n'] | Bool[Any, ' n'],
    offset: Float32[Array, ''] | Float32[Array, ' k'],
    error_scale: Float32[Any, ' n'] | None,
    error_cov_df: float | Float32[Any, ''] | None,
    error_cov_scale: float | Float32[Any, ''] | Float32[Any, 'k k'] | None,
    leaf_prior_cov_inv: Float32[Array, ''] | Float32[Array, 'k k'],
) -> tuple[
    Kind | str,
    None | int,
    None | Float32[Array, ''],
    None | Float32[Array, ''],
    None | Float32[Array, ''],
]:
    """
    Determine 'kind' and initialize/validate kind-specific params.

    Parameters
    ----------
    kind
        The regression kind, or None to infer from `y`.
    y
        The response variable.
    offset
        The offset to add to the predictions.
    error_scale
        Per-observation error scale (univariate only).
    error_cov_df
        The error covariance degrees of freedom.
    error_cov_scale
        The error covariance scale.
    leaf_prior_cov_inv
        The inverse of the leaf prior covariance.

    Returns
    -------
    kind
        The regression kind.
    k
        The number of output dimensions (None for uv).
    error_cov_inv
        The initialized error covariance inverse.
    error_cov_df
        The error covariance degrees of freedom (as array).
    error_cov_scale
        The error covariance scale (as array).

    Raises
    ------
    ValueError
        If `kind` is 'binary' and `y` is multivariate.
    """
    is_binary_y = y.dtype == bool
    k = None if y.ndim == 1 else y.shape[0]

    # Infer kind if not specified
    if kind is None:
        if is_binary_y:
            kind = 'binary'
        elif k is None:
            kind = 'uv'
        else:
            kind = 'mv'

    assert kind in ('binary', 'uv', 'mv')

    # Binary vs continuous
    if kind == 'binary':
        if k is not None:
            msg = 'Binary multivariate regression not supported, open an issue at https://github.com/Gattocrucco/bartz/issues if you need it.'
            raise ValueError(msg)
        assert is_binary_y
        assert error_scale is None
        assert error_cov_df is None
        assert error_cov_scale is None
        error_cov_inv = None
    else:
        assert not is_binary_y
        error_cov_df = jnp.asarray(error_cov_df)
        error_cov_scale = jnp.asarray(error_cov_scale)

    # Multivariate vs univariate
    if kind == 'mv':
        assert y.ndim == 2
        assert y.shape[0] == k
        assert leaf_prior_cov_inv.shape == (k, k)
        assert offset.shape == (k,)
        if kind != 'binary':
            assert error_cov_scale.shape == (k, k)
            error_cov_inv = error_cov_df * _inv_via_chol_with_gersh(error_cov_scale)
    else:
        assert y.ndim == 1
        assert leaf_prior_cov_inv.ndim == 0
        assert offset.ndim == 0
        if kind != 'binary':
            assert error_cov_scale.ndim == 0
            # inverse gamma prior: alpha = df / 2, beta = scale / 2
            error_cov_inv = error_cov_df / error_cov_scale

    return kind, k, error_cov_inv, error_cov_df, error_cov_scale


def init(
    *,
    X: UInt[Any, 'p n'],
    y: Float32[Any, ' n'] | Float32[Array, ' k n'] | Bool[Any, ' n'],
    offset: float | Float32[Any, ''] | Float32[Any, ' k'],
    max_split: UInt[Any, ' p'],
    num_trees: int,
    p_nonterminal: Float32[Any, ' d-1'],
    leaf_prior_cov_inv: float | Float32[Any, ''] | Float32[Array, 'k k'],
    error_cov_df: float | Float32[Any, ''] | None = None,
    error_cov_scale: float | Float32[Any, ''] | Float32[Array, 'k k'] | None = None,
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
    kind: Kind | str | None = None,
) -> State:
    """
    Make a BART posterior sampling MCMC initial state.

    Parameters
    ----------
    X
        The predictors. Note this is trasposed compared to the usual convention.
    y
        The response. If the data type is `bool`, the regression model is binary
        regression with probit. If two-dimensional, the outcome is multivariate
        with the first axis indicating the component.
    offset
        Constant shift added to the sum of trees. 0 if not specified.
    max_split
        The maximum split index for each variable. All split ranges start at 1.
    num_trees
        The number of trees in the forest.
    p_nonterminal
        The probability of a nonterminal node at each depth. The maximum depth
        of trees is fixed by the length of this array.
    leaf_prior_cov_inv
        The prior precision matrix of a leaf, conditional on the tree structure.
        For the univariate case (k=1), this is a scalar (the inverse variance).
        The prior covariance of the sum of trees is
        ``num_trees * leaf_prior_cov_inv^-1``. The prior mean of leaves is
        always zero.
    error_cov_df
    error_cov_scale
        The df and scale parameters of the inverse Wishart prior on the error
        covariance. For the univariate case, the relationship to the inverse
        gamma prior parameters is ``alpha = df / 2``, ``beta = scale / 2``.
        Leave unspecified for binary regression.
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
    kind
        Inidicator of regression type. If not specified, it's inferred from `y`.

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
    leaf_prior_cov_inv = jnp.asarray(leaf_prior_cov_inv)

    resid_batch_size, count_batch_size = _choose_suffstat_batch_size(
        resid_batch_size, count_batch_size, y, 2**max_depth * num_trees
    )

    kind, k, error_cov_inv, error_cov_df, error_cov_scale = _init_kind_parameters(
        kind, y, offset, error_scale, error_cov_df, error_cov_scale, leaf_prior_cov_inv
    )
    is_binary = kind == 'binary'

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

    if kind == 'mv':
        leaf_tree = jax.vmap(
            make_forest, in_axes=(None, None), out_axes=1, axis_size=k
        )(max_depth, jnp.float32)
    else:
        leaf_tree = make_forest(max_depth, jnp.float32)

    return State(
        X=jnp.asarray(X),
        y=y,
        z=jnp.full(y.shape, offset) if is_binary else None,
        offset=offset,
        resid=jnp.zeros(y.shape) if is_binary else y - offset[..., None],
        error_cov_inv=error_cov_inv,
        prec_scale=(
            None if error_scale is None else lax.reciprocal(jnp.square(error_scale))
        ),
        error_cov_df=error_cov_df,
        error_cov_scale=error_cov_scale,
        kind=kind,
        forest=Forest(
            leaf_tree=leaf_tree,
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
                (num_trees, y.shape[-1]), minimal_unsigned_dtype(2**max_depth - 1)
            ),
            min_points_per_decision_node=_asarray_or_none(min_points_per_decision_node),
            min_points_per_leaf=_asarray_or_none(min_points_per_leaf),
            resid_batch_size=resid_batch_size,
            count_batch_size=count_batch_size,
            log_trans_prior=jnp.zeros(num_trees) if save_ratios else None,
            log_likelihood=jnp.zeros(num_trees) if save_ratios else None,
            leaf_prior_cov_inv=leaf_prior_cov_inv,
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


def _get_platform(x: Array):
    """Get the platform of the device where `x` is located, or the default device if that's not possible."""
    try:
        device = x.devices().pop()
    except jax.errors.ConcretizationTypeError:
        device = get_default_device()
    platform = device.platform
    if platform not in ('cpu', 'gpu'):
        msg = f'Unknown platform: {platform}'
        raise KeyError(msg)
    return platform


def _choose_suffstat_batch_size(
    resid_batch_size: int | None | Literal['auto'],
    count_batch_size: int | None | Literal['auto'],
    y: Float32[Array, 'k n'] | Float32[Array, ' n'],
    forest_size: int,
) -> tuple[int | None, int | None]:
    if y.ndim == 2:
        k, n = y.shape
    else:
        k = 1
        (n,) = y.shape
    n = max(1, n)

    @cache
    def get_platform():
        return _get_platform(y)

    if resid_batch_size == 'auto':
        platform = get_platform()
        if platform == 'cpu':
            resid_batch_size = 2 ** round(math.log2(n / 6))  # n/6
        elif platform == 'gpu':
            resid_batch_size = 2 ** round((1 + math.log2(n)) / 3)  # n^1/3
        resid_batch_size *= k
        resid_batch_size = max(1, resid_batch_size)

    if count_batch_size == 'auto':
        platform = get_platform()
        if platform == 'cpu':
            count_batch_size = None
        elif platform == 'gpu':
            count_batch_size = 2 ** round(math.log2(n) / 2 - 2)  # n^1/2
            # /4 is good on V100, /2 on L4/T4, still haven't tried A100
            count_batch_size *= k
            max_memory = 2**29
            itemsize = 4
            min_batch_size = math.ceil(forest_size * itemsize * n / max_memory)
            count_batch_size = max(count_batch_size, min_batch_size)
            count_batch_size = max(1, count_batch_size)

    return resid_batch_size, count_batch_size


def chol_with_gersh(
    mat: Float32[Array, '... k k'], absolute_eps: bool = False
) -> Float32[Array, '... k k']:
    """Cholesky with Gershgorin stabilization, supports batching."""
    return _chol_with_gersh_impl(mat, absolute_eps)


@partial(jnp.vectorize, signature='(k,k)->(k,k)', excluded=(1,))
def _chol_with_gersh_impl(
    mat: Float32[Array, '... k k'], absolute_eps: bool
) -> Float32[Array, '... k k']:
    rho = jnp.max(jnp.sum(jnp.abs(mat), axis=1))
    eps = jnp.finfo(mat.dtype).eps
    u = mat.shape[0] * rho * eps
    if absolute_eps:
        u += eps
    mat = mat.at[jnp.diag_indices_from(mat)].add(u)
    return jnp.linalg.cholesky(mat)


def _inv_via_chol_with_gersh(mat: Float32[Array, 'k k']) -> Float32[Array, 'k k']:
    """Compute matrix inverse via Cholesky with Gershgorin stabilization.

    DO NOT USE THIS FUNCTION UNLESS YOU REALLY NEED TO.
    """
    L = chol_with_gersh(mat)
    I = jnp.eye(mat.shape[0], dtype=mat.dtype)
    L_inv = solve_triangular(L, I, lower=True)
    return L_inv.T @ L_inv
