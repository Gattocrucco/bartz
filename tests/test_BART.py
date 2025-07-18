# bartz/tests/test_BART.py
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

"""Test `bartz.BART`.

This is the main suite of tests.
"""

import os
import signal
import threading
import time
from collections.abc import Callable
from functools import partial
from typing import Any, Literal

import jax
import numpy
import polars as pl
import pytest
from jax import debug_nans, random, vmap
from jax import numpy as jnp
from jax.scipy.linalg import solve_triangular
from jax.scipy.special import ndtr
from jaxtyping import Array, Float, Float32, Int32, Key, Real, UInt
from numpy.testing import assert_allclose, assert_array_equal

from bartz.debug import (
    TraceWithOffset,
    check_trace,
    sample_prior,
    trace_depth_distr,
    tree_actual_depth,
    trees_BART_to_bartz,
)
from bartz.debug import debug_gbart as gbart
from bartz.grove import is_actual_leaf, tree_depth, tree_depths
from bartz.jaxext import split
from bartz.mcmcloop import (
    PrintCallbackState,
    SparseCallbackState,
    compute_varcount,
    evaluate_trace,
)
from tests.rbartpackages import BART
from tests.util import assert_close_matrices


def gen_X(key, p, n, kind):
    """Generate a matrix of predictors."""
    if kind == 'continuous':
        return random.uniform(key, (p, n), float, -2, 2)
    elif kind == 'binary':
        return random.bernoulli(key, 0.5, (p, n)).astype(float)
    else:
        raise KeyError(kind)


def f(x: Real[Array, 'p n'], s: Real[Array, ' p']) -> Float32[Array, ' n']:
    """Conditional mean of the DGP."""
    T = 2
    return s @ jnp.cos(2 * jnp.pi / T * x) / jnp.sqrt(s @ s)


def gen_w(key, n):
    """Generate a vector of error weights."""
    return jnp.exp(random.uniform(key, (n,), float, -1, 1))


def gen_y(
    key,
    X,
    w,
    kind: Literal['continuous', 'probit'],
    *,
    s: Real[Array, ' p'] | Literal['uniform', 'random'] = 'uniform',
):
    """Generate responses given predictors."""
    keys = split(key, 3)

    p, n = X.shape
    if isinstance(s, jax.Array):
        pass
    elif s == 'random':
        s = jnp.exp(random.uniform(keys.pop(), (p,), float, -1, 1))
    elif s == 'uniform':
        s = jnp.ones(p)

    match kind:
        case 'continuous':
            sigma = 0.1
            error = sigma * random.normal(keys.pop(), (n,))
            if w is not None:
                error *= w
            return f(X, s) + error

        case 'probit':
            assert w is None
            _, n = X.shape
            error = random.normal(keys.pop(), (n,))
            prob = ndtr(f(X, s) + error)
            return random.bernoulli(keys.pop(), prob, (n,))


@pytest.fixture(params=[1, 2, 3])
def kw(keys, request):
    """Return a dictionary of keyword arguments for BART."""
    match request.param:
        # continuous regression with some settings that induce large types,
        # sparsity with free theta
        case 1:
            X = gen_X(keys.pop(), 2, 30, 'continuous')
            Xt = gen_X(keys.pop(), 2, 31, 'continuous')
            y = gen_y(keys.pop(), X, None, 'continuous', s='random')
            return dict(
                x_train=X,
                y_train=y,
                x_test=Xt,
                sparse=True,
                ntree=20,
                ndpost=100,
                nskip=50,
                printevery=50,
                usequants=False,
                numcut=256,  # > 255 to use uint16 for X and split_trees
                maxdepth=9,  # > 8 to use uint16 for leaf_indices
                seed=keys.pop(),
                init_kw=dict(
                    resid_batch_size=None, count_batch_size=None, save_ratios=True
                ),
            )

        # binary regression with binary X and high p
        case 2:
            p = 257  # > 256 to use uint16 for var_trees.
            X = gen_X(keys.pop(), p, 30, 'binary')
            Xt = gen_X(keys.pop(), p, 31, 'binary')
            y = gen_y(keys.pop(), X, None, 'probit')
            return dict(
                x_train=X,
                y_train=y,
                x_test=Xt,
                type='pbart',
                ntree=20,
                ndpost=100,
                nskip=50,
                keepevery=1,  # the default with binary would be 10
                printevery=None,
                usequants=True,
                # usequants=True with binary X to check the case in which the
                # splits are less than the statically known maximum
                numcut=255,
                maxdepth=6,
                seed=keys.pop(),
                init_kw=dict(
                    resid_batch_size=16,
                    count_batch_size=16,
                    save_ratios=False,
                    min_points_per_decision_node=None,
                    min_points_per_leaf=None,
                ),
            )

        # continuous regression with error weights and sparsity with fixed theta
        case 3:
            X = gen_X(keys.pop(), 2, 30, 'continuous')
            Xt = gen_X(keys.pop(), 2, 31, 'continuous')
            w = gen_w(keys.pop(), X.shape[1])
            y = gen_y(keys.pop(), X, w, 'continuous', s='random')
            return dict(
                x_train=X,
                x_test=Xt,
                y_train=y,
                w=w,
                sparse=True,
                theta=2,
                ntree=20,
                ndpost=100,
                nskip=50,
                printevery=50,
                usequants=True,
                numcut=10,
                maxdepth=8,  # 8 to check if leaf_indices changes type too soon
                seed=keys.pop(),
                init_kw=dict(
                    resid_batch_size=None, count_batch_size=None, save_ratios=True
                ),
            )


def test_sequential_guarantee(kw):
    """Check that the way iterations are saved does not influence the result."""
    kw['keepevery'] = 1
    bart1 = gbart(**kw)

    kw['seed'] = random.clone(kw['seed'])
    if kw.get('sparse', False):
        callback_state = (
            PrintCallbackState(None, None),
            SparseCallbackState(kw['nskip'] // 2),
        )
        # see `mcmcloop.make_default_callback`
        kw.setdefault('run_mcmc_kw', {}).setdefault('callback_state', callback_state)
    kw['nskip'] -= 1
    kw['ndpost'] += 1
    bart2 = gbart(**kw)
    assert_array_equal(bart1.yhat_train, bart2.yhat_train[1:])

    kw['seed'] = random.clone(kw['seed'])
    kw['keepevery'] = 2
    bart3 = gbart(**kw)
    yhat_train = bart2.yhat_train[1::2]
    assert_array_equal(yhat_train, bart3.yhat_train[: len(yhat_train)])


def test_output_shapes(kw):
    """Check the output shapes of all the array attributes of BART.gbart."""
    bart = gbart(**kw)

    ndpost = kw['ndpost']
    nskip = kw['nskip']
    p, n = kw['x_train'].shape
    _, m = kw['x_test'].shape

    binary = kw['y_train'].dtype == bool

    assert bart.offset.shape == ()
    if binary:
        assert bart.prob_test.shape == (ndpost, m)
        assert bart.prob_test_mean.shape == (m,)
        assert bart.prob_train.shape == (ndpost, n)
        assert bart.prob_train_mean.shape == (n,)
        assert bart.sigma is None
        assert bart.sigma_mean is None
    else:
        assert bart.prob_test is None
        assert bart.prob_test_mean is None
        assert bart.prob_train is None
        assert bart.prob_train_mean is None
        assert bart.sigma.shape == (nskip + ndpost,)
        assert bart.sigma_mean.shape == ()
    assert bart.varcount.shape == (ndpost, p)
    assert bart.varcount_mean.shape == (p,)
    assert bart.varprob.shape == (ndpost, p)
    assert bart.varprob_mean.shape == (p,)
    assert bart.yhat_test.shape == (ndpost, m)
    if binary:
        assert bart.yhat_test_mean is None
    else:
        assert bart.yhat_test_mean.shape == (m,)
    assert bart.yhat_train.shape == (ndpost, n)
    if binary:
        assert bart.yhat_train_mean is None
    else:
        assert bart.yhat_train_mean.shape == (n,)


def test_output_types(kw):
    """Check the output types of all the attributes of BART.gbart."""
    bart = gbart(**kw)

    binary = kw['y_train'].dtype == bool

    assert bart.offset.dtype == jnp.float32
    assert isinstance(bart.ndpost, int)
    if binary:
        assert bart.prob_test.dtype == jnp.float32
        assert bart.prob_test_mean.dtype == jnp.float32
        assert bart.prob_train.dtype == jnp.float32
        assert bart.prob_train_mean.dtype == jnp.float32
    else:
        assert bart.sigma.dtype == jnp.float32
        assert bart.sigma_mean.dtype == jnp.float32
    assert bart.varcount.dtype == jnp.int32
    assert bart.varcount_mean.dtype == jnp.float32
    assert bart.varprob.dtype == jnp.float32
    assert bart.varprob_mean.dtype == jnp.float32
    assert bart.yhat_test.dtype == jnp.float32
    if not binary:
        assert bart.yhat_test_mean.dtype == jnp.float32
    assert bart.yhat_train.dtype == jnp.float32
    if not binary:
        assert bart.yhat_train_mean.dtype == jnp.float32


def test_predict(kw):
    """Check that the public BART.gbart.predict method works."""
    bart = gbart(**kw)
    yhat_train = bart.predict(kw['x_train'])
    assert_array_equal(bart.yhat_train, yhat_train)


def test_varprob(kw):
    """Basic checks of the `varprob` attribute."""
    bart = gbart(**kw)

    # basic properties of probabilities
    assert jnp.all(bart.varprob >= 0)
    assert jnp.all(bart.varprob <= 1)
    assert_allclose(bart.varprob.sum(axis=1), 1, rtol=1e-6)

    # probabilities are either 0 or 1/peff if sparsity is disabled
    sparse = kw.get('sparse', False)
    if not sparse:
        unique = jnp.unique(bart.varprob)
        assert unique.size in (1, 2)
        if unique.size == 2:
            assert unique[0] == 0

    # the mean is the mean
    assert_array_equal(bart.varprob_mean, bart.varprob.mean(axis=0))


def test_varprob_blocked_vars(keys):
    """Check that varprob = 0 on predictors blocked a priori."""
    X = gen_X(keys.pop(), 2, 30, 'continuous')
    y = gen_y(keys.pop(), X, None, 'continuous')
    with debug_nans(False):
        xinfo = jnp.array([[jnp.nan], [0]])
    bart = gbart(x_train=X, y_train=y, xinfo=xinfo, seed=keys.pop())
    assert_array_equal(bart._mcmc_state.forest.max_split, [0, 1])
    assert_array_equal(bart.varprob_mean, [0, 1])
    assert jnp.all(bart.varprob_mean == bart.varprob)


@pytest.mark.parametrize('theta', ['fixed', 'free'])
def test_variable_selection(keys: split, theta: Literal['fixed', 'free']):
    """Check that variable selection works."""
    # data config
    p = 100  # number of predictors
    peff = 5  # number of actually used predictors
    n = 1000

    # generate sparsity pattern
    mask = jnp.zeros(p, bool).at[:peff].set(True)
    mask = random.permutation(keys.pop(), mask)
    s = mask.astype(float)

    # generate data
    X = gen_X(keys.pop(), p, n, 'continuous')
    y = gen_y(keys.pop(), X, None, 'continuous', s=s)

    # run bart
    bart = gbart(
        x_train=X,
        y_train=y,
        nskip=1000,
        sparse=True,
        theta=peff if theta == 'fixed' else None,
        seed=keys.pop(),
    )

    # check that the variables have been identified
    assert bart.varprob_mean[mask].sum() >= 0.9
    assert bart.varprob_mean[mask].min().item() > 0.5 / peff
    assert bart.varprob_mean[~mask].max().item() < 1 / (p - peff)


def test_scale_shift(kw):
    """Check self-consistency of rescaling the inputs."""
    if kw['y_train'].dtype == bool:
        pytest.skip('Cannot rescale binary responses.')

    bart1 = gbart(**kw)

    offset = 0.4703189
    scale = 0.5294714
    kw.update(y_train=offset + kw['y_train'] * scale, seed=random.clone(kw['seed']))
    # note: using the same seed does not guarantee stable error because the mcmc
    # makes discrete choices based on thresholds on floats, so numerical error
    # can be amplified.
    bart2 = gbart(**kw)

    assert_allclose(bart1.offset, (bart2.offset - offset) / scale, rtol=1e-6, atol=1e-6)
    assert_allclose(
        bart1._mcmc_state.forest.sigma_mu2,
        bart2._mcmc_state.forest.sigma_mu2 / scale**2,
        rtol=1e-6,
        atol=0,
    )
    assert_allclose(bart1.sigest, bart2.sigest / scale, rtol=1e-6)
    assert_array_equal(bart1._mcmc_state.sigma2_alpha, bart2._mcmc_state.sigma2_alpha)
    assert_allclose(
        bart1._mcmc_state.sigma2_beta,
        bart2._mcmc_state.sigma2_beta / scale**2,
        rtol=1e-6,
    )
    assert_close_matrices(
        bart1.yhat_train, (bart2.yhat_train - offset) / scale, rtol=1e-5
    )
    assert_close_matrices(
        bart1.yhat_train_mean, (bart2.yhat_train_mean - offset) / scale, rtol=1e-5
    )
    assert_close_matrices(
        bart1.yhat_test, (bart2.yhat_test - offset) / scale, rtol=1e-5
    )
    assert_close_matrices(
        bart1.yhat_test_mean, (bart2.yhat_test_mean - offset) / scale, rtol=1e-5
    )
    assert_close_matrices(bart1.sigma, bart2.sigma / scale, rtol=1e-5)
    assert_allclose(bart1.sigma_mean, bart2.sigma_mean / scale, atol=1e-6)


def test_min_points_per_decision_node(kw):
    """Check that the limit of at least 10 datapoints per decision node is respected."""
    kw['init_kw'].update(min_points_per_leaf=None)
    bart = gbart(**kw)
    distr = bart.points_per_decision_node_distr()
    distr_marg = distr.sum(axis=(0, 1))

    min_points = kw['init_kw'].get('min_points_per_decision_node', 10)

    if min_points is None:
        assert distr_marg[9] > 0
    else:
        assert jnp.all(distr_marg[:min_points] == 0)
        assert distr_marg[min_points] > 0


def test_min_points_per_leaf(kw):
    """Check that the limit of at least 5 datapoints per leaf is respected."""
    kw['init_kw'].update(min_points_per_decision_node=None)
    bart = gbart(**kw)
    distr = bart.points_per_leaf_distr()
    distr_marg = distr.sum(axis=(0, 1))

    min_points = kw['init_kw'].get('min_points_per_leaf', 5)

    if min_points is None:
        assert distr_marg[4] > 0
    else:
        assert jnp.all(distr_marg[:min_points] == 0)
        assert distr_marg[min_points] > 0


def test_residuals_accuracy(kw):
    """Check that running residuals are close to the recomputed final residuals."""
    kw.update(ntree=200, ndpost=1000, nskip=0)
    bart = gbart(**kw)
    accum_resid, actual_resid = bart.compare_resid()
    assert_close_matrices(accum_resid, actual_resid, rtol=1e-4)


def set_num_datapoints(kw: dict, n):
    """Set the number of datapoints in the kw dictionary."""
    assert n <= kw['y_train'].size
    kw = kw.copy()
    kw['x_train'] = kw['x_train'][:, :n]
    kw['y_train'] = kw['y_train'][:n]
    if kw.get('w') is not None:
        kw['w'] = kw['w'][:n]
    return kw


def test_no_datapoints(kw):
    """Check automatic data scaling with 0 datapoints."""
    # remove all datapoints
    kw = set_num_datapoints(kw, 0)

    # set the split grid manually because automatic setting relies on datapoints
    p, _ = kw['x_train'].shape
    nsplits = 10
    xinfo = jnp.broadcast_to(jnp.arange(nsplits, dtype=jnp.float32), (p, nsplits))
    kw.update(xinfo=xinfo)

    bart = gbart(**kw)
    ndpost = kw['ndpost']
    assert bart.yhat_train.shape == (ndpost, 0)
    assert bart.offset == 0
    if kw['y_train'].dtype == bool:
        tau_num = 3
        assert bart.sigest is None
    else:
        tau_num = 1
        assert bart.sigest == 1
    assert_allclose(
        bart._mcmc_state.forest.sigma_mu2, tau_num**2 / (2**2 * kw['ntree']), rtol=1e-6
    )


def test_one_datapoint(kw):
    """Check automatic data scaling with 1 datapoint."""
    kw = set_num_datapoints(kw, 1)

    # set the split grid manually because otherwise there would be 0 cutpoints
    # when usequants=True, and computing varprob produces nans in that case
    if kw.get('usequants', False):
        p, _ = kw['x_train'].shape
        nsplits = 10
        xinfo = jnp.broadcast_to(jnp.arange(nsplits, dtype=jnp.float32), (p, nsplits))
        kw.update(xinfo=xinfo)

    bart = gbart(**kw)
    if kw['y_train'].dtype == bool:
        tau_num = 3
        assert bart.sigest is None
        assert bart.offset == 0
    else:
        tau_num = 1
        assert bart.sigest == 1
        assert bart.offset == kw['y_train'].item()
    assert_allclose(
        bart._mcmc_state.forest.sigma_mu2, tau_num**2 / (2**2 * kw['ntree']), rtol=1e-6
    )


def test_two_datapoints(kw):
    """Check automatic data scaling with 2 datapoints."""
    kw = set_num_datapoints(kw, 2)
    bart = gbart(**kw)
    if kw['y_train'].dtype != bool:
        assert_allclose(bart.sigest, kw['y_train'].std(), rtol=1e-6)
    if kw['usequants']:
        assert jnp.all(bart._mcmc_state.forest.max_split <= 1)


def test_few_datapoints(kw):
    """Check that the trees cannot grow if there are not enough datapoints.

    If there are less than 10 datapoints, it is not possible to satisfy the 10
    points per decision node requirement, neither the 5 datapoints per leaf
    constraint.
    """
    kw.setdefault('init_kw', {}).update(
        min_points_per_decision_node=10, min_points_per_leaf=None
    )
    kw = set_num_datapoints(kw, 9)  # < 10 = 2 * 5
    bart = gbart(**kw)
    assert jnp.all(bart.yhat_train == bart.yhat_train[:, :1])

    kw['init_kw'].update(min_points_per_decision_node=None, min_points_per_leaf=5)
    kw['seed'] = random.clone(kw['seed'])
    bart = gbart(**kw)
    assert jnp.all(bart.yhat_train == bart.yhat_train[:, :1])


def kw_bartz_to_BART(key: Key[Array, ''], kw: dict, bart: gbart) -> dict:
    """Convert bartz keyword arguments to R BART keyword arguments."""
    kw_BART = dict(**kw, rm_const=False, mc_cores=1)
    kw_BART.pop('init_kw')
    kw_BART.pop('maxdepth', None)
    for arg in 'w', 'printevery':
        if arg in kw_BART and kw_BART[arg] is None:
            kw_BART.pop(arg)
    kw_BART['seed'] = random.randint(key, (), 0, jnp.uint32(2**31)).item()

    # Set BART cutpoints manually. This means I am not checking that the
    # automatic cutpoint determination of BART is the same of my package. They
    # are similar but have some differences, and having exactly the same
    # cutpoints is more important for the test.
    kw_BART['transposed'] = True  # this disables predictors pre-processing
    kw_BART['numcut'] = bart._mcmc_state.forest.max_split
    kw_BART['xinfo'] = bart._splits

    return kw_BART


def check_rbart(kw, bart, rbart):
    """Subroutine for `test_rbart`, check that the R BART output is self-consistent."""
    # convert the trees to bartz format
    trees = rbart.treedraws['trees']
    trace, meta = trees_BART_to_bartz(trees, offset=rbart.offset)

    # check the trees are valid
    assert jnp.all(meta.numcut <= bart._mcmc_state.forest.max_split)
    bad = check_trace(trace, meta.numcut)
    num_bad = jnp.count_nonzero(bad)
    assert num_bad == 0

    # check varcount
    varcount = compute_varcount(meta.numcut.size, trace)
    assert jnp.all(varcount == rbart.varcount)

    # chech yhat_train
    yhat_train = evaluate_trace(trace, bart._mcmc_state.X)
    assert_close_matrices(yhat_train, rbart.yhat_train, rtol=1e-6)

    # check yhat_test
    Xt = bart._bin_predictors(kw['x_test'], bart._splits)
    yhat_test = evaluate_trace(trace, Xt)
    assert_close_matrices(yhat_test, rbart.yhat_test, rtol=1e-6)

    if kw['y_train'].dtype == bool:
        # check prob_train
        prob_train = ndtr(yhat_train)
        assert_close_matrices(prob_train, rbart.prob_train, rtol=1e-7)

        # check prob_test
        prob_test = ndtr(yhat_test)
        assert_close_matrices(prob_test, rbart.prob_test, rtol=1e-7)


def test_rbart(kw, keys):
    """Check bartz.BART gives the same results as R package BART."""
    p, n = kw['x_train'].shape
    kw.update(ntree=max(2 * n, p), nskip=3000, ndpost=1000, keepevery=1)
    # R BART can't change the min_points_per_leaf per leaf setting
    kw['init_kw'].update(min_points_per_decision_node=10, min_points_per_leaf=5)

    # run bart with both packages
    bart = gbart(**kw)
    kw_BART = kw_bartz_to_BART(keys.pop(), kw, bart)
    rbart = BART.mc_gbart(**kw_BART)
    # use mc_gbart instead of gbart because gbart does not use the seed

    # first cross-check the outputs of R BART alone
    check_rbart(kw, bart, rbart)

    # compare results of bartz and BART

    # check offset
    assert_allclose(bart.offset, rbart.offset, rtol=1e-6, atol=1e-7)
    # I would check sigest as well, but it's not in the R object despite what
    # the documentation says

    # check yhat_train
    rhat_yhat_train = multivariate_rhat([bart.yhat_train, rbart.yhat_train])
    assert rhat_yhat_train < 1.8

    # check yhat_test
    rhat_yhat_test = multivariate_rhat([bart.yhat_test, rbart.yhat_test])
    assert rhat_yhat_test < 1.8

    if kw['y_train'].dtype == bool:  # binary regression
        # check prob_train
        rhat_prob_train = multivariate_rhat([bart.prob_train, rbart.prob_train])
        assert rhat_prob_train < 1.2

        # check prob_test
        rhat_prob_test = multivariate_rhat([bart.prob_test, rbart.prob_test])
        assert rhat_prob_test < 1.2

    else:  # continuous regression
        # check yhat_train_mean
        assert_close_matrices(bart.yhat_train_mean, rbart.yhat_train_mean, rtol=0.2)

        # check yhat_test_mean
        assert_close_matrices(bart.yhat_test_mean, rbart.yhat_test_mean, rtol=0.4)

        # check sigma
        rhat_sigma = rhat([bart.sigma[-bart.ndpost :], rbart.sigma[-rbart.ndpost :]])
        assert rhat_sigma < 1.02

        # check sigma_mean
        assert_allclose(bart.sigma_mean, rbart.sigma_mean, rtol=0.05)

    # check number of tree nodes in forest
    bart_count = bart.varcount.sum(axis=1)
    rbart_count = rbart.varcount.sum(axis=1)
    rhat_count = rhat([bart_count, rbart_count])
    assert rhat_count < 30  # genuinely bad, see below
    assert_allclose(bart_count.mean(), rbart_count.mean(), rtol=0.2)

    if p < n:
        # skip if p is large because it would be difficult for the MCMC to get
        # stuff about predictors right

        # check varcount
        rhat_varcount = multivariate_rhat([bart.varcount, rbart.varcount])
        # there is a visible discrepancy on the number of nodes, with bartz
        # having deeper trees, this 5 is not just "not good to sampling
        # accuracy but close in practice.""
        assert rhat_varcount < 5
        assert_allclose(bart.varcount_mean, rbart.varcount_mean, rtol=0.2, atol=4)

        # check varprob
        if kw.get('sparse', False):
            rhat_varprob = multivariate_rhat([bart.varprob, rbart.varprob])
            assert rhat_varprob < 1.7
            assert_allclose(bart.varprob_mean, rbart.varprob_mean, atol=0.1)


def test_xinfo():
    """Simple check that the `xinfo` parameter works."""
    with debug_nans(False):
        xinfo = jnp.array(
            [[1.1, 2.3, jnp.nan], [-50, 10, 20], [jnp.nan, jnp.nan, jnp.nan]]
        )
    kw = dict(
        x_train=jnp.empty((3, 0)),
        y_train=jnp.empty(0),
        ndpost=0,
        nskip=0,
        # these `usequants` and `numcut` values would lead to an error, so this
        # checks they are ignored if `xinfo` is specified
        usequants=True,
        numcut=0,
        xinfo=xinfo,
    )
    bart = gbart(**kw)

    xinfo_wo_nan = jnp.where(jnp.isnan(xinfo), jnp.finfo(jnp.float32).max, xinfo)
    assert_array_equal(bart._splits, xinfo_wo_nan)
    assert_array_equal(bart._mcmc_state.forest.max_split, [2, 3, 0])


def test_xinfo_wrong_p():
    """Check that `xinfo` must have the same number of rows as `X`."""
    with debug_nans(False):
        xinfo = jnp.array(
            [[1.1, 2.3, jnp.nan], [-50, 10, 20], [jnp.nan, jnp.nan, jnp.nan]]
        )
    kw = dict(
        x_train=jnp.empty((5, 0)), y_train=jnp.empty(0), ndpost=0, nskip=0, xinfo=xinfo
    )
    with pytest.raises(ValueError, match=r'xinfo\.shape'):
        gbart(**kw)


@pytest.mark.parametrize(
    ('p', 'nsplits'),
    [
        (1, 1),  # sure that trees do not grow beyond depth 2
        (3, 2),  # likely to have no available decision rules on some nodes
        (10, 1),  # always available decision rules, but never on the same variable
        (10, 255),  # likely always available decision rules for all variables
    ],
)
def test_prior(keys, p, nsplits):
    """Check that the posterior without data is equivalent to the prior."""
    # sample from posterior without data
    xinfo = jnp.broadcast_to(jnp.arange(nsplits, dtype=jnp.float32), (p, nsplits))
    # set the split grid manually because automatic setting relies on datapoints
    kw = dict(
        x_train=jnp.empty((p, 0)),
        y_train=jnp.empty(0),
        ntree=20,
        ndpost=1000,
        nskip=3000,
        printevery=None,
        xinfo=xinfo,
        seed=keys.pop(),
        init_kw=dict(min_points_per_decision_node=None, min_points_per_leaf=None),
        # unset limits on datapoints per node because there's no data
    )
    bart = gbart(**kw)

    # extract p_nonterminal in original format from mcmc state
    p_nonterminal = bart._mcmc_state.forest.p_nonterminal
    max_depth = tree_depth(p_nonterminal)
    indices = 2 ** jnp.arange(max_depth - 1)
    p_nonterminal = p_nonterminal[indices]

    # sample from prior
    prior_trees = sample_prior(
        keys.pop(),
        kw['ndpost'],
        kw['ntree'],
        bart._mcmc_state.forest.max_split,
        p_nonterminal,
        jnp.sqrt(bart._mcmc_state.forest.sigma_mu2),
    )
    prior_trace = TraceWithOffset.from_trees_trace(prior_trees, bart.offset)

    # check prior samples
    bad = check_trace(prior_trees, bart._mcmc_state.forest.max_split)
    bad_count = jnp.count_nonzero(bad)
    assert bad_count == 0

    # compare number of stub trees
    nstub_mcmc = count_stub_trees(bart._main_trace.split_tree)
    nstub_prior = count_stub_trees(prior_trace.split_tree)
    rhat_nstub = rhat([nstub_mcmc.squeeze(0), nstub_prior])
    assert rhat_nstub < 1.01

    if (p, nsplits) != (1, 1):
        # all the following are equivalent to nstub in the 1-1 case

        # compare number of "simple" trees
        nsimple_mcmc = count_simple_trees(bart._main_trace.split_tree)
        nsimple_prior = count_simple_trees(prior_trace.split_tree)
        rhat_nsimple = rhat([nsimple_mcmc.squeeze(0), nsimple_prior])
        assert rhat_nsimple < 1.01

        # compare varcount
        varcount_prior = compute_varcount(
            bart._mcmc_state.forest.max_split.size, prior_trace
        )
        rhat_varcount = multivariate_rhat([bart.varcount, varcount_prior])
        if p == 10:
            # varcount is p-dimensional
            assert rhat_varcount < 1.4
        else:
            assert rhat_varcount < 1.05

        # compare number of nodes. since #leaves = 1 + #(internal nodes) in binary
        # trees, I only check #(internal nodes) = sum(varcount).
        sum_varcount_mcmc = bart.varcount.sum(axis=1)
        sum_varcount_prior = varcount_prior.sum(axis=1)
        rhat_sum_varcount = rhat([sum_varcount_mcmc, sum_varcount_prior])
        assert rhat_sum_varcount < 1.01

        # compare imbalance index
        imb_mcmc = avg_imbalance_index(bart._main_trace.split_tree)
        imb_prior = avg_imbalance_index(prior_trace.split_tree)
        rhat_imb = rhat([imb_mcmc.squeeze(0), imb_prior])
        assert rhat_imb < 1.01

        # compare average max tree depth
        maxd_mcmc = avg_max_tree_depth(bart._main_trace.split_tree)
        maxd_prior = avg_max_tree_depth(prior_trace.split_tree)
        rhat_maxd = rhat([maxd_mcmc.squeeze(0), maxd_prior])
        assert rhat_maxd < 1.01

        # compare max tree depth distribution
        dd_mcmc = bart.depth_distr()
        dd_prior = trace_depth_distr(prior_trace.split_tree)
        rhat_dd = multivariate_rhat([dd_mcmc.squeeze(0), dd_prior])
        assert rhat_dd < 1.05

    # compare y
    X = random.randint(keys.pop(), (p, 30), 0, nsplits + 1)
    yhat_mcmc = bart._predict(X)
    yhat_prior = evaluate_trace(prior_trace, X)
    rhat_yhat = multivariate_rhat([yhat_mcmc, yhat_prior])
    assert rhat_yhat < 1.05


def count_stub_trees(
    split_tree: UInt[Array, '*batch_shape ntree 2**(d-1)'],
) -> Int32[Array, '*batch_shape']:
    """Count the number of trees with only a root node."""
    return (~split_tree.any(-1)).sum(-1)


def count_simple_trees(
    split_tree: UInt[Array, '*batch_shape ntree 2**(d-1)'],
) -> Int32[Array, '*batch_shape']:
    """Count the number of trees with 2 layers."""
    return (split_tree.astype(bool).sum(-1) == 1).sum(-1)


@partial(jnp.vectorize, signature='(n,m)->()')
def avg_imbalance_index(
    split_tree: UInt[Array, '*batch_shape ntree 2**(d-1)'],
) -> Float32[Array, '*batch_shape']:
    """Measure average tree imbalance in the forest.

    The imbalance is measured as the standard deviation of the depth of the
    leaves.
    """
    is_leaf = vmap(partial(is_actual_leaf, add_bottom_level=True))(split_tree)
    depths = tree_depths(is_leaf.shape[-1])
    depths = jnp.broadcast_to(depths, is_leaf.shape)
    index = jnp.std(depths, where=is_leaf, axis=-1)
    return index.mean(-1)


@partial(jnp.vectorize, signature='(n,m)->()')
def avg_max_tree_depth(
    split_tree: UInt[Array, '*batch_shape ntree 2**(d-1)'],
) -> Float32[Array, '*batch_shape']:
    """Measure average maximum tree depth in the forest."""
    depth = vmap(tree_actual_depth)(split_tree)
    return depth.mean(-1)


def multivariate_rhat(chains: Real[Any, 'chain sample dim']) -> Float[Array, '']:
    """
    Compute the multivariate Gelman-Rubin R-hat.

    Parameters
    ----------
    chains
        Independent chains of samples of a vector.

    Returns
    -------
    Multivariate R-hat statistic.

    Raises
    ------
    ValueError
        If there are not enough chains or samples.
    """
    chains = jnp.asarray(chains)
    m, n, p = chains.shape

    if m < 2:
        msg = 'Need at least 2 chains'
        raise ValueError(msg)
    if n < 2:
        msg = 'Need at least 2 samples per chain'
        raise ValueError(msg)

    chain_means = jnp.mean(chains, axis=1)

    def compute_chain_cov(chain_samples, chain_mean):
        centered = chain_samples - chain_mean
        return jnp.dot(centered.T, centered) / (n - 1)

    within_chain_covs = vmap(compute_chain_cov)(chains, chain_means)
    W = jnp.mean(within_chain_covs, axis=0)

    overall_mean = jnp.mean(chain_means, axis=0)
    chain_mean_diffs = chain_means - overall_mean
    B = (n / (m - 1)) * jnp.dot(chain_mean_diffs.T, chain_mean_diffs)

    V_hat = ((n - 1) / n) * W + ((m + 1) / (m * n)) * B

    # Add regularization to W for numerical stability
    gershgorin = jnp.max(jnp.sum(jnp.abs(W), axis=1))
    regularization = jnp.finfo(W.dtype).eps * len(W) * gershgorin
    W_reg = W + regularization * jnp.eye(p)

    # Compute max(eigvals(W^-1 V_hat))
    L = jnp.linalg.cholesky(W_reg)
    # Solve L @ L.T @ x = V_hat @ x = λ @ W @ x
    # This is equivalent to solving (L^-1 V_hat L^-T) @ y = λ @ y
    L_1V = solve_triangular(L, V_hat, lower=True)
    L_1VL_T = solve_triangular(L, L_1V.T, lower=True).T
    eigenvals = jnp.linalg.eigvalsh(L_1VL_T)

    return jnp.max(eigenvals)


def rhat(chains: Real[Any, 'chain sample']) -> Float[Array, '']:
    """
    Compute the univariate Gelman-Rubin R-hat.

    Parameters
    ----------
    chains
        Independent chains of samples of a scalar.

    Returns
    -------
    Univariate R-hat statistic.
    """
    chains = jnp.asarray(chains)
    return multivariate_rhat(chains[:, :, None])


def test_rhat(keys):
    """Test the multivariate R-hat implementation."""
    chains, divergent_chains = random.normal(keys.pop(), (2, 2, 1000, 10))
    mean_offset = jnp.arange(len(chains))
    divergent_chains += mean_offset[:, None, None]
    rhat = multivariate_rhat(chains)
    rhat_divergent = multivariate_rhat(divergent_chains)
    assert rhat < 1.02
    assert rhat_divergent > 5


def test_jit(kw):
    """Test that jitting around the whole interface works."""
    # set printevery to None to move all iterations to the inner loop and avoid
    # multiple compilation
    kw.update(printevery=None)

    # do not check trees because the assert breaks abstract tracing
    kw.update(check_trees=False)

    # do not check for splitless variables because it breaks tracing, I check it
    # later in this test
    kw.update(rm_const=None)

    X = kw.pop('x_train')
    y = kw.pop('y_train')
    w = kw.pop('w', None)
    key = kw.pop('seed')

    def task(X, y, w, key):
        bart = gbart(X, y, w=w, **kw, seed=key)
        return bart._mcmc_state, bart.yhat_train

    task_compiled = jax.jit(task)

    state1, pred1 = task(X, y, w, key)
    state2, pred2 = task_compiled(X, y, w, random.clone(key))

    # because the check is disabled for traceability
    assert jnp.all(state1.forest.max_split > 0)
    assert jnp.all(state2.forest.max_split > 0)

    assert_close_matrices(pred1, pred2, rtol=1e-5)


def call_with_timed_interrupt(
    time_to_sigint: float, func: Callable, *args: Any, **kw: Any
):
    """
    Call a function and send SIGINT after a certain time.

    This simulates a user pressing ^C during the function execution.

    Parameters
    ----------
    time_to_sigint
        Time in seconds after which to send SIGINT.
    func
        An arbitrary callable.
    *args
    **kw
        Arguments to pass to `func`.

    Returns
    -------
    result : any
        The return value of `func`.

    Notes
    -----
    This function does not disable the SIGINT timer if `func` returns before
    the signal is triggered. This is intentional to prevent silently ignoring a
    case in which the signal is not triggered while the function is running.
    """
    pid = os.getpid()

    def send_sigint():
        time.sleep(time_to_sigint)
        os.kill(pid, signal.SIGINT)

    timer = threading.Thread(target=send_sigint, daemon=True)
    timer.start()
    return func(*args, **kw)


@pytest.mark.timeout(6)
def test_interrupt(kw):
    """Test that the MCMC can be interrupted with ^C."""
    if kw['printevery'] is None:
        kw['printevery'] = 50
    kw.update(ndpost=0, nskip=10000)
    with pytest.raises(KeyboardInterrupt):
        call_with_timed_interrupt(3, gbart, **kw)


def test_polars(kw):
    """Test passing data as DataFrame and Series."""
    bart = gbart(**kw)
    pred = bart.predict(kw['x_test'])

    kw.update(
        seed=random.clone(kw['seed']),
        x_train=pl.DataFrame(numpy.array(kw['x_train']).T),
        x_test=pl.DataFrame(numpy.array(kw['x_test']).T),
        y_train=pl.Series(numpy.array(kw['y_train'])),
        w=None if kw.get('w') is None else pl.Series(numpy.array(kw['w'])),
    )
    bart2 = gbart(**kw)
    pred2 = bart2.predict(kw['x_test'])

    assert_array_equal(bart.yhat_train, bart2.yhat_train)
    assert_array_equal(bart.sigma, bart2.sigma)
    assert_array_equal(pred, pred2)


def test_data_format_mismatch(kw):
    """Test that passing predictors with mismatched formats raises an error."""
    kw.update(
        x_train=pl.DataFrame(numpy.array(kw['x_train']).T),
        x_test=pl.DataFrame(numpy.array(kw['x_test']).T),
        w=None if kw.get('w') is None else pl.Series(numpy.array(kw['w'])),
    )
    bart = gbart(**kw)
    with pytest.raises(ValueError, match='format mismatch'):
        bart.predict(kw['x_test'].to_numpy().T)


def test_automatic_integer_types(kw):
    """Test that integer variables in the MCMC state have the correct type.

    Some integer variables change type automatically to be as small as possible.
    """
    bart = gbart(**kw)

    def select_type(cond):
        return jnp.uint8 if cond else jnp.uint16

    leaf_indices_type = select_type(kw['maxdepth'] <= 8)
    split_trees_type = X_type = select_type(kw['numcut'] <= 255)
    var_trees_type = select_type(kw['x_train'].shape[0] <= 256)

    assert bart._mcmc_state.forest.var_tree.dtype == var_trees_type
    assert bart._mcmc_state.forest.split_tree.dtype == split_trees_type
    assert bart._mcmc_state.forest.leaf_indices.dtype == leaf_indices_type
    assert bart._mcmc_state.X.dtype == X_type
    assert bart._mcmc_state.forest.max_split.dtype == split_trees_type
