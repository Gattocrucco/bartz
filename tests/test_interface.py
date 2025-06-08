# bartz/tests/test_interface.py
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

"""Test the BART.gbart simplified user interface."""

import os
import signal
import threading
import time
from collections.abc import Callable
from typing import Any

import jax
import numpy
import polars as pl
import pytest
from jax import numpy as jnp
from jax import random
from jax.scipy.linalg import solve_triangular
from jax.scipy.special import ndtr
from jaxtyping import Array, Float, Key, Real
from numpy.testing import assert_allclose, assert_array_equal

import bartz
from bartz.debug import check_trace, trees_BART_to_bartz
from bartz.jaxext import split
from bartz.mcmcloop import compute_varcount, evaluate_trace

from .rbartpackages import BART
from .util import assert_close_matrices


def gen_X(key, p, n, kind):
    """Generate a matrix of predictors."""
    if kind == 'continuous':
        return random.uniform(key, (p, n), float, -2, 2)
    elif kind == 'binary':
        return random.bernoulli(key, 0.5, (p, n)).astype(float)
    else:
        raise KeyError(kind)


def f(x):
    """Conditional mean of the DGP."""
    T = 2
    p, _ = x.shape
    return jnp.sum(jnp.cos(2 * jnp.pi / T * x), axis=0) / jnp.sqrt(p)


def gen_w(key, n):
    """Generate a vector of error weights."""
    return jnp.exp(random.uniform(key, (n,), float, -1, 1))


def gen_y(key, X, w, kind):
    """Generate responses given predictors."""
    keys = split(key)
    match kind:
        case 'continuous':
            sigma = 0.1
            error = sigma * random.normal(keys.pop(), (X.shape[1],))
            if w is not None:
                error *= w
            return f(X) + error

        case 'probit':
            assert w is None
            _, n = X.shape
            error = random.normal(keys.pop(), (n,))
            prob = ndtr(f(X) + error)
            return random.bernoulli(keys.pop(), prob, (n,))


@pytest.fixture(params=[1, 2, 3])
def kw(keys, request):
    """Return a dictionary of keyword arguments for BART."""
    match request.param:
        # continuous regression with close to default settings
        case 1:
            X = gen_X(keys.pop(), 2, 30, 'continuous')
            Xt = gen_X(keys.pop(), 2, 31, 'continuous')
            y = gen_y(keys.pop(), X, None, 'continuous')
            return dict(
                x_train=X,
                y_train=y,
                x_test=Xt,
                w=None,
                ntree=20,
                ndpost=100,
                nskip=50,
                printevery=50,
                usequants=True,
                numcut=255,
                maxdepth=6,
                seed=keys.pop(),
                init_kw=dict(
                    resid_batch_size=None, count_batch_size=None, save_ratios=True
                ),
            )

        # continuous regression with weird settings
        case 2:
            p = 257  # > 256 to use uint16 for var_trees.
            X = gen_X(keys.pop(), p, 30, 'binary')
            Xt = gen_X(keys.pop(), p, 31, 'binary')
            w = gen_w(keys.pop(), X.shape[1])
            y = gen_y(keys.pop(), X, w, 'continuous')
            return dict(
                x_train=X,
                y_train=y,
                x_test=Xt,
                w=w,
                ntree=20,
                ndpost=100,
                nskip=50,
                printevery=None,
                # with usequants=True, numcut would have no effect
                usequants=False,
                numcut=256,  # > 255 to use uint16 for X and split_trees
                maxdepth=9,  # > 8 to use uint16 for leaf_indices
                seed=keys.pop(),
                init_kw=dict(
                    resid_batch_size=16,
                    count_batch_size=16,
                    save_ratios=False,
                    min_points_per_leaf=None,
                ),
            )

        # binary regression
        case 3:
            X = gen_X(keys.pop(), 2, 30, 'continuous')
            Xt = gen_X(keys.pop(), 2, 31, 'continuous')
            y = gen_y(keys.pop(), X, None, 'probit')
            return dict(
                x_train=X,
                x_test=Xt,
                y_train=y,
                type='pbart',
                w=None,
                ntree=20,
                ndpost=100,
                nskip=50,
                printevery=50,
                usequants=True,
                numcut=255,
                maxdepth=6,
                seed=keys.pop(),
                init_kw=dict(
                    resid_batch_size=None, count_batch_size=None, save_ratios=True
                ),
            )


def test_bad_trees(kw):
    """Check all trees produced by the MCMC respect the tree format."""
    bart = bartz.BART.gbart(**kw)
    bad = bart._check_trees()
    bad_count = jnp.count_nonzero(bad)
    assert bad_count == 0


def test_sequential_guarantee(kw):
    """Check that the way iterations are saved does not influence the result."""
    bart1 = bartz.BART.gbart(**kw)

    kw['seed'] = random.clone(kw['seed'])

    kw['nskip'] -= 1
    kw['ndpost'] += 1
    bart2 = bartz.BART.gbart(**kw)
    assert_array_equal(bart1.yhat_train, bart2.yhat_train[1:])

    kw['keepevery'] = 2
    bart3 = bartz.BART.gbart(**kw)
    yhat_train = bart2.yhat_train[1::2]
    assert_array_equal(yhat_train, bart3.yhat_train[: len(yhat_train)])


def test_output_shapes(kw):
    """Check the output shapes of all the attributes of BART.gbart."""
    bart = bartz.BART.gbart(**kw)

    ndpost = kw['ndpost']
    nskip = kw['nskip']
    p, n = kw['x_train'].shape
    _, m = kw['x_test'].shape

    assert bart.yhat_train.shape == (ndpost, n)
    assert bart.yhat_train_mean.shape == (n,)
    assert bart.yhat_test.shape == (ndpost, m)
    assert bart.yhat_test_mean.shape == (m,)
    if kw['y_train'].dtype == bool:
        assert bart.sigma is None
        assert bart.first_sigma is None
    else:
        assert bart.sigma.shape == (ndpost,)
        assert bart.first_sigma.shape == (nskip,)
    assert bart.offset.shape == ()
    assert bart.varcount.shape == (ndpost, p)
    assert bart.varcount_mean.shape == (p,)


def test_predict(kw):
    """Check that the public BART.gbart.predict method works."""
    bart = bartz.BART.gbart(**kw)
    yhat_train = bart.predict(kw['x_train'])
    assert_array_equal(bart.yhat_train, yhat_train)


def test_scale_shift(kw):
    """Check self-consistency of rescaling the inputs."""
    if kw['y_train'].dtype == bool:
        pytest.skip('Cannot rescale binary responses.')

    bart1 = bartz.BART.gbart(**kw)

    offset = 0.4703189
    scale = 0.5294714
    kw.update(y_train=offset + kw['y_train'] * scale, seed=random.clone(kw['seed']))
    # note: using the same seed does not guarantee stable error because the mcmc
    # makes discrete choices based on thresholds on floats, so numerical error
    # can be amplified.
    bart2 = bartz.BART.gbart(**kw)

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
    assert_close_matrices(bart1.sigma, bart2.sigma / scale, rtol=1e-5)
    assert_close_matrices(bart1.first_sigma, bart2.first_sigma / scale, rtol=1e-6)


def test_min_points_per_leaf(kw):
    """Check that the limit of at least 5 datapoints per leaf is respected."""
    bart = bartz.BART.gbart(**kw)
    distr = bart._points_per_leaf_distr()
    distr_marg = distr.sum(axis=0)
    limit_inactive = kw['init_kw'].get('min_points_per_leaf', 5) is None
    assert jnp.all(distr_marg[:5] == 0) == (not limit_inactive)
    assert limit_inactive or jnp.all(distr_marg[-5:-1] == 0)


def test_residuals_accuracy(kw):
    """Check that running residuals are close to the recomputed final residuals."""
    kw.update(ntree=200, ndpost=1000, nskip=0)
    bart = bartz.BART.gbart(**kw)
    accum_resid, actual_resid = bart._compare_resid()
    assert_close_matrices(accum_resid, actual_resid, rtol=1e-4)


def set_num_datapoints(kw, n):
    """Set the number of datapoints in the kw dictionary."""
    assert n <= kw['y_train'].size
    kw = kw.copy()
    kw['x_train'] = kw['x_train'][:, :n]
    kw['y_train'] = kw['y_train'][:n]
    if kw['w'] is not None:
        kw['w'] = kw['w'][:n]
    return kw


def test_no_datapoints(kw):
    """Check automatic data scaling with 0 datapoints."""
    kw = set_num_datapoints(kw, 0)
    kw.update(usequants=True)  # because the uniform grid requires endpoints
    bart = bartz.BART.gbart(**kw)
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
    bart = bartz.BART.gbart(**kw)
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
    bart = bartz.BART.gbart(**kw)
    if kw['y_train'].dtype != bool:
        assert_allclose(bart.sigest, kw['y_train'].std(), rtol=1e-6)


def test_few_datapoints(kw):
    """Check that the trees cannot grow if there are not enough datapoints.

    If there are less than 10 datapoints, it is not possible to satisfy the 5
    points per leaf requirement.
    """
    kw.setdefault('init_kw', {}).update(min_points_per_leaf=5)
    kw = set_num_datapoints(kw, 9)  # < 2 * 5
    bart = bartz.BART.gbart(**kw)
    assert jnp.all(bart.yhat_train == bart.yhat_train[:, :1])


def kw_bartz_to_BART(key: Key[Array, ''], kw: dict, bart: bartz.BART.gbart) -> dict:
    """Convert bartz keyword arguments to R BART keyword arguments."""
    kw_BART = dict(
        **kw,
        rm_const=False,
        mc_cores=1,
    )
    kw_BART.pop('init_kw')
    kw_BART.pop('maxdepth', None)
    for arg in 'w', 'printevery':
        if kw_BART[arg] is None:
            kw_BART.pop(arg)
    kw_BART['seed'] = random.randint(key, (), 0, jnp.uint32(2**31)).item()

    # Set BART cutpoints manually. This means I am not checking that the
    # automatic cutpoint determination of BART is the same of my package. They
    # are similar but have some differences, and having exactly the same
    # cutpoints is more important for the test.
    kw_BART['transposed'] = True  # this disables predictors pre-processing
    kw_BART['numcut'] = bart._mcmc_state.max_split
    kw_BART['xinfo'] = bart._splits

    return kw_BART


def test_rbart(kw, keys):
    """Check bartz.BART gives the same results as R package BART."""
    p, n = kw['x_train'].shape
    kw.update(
        ntree=max(2 * n, p),
        nskip=3000,
        ndpost=1000,
    )
    # R BART can't change the min_points_per_leaf per leaf setting
    kw['init_kw'].update(min_points_per_leaf=5)

    # run bart with both packages
    bart = bartz.BART.gbart(**kw)
    kw_BART = kw_bartz_to_BART(keys.pop(), kw, bart)
    rbart = BART.mc_gbart(**kw_BART)

    # first cross-check the outputs of R BART alone

    # convert the trees to bartz format
    trees: str = rbart.treedraws['trees'].item()
    trace, meta = trees_BART_to_bartz(trees, offset=rbart.offset)

    varcount = compute_varcount(meta.p, trace)
    assert jnp.all(varcount == rbart.varcount)

    yhat_train = evaluate_trace(trace, bart._mcmc_state.X)
    assert_close_matrices(yhat_train, rbart.yhat_train, rtol=1e-6)

    Xt = bart._bin_predictors(kw['x_test'], bart._splits)
    yhat_test = evaluate_trace(trace, Xt)
    assert_close_matrices(yhat_test, rbart.yhat_test, rtol=1e-6)

    bad = check_trace(trace, bart._mcmc_state.max_split)
    num_bad = jnp.count_nonzero(bad)
    assert num_bad == 0

    # compare results

    assert_allclose(bart.offset, rbart.offset, rtol=1e-6, atol=1e-7)
    # I would check sigest as well, but it's not in the R object despite what
    # the documentation says

    rhat_yhat_train = multivariate_rhat(jnp.stack([bart.yhat_train, rbart.yhat_train]))
    assert rhat_yhat_train < 1.2

    if kw['y_train'].dtype != bool:
        rhat_sigma = multivariate_rhat(
            jnp.stack([bart.sigma[:, None], rbart.sigma[:, None]])
        )
        assert rhat_sigma < 1.05

    if p < n:
        # skip if p is large because it would be difficult for the MCMC to get
        # stuff about predictors right
        rhat_varcount = multivariate_rhat(jnp.stack([bart.varcount, rbart.varcount]))
        assert rhat_varcount < 100  # TODO !!! # noqa: FIX002, will take a while
        # loose criterion as a patch
        assert_allclose(bart.varcount_mean, rbart.varcount_mean, rtol=0.15)


def multivariate_rhat(chains: Real[Array, 'chain sample dim']) -> Float[Array, '']:
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

    within_chain_covs = jax.vmap(compute_chain_cov)(chains, chain_means)
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
    kw.update(printevery=None)
    # set printevery to None to move all iterations to the inner loop and avoid
    # multiple compilation

    X = kw.pop('x_train')
    y = kw.pop('y_train')
    w = kw.pop('w')
    key = kw.pop('seed')

    def task(X, y, w, key):
        bart = bartz.BART.gbart(X, y, w=w, **kw, seed=key)
        return bart._mcmc_state, bart.yhat_train

    task_compiled = jax.jit(task)

    state1, pred1 = task(X, y, w, key)
    state2, pred2 = task_compiled(X, y, w, random.clone(key))

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
        call_with_timed_interrupt(3, bartz.BART.gbart, **kw)


def test_polars(kw):
    """Test passing data as DataFrame and Series."""
    bart = bartz.BART.gbart(**kw)
    pred = bart.predict(kw['x_test'])

    kw.update(
        seed=random.clone(kw['seed']),
        x_train=pl.DataFrame(numpy.array(kw['x_train']).T),
        x_test=pl.DataFrame(numpy.array(kw['x_test']).T),
        y_train=pl.Series(numpy.array(kw['y_train'])),
        w=None if kw['w'] is None else pl.Series(numpy.array(kw['w'])),
    )
    bart2 = bartz.BART.gbart(**kw)
    pred2 = bart2.predict(kw['x_test'])

    assert_array_equal(bart.yhat_train, bart2.yhat_train)
    assert_array_equal(bart.sigma, bart2.sigma)
    assert_array_equal(pred, pred2)


def test_data_format_mismatch(kw):
    """Test that passing predictors with mismatched formats raises an error."""
    kw.update(
        x_train=pl.DataFrame(numpy.array(kw['x_train']).T),
        x_test=pl.DataFrame(numpy.array(kw['x_test']).T),
        w=None if kw['w'] is None else pl.Series(numpy.array(kw['w'])),
    )
    bart = bartz.BART.gbart(**kw)
    with pytest.raises(ValueError, match='format mismatch'):
        bart.predict(kw['x_test'].to_numpy().T)


def test_automatic_integer_types(kw):
    """Test that integer variables in the MCMC state have the correct type.

    Some integer variables change type automatically to be as small as possible.
    """
    bart = bartz.BART.gbart(**kw)

    def select_type(cond):
        return jnp.uint8 if cond else jnp.uint16

    leaf_indices_type = select_type(kw['maxdepth'] <= 8)
    split_trees_type = X_type = select_type(kw['numcut'] <= 255)
    var_trees_type = select_type(kw['x_train'].shape[0] <= 256)

    assert bart._mcmc_state.forest.var_tree.dtype == var_trees_type
    assert bart._mcmc_state.forest.split_tree.dtype == split_trees_type
    assert bart._mcmc_state.forest.leaf_indices.dtype == leaf_indices_type
    assert bart._mcmc_state.X.dtype == X_type
    assert bart._mcmc_state.max_split.dtype == split_trees_type
