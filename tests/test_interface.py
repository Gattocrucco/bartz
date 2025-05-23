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

import jax
import numpy
import polars as pl
import pytest
from jax import numpy as jnp
from jax import random

import bartz

from . import util
from .rbartpackages import BART


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
    return jnp.sum(jnp.cos(2 * jnp.pi / T * x), axis=0)


def gen_w(key, n):
    """Generate a vector of error weights."""
    return jnp.exp(random.uniform(key, (n,), float, -1, 1))


def gen_y(key, X, w):
    """Generate responses given predictors."""
    sigma = 0.1
    error = sigma * random.normal(key, (X.shape[1],))
    if w is not None:
        error *= w
    return f(X) + error


@pytest.fixture(params=[1, 2])
def kw(keys, request):
    """Return a dictionary of keyword arguments for BART."""
    match request.param:
        case 1:
            X = gen_X(keys.pop(), 2, 30, 'continuous')
            y = gen_y(keys.pop(), X, None)
            return dict(
                x_train=X,
                y_train=y,
                w=None,
                ntree=20,
                ndpost=100,
                nskip=50,
                printevery=50,
                usequants=True,
                seed=keys.pop(),
                init_kw=dict(
                    resid_batch_size=None, count_batch_size=None, save_ratios=True
                ),
            )

        case 2:
            X = gen_X(keys.pop(), 2, 30, 'binary')
            w = gen_w(keys.pop(), X.shape[1])
            y = gen_y(keys.pop(), X, w)
            return dict(
                x_train=X,
                y_train=y,
                w=w,
                ntree=20,
                ndpost=100,
                nskip=50,
                printevery=None,
                usequants=False,
                numcut=5,
                seed=keys.pop(),
                init_kw=dict(
                    resid_batch_size=16, count_batch_size=16, save_ratios=False
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
    numpy.testing.assert_array_equal(bart1.yhat_train, bart2.yhat_train[1:])

    kw['keepevery'] = 2
    bart3 = bartz.BART.gbart(**kw)
    yhat_train = bart2.yhat_train[1::2]
    numpy.testing.assert_array_equal(yhat_train, bart3.yhat_train[: len(yhat_train)])


def test_finite(kw):
    """Check all numerical outputs are not inf or nan."""
    # XXX: maybe this could be replaced by a global jax debug option
    bart = bartz.BART.gbart(**kw)
    assert jnp.all(jnp.isfinite(bart.yhat_train))
    assert jnp.all(jnp.isfinite(bart.sigma))


def test_output_shapes(kw):
    """Check the output shapes of all the attributes of BART.gbart."""
    bart = bartz.BART.gbart(x_test=kw['x_train'], **kw)

    ndpost = kw['ndpost']
    nskip = kw['nskip']
    n = kw['y_train'].size

    assert bart.offset.shape == ()
    assert bart.scale.shape == ()
    assert bart.lamda.shape == ()
    assert bart.yhat_train.shape == (ndpost, n)
    assert bart.yhat_test.shape == (ndpost, n)
    assert bart.yhat_train_mean.shape == (n,)
    assert bart.yhat_test_mean.shape == (n,)
    assert bart.sigma.shape == (ndpost,)
    assert bart.first_sigma.shape == (nskip,)


def test_predict(kw):
    """Check that the public BART.gbart.predict method works."""
    bart = bartz.BART.gbart(**kw)
    yhat_train = bart.predict(kw['x_train'])
    numpy.testing.assert_array_equal(bart.yhat_train, yhat_train)


def test_scale_shift(kw):
    """Check self-consistency of rescaling the inputs."""
    bart1 = bartz.BART.gbart(**kw)

    offset = 0.4703189
    scale = 0.5294714
    kw.update(y_train=offset + kw['y_train'] * scale, seed=random.clone(kw['seed']))
    bart2 = bartz.BART.gbart(**kw)

    numpy.testing.assert_allclose(
        bart1.offset, (bart2.offset - offset) / scale, rtol=1e-6, atol=1e-7
    )
    numpy.testing.assert_allclose(bart1.scale, bart2.scale / scale, rtol=1e-6)
    numpy.testing.assert_allclose(bart1.sigest, bart2.sigest / scale, rtol=1e-6)
    numpy.testing.assert_allclose(bart1.lamda, bart2.lamda / scale**2, rtol=1e-6)
    numpy.testing.assert_allclose(
        bart1.yhat_train, (bart2.yhat_train - offset) / scale, atol=1e-5, rtol=1e-5
    )
    numpy.testing.assert_allclose(
        bart1.yhat_train_mean,
        (bart2.yhat_train_mean - offset) / scale,
        atol=1e-5,
        rtol=1e-5,
    )
    util.assert_close_matrices(bart1.sigma, bart2.sigma / scale, rtol=1e-5)
    util.assert_close_matrices(bart1.first_sigma, bart2.first_sigma / scale, rtol=1e-6)


def test_min_points_per_leaf(kw):
    """Check that the limit of at least 5 datapoints per leaves is respected."""
    bart = bartz.BART.gbart(**kw)
    distr = bart._points_per_leaf_distr()
    distr_marg = distr.sum(axis=0)
    assert jnp.all(distr_marg[:5] == 0)
    assert jnp.all(distr_marg[-5:-1] == 0)


def test_residuals_accuracy(kw):
    """Check that running residuals are close to the recomputed final residuals."""
    kw.update(ntree=200, ndpost=1000, nskip=0)
    bart = bartz.BART.gbart(**kw)
    accum_resid, actual_resid = bart._compare_resid()
    util.assert_close_matrices(accum_resid, actual_resid, rtol=1e-4)


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
    assert bart.scale == 1
    assert bart.sigest == 1


def test_one_datapoint(kw):
    """Check automatic data scaling with 1 datapoint."""
    kw = set_num_datapoints(kw, 1)
    bart = bartz.BART.gbart(**kw)
    assert bart.scale == 1
    assert bart.sigest == 1


def test_two_datapoints(kw):
    """Check automatic data scaling with 2 datapoints."""
    kw = set_num_datapoints(kw, 2)
    bart = bartz.BART.gbart(**kw)
    numpy.testing.assert_allclose(bart.sigest, kw['y_train'].std(), rtol=1e-6)


def test_few_datapoints(kw):
    """Check that the trees cannot grow if there are not enough datapoints.

    If there are less than 10 datapoints, it is not possible to satisfy the 5
    points per leaf requirement.
    """
    kw = set_num_datapoints(kw, 9)  # < 2 * 5
    bart = bartz.BART.gbart(**kw)
    assert jnp.all(bart.yhat_train == bart.yhat_train[:, :1])


def test_comparison_rbart(kw, keys):
    """Check bartz.BART gives the same results as R package BART."""
    X = kw.pop('x_train')
    kw.update(
        ntree=2 * X.shape[1],
        nskip=2000,
        ndpost=2000,
        numcut=255,
    )

    kw_BART = dict(
        **kw,
        rm_const=False,
        mc_cores=1,
    )
    kw_BART.pop('init_kw')
    for key in 'w', 'printevery':
        if kw_BART[key] is None:
            kw_BART.pop(key)
    kw_BART['seed'] = random.randint(keys.pop(), (), 0, jnp.uint32(2**31)).item()

    bart = bartz.BART.gbart(X, **kw)
    rbart = BART.mc_gbart(X.T, **kw_BART)

    numpy.testing.assert_allclose(bart.offset, rbart.offset, rtol=1e-6, atol=1e-7)
    # I would check sigest as well, but it's not in the R object despite what
    # the documentation says

    # TODO: instead of this comparison function which I made up without enough
    # care, use the multivariate Gelman-Rubin R-hat, which is something similar
    # but very likely better thought out.
    dist2, rank = mahalanobis_distance2(bart.yhat_train, rbart.yhat_train)
    assert dist2 < rank / 10

    dist2, rank = mahalanobis_distance2(bart.sigma[:, None], rbart.sigma[:, None])
    assert dist2 < rank / 10


def mahalanobis_distance2(x, y):
    """Compute the Mahalanobis squared distance, given observations.

    Parameters
    ----------
    x : array (nobs, dim)
    y : array (nobs, dim)
        The two ``dim``-dimensional variables to compare.

    Returns
    -------
    dist2 : float
        The Mahalanobis squared distance between the means of the two variables.
        The distance is the euclidean distance with metric given by the inverse
        of the sample covariance matrix of the average of `x` and `y`.
    rank : int
        The estimated rank of the covariance matrix.
    """
    avg = (x + y) / 2
    cov = jnp.atleast_2d(jnp.cov(avg, rowvar=False))

    w, O = jnp.linalg.eigh(cov)  # cov = O w O^T
    eps = len(w) * jnp.max(jnp.abs(w)) * jnp.finfo(w.dtype).eps
    nonzero = w > eps
    w = w[nonzero]
    O = O[:, nonzero]
    rank = len(w)

    d = x.mean(0) - y.mean(0)
    Od = O.T @ d
    dist2 = Od @ (Od / w)

    return dist2, rank


def test_jit(keys, kw):
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

    numpy.testing.assert_allclose(pred1[5], pred2[5], rtol=1e-6, atol=1e-6)


def call_with_timed_interrupt(time_to_sigint, func, *args, **kw):
    """
    Call a function and send SIGINT after a certain time.

    This simulates a user pressing ^C during the function execution.

    Parameters
    ----------
    time_to_sigint : float
        Time in seconds after which to send SIGINT.
    func : callable
        An arbitrary callable.
    *args : any
    **kw : any
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
        pytest.skip('With logging disabled, the MCMC cannot be interrupted.')
    kw.update(ndpost=0, nskip=10000)
    with pytest.raises(KeyboardInterrupt):
        call_with_timed_interrupt(3, bartz.BART.gbart, **kw)


def test_polars(kw):
    """Test passing data as DataFrame and Series."""
    bart = bartz.BART.gbart(**kw)
    pred = bart.predict(kw['x_train'])

    kw.update(
        seed=random.clone(kw['seed']),
        x_train=pl.DataFrame(numpy.array(kw['x_train']).T),
        y_train=pl.Series(numpy.array(kw['y_train'])),
        w=None if kw['w'] is None else pl.Series(numpy.array(kw['w'])),
    )
    bart2 = bartz.BART.gbart(**kw)
    pred2 = bart2.predict(kw['x_train'])

    numpy.testing.assert_array_equal(bart.yhat_train, bart2.yhat_train)
    numpy.testing.assert_array_equal(bart.sigma, bart2.sigma)
    numpy.testing.assert_array_equal(pred, pred2)


def test_data_format_mismatch(kw):
    """Test that passing predictors with mismatched formats raises an error."""
    kw.update(
        x_train=pl.DataFrame(numpy.array(kw['x_train']).T),
        w=None if kw['w'] is None else pl.Series(numpy.array(kw['w'])),
    )
    bart = bartz.BART.gbart(**kw)
    with pytest.raises(ValueError):
        bart.predict(kw['x_train'].to_numpy().T)
