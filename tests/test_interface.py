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

import pytest
import jax
from jax import numpy as jnp
from jax import random
import numpy

import bartz

from .rbartpackages import BART
from . import util

@pytest.fixture
def n():
    return 30

@pytest.fixture
def p():
    return 2

def gen_X(key, p, n, kind):
    if kind == 'continuous':
        return random.uniform(key, (p, n), float, -2, 2)
    elif kind == 'binary':
        return random.bernoulli(key, 0.5, (p, n)).astype(float)
    else:
        raise KeyError(kind)

@pytest.fixture
def X_continuous(n, p, key):
    key = random.fold_in(key, 0xd9b0963d)
    return gen_X(key, p, n, 'continuous')

@pytest.fixture
def X_binary(n, p, key):
    key = random.fold_in(key, 0x1a7c4e8d)
    return gen_X(key, p, n, 'binary')

@pytest.fixture(params=['X_continuous', 'X_binary'])
def X(request, X_continuous, X_binary):
    return eval(request.param)

def f(x): # conditional mean
    T = 2
    return jnp.sum(jnp.cos(2 * jnp.pi / T * x), axis=0)

def gen_y(key, X):
    sigma = 0.1
    return f(X) + sigma * random.normal(key, (X.shape[1],))

@pytest.fixture
def y(X, key):
    key = random.fold_in(key, 0x1391bc96)
    return gen_y(key, X)

@pytest.fixture
def kw():
    return dict(ntree=20, ndpost=100, nskip=50, usequants=True)

def test_bad_trees(X, y, key, kw):
    bart = bartz.BART.gbart(X, y, **kw, seed=key)
    bad = bart._check_trees()
    bad_count = jnp.count_nonzero(bad)
    assert bad_count == 0

def test_sequential_guarantee(X, y, key, kw):
    bart1 = bartz.BART.gbart(X, y, **kw, seed=key)

    kw['nskip'] -= 1
    kw['ndpost'] += 1
    bart2 = bartz.BART.gbart(X, y, **kw, seed=key)

    numpy.testing.assert_array_equal(bart1.yhat_train, bart2.yhat_train[1:])

    kw['keepevery'] = 2
    bart3 = bartz.BART.gbart(X, y, **kw, seed=key)
    yhat_train = bart2.yhat_train[1::2]
    numpy.testing.assert_array_equal(yhat_train, bart3.yhat_train[:len(yhat_train)])

def test_finite(X, y, key, kw):
    bart = bartz.BART.gbart(X, y, **kw, seed=key)
    assert jnp.all(jnp.isfinite(bart.yhat_train))
    assert jnp.all(jnp.isfinite(bart.sigma))

def test_output_shapes(X, y, key, kw):
    bart = bartz.BART.gbart(X, y, x_test=X, **kw, seed=key)

    ndpost = kw['ndpost']
    nskip = kw['nskip']
    _, n = X.shape

    assert bart.offset.shape == ()
    assert bart.scale.shape == ()
    assert bart.lamda.shape == ()
    assert bart.yhat_train.shape == (ndpost, n)
    assert bart.yhat_test.shape == (ndpost, n)
    assert bart.yhat_train_mean.shape == (n,)
    assert bart.yhat_test_mean.shape == (n,)
    assert bart.sigma.shape == (ndpost,)
    assert bart.first_sigma.shape == (nskip,)

def test_predict(X, y, key, kw):
    bart = bartz.BART.gbart(X, y, **kw, seed=key)
    yhat_train = bart.predict(X)
    numpy.testing.assert_array_equal(bart.yhat_train, yhat_train)

def test_scale_shift(X, y, key, kw):
    bart1 = bartz.BART.gbart(X, y, **kw, seed=key)

    offset = 0.4703189
    scale = 0.5294714
    bart2 = bartz.BART.gbart(X, offset + y * scale, **kw, seed=key)

    numpy.testing.assert_allclose(bart1.offset, (bart2.offset - offset) / scale, rtol=1e-6)
    numpy.testing.assert_allclose(bart1.scale, bart2.scale / scale, rtol=1e-6)
    numpy.testing.assert_allclose(bart1.sigest, bart2.sigest / scale, atol=1e-7)
    numpy.testing.assert_allclose(bart1.lamda, bart2.lamda / scale ** 2, rtol=1e-6)
    numpy.testing.assert_allclose(bart1.yhat_train, (bart2.yhat_train - offset) / scale, atol=1e-5, rtol=1e-5)
    numpy.testing.assert_allclose(bart1.yhat_train_mean, (bart2.yhat_train_mean - offset) / scale, atol=1e-5, rtol=1e-5)
    util.assert_close_matrices(bart1.sigma, bart2.sigma / scale, rtol=1e-5)
    util.assert_close_matrices(bart1.first_sigma, bart2.first_sigma / scale, rtol=1e-6)

def test_min_points_per_leaf(X, y, key, kw):
    bart = bartz.BART.gbart(X, y, **kw, seed=key)
    distr = bart._points_per_leaf_distr()
    distr_marg = distr.sum(axis=0)
    assert jnp.all(distr_marg[:5] == 0)
    assert jnp.all(distr_marg[-5:-1] == 0)

def test_residuals_accuracy(key):
    key1, key2, key3 = random.split(key, 3)
    n = 100
    p = 1
    X = gen_X(key1, p, n, 'continuous')
    y = gen_y(key2, X)
    bart = bartz.BART.gbart(X, y, ntree=200, ndpost=1000, nskip=0, seed=key3)
    accum_resid, actual_resid = bart._compare_resid()
    util.assert_close_matrices(accum_resid, actual_resid, rtol=1e-4)

def test_no_datapoints(X, y, kw, key):
    X = X[:, :0]
    y = y[:0]
    bart = bartz.BART.gbart(X, y, **kw, seed=key)
    ndpost = kw['ndpost']
    assert bart.yhat_train.shape == (ndpost, 0)
    assert bart.offset == 0
    assert bart.scale == 1
    assert bart.sigest == 1

def test_one_datapoint(X, y, kw, key):
    X = X[:, :1]
    y = y[:1]
    bart = bartz.BART.gbart(X, y, **kw, seed=key)
    ndpost = kw['ndpost']
    assert bart.scale == 1
    assert bart.sigest == 1

def test_two_datapoints(X, y, kw, key):
    X = X[:, :2]
    y = y[:2]
    bart = bartz.BART.gbart(X, y, **kw, seed=key)
    ndpost = kw['ndpost']
    numpy.testing.assert_allclose(bart.sigest, y.std())

def test_few_datapoints(X, y, kw, key):
    X = X[:, :9] # < 2 * 5
    y = y[:9]
    bart = bartz.BART.gbart(X, y, **kw, seed=key)
    assert jnp.all(bart.yhat_train == bart.yhat_train[:, :1])

@pytest.mark.parametrize('kw_shared,initkw', [
    [dict(usequants=True), dict(resid_batch_size=None, count_batch_size=None, save_ratios=True)],
    [dict(usequants=False, numcut=5), dict(resid_batch_size=16, count_batch_size=16, save_ratios=False)],
])
def test_comparison_rbart(X, y, key, kw_shared, initkw):
    key1, key2 = random.split(key, 2)

    kw = dict(
        ntree=2 * X.shape[1],
        nskip=1000,
        ndpost=1000,
        numcut=255,
    )
    kw.update(kw_shared)

    kw_bartz = dict(**kw, initkw=initkw)

    kw_BART = dict(
        **kw,
        rm_const=False,
        mc_cores=1,
    )

    bart = bartz.BART.gbart(X, y, **kw_bartz, seed=key1)
    seed = random.randint(key2, (), 0, jnp.uint32(2 ** 31)).item()
    rbart = BART.mc_gbart(X.T, y, **kw_BART, seed=seed)

    numpy.testing.assert_allclose(bart.offset, rbart.offset, rtol=1e-6, atol=1e-7)

    dist2, rank = mahalanobis_distance2(bart.yhat_train, rbart.yhat_train)
    assert dist2 < rank / 10

    dist2, rank = mahalanobis_distance2(bart.sigma[:, None], rbart.sigma[:, None])
    assert dist2 < rank / 10

def mahalanobis_distance2(x, y):
    avg = (x + y) / 2
    cov = jnp.atleast_2d(jnp.cov(avg, rowvar=False))

    w, O = jnp.linalg.eigh(cov) # cov = O w O^T
    eps = len(w) * jnp.max(jnp.abs(w)) * jnp.finfo(w.dtype).eps
    nonzero = w > eps
    w = w[nonzero]
    O = O[:, nonzero]
    rank = len(w)

    d = x.mean(0) - y.mean(0)
    Od = O.T @ d
    dist2 = Od @ (Od / w)

    return dist2, rank

def test_jit(X, y, key, kw):
    """ test that jitting around the whole interface works """

    def task(X, y, key):
        bart = bartz.BART.gbart(X, y, **kw, seed=key)
        return bart._mcmc_state, bart.yhat_train
    task_compiled = jax.jit(task)

    state1, pred1 = task(X, y, key)
    state2, pred2 = task_compiled(X, y, key)

    numpy.testing.assert_allclose(pred1[5], pred2[5], rtol=0, atol=1e-6)
