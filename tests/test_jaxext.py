# bartz/tests/test_jaxext.py
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

import pytest
from jax import numpy as jnp
from jax import random
import numpy

from bartz import jaxext

def test_unique():
    x = jnp.arange(10)[::-1]
    out, length = jaxext.unique(x, x.size, 666)
    numpy.testing.assert_array_equal(jnp.sort(x), out)
    assert out.dtype == x.dtype
    assert length == x.size

def test_unique_short():
    x = jnp.ones(10)
    out, length = jaxext.unique(x, x.size, 666)
    numpy.testing.assert_array_equal([1] + 9 * [666], out)
    assert out.dtype == x.dtype
    assert length == 1

def test_unique_empty_input():
    x = jnp.array([])
    out, length = jaxext.unique(x, 2, 666)
    numpy.testing.assert_array_equal([666, 666], out)
    assert out.dtype == x.dtype
    assert length == 0

def test_unique_empty_output():
    x = jnp.array([1, 1, 1])
    out, length = jaxext.unique(x, 0, 666)
    numpy.testing.assert_array_equal([], out)
    assert out.dtype == x.dtype
    assert length == 0

class TestAutoBatch:

    @pytest.mark.parametrize('target_nbatches', [1, 7])
    @pytest.mark.parametrize('with_margin', [False, True])
    @pytest.mark.parametrize('additional_size', [3, 0])
    def test_batch_size(self, key, target_nbatches, with_margin, additional_size):

        def func(a, b, c):
            return (a * b[:, None]).sum(1), c * b[None, :]

        atomic_batch_size = additional_size + 12
        multiplier = 2
        batch_size = multiplier * atomic_batch_size
        if with_margin:
            batch_size += 1
        size = target_nbatches * multiplier

        key = random.split(key, 3)
        a = random.uniform(key[0], (size, additional_size))
        b = random.uniform(key[1], (size,))
        c = random.uniform(key[2], (5, size))

        assert atomic_batch_size == a.shape[1] + 1 + c.shape[0] + 1 + c.shape[0]

        batch_nbytes = batch_size * a.itemsize
        batched_func = jaxext.autobatch(func, batch_nbytes, (0, 0, 1), (0, 1), return_nbatches=True)
        batched_func_nobatches = jaxext.autobatch(func, batch_nbytes, (0, 0, 1), (0, 1))

        out1 = func(a, b, c)
        out2, nbatches = batched_func(a, b, c)
        out3 = batched_func_nobatches(a, b, c)

        assert nbatches == target_nbatches

        for o2, o3 in zip(out2, out3):
            numpy.testing.assert_array_max_ulp(o2, o3)
        for o1, o2 in zip(out1, out2):
            numpy.testing.assert_array_max_ulp(o1, o2)

    def test_unbatched_arg(self):

        def func(a, b):
            return a + b
        batched_func = jaxext.autobatch(func, 32, (0, None))

        a = jnp.arange(100)
        b = 2

        out1 = func(a, b)
        out2 = batched_func(a, b)

        numpy.testing.assert_array_max_ulp(out1, out2)

    def test_large_batch_warning(self):
        x = jnp.arange(10_000).reshape(10, 1000)
        def f(x):
            return x
        g = jaxext.autobatch(f, 100)
        with pytest.warns(UserWarning, match=' > max_io_nbytes = '):
            g(x)

    def test_empty_values(self):
        x = jnp.empty((10, 0))
        def f(x):
            return x
        g = jaxext.autobatch(f, 100, return_nbatches=True)
        y, nbatches = g(x)
        assert nbatches == 1
        assert jnp.all(y == x)

    def test_zero_size(self):
        x = jnp.empty((0, 10))
        def f(x):
            return x
        g = jaxext.autobatch(f, 100, return_nbatches=True)
        y, nbatches = g(x)
        assert nbatches == 1
        assert jnp.all(y == x)

def test_leaf_dict_repr():
    x = jaxext.LeafDict(a=1)
    assert repr(x) == str(x) == "LeafDict({'a': 1})"
