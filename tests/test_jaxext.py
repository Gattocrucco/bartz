# bartz/tests/test_jaxext.py
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

"""Test bartz.jaxext."""

import numpy
import pytest
from jax import numpy as jnp
from jax import random

from bartz import jaxext


class TestUnique:
    """Test jaxext.unique."""

    def test_sort(self):
        """Check that it's equivalent to sort if no values are repeated."""
        x = jnp.arange(10)[::-1]
        out, length = jaxext.unique(x, x.size, 666)
        numpy.testing.assert_array_equal(jnp.sort(x), out)
        assert out.dtype == x.dtype
        assert length == x.size

    def test_fill(self):
        """Check that the trailing fill value is used correctly."""
        x = jnp.ones(10)
        out, length = jaxext.unique(x, x.size, 666)
        numpy.testing.assert_array_equal([1] + 9 * [666], out)
        assert out.dtype == x.dtype
        assert length == 1

    def test_empty_input(self):
        """Check that the function works on empty input."""
        x = jnp.array([])
        out, length = jaxext.unique(x, 2, 666)
        numpy.testing.assert_array_equal([666, 666], out)
        assert out.dtype == x.dtype
        assert length == 0

    def test_empty_output(self):
        """Check that the function works if the output is forced to be empty."""
        x = jnp.array([1, 1, 1])
        out, length = jaxext.unique(x, 0, 666)
        numpy.testing.assert_array_equal([], out)
        assert out.dtype == x.dtype
        assert length == 0


class TestAutoBatch:
    """Test jaxext.autobatch."""

    @pytest.mark.parametrize('target_nbatches', [1, 7])
    @pytest.mark.parametrize('with_margin', [False, True])
    @pytest.mark.parametrize('additional_size', [3, 0])
    def test_batch_size(self, keys, target_nbatches, with_margin, additional_size):
        """Check batch sizes are correct in various conditions."""

        def func(a, b, c):
            return (a * b[:, None]).sum(1), c * b[None, :]

        atomic_batch_size = additional_size + 12
        multiplier = 2
        batch_size = multiplier * atomic_batch_size
        if with_margin:
            batch_size += 1
        size = target_nbatches * multiplier

        a = random.uniform(keys.pop(), (size, additional_size))
        b = random.uniform(keys.pop(), (size,))
        c = random.uniform(keys.pop(), (5, size))

        assert atomic_batch_size == a.shape[1] + 1 + c.shape[0] + 1 + c.shape[0]

        batch_nbytes = batch_size * a.itemsize
        batched_func = jaxext.autobatch(
            func, batch_nbytes, (0, 0, 1), (0, 1), return_nbatches=True
        )
        batched_func_nobatches = jaxext.autobatch(func, batch_nbytes, (0, 0, 1), (0, 1))

        out1 = func(a, b, c)
        out2, nbatches = batched_func(a, b, c)
        out3 = batched_func_nobatches(a, b, c)

        assert nbatches == target_nbatches

        for o2, o3 in zip(out2, out3, strict=True):
            numpy.testing.assert_array_max_ulp(o2, o3)
        for o1, o2 in zip(out1, out2, strict=True):
            numpy.testing.assert_array_max_ulp(o1, o2)

    def test_unbatched_arg(self):
        """Check the function with batching disabled on a scalar argument."""

        def func(a, b):
            return a + b

        batched_func = jaxext.autobatch(func, 32, (0, None))

        a = jnp.arange(100)
        b = 2

        out1 = func(a, b)
        out2 = batched_func(a, b)

        numpy.testing.assert_array_max_ulp(out1, out2)

    def test_large_batch_warning(self):
        """Check the function emits a warning if the size limit can't be honored."""
        x = jnp.arange(10_000).reshape(10, 1000)

        def f(x):
            return x

        g = jaxext.autobatch(f, 100)
        with pytest.warns(UserWarning, match=' > max_io_nbytes = '):
            g(x)

    def test_empty_values(self):
        """Check that the function works with batchable empty arrays."""
        x = jnp.empty((10, 0))

        def f(x):
            return x

        g = jaxext.autobatch(f, 100, return_nbatches=True)
        y, nbatches = g(x)
        assert nbatches == 1
        assert jnp.all(y == x)

    def test_zero_size(self):
        """Check the function works with a batch axis with length 0."""
        x = jnp.empty((0, 10))

        def f(x):
            return x

        g = jaxext.autobatch(f, 100, return_nbatches=True)
        y, nbatches = g(x)
        assert nbatches == 1
        assert jnp.all(y == x)


def different_keys(keya, keyb):
    """Return True iff two jax random keys are different."""
    return jnp.any(random.key_data(keya) != random.key_data(keyb)).item()


def test_split(keys):
    """Test jaxext.split."""
    key = keys.pop()
    ks = jaxext.split(key, 3)

    assert len(ks) == 3
    key1 = ks.pop()
    assert len(ks) == 2
    key2 = ks.pop()
    assert len(ks) == 1
    key3 = ks.pop()
    assert len(ks) == 0

    with pytest.raises(IndexError):
        ks.pop()

    assert different_keys(key, key1)
    assert different_keys(key, key2)
    assert different_keys(key, key3)
    assert different_keys(key1, key2)
    assert different_keys(key1, key3)
    assert different_keys(key2, key3)

    ks = jaxext.split(random.clone(key), 3)
    key1a = ks.pop()
    key23 = ks.pop(2)

    assert not different_keys(key1, key1a)
    assert not different_keys(key2, key23[0])
    assert not different_keys(key3, key23[1])
