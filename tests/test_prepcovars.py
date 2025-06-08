# bartz/tests/test_prepcovars.py
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

"""Test the `bartz.prepcovars` module."""

import numpy
import pytest
from jax import debug_infs
from jax import numpy as jnp

import bartz


class TestQuantilizer:
    """Test `prepcovars.quantilized_splits_from_matrix`."""

    @pytest.mark.parametrize('fill_value', [jnp.inf, 2**31 - 1])
    def test_splits_fill(self, fill_value):
        """Check that predictors with less unique values are right-padded."""
        with debug_infs(not jnp.isinf(fill_value)):
            fill_value = jnp.array(fill_value)
            x = jnp.array([[1, 3, 3, 5], [1, 3, 5, 7]], fill_value.dtype)
            splits, _ = bartz.prepcovars.quantilized_splits_from_matrix(x, 100)
        expected_splits = [[2, 4, fill_value], [2, 4, 6]]
        numpy.testing.assert_array_equal(splits, expected_splits)

    def test_max_splits(self):
        """Check that the number of splits per predictor is counted correctly."""
        x = jnp.array(
            [
                [1, 1, 1, 1],
                [4, 4, 1, 1],
                [2, 1, 3, 2],
                [1, 4, 2, 3],
            ]
        )
        _, max_split = bartz.prepcovars.quantilized_splits_from_matrix(x, 100)
        numpy.testing.assert_array_equal(max_split, jnp.arange(4))

    def test_integer_splits_overflow(self):
        """Check that the splits are computed correctly at the limit of overflow."""
        x = jnp.array([[-(2**31), 2**31 - 2]])
        splits, _ = bartz.prepcovars.quantilized_splits_from_matrix(x, 100)
        expected_splits = [[-1]]
        numpy.testing.assert_array_equal(splits, expected_splits)

    @pytest.mark.parametrize('dtype', [int, float])
    def test_splits_type(self, dtype):
        """Check that the input type is preserved."""
        x = jnp.arange(10, dtype=dtype)[None, :]
        splits, _ = bartz.prepcovars.quantilized_splits_from_matrix(x, 100)
        assert splits.dtype == x.dtype

    def test_splits_length(self):
        """Check that the correct number of splits is returned in corner cases."""
        x = jnp.linspace(0, 1, 10)[None, :]

        short_splits, _ = bartz.prepcovars.quantilized_splits_from_matrix(x, 2)
        assert short_splits.shape == (1, 1)

        long_splits, _ = bartz.prepcovars.quantilized_splits_from_matrix(x, 100)
        assert long_splits.shape == (1, 9)

        just_right_splits, _ = bartz.prepcovars.quantilized_splits_from_matrix(x, 10)
        assert just_right_splits.shape == (1, 9)

        no_splits, _ = bartz.prepcovars.quantilized_splits_from_matrix(x, 1)
        assert no_splits.shape == (1, 0)

    def test_round_trip(self):
        """Check that `bin_predictors` is the ~inverse of `quantilized_splits_from_matrix`."""
        x = jnp.arange(10)[None, :]
        splits, _ = bartz.prepcovars.quantilized_splits_from_matrix(x, 100)
        b = bartz.prepcovars.bin_predictors(x, splits)
        numpy.testing.assert_array_equal(x, b)


def test_binner_left_boundary():
    """Check that the first bin is right-closed."""
    splits = jnp.array([[1, 2, 3]])

    x = jnp.array([[0, 1]])
    b = bartz.prepcovars.bin_predictors(x, splits)
    numpy.testing.assert_array_equal(b, [[0, 0]])


def test_binner_right_boundary():
    """Check that the next-to-last bin is right-closed."""
    splits = jnp.array([[1, 2, 3, 2**31 - 1]])

    x = jnp.array([[2**31 - 1]])
    b = bartz.prepcovars.bin_predictors(x, splits)
    numpy.testing.assert_array_equal(b, [[3]])
