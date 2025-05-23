# bartz/tests/test_meta.py
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

"""Test properties of pytest itself or other utilities."""

import jax
import pytest
from jax import random


@pytest.fixture
def keys1(keys):
    """Pass-through the `keys` fixture."""
    return keys


@pytest.fixture
def keys2(keys):
    """Pass-through the `keys` fixture."""
    return keys


def test_random_keys_do_not_depend_on_fixture(keys1, keys2):
    """Check that the `keys` fixture is per-test-case, not per-fixture."""
    assert keys1 is keys2


def test_number_of_random_keys(keys):
    """Check the fixed number of available keys.

    This is here just as reference for the `test_random_keys_are_consumed` test
    below.
    """
    assert len(keys) == 128


@pytest.fixture
def consume_one_key(keys):  # noqa: D103
    return keys.pop()


@pytest.fixture
def consume_another_key(keys):  # noqa: D103
    return keys.pop()


def test_random_keys_are_consumed(consume_one_key, consume_another_key, keys):
    """Check that the random keys in `keys` can't be used more than once."""
    assert len(keys) == 126


def test_debug_key_reuse(keys):
    """Check that the jax debug_key_reuse option works."""
    with pytest.raises(jax.errors.KeyReuseError):
        key = keys.pop()
        random.uniform(key)
        random.uniform(key)


def test_debug_key_reuse_within_jit(keys):
    """Check that the jax debug_key_reuse option works within a jitted function."""

    @jax.jit
    def func(key):
        return random.uniform(key) + random.uniform(key)

    with pytest.raises(jax.errors.KeyReuseError):
        func(keys.pop())
