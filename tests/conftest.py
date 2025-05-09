# bartz/tests/conftest.py
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
import numpy as np
import jax


@pytest.fixture
def rng(request):
    """A deterministic per-test numpy random number generator"""
    nodeid = request.node.nodeid
    seed = np.array([nodeid], np.bytes_).view(np.uint8)
    return np.random.default_rng(seed)


class ListOfRandomKeys:
    # I could move this into a public jaxext.split function (with a shorter
    # name)

    def __init__(self, key):
        self._keys = list(jax.random.split(key, 128))

    def __len__(self):
        return len(self._keys)

    def pop(self):
        if not self._keys:
            raise IndexError('No more keys available')
        return self._keys.pop()


@pytest.fixture
def keys(rng):
    """
    A deterministic per-test-case list of jax random keys. To use a key, do
    `keys.pop()`. If consumed this way, this list of keys can be safely used by
    multiple fixtures involved in the test case.
    """
    seed = np.array(rng.bytes(4)).view(np.uint32)
    key = jax.random.key(seed)
    key = jax.random.fold_in(key, 0xCC755E92)
    return ListOfRandomKeys(key)
