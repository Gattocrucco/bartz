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

"""Pytest configuration."""

import jax
import numpy as np
import pytest

from bartz import jaxext

# XXX: does this option test key reuse within the MCMC loop? Maybe it doesn't
# because it's always run compiled as a whole. If so, maybe that could be
# circumvented by a single short run with jit disabled. But would the option
# implementation do something in that case? (The docs say it checks on jit
# boundaries.) If that worked, I would disable this globally and enable it only
# on the test case that disables the jit.
jax.config.update('jax_debug_key_reuse', True)


# XXX: I'm not using this any more. Is there a jax api to convert an array of
# bytes into a seed?
@pytest.fixture
def rng(request):
    """Return a deterministic per-test numpy random number generator."""
    nodeid = request.node.nodeid
    seed = np.array([nodeid], np.bytes_).view(np.uint8)
    return np.random.default_rng(seed)


@pytest.fixture
def keys(rng):
    """
    Return a deterministic per-test-case list of jax random keys.

    To use a key, do `keys.pop()`. If consumed this way, this list of keys can
    be safely used by multiple fixtures involved in the test case.
    """
    seed = np.array(rng.bytes(4)).view(np.uint32)
    key = jax.random.key(seed)
    key = jax.random.fold_in(key, 0xCC755E92)
    return jaxext.split(key, 128)
