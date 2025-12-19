# bartz/tests/conftest.py
#
# Copyright (c) 2024-2025, The Bartz Contributors
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

from contextlib import nullcontext
from re import fullmatch

import jax
import numpy as np
import pytest

from bartz import jaxext

jax.config.update('jax_debug_key_reuse', True)
jax.config.update('jax_debug_nans', True)
jax.config.update('jax_debug_infs', True)
jax.config.update('jax_legacy_prng_key', 'error')


@pytest.fixture
def keys(request):
    """
    Return a deterministic per-test-case list of jax random keys.

    To use a key, do `keys.pop()`. If consumed this way, this list of keys can
    be safely used by multiple fixtures involved in the test case.
    """
    nodeid = request.node.nodeid
    # exclude xdist_group suffixes because they are active only under xdist
    match = fullmatch(r'(.+?\.py::.+?(\[.+?\])?)(@.+)?', nodeid)
    nodeid = match.group(1)
    seed = np.array([nodeid], np.bytes_).view(np.uint8)
    rng = np.random.default_rng(seed)
    seed = np.array(rng.bytes(4)).view(np.uint32)
    key = jax.random.key(seed)
    return jaxext.split(key, 128)


def pytest_sessionstart(session: pytest.Session) -> None:
    """Print information before pytest starts."""
    # Get the capture manager plugin
    capman = session.config.pluginmanager.get_plugin('capturemanager')

    # Suspend capturing temporarily
    if capman:
        ctx = capman.global_and_fixture_disabled()
    else:
        ctx = nullcontext()

    with ctx:
        device_kind = jax.devices()[0].device_kind
        print(f'jax default device: {device_kind}')
