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

def test_unique_empty():
    x = jnp.array([])
    out, length = jaxext.unique(x, 2, 666)
    numpy.testing.assert_array_equal([666, 666], out)
    assert out.dtype == x.dtype
    assert length == 0
