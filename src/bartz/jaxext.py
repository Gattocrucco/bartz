# bartz/src/bartz/jaxext.py
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

import functools

from scipy import special
import jax
from jax import numpy as jnp

def float_type(*args):
    t = jnp.result_type(*args)
    return jnp.sin(jnp.empty(0, t)).dtype

def castto(func, type):
    @functools.wraps(func)
    def newfunc(*args, **kw):
        return func(*args, **kw).astype(type)
    return newfunc

def pure_callback_ufunc(callback, dtype, *args, excluded=None, **kwargs):
    """ version of `jax.pure_callback` that deals correctly with ufuncs,
    see `<https://github.com/google/jax/issues/17187>`_ """
    if excluded is None:
        excluded = ()
    shape = jnp.broadcast_shapes(*(
        a.shape
        for i, a in enumerate(args)
        if i not in excluded
    ))
    ndim = len(shape)
    padded_args = [
        a if i in excluded
        else jnp.expand_dims(a, tuple(range(ndim - a.ndim)))
        for i, a in enumerate(args)
    ]
    result = jax.ShapeDtypeStruct(shape, dtype)
    return jax.pure_callback(callback, result, *padded_args, vectorized=True, **kwargs)

    # TODO when jax solves this, check version and piggyback on original if new

class scipy:

    class special:

        def gammainccinv(a, y):
            a = jnp.asarray(a)
            y = jnp.asarray(y)
            dtype = float_type(a.dtype, y.dtype)
            ufunc = castto(special.gammainccinv, dtype)
            return pure_callback_ufunc(ufunc, dtype, a, y)

    class stats:

        class invgamma:
            
            def ppf(q, a):
                return 1 / scipy.special.gammainccinv(a, q)

@functools.wraps(jax.vmap)
def vmap_nodoc(fun, *args, **kw):
    """
    Version of `jax.vmap` that preserves the docstring of the input function.
    """
    doc = fun.__doc__
    fun = jax.vmap(fun, *args, **kw)
    fun.__doc__ = doc
    return fun

def huge_value(x):
    """
    Return the maximum value that can be stored in `x`.

    Parameters
    ----------
    x : array
        A numerical numpy or jax array.

    Returns
    -------
    maxval : scalar
        The maximum value allowed by `x`'s type (+inf for floats).
    """
    if jnp.issubdtype(x.dtype, jnp.integer):
        return jnp.iinfo(x.dtype).max
    else:
        return jnp.inf

def minimal_unsigned_dtype(max_value):
    """
    Return the smallest unsigned integer dtype that can represent a given
    maximum value.
    """
    if max_value < 2 ** 8:
        return jnp.uint8
    if max_value < 2 ** 16:
        return jnp.uint16
    if max_value < 2 ** 32:
        return jnp.uint32
    return jnp.uint64

def signed_to_unsigned(int_dtype):
    """
    Map a signed integer type to its unsigned counterpart. Unsigned types are
    passed through.
    """
    assert jnp.issubdtype(int_dtype, jnp.integer)
    if jnp.issubdtype(int_dtype, jnp.unsignedinteger):
        return int_dtype
    if int_dtype == jnp.int8:
        return jnp.uint8
    if int_dtype == jnp.int16:
        return jnp.uint16
    if int_dtype == jnp.int32:
        return jnp.uint32
    if int_dtype == jnp.int64:
        return jnp.uint64

def ensure_unsigned(x):
    """
    If x has signed integer type, cast it to the unsigned dtype of the same size.
    """
    return x.astype(signed_to_unsigned(x.dtype))
