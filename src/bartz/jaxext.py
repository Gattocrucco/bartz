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
import math
import warnings

from scipy import special
import jax
from jax import numpy as jnp
from jax import tree_util
from jax import lax

def float_type(*args):
    """
    Determine the jax floating point result type given operands/types.
    """
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

        @functools.wraps(special.gammainccinv)
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
    maximum value (inclusive).
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

@functools.partial(jax.jit, static_argnums=(1,))
def unique(x, size, fill_value):
    """
    Restricted version of `jax.numpy.unique` that uses less memory.

    Parameters
    ----------
    x : 1d array
        The input array.
    size : int
        The length of the output.
    fill_value : scalar
        The value to fill the output with if `size` is greater than the number
        of unique values in `x`.

    Returns
    -------
    out : array (size,)
        The unique values in `x`, sorted, and right-padded with `fill_value`.
    actual_length : int
        The number of used values in `out`.
    """
    if x.size == 0:
        return jnp.full(size, fill_value, x.dtype), 0
    if size == 0:
        return jnp.empty(0, x.dtype), 0
    x = jnp.sort(x)
    def loop(carry, x):
        i_out, i_in, last, out = carry
        i_out = jnp.where(x == last, i_out, i_out + 1)
        out = out.at[i_out].set(x)
        return (i_out, i_in + 1, x, out), None
    carry = 0, 0, x[0], jnp.full(size, fill_value, x.dtype)
    (actual_length, _, _, out), _ = jax.lax.scan(loop, carry, x[:size])
    return out, actual_length + 1

def autobatch(func, max_io_nbytes, in_axes=0, out_axes=0, return_nbatches=False):
    """
    Batch a function such that each batch is smaller than a threshold.

    Parameters
    ----------
    func : callable
        A jittable function with positional arguments only, with inputs and
        outputs pytrees of arrays.
    max_io_nbytes : int
        The maximum number of input + output bytes in each batch.
    in_axes : pytree of ints, default 0
        A tree matching the structure of the function input, indicating along
        which axes each array should be batched. If a single integer, it is
        used for all arrays.
    out_axes : pytree of ints, default 0
        The same for outputs.
    return_nbatches : bool, default False
        If True, the number of batches is returned as a second output.

    Returns
    -------
    batched_func : callable
        A function with the same signature as `func`, but that processes the
        input and output in batches in a loop.
    """

    def expand_axes(axes, tree):
        if isinstance(axes, int):
            return tree_util.tree_map(lambda _: axes, tree)
        return tree_util.tree_map(lambda _, axis: axis, tree, axes)

    def extract_size(axes, tree):
        sizes = tree_util.tree_map(lambda x, axis: x.shape[axis], tree, axes)
        sizes, _ = tree_util.tree_flatten(sizes)
        assert all(s == sizes[0] for s in sizes)
        return sizes[0]

    def sum_nbytes(tree):
        def nbytes(x):
            return math.prod(x.shape) * x.dtype.itemsize
        return tree_util.tree_reduce(lambda size, x: size + nbytes(x), tree, 0)

    def next_divisor_small(dividend, min_divisor):
        for divisor in range(min_divisor, int(math.sqrt(dividend)) + 1):
            if dividend % divisor == 0:
                return divisor
        return dividend

    def next_divisor_large(dividend, min_divisor):
        max_inv_divisor = dividend // min_divisor
        for inv_divisor in range(max_inv_divisor, 0, -1):
            if dividend % inv_divisor == 0:
                return dividend // inv_divisor
        return dividend

    def next_divisor(dividend, min_divisor):
        if min_divisor * min_divisor <= dividend:
            return next_divisor_small(dividend, min_divisor)
        return next_divisor_large(dividend, min_divisor)

    def move_axes_out(axes, tree):
        def move_axis_out(axis, x):
            if axis != 0:
                return jnp.moveaxis(x, axis, 0)
            return x
        return tree_util.tree_map(move_axis_out, axes, tree)

    def move_axes_in(axes, tree):
        def move_axis_in(axis, x):
            if axis != 0:
                return jnp.moveaxis(x, 0, axis)
            return x
        return tree_util.tree_map(move_axis_in, axes, tree)

    def batch(tree, nbatches):
        def batch(x):
            return x.reshape((nbatches, x.shape[0] // nbatches) + x.shape[1:])
        return tree_util.tree_map(batch, tree)

    def unbatch(tree):
        def unbatch(x):
            return x.reshape((x.shape[0] * x.shape[1],) + x.shape[2:])
        return tree_util.tree_map(unbatch, tree)

    def check_same(tree1, tree2):
        def check_same(x1, x2):
            assert x1.shape == x2.shape
            assert x1.dtype == x2.dtype
        tree_util.tree_map(check_same, tree1, tree2)

    initial_in_axes = in_axes
    initial_out_axes = out_axes

    @jax.jit
    @functools.wraps(func)
    def batched_func(*args):
        example_result = jax.eval_shape(func, *args)

        in_axes = expand_axes(initial_in_axes, args)
        out_axes = expand_axes(initial_out_axes, example_result)

        in_size = extract_size(in_axes, args)
        out_size = extract_size(out_axes, example_result)
        assert in_size == out_size
        size = in_size

        total_nbytes = sum_nbytes(args) + sum_nbytes(example_result)
        min_nbatches = total_nbytes // max_io_nbytes + bool(total_nbytes % max_io_nbytes)
        nbatches = next_divisor(size, min_nbatches)
        assert 1 <= nbatches <= size
        assert size % nbatches == 0
        assert total_nbytes % nbatches == 0

        batch_nbytes = total_nbytes // nbatches
        if batch_nbytes > max_io_nbytes:
            assert size == nbatches
            warnings.warn(f'batch_nbytes = {batch_nbytes} > max_io_nbytes = {max_io_nbytes}')

        def loop(_, args):
            args = move_axes_in(in_axes, args)
            result = func(*args)
            result = move_axes_out(out_axes, result)
            return None, result

        args = move_axes_out(in_axes, args)
        args = batch(args, nbatches)
        _, result = lax.scan(loop, None, args)
        result = unbatch(result)
        result = move_axes_in(out_axes, result)

        check_same(example_result, result)

        if return_nbatches:
            return result, nbatches
        return result

    return batched_func

@tree_util.register_pytree_node_class
class LeafDict(dict):
    """ dictionary that acts as a leaf in jax pytrees, to store compile-time
    values """

    def tree_flatten(self):
        return (), self

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return aux_data

    def __repr__(self):
        return f'{__class__.__name__}({super().__repr__()})'
