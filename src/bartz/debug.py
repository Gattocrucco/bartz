# bartz/src/bartz/debug.py
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

from dataclasses import replace
from functools import partial
from inspect import signature
from math import ceil, log2
from re import fullmatch

import numpy
from equinox import Module, field
from jax import jit, lax, random, vmap
from jax import numpy as jnp
from jax.tree_util import tree_map
from jaxtyping import Array, Bool, Float, Float32, Int32, Key, UInt

from . import grove, jaxext
from .BART import gbart
from .mcmcloop import Trace, TreesTrace
from .mcmcstep import randint_masked


def format_tree(tree: grove.TreeHeaps, *, print_all=False) -> str:
    tee = '├──'
    corner = '└──'
    join = '│  '
    space = '   '
    down = '┐'
    bottom = '╢'  # '┨' #

    def traverse_tree(lines, index, depth, indent, first_indent, next_indent, unused):
        if index >= len(tree.leaf_tree):
            return

        var = tree.var_tree.at[index].get(mode='fill', fill_value=0)
        split = tree.split_tree.at[index].get(mode='fill', fill_value=0)

        is_leaf = split == 0
        left_child = 2 * index
        right_child = 2 * index + 1

        if print_all:
            if unused:
                category = 'unused'
            elif is_leaf:
                category = 'leaf'
            else:
                category = 'decision'
            node_str = f'{category}({var}, {split}, {tree.leaf_tree[index]})'
        else:
            assert not unused
            if is_leaf:
                node_str = f'{tree.leaf_tree[index]:#.2g}'
            else:
                node_str = f'x{var} < {split}'

        if not is_leaf or (print_all and left_child < len(tree.leaf_tree)):
            link = down
        elif not print_all and left_child >= len(tree.leaf_tree):
            link = bottom
        else:
            link = ' '

        max_number = len(tree.leaf_tree) - 1
        ndigits = len(str(max_number))
        number = str(index).rjust(ndigits)

        lines.append(f' {number} {indent}{first_indent}{link}{node_str}')

        indent += next_indent
        unused = unused or is_leaf

        if unused and not print_all:
            return

        traverse_tree(lines, left_child, depth + 1, indent, tee, join, unused)
        traverse_tree(lines, right_child, depth + 1, indent, corner, space, unused)

    lines = []
    traverse_tree(lines, 1, 0, '', '', '', False)
    return '\n'.join(lines)


def tree_actual_depth(split_tree):
    # this could be done just with split_tree != 0
    is_leaf = grove.is_actual_leaf(split_tree, add_bottom_level=True)
    depth = grove.tree_depths(is_leaf.size)
    depth = jnp.where(is_leaf, depth, 0)
    return jnp.max(depth)


def forest_depth_distr(split_trees):
    depth = grove.tree_depth(split_trees) + 1
    depths = vmap(tree_actual_depth)(split_trees)
    return jnp.bincount(depths, length=depth)


def trace_depth_distr(split_trees_trace):
    return vmap(forest_depth_distr)(split_trees_trace)


def points_per_leaf_distr(var_tree, split_tree, X):
    traverse_tree = vmap(grove.traverse_tree, in_axes=(1, None, None))
    indices = traverse_tree(X, var_tree, split_tree)
    count_tree = jnp.zeros(
        2 * split_tree.size, dtype=jaxext.minimal_unsigned_dtype(indices.size)
    )
    count_tree = count_tree.at[indices].add(1)
    is_leaf = grove.is_actual_leaf(split_tree, add_bottom_level=True).view(jnp.uint8)
    return jnp.bincount(count_tree, is_leaf, length=X.shape[1] + 1)


def forest_points_per_leaf_distr(trees: grove.TreeHeaps, X):
    distr = jnp.zeros(X.shape[1] + 1, int)

    def loop(distr, heaps: tuple[Array, Array]):
        return distr + points_per_leaf_distr(*heaps, X), None

    distr, _ = lax.scan(loop, distr, (trees.var_tree, trees.split_tree))
    return distr


def trace_points_per_leaf_distr(trace: grove.TreeHeaps, X):
    def loop(_, trace):
        return None, forest_points_per_leaf_distr(trace, X)

    _, distr = lax.scan(loop, None, trace)
    return distr


check_functions = []


BoolLike = bool | Bool[Array, '']


def check(func):
    """Check the signature and add the function to the list `check_functions`."""
    sig = signature(func)
    assert str(sig) == f'(tree: bartz.grove.TreeHeaps, max_split) -> {BoolLike}', str(
        sig
    )
    check_functions.append(func)
    return func


@check
def check_types(tree: grove.TreeHeaps, max_split) -> BoolLike:
    expected_var_dtype = jaxext.minimal_unsigned_dtype(max_split.size - 1)
    expected_split_dtype = max_split.dtype
    return (
        tree.var_tree.dtype == expected_var_dtype
        and tree.split_tree.dtype == expected_split_dtype
    )


@check
def check_sizes(tree: grove.TreeHeaps, max_split) -> BoolLike:  # noqa: ARG001
    return tree.leaf_tree.size == 2 * tree.var_tree.size == 2 * tree.split_tree.size


@check
def check_unused_node(tree: grove.TreeHeaps, max_split) -> BoolLike:  # noqa: ARG001
    return (tree.var_tree[0] == 0) & (tree.split_tree[0] == 0)


@check
def check_leaf_values(tree: grove.TreeHeaps, max_split) -> BoolLike:  # noqa: ARG001
    return jnp.all(jnp.isfinite(tree.leaf_tree))


@check
def check_stray_nodes(tree: grove.TreeHeaps, max_split) -> BoolLike:  # noqa: ARG001
    """Check if there is any node marked-non-leaf with a marked-leaf parent."""
    index = jnp.arange(
        2 * tree.split_tree.size,
        dtype=jaxext.minimal_unsigned_dtype(2 * tree.split_tree.size - 1),
    )
    parent_index = index >> 1
    is_not_leaf = tree.split_tree.at[index].get(mode='fill', fill_value=0) != 0
    parent_is_leaf = tree.split_tree[parent_index] == 0
    stray = is_not_leaf & parent_is_leaf
    stray = stray.at[1].set(False)
    return ~jnp.any(stray)


@check
def check_rule_consistency(tree: grove.TreeHeaps, max_split) -> BoolLike:
    """Check that decision rules define proper subsets of ancestor rules."""
    if tree.var_tree.size < 4:
        return True

    # initial boundaries of decision rules. use extreme integers instead of 0,
    # max_split to avoid checking if there is something out of bounds.
    small = jnp.iinfo(jnp.int32).min
    large = jnp.iinfo(jnp.int32).max
    lower = jnp.full(max_split.size, small, jnp.int32)
    upper = jnp.full(max_split.size, large, jnp.int32)
    # specify the type explicitly, otherwise they are weakly types and get
    # implicitly converted to split.dtype (typically uint8) in the expressions

    def _check_recursive(node, lower, upper):
        # read decision rule
        var = tree.var_tree[node]
        split = tree.split_tree[node]

        # get rule boundaries from ancestors. use fill value in case var is
        # out of bounds, we don't want to check out of bounds in this function
        lower_var = lower.at[var].get(mode='fill', fill_value=small)
        upper_var = upper.at[var].get(mode='fill', fill_value=large)

        # check rule is in bounds
        bad = jnp.where(split, (split <= lower_var) | (split >= upper_var), False)

        # recurse
        if node < tree.var_tree.size // 2:
            bad |= _check_recursive(
                2 * node,
                lower,
                upper.at[jnp.where(split, var, max_split.size)].set(split),
            )
            bad |= _check_recursive(
                2 * node + 1,
                lower.at[jnp.where(split, var, max_split.size)].set(split),
                upper,
            )
        return bad

    return ~_check_recursive(1, lower, upper)


@check
def check_num_nodes(tree: grove.TreeHeaps, max_split) -> BoolLike:  # noqa: ARG001
    """Check that #leaves = 1 + #(internal nodes)."""
    is_leaf = grove.is_actual_leaf(tree.split_tree, add_bottom_level=True)
    num_leaves = jnp.count_nonzero(is_leaf)
    num_internal = jnp.count_nonzero(tree.split_tree)
    return num_leaves == num_internal + 1


@check
def check_var_in_bounds(tree: grove.TreeHeaps, max_split) -> BoolLike:
    """Check that variables are in [0, max_split.size)."""
    decision_node = tree.split_tree.astype(bool)
    in_bounds = (tree.var_tree >= 0) & (tree.var_tree < max_split.size)
    return jnp.all(in_bounds | ~decision_node)


@check
def check_split_in_bounds(tree: grove.TreeHeaps, max_split) -> BoolLike:
    """Check that splits are in [0, max_split[var]]."""
    max_split_var = (
        max_split.astype(jnp.int32)
        .at[tree.var_tree]
        .get(mode='fill', fill_value=jnp.iinfo(jnp.int32).max)
    )
    return jnp.all((tree.split_tree >= 0) & (tree.split_tree <= max_split_var))


def check_tree(tree: grove.TreeHeaps, max_split) -> Bool[Array, '']:
    error_type = jaxext.minimal_unsigned_dtype(2 ** len(check_functions) - 1)
    error = error_type(0)
    for i, func in enumerate(check_functions):
        ok = func(tree, max_split)
        ok = jnp.bool_(ok)
        bit = (~ok) << i
        error |= bit
    return error


def describe_error(error):
    return [func.__name__ for i, func in enumerate(check_functions) if error & (1 << i)]


check_forest = vmap(check_tree, in_axes=(0, None))


@jit
@partial(vmap, in_axes=(0, None))
def check_trace(trace: Trace, max_split: UInt[Array, ' p']):
    trees = TreesTrace.from_dataclass(trace)
    return check_forest(trees, max_split)


def get_next_line(s: str, i: int) -> tuple[str, int]:
    """Get the next line from a string and the new index."""
    i_new = s.find('\n', i)
    if i_new == -1:
        return s[i:], len(s)
    return s[i:i_new], i_new + 1


class BARTTraceMeta(Module):
    ndpost: int = field(static=True)
    ntree: int = field(static=True)
    numcut: UInt[Array, ' p']
    heap_size: int = field(static=True)


def scan_BART_trees(trees: str) -> BARTTraceMeta:
    # parse first line
    line, i_char = get_next_line(trees, 0)
    i_line = 1
    match = fullmatch(r'(\d+) (\d+) (\d+)', line)
    if match is None:
        msg = f'Malformed header at {i_line=}'
        raise ValueError(msg)
    ndpost, ntree, p = map(int, match.groups())

    # initial values for maxima
    max_heap_index = 0
    numcut = numpy.zeros(p, int)

    # cycle over iterations and trees
    for i_iter in range(ndpost):
        for i_tree in range(ntree):
            # parse first line of tree definition
            line, i_char = get_next_line(trees, i_char)
            i_line += 1
            match = fullmatch(r'(\d+)', line)
            if match is None:
                msg = f'Malformed tree header at {i_iter=} {i_tree=} {i_line=}'
                raise ValueError(msg)
            num_nodes = int(line)

            # cycle over nodes
            for i_node in range(num_nodes):
                # parse node definition
                line, i_char = get_next_line(trees, i_char)
                i_line += 1
                match = fullmatch(
                    r'(\d+) (\d+) (\d+) (-?\d+(\.\d+)?(e(\+|-|)\d+)?)', line
                )
                if match is None:
                    msg = f'Malformed node definition at {i_iter=} {i_tree=} {i_node=} {i_line=}'
                    raise ValueError(msg)
                i_heap = int(match.group(1))
                var = int(match.group(2))
                split = int(match.group(3))

                # update maxima
                numcut[var] = max(numcut[var], split)
                max_heap_index = max(max_heap_index, i_heap)

    assert i_char <= len(trees)
    if i_char < len(trees):
        msg = f'Leftover {len(trees) - i_char} characters in string'
        raise ValueError(msg)

    # determine minimal integer type for numcut
    numcut += 1  # because BART is 0-based
    split_dtype = jaxext.minimal_unsigned_dtype(numcut.max())
    numcut = jnp.array(numcut.astype(split_dtype))

    # determine minimum heap size to store the trees
    heap_size = 2 ** ceil(log2(max_heap_index + 1))

    return BARTTraceMeta(
        ndpost=ndpost,
        ntree=ntree,
        numcut=numcut,
        heap_size=heap_size,
    )


class TraceWithOffset(Module):
    """Implementation of `mcmcloop.Trace`."""

    leaf_tree: Float32[Array, 'ndpost ntree 2**d']
    var_tree: UInt[Array, 'ndpost ntree 2**(d-1)']
    split_tree: UInt[Array, 'ndpost ntree 2**(d-1)']
    offset: Float32[Array, ' ndpost']

    @classmethod
    def from_trees_trace(
        cls, trees: grove.TreeHeaps, offset: Float32[Array, '']
    ) -> 'TraceWithOffset':
        """Create a `TraceWithOffset` from a `TreeHeaps`."""
        ndpost, _, _ = trees.leaf_tree.shape
        return cls(
            leaf_tree=trees.leaf_tree,
            var_tree=trees.var_tree,
            split_tree=trees.split_tree,
            offset=jnp.full(ndpost, offset),
        )


def trees_BART_to_bartz(
    trees: str,
    *,
    min_maxdepth: int = 0,
    offset: float | Float[Array, ''] | None = None,
) -> tuple[TraceWithOffset, BARTTraceMeta]:
    """Convert trees from the R BART format to bartz format.

    Parameters
    ----------
    trees
        The string representation of a trace of trees of the R BART package.
        Can be accessed from ``mc_gbart(...).treedraws['trees']``.
    min_maxdepth
        The maximum tree depth of the output will be set to the maximum
        observed depth in the input trees. Use this parameter to require at
        least this maximum depth in the output format.
    offset
        The trace returned by `run_mcmc` contains an offset to be summed to the
        sum of trees. To match that behavior, this function returns an offset
        as well, zero by default. Set with this parameter otherwise.

    Returns
    -------
    trace : TraceWithOffset
        A representation of the trees compatible with the trace returned by
        `run_mcmc`.
    meta : BARTTraceMeta
        The metadata of the trace, containing the number of iterations,
        trees, and the maximum split value.
    """
    # scan all the string checking for errors and determining sizes
    meta = scan_BART_trees(trees)

    # skip first line
    _, i_char = get_next_line(trees, 0)

    heap_size = max(meta.heap_size, 2**min_maxdepth)
    leaf_trees = numpy.zeros((meta.ndpost, meta.ntree, heap_size), dtype=numpy.float32)
    var_trees = numpy.zeros(
        (meta.ndpost, meta.ntree, heap_size // 2),
        dtype=jaxext.minimal_unsigned_dtype(meta.numcut.size - 1),
    )
    split_trees = numpy.zeros(
        (meta.ndpost, meta.ntree, heap_size // 2), dtype=meta.numcut.dtype
    )

    # cycle over iterations and trees
    for i_iter in range(meta.ndpost):
        for i_tree in range(meta.ntree):
            # parse first line of tree definition
            line, i_char = get_next_line(trees, i_char)
            num_nodes = int(line)

            is_internal = numpy.zeros(heap_size // 2, dtype=bool)

            # cycle over nodes
            for _ in range(num_nodes):
                # parse node definition
                line, i_char = get_next_line(trees, i_char)
                values = line.split()
                i_heap = int(values[0])
                var = int(values[1])
                split = int(values[2])
                leaf = float(values[3])

                # update values
                leaf_trees[i_iter, i_tree, i_heap] = leaf
                is_internal[i_heap // 2] = True
                if i_heap < heap_size // 2:
                    var_trees[i_iter, i_tree, i_heap] = var
                    split_trees[i_iter, i_tree, i_heap] = split + 1

            is_internal[0] = False
            split_trees[i_iter, i_tree, ~is_internal] = 0

    return TraceWithOffset(
        leaf_tree=jnp.array(leaf_trees),
        var_tree=jnp.array(var_trees),
        split_tree=jnp.array(split_trees),
        offset=jnp.zeros(meta.ndpost)
        if offset is None
        else jnp.full(meta.ndpost, offset),
    ), meta


class SamplePriorStack(Module):
    nonterminal: Bool[Array, ' d-1']
    lower: UInt[Array, 'd-1 p']
    upper: UInt[Array, 'd-1 p']
    var: UInt[Array, ' d-1']
    split: UInt[Array, ' d-1']

    @classmethod
    def initial(
        cls, p_nonterminal: Float32[Array, ' d-1'], max_split: UInt[Array, ' p']
    ) -> 'SamplePriorStack':
        var_dtype = jaxext.minimal_unsigned_dtype(max_split.size - 1)
        return cls(
            nonterminal=jnp.ones(p_nonterminal.size, bool),
            lower=jnp.zeros((p_nonterminal.size, max_split.size), max_split.dtype),
            upper=jnp.broadcast_to(max_split, (p_nonterminal.size, max_split.size)),
            var=jnp.zeros(p_nonterminal.size, var_dtype),
            split=jnp.zeros(p_nonterminal.size, max_split.dtype),
        )


class SamplePriorTrees(Module):
    leaf_tree: Float32[Array, '* 2**d']
    var_tree: UInt[Array, '* 2**(d-1)']
    split_tree: UInt[Array, '* 2**(d-1)']

    @classmethod
    def initial(
        cls,
        key: Key[Array, ''],
        sigma_mu: Float32[Array, ''],
        p_nonterminal: Float32[Array, ' d-1'],
        max_split: UInt[Array, ' p'],
    ) -> 'SamplePriorTrees':
        heap_size = 2 ** (p_nonterminal.size + 1)
        return cls(
            leaf_tree=sigma_mu * random.normal(key, (heap_size,)),
            var_tree=jnp.zeros(
                heap_size // 2, dtype=jaxext.minimal_unsigned_dtype(max_split.size - 1)
            ),
            split_tree=jnp.zeros(heap_size // 2, dtype=max_split.dtype),
        )


class SamplePriorCarry(Module):
    key: Key[Array, '']
    stack: SamplePriorStack
    trees: SamplePriorTrees

    @classmethod
    def initial(
        cls,
        key: Key[Array, ''],
        sigma_mu: Float32[Array, ''],
        p_nonterminal: Float32[Array, ' d-1'],
        max_split: UInt[Array, ' p'],
    ) -> 'SamplePriorCarry':
        keys = jaxext.split(key)
        return cls(
            keys.pop(),
            SamplePriorStack.initial(p_nonterminal, max_split),
            SamplePriorTrees.initial(keys.pop(), sigma_mu, p_nonterminal, max_split),
        )


class SamplePriorX(Module):
    node: Int32[Array, ' 2**(d-1)-1']
    depth: Int32[Array, ' 2**(d-1)-1']
    next_depth: Int32[Array, ' 2**(d-1)-1']

    @classmethod
    def initial(cls, p_nonterminal: Float32[Array, ' d-1']) -> 'SamplePriorX':
        seq = cls._sequence(p_nonterminal.size)
        assert len(seq) == 2**p_nonterminal.size - 1
        node = [node for node, depth in seq]
        depth = [depth for node, depth in seq]
        next_depth = depth[1:] + [p_nonterminal.size]
        return cls(
            node=jnp.array(node),
            depth=jnp.array(depth),
            next_depth=jnp.array(next_depth),
        )

    @classmethod
    def _sequence(
        cls, max_depth: int, depth: int = 0, node: int = 1
    ) -> tuple[tuple[int, int], ...]:
        if depth < max_depth:
            out = ((node, depth),)
            out += cls._sequence(max_depth, depth + 1, 2 * node)
            out += cls._sequence(max_depth, depth + 1, 2 * node + 1)
            return out
        return ()


def sample_prior_onetree(
    key: Key[Array, ''],
    max_split: UInt[Array, ' p'],
    p_nonterminal: Float32[Array, ' d-1'],
    sigma_mu: Float32[Array, ''],
) -> SamplePriorTrees:
    carry = SamplePriorCarry.initial(key, sigma_mu, p_nonterminal, max_split)
    xs = SamplePriorX.initial(p_nonterminal)

    def loop(carry: SamplePriorCarry, x: SamplePriorX):
        keys = jaxext.split(carry.key, 4)

        # get variables at current stack level
        stack = carry.stack
        nonterminal = stack.nonterminal[x.depth]
        lower = stack.lower[x.depth, :]
        upper = stack.upper[x.depth, :]

        # sample a random decision rule
        available: Bool[Array, ' p'] = lower < upper
        allowed = jnp.any(available)
        var = randint_masked(keys.pop(), available)
        split = 1 + random.randint(keys.pop(), (), lower[var], upper[var])

        # cast to shorter integer types
        var = var.astype(carry.trees.var_tree.dtype)
        split = split.astype(carry.trees.split_tree.dtype)

        # decide whether to try to grow the node if it is growable
        pnt = p_nonterminal[x.depth]
        try_nonterminal: Bool[Array, ''] = random.bernoulli(keys.pop(), pnt)
        nonterminal &= try_nonterminal & allowed

        # update trees
        trees = carry.trees
        trees = replace(
            trees,
            var_tree=trees.var_tree.at[x.node].set(var),
            split_tree=trees.split_tree.at[x.node].set(
                jnp.where(nonterminal, split, 0)
            ),
        )

        def write_push_stack() -> SamplePriorStack:
            """Update the stack to go to the left child."""
            return replace(
                stack,
                nonterminal=stack.nonterminal.at[x.next_depth].set(nonterminal),
                lower=stack.lower.at[x.next_depth, :].set(lower),
                upper=stack.upper.at[x.next_depth, :].set(upper.at[var].set(split - 1)),
                var=stack.var.at[x.depth].set(var),
                split=stack.split.at[x.depth].set(split),
            )

        def pop_push_stack() -> SamplePriorStack:
            """Update the stack to go to the right sibling, possibly at lower depth."""
            var = stack.var[x.next_depth - 1]
            split = stack.split[x.next_depth - 1]
            lower = stack.lower[x.next_depth - 1, :]
            upper = stack.upper[x.next_depth - 1, :]
            return replace(
                stack,
                lower=stack.lower.at[x.next_depth, :].set(lower.at[var].set(split)),
                upper=stack.upper.at[x.next_depth, :].set(upper),
            )

        # update stack
        stack = lax.cond(x.next_depth > x.depth, write_push_stack, pop_push_stack)

        # update carry
        carry = replace(
            carry,
            key=keys.pop(),
            stack=stack,
            trees=trees,
        )
        return carry, None

    carry, _ = lax.scan(loop, carry, xs)
    return carry.trees


@partial(jaxext.vmap_nodoc, in_axes=(0, None, None, None))
def sample_prior_forest(
    keys: Key[Array, ' num_trees'],
    max_split: UInt[Array, ' p'],
    p_nonterminal: Float32[Array, ' d-1'],
    sigma_mu: Float32[Array, ''],
) -> SamplePriorTrees:
    return sample_prior_onetree(keys, max_split, p_nonterminal, sigma_mu)


@partial(jit, static_argnums=(1, 2))
def sample_prior(
    key: Key[Array, ''],
    trace_length: int,
    num_trees: int,
    max_split: UInt[Array, ' p'],
    p_nonterminal: Float32[Array, ' d-1'],
    sigma_mu: Float32[Array, ''],
) -> SamplePriorTrees:
    keys = random.split(key, trace_length * num_trees)
    trees = sample_prior_forest(keys, max_split, p_nonterminal, sigma_mu)
    return tree_map(lambda x: x.reshape(trace_length, num_trees, -1), trees)


class debug_gbart(gbart):
    """A subclass of `gbart` that adds debugging functionality."""

    def __init__(self, *args, check_trees: bool = True, **kw):
        super().__init__(*args, **kw)
        if check_trees:
            bad = self.check_trees()
            bad_count = jnp.count_nonzero(bad)
            assert bad_count == 0

    def show_tree(self, i_sample, i_tree, print_all=False):
        from .debug import format_tree

        tree = TreesTrace(
            leaf_tree=self._main_trace.leaf_tree,
            var_tree=self._main_trace.var_tree,
            split_tree=self._main_trace.split_tree,
        )
        tree = tree_map(lambda x: x[i_sample, i_tree, :], tree)
        s = format_tree(tree, print_all=print_all)
        print(s)  # noqa: T201, this method is intended for debug

    def sigma_harmonic_mean(self, prior=False):
        bart = self._mcmc_state
        assert bart.sigma2_alpha is not None
        assert bart.z is None
        if prior:
            alpha = bart.sigma2_alpha
            beta = bart.sigma2_beta
        else:
            resid = bart.resid
            alpha = bart.sigma2_alpha + resid.size / 2
            norm2 = resid @ resid
            beta = bart.sigma2_beta + norm2 / 2
        sigma2 = beta / alpha
        return jnp.sqrt(sigma2)

    def compare_resid(self):
        bart = self._mcmc_state
        resid1 = bart.resid

        trees = grove.evaluate_forest(bart.X, bart.forest)

        if bart.z is not None:
            ref = bart.z
        else:
            ref = bart.y
        resid2 = ref - (trees + bart.offset)

        return resid1, resid2

    def avg_acc(self):
        trace = self._main_trace

        def acc(prefix):
            acc = getattr(trace, f'{prefix}_acc_count')
            prop = getattr(trace, f'{prefix}_prop_count')
            return acc.sum() / prop.sum()

        return acc('grow'), acc('prune')

    def avg_prop(self):
        trace = self._main_trace

        def prop(prefix):
            return getattr(trace, f'{prefix}_prop_count').sum()

        pgrow = prop('grow')
        pprune = prop('prune')
        total = pgrow + pprune
        return pgrow / total, pprune / total

    def avg_move(self):
        agrow, aprune = self.avg_acc()
        pgrow, pprune = self.avg_prop()
        return agrow * pgrow, aprune * pprune

    def depth_distr(self):
        return trace_depth_distr(self._main_trace.split_tree)

    def points_per_leaf_distr(self):
        return trace_points_per_leaf_distr(self._main_trace, self._mcmc_state.X)

    def check_trees(self):
        return check_trace(self._main_trace, self._mcmc_state.max_split)

    def tree_goes_bad(self):
        bad = self.check_trees().astype(bool)
        bad_before = jnp.pad(bad[:-1], [(1, 0), (0, 0)])
        return bad & ~bad_before
