import time
import functools

import jax
from jax import numpy as jnp
from jax import random
from jax import lax
import bartz
from bartz import jaxext
from bartz.grove import tree_depth

# BART config
kw = dict(ntree=200, nskip=0, ndpost=1000, numcut=255, maxdepth=6)

# benchmark config
reps = 100

# DGP definition
n = 1000 # number of datapoints
p = 100 # number of covariates
def f(x): # conditional mean
    T = 2
    return jnp.sum(jnp.cos(2 * jnp.pi / T * x), axis=0)
def gen_X(key, p, n):
    return random.uniform(key, (p, n), float, -2, 2)
def gen_y(key, X):
    return f(X) + 0.1 * random.normal(key, X.shape[1:])

# set up random seed
key = random.key(202403281539)

class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self
    def __exit__(self, *_):
        self.time = time.perf_counter() - self.start

key, key1, key2, key3, key4 = random.split(key, 5)

# generate data
X = gen_X(key1, p, n)
y = gen_y(key2, X)

# compile bartz
def run_bart(X, y, key):
    return bartz.BART.gbart(X, y, **kw, seed=key)._mcmc_state
run_bart = jax.jit(run_bart).lower(X, y, key).compile()

# run bartz
with Timer() as timer:
    state = jax.block_until_ready(run_bart(X, y, key3))
niter = kw['ndpost'] + kw['nskip']
print(f'{niter} iterations in {timer.time:#.2g}s ({timer.time / niter:#.2g}s per iteration)')

# old traverse implementation
@functools.partial(jax.vmap, in_axes=(None, 0, 0))
@functools.partial(jax.vmap, in_axes=(1, None, None))
def traverse_tree_1(x, var_tree, split_tree):
    carry = (
        jnp.zeros((), bool),
        jnp.ones((), jaxext.minimal_unsigned_dtype(2 * var_tree.size - 1)),
    )

    def loop(carry, _):
        leaf_found, index = carry

        split = split_tree.at[index].get(mode='fill', fill_value=0)
        var = var_tree.at[index].get(mode='fill', fill_value=0)
        
        leaf_found |= split == 0
        child_index = (index << 1) + (x[var] >= split)
        index = jnp.where(leaf_found, index, child_index)

        return (leaf_found, index), None

    depth = 1 + tree_depth(var_tree)
    (_, index), _ = lax.scan(loop, carry, None, depth)
    return index

# new traverse implementation
@functools.partial(jax.vmap, in_axes=(None, 0, 0))
@functools.partial(jax.vmap, in_axes=(1, None, None))
def traverse_tree_2a(x, var_tree, split_tree):
    carry = (
        jnp.zeros((), bool),
        jnp.ones((), jaxext.minimal_unsigned_dtype(2 * var_tree.size - 1)),
    )

    def loop(carry, _):
        leaf_found, index = carry

        split = split_tree[index]
        var = var_tree[index]
        
        leaf_found |= split == 0
        child_index = (index << 1) + (x[var] >= split)
        index = jnp.where(leaf_found, index, child_index)

        return (leaf_found, index), None

    depth = tree_depth(var_tree)
    (_, index), _ = lax.scan(loop, carry, None, depth, unroll=5)
    return index

# new traverse implementation
@functools.partial(jax.vmap, in_axes=(None, 0, 0))
@functools.partial(jax.vmap, in_axes=(1, None, None))
def traverse_tree_2(x, var_tree, split_tree):
    depth = tree_depth(var_tree)

    carry = (
        jnp.zeros((), bool),
        jnp.ones((), jaxext.minimal_unsigned_dtype(2 * var_tree.size - 1)),
        jnp.zeros((), jaxext.minimal_unsigned_dtype(depth)),
    )

    def cond(carry):
        leaf_found, _, level = carry
        return (level < depth) & ~leaf_found

    def loop(carry):
        leaf_found, index, level = carry

        split = split_tree[index]
        var = var_tree[index]
        
        leaf_found |= split == 0
        child_index = (index << 1) + (x[var] >= split)
        index = jnp.where(leaf_found, index, child_index)

        return leaf_found, index, level + 1

    _, index, _ = lax.while_loop(cond, loop, carry)
    return index

# prepare pre-compiled tasks to run
def make_task(traverse_func):
    X = state['X']
    var_trees = state['var_trees']
    split_trees = state['split_trees']
    traverse_func = jax.jit(traverse_func).lower(X, var_trees, split_trees).compile()
    def task():
        jax.block_until_ready(traverse_func(X, var_trees, split_trees))
    return task

task1 = make_task(traverse_tree_1)
task2 = make_task(traverse_tree_2)

# time implementations
def clock(task, reps):
    times = []
    for _ in range(reps):
        with Timer() as timer:
            task()
        times.append(timer.time)
    return min(times)

t1 = clock(task1, reps)
t2 = clock(task2, reps)

print(f'1: {t1:#.2g}s')
print(f'2: {t2:#.2g}s')
