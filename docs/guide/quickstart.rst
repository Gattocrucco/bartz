.. bartz/docs/quickstart.rst
..
.. Copyright (c) 2024-2025, The Bartz Contributors
..
.. This file is part of bartz.
..
.. Permission is hereby granted, free of charge, to any person obtaining a copy
.. of this software and associated documentation files (the "Software"), to deal
.. in the Software without restriction, including without limitation the rights
.. to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
.. copies of the Software, and to permit persons to whom the Software is
.. furnished to do so, subject to the following conditions:
..
.. The above copyright notice and this permission notice shall be included in all
.. copies or substantial portions of the Software.
..
.. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
.. IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
.. FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
.. AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
.. LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
.. OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
.. SOFTWARE.

Quickstart
==========

Basics
------

Import and use the `bartz.BART.gbart` class:

.. code-block:: python

    from bartz.BART import gbart
    bart = gbart(X, y, ...)
    y_pred = bart.predict(X_test)

The interface hews to the R package `BART <https://cran.r-project.org/package=BART>`_, with a few differences explained in the documentation of `bartz.BART.gbart`.

JAX
---

`bartz` is implemented using `jax`, a Google library for machine learning. It allows to run the code on GPU or TPU and do various other things.

For basic usage, JAX is just an alternative implementation of `numpy`. The arrays returned by `~bartz.BART.gbart` are "jax arrays" instead of "numpy arrays", but there is no perceived difference in their functionality. If you pass numpy arrays to `bartz`, they will be converted automatically. You don't have to deal with `jax` in any way.

For advanced usage, refer to the `jax documentation <https://docs.jax.dev>`_.

Advanced
--------

`bartz` exposes the various functions that implement the MCMC of BART. You can use those yourself to try to make your own variant of BART. See the rest of the documentation for reference; the main entry points are `bartz.mcmcstep.init` and `bartz.mcmcloop.run_mcmc`. Using the internals is the only way to change the device used by each step of the algorithm, which is useful to pre-process data on CPU and move to GPU only the state of the MCMC if the data preprocessing step does not fit in the GPU memory.
