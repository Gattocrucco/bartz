.. bartz/docs/installation.rst
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

Installation
============

Install and set up Python. There are various ways to do it; my favorite one is to use `uv <https://docs.astral.sh/uv/>`_. Then:

.. code-block:: sh

    pip install bartz

To install the latest development version, do instead

.. code-block:: sh

    pip install git+https://github.com/Gattocrucco/bartz.git

To install a specific commit, do

.. code-block:: sh

    pip install git+https://github.com/Gattocrucco/bartz.git@<commit hash>

To use on GPU on a system that doesn't provide `jax` pre-installed, read how to install jax `in its manual <https://docs.jax.dev/en/latest/installation.html>`_.
