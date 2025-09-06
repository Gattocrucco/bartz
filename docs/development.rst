.. bartz/docs/development.rst
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

Development
===========

Initial setup
-------------

`Fork <https://github.com/Gattocrucco/bartz/fork>`_ the repository on Github, then clone the fork:

.. code-block:: shell

    git clone git@github.com:YourGithubUserName/bartz.git
    cd bartz

Install `uv <https://docs.astral.sh/uv/getting-started/installation/>`_. My favorite installation route on macOS would be to install `brew <https://brew.sh/>`_ and then :literal:`brew install uv`.

Then install `micromamba <https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html>`_; for example with Homebrew:

.. code-block:: shell

    brew install micromamba
    micromamba shell init

Finally, run

.. code-block:: shell

    make setup

This creates a micromamba virtual environment in :literal:`./.venv`, which is then managed by :literal:`uv`. To run commands that involve the python installation, do :literal:`uv run --no-sync <command>`. For example, to start an IPython shell, do :literal:`uv run --no-sync ipython`.

Pre-defined commands
--------------------

Development commands are defined in a makefile. Run :literal:`make` without arguments to list the targets.

Benchmarks
----------

The benchmarks are managed with `asv <https://asv.readthedocs.io/en/latest>`_. Basic workflow:

.. code-block:: shell

    uv run asv run
    uv run asv publish
    uv run asv preview

:literal:`asv run` writes the results into files saved in :literal:`./benchmarks`. These files are tracked by git; consider deliberately not committing all results generated while developing.

Documentation
-------------

To build the documentation for the current working copy, do

.. code-block:: shell

    make docs

To build the documentation for the latest release tag, do

.. code-block:: shell

    make docs-latest

To debug the documentation build, do

.. code-block:: shell

    make docs SPHINXOPTS='--fresh-env --pdb'
