.. bartz/docs/development.rst
..
.. Copyright (c) 2024, Giacomo Petrillo
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

Install `poetry <https://python-poetry.org/docs/#installation>`_. My favorite complete installation route on macOS:

* Install `brew <https://brew.sh/>`_, then add :literal:`brew` to the :literal:`PATH`
* :literal:`brew install pipx`, then add :literal:`pipx`'s directory to the :literal:`PATH`
* :literal:`pipx install poetry`

Install `conda <https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html>`_ or an equivalent package manager like :literal:`mamba` or :literal:`micromamba`. My favorite is :literal:`micromamba`:

.. code-block:: shell

    brew install micromamba

Create a virtual environment from the file spec:

.. code-block:: shell

    micromamba env create --file condaenv.yml
    micromamba activate bartz
    poetry config virtualenvs.create false --local # to make sure poetry does not create another virtualenv

Finally, install the package with

.. code-block:: shell

    poetry install
    pre-commit install

Routine setup
-------------

Each time you want to work on `bartz` in a terminal, do

.. code-block:: shell

    cd <...>/bartz
    micromamba activate bartz # or the activation command for your env

Commands
--------

Development commands are defined in a makefile. Run :literal:`make` without arguments to list the targets.
