# bartz/tests/util.py
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

"""Functions intended to be shared across the test suite."""

from pathlib import Path

import numpy as np
import tomli
from jaxtyping import ArrayLike
from scipy import linalg


def assert_close_matrices(
    actual: ArrayLike,
    desired: ArrayLike,
    *,
    rtol: float = 0.0,
    atol: float = 0.0,
    tozero: bool = False,
):
    """
    Check if two matrices are similar.

    Parameters
    ----------
    actual
    desired
        The two matrices to be compared. Must be scalars, vectors, or 2d arrays.
        Scalars and vectors are intepreted as 1x1 and Nx1 matrices, but the two
        arrays must have the same shape beforehand.
    rtol
    atol
        Relative and absolute tolerances for the comparison. The closeness
        condition is:

            ||actual - desired|| <= atol + rtol * ||desired||,

        where the norm is the matrix 2-norm, i.e., the maximum (in absolute
        value) singular value.
    tozero
        If True, use the following codition instead:

            ||actual|| <= atol + rtol * ||desired||

        So `actual` is compared to zero, and `desired` is only used as a
        reference to set the threshold.

    Raises
    ------
    ValueError
        If the two matrices have different shapes.
    """
    actual = np.asarray(actual)
    desired = np.asarray(desired)
    if actual.shape != desired.shape:
        msg = f'{actual.shape=} != {desired.shape=}'
        raise ValueError(msg)
    if actual.size > 0:
        actual = np.atleast_1d(actual)
        desired = np.atleast_1d(desired)

        if tozero:
            expr = 'actual'
            ref = 'zero'
        else:
            expr = 'actual - desired'
            ref = 'desired'

        dnorm = linalg.norm(desired, 2)
        adnorm = linalg.norm(eval(expr), 2)  # noqa: S307, expr is a literal
        ratio = adnorm / dnorm if dnorm else np.nan

        msg = f"""\
matrices actual and {ref} are not close in 2-norm
matrix shape: {desired.shape}
norm(desired) = {dnorm:.2g}
norm({expr}) = {adnorm:.2g}  (atol = {atol:.2g})
ratio = {ratio:.2g}  (rtol = {rtol:.2g})"""

        assert adnorm <= atol + rtol * dnorm, msg


def get_old_python_str() -> str:
    """Read the oldest supported Python from pyproject.toml."""
    with Path('pyproject.toml').open('rb') as file:
        return tomli.load(file)['project']['requires-python'].removeprefix('>=')


def get_old_python_tuple() -> tuple[int, int]:
    """Read the oldest supported Python from pyproject.toml as a tuple."""
    ver_str = get_old_python_str()
    major, minor = ver_str.split('.')
    return int(major), int(minor)


def get_version() -> str:
    """Read the bartz version from pyproject.toml."""
    with Path('pyproject.toml').open('rb') as file:
        return tomli.load(file)['project']['version']


def update_version():
    """Update the version file."""
    version = get_version()
    Path('src/bartz/_version.py').write_text(f'__version__ = {version!r}\n')
