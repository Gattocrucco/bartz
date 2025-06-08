# bartz/tests/rbartpackages/_base.py
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

from collections.abc import Callable

import numpy as np
from rpy2 import robjects
from rpy2.robjects import conversion, methods, numpy2ri

# converter for pandas
pandas_converter = conversion.Converter('pandas')
try:
    from rpy2.robjects import pandas2ri
except ImportError:
    pass
else:
    pandas_converter = pandas2ri.converter

# converter for polars
polars_converter = conversion.Converter('polars')
try:
    import polars
    from rpy2.robjects import pandas2ri
except ImportError:
    pass
else:

    def polars_to_r(df):
        df = df.to_pandas()
        return pandas2ri.py2rpy(df)

    polars_converter.py2rpy.register(polars.DataFrame, polars_to_r)
    polars_converter.py2rpy.register(polars.Series, polars_to_r)

# converter for jax
jax_converter = conversion.Converter('jax')
try:
    import jax
except ImportError:
    pass
else:

    def jax_to_r(x):
        x = np.asarray(x)
        if x.ndim == 0:
            x = x[()]
        return numpy2ri.py2rpy(x)

    jax_converter.py2rpy.register(jax.Array, jax_to_r)

numpy_converter = numpy2ri.converter

# converter for python dictionaries
dict_converter = conversion.Converter('dict')


def dict_to_r(x):
    return robjects.ListVector(x)


dict_converter.py2rpy.register(dict, dict_to_r)


class RObjectBase:
    """
    Base class for Python wrappers of R objects creators.

    Subclasses should define the class attributes `_rfuncname` and `_methods`.

    _rfuncname : str
        An R function in the format ``'<package>::<function>``. The function is
        called with the initialization arguments, converted to R objects, and is
        expected to return an R object. The attributes of the R object are
        converted to equivalent Python values and set as attributes of the
        Python object. The R object itself is assigned to the member `_robject`.

    _methods : list[str] (optional)
        List of R function names (without package) that take as first argument
        the object returned by `_rfuncname`. They are wrapped as instance
        methods of the subclass.

    Initialization has additional arguments `timeout` and `retries` handled by
    the wrapper. The R function is terminated if it takes more than `timeout` to
    run. Re-execution is attempted at most `retries` times.

    """

    _converter = (
        robjects.default_converter
        + pandas_converter
        + polars_converter
        + numpy_converter
        + jax_converter
        + dict_converter
    )
    _convctx = conversion.localconverter(_converter)

    def _py2r(self, x):
        if isinstance(x, __class__):
            return x._robject
        with self._convctx:
            return self._converter.py2rpy(x)

    def _r2py(self, x):
        with self._convctx:
            return self._converter.rpy2py(x)

    def _args2r(self, args):
        return tuple(map(self._py2r, args))

    def _kw2r(self, kw):
        return {key: self._py2r(value) for key, value in kw.items()}

    _rfuncname: str = NotImplemented
    _methods: tuple[str, ...] = ()

    def __init__(self, *args, **kw):
        library, _ = self._rfuncname.split('::')
        robjects.r(f'loadNamespace("{library}")')
        self._library = library
        func = robjects.r(self._rfuncname)
        obj = func(*self._args2r(args), **self._kw2r(kw))
        self._robject = obj
        if hasattr(obj, 'items'):
            for s, v in obj.items():
                setattr(self, s.replace('.', '_'), self._r2py(v))

    def __init_subclass__(cls, **kw):
        """Automatically modify subclasses."""

        def implof(method: str) -> Callable:
            def impl(self, *args, **kw):
                if isinstance(self._robject, methods.RS4):
                    func = robjects.r['$'](self._robject, method)
                    out = func(*self._args2r(args), **self._kw2r(kw))
                else:
                    func = robjects.r(f'{self._library}::{method}')
                    out = func(self._robject, *self._args2r(args), **self._kw2r(kw))
                return self._r2py(out)

            return impl

        # if the subclass lists method names, create automatically Python
        # wrappers for those methods if they are not already defined
        for method in cls._methods:
            if not hasattr(cls, method):
                setattr(cls, method, implof(method))

        # set the docstring from R help
        library, name = cls._rfuncname.split('::')
        package_help = robjects.help.Package(library)
        page = package_help.fetch(name)
        if cls.__doc__ is None:
            cls.__doc__ = ''
        cls.__doc__ += 'R documentation:\n' + page.to_docstring()
