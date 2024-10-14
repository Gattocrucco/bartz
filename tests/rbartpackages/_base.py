import functools
import abc

from rpy2 import robjects
from rpy2.robjects import conversion, numpy2ri, methods
import numpy as np

# converter for pandas
pandas_converter = conversion.Converter('pandas')
try:
    from rpy2.robjects import pandas2ri
    pandas_converter = pandas2ri.converter
except ImportError:
    pandas_converter = conversion.Converter('pandas')

# converter for polars
# TODO i'd like to copy the code of the pandas converter to relinquish the dependency on pandas
polars_converter = conversion.Converter('polars')
try:
    import polars
    from rpy2.robjects import pandas2ri
    def polars_to_r(df):
        df = df.to_pandas()
        return pandas2ri.py2rpy(df)
    polars_converter.py2rpy.register(polars.DataFrame, polars_to_r)
    polars_converter.py2rpy.register(polars.Series, polars_to_r)
except ImportError:
    pass

# converter for jax
jax_converter = conversion.Converter('jax')
try:
    import jax
    def jax_to_r(x):
        x = np.asarray(x)
        if x.ndim == 0:
            x = x[()]
        return numpy2ri.py2rpy(x)
    jax_converter.py2rpy.register(jax.Array, jax_to_r)
except ImportError:
    pass

# alternative numpy converter because the default one produces conflicts
# => the problem with this is that it does not handle r -> numpy
# numpy_converter = conversion.Converter('numpy')
# def numpy_to_r(x):
#     return numpy2ri.py2rpy(x)
# numpy_converter.py2rpy.register(np.ndarray, numpy_to_r)
# numpy_converter.py2rpy.register(np.generic, numpy_to_r)
numpy_converter = numpy2ri.converter

# converter for python dictionaries
dict_converter = conversion.Converter('dict')
def dict_to_r(x):
    return robjects.ListVector(x)
dict_converter.py2rpy.register(dict, dict_to_r)

class RObjectABC(abc.ABC):
    """

    Abstract base class for Python wrappers of R objects creators.

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

    @property
    @abc.abstractmethod
    def _rfuncname(self):
        """ function/class to be wrapped written as library::name, override
        this property with a class string attribute """
        raise NotImplementedError

    def __init__(self, *args, timeout=None, retries=0, **kw):
        library, _ = self._rfuncname.split('::')
        robjects.r(f'library({library})')
            # TODO I would like to do loadNamespace('<library>') and then always use <library>::<thing>. However this does not work with methods (see __init_subclass__). I have to look up how to reference methods directly in a namespace. Alternatively, I could do library(<library>, quietly=TRUE), but I prefer not to suppress eventual errors.
        func = robjects.r(self._rfuncname)
        dofunc = lambda: func(*self._args2r(args), **self._kw2r(kw))
        if timeout is not None:
            dofunc = self._tryagain_withtimeout(dofunc, timeout, retries)
        obj = dofunc()
        self._robject = obj
        if hasattr(obj, 'items'):
            for s, v in obj.items():
                setattr(self, s.replace('.', '_'), self._r2py(v))

    @staticmethod
    def _tryagain_withtimeout(func, timeoutpercall, maxretries):
        """ decorate `func` to time its execution, time out over a threshold,
        and optionally retries up to a maximum number of calls """
        import wrapt_timeout_decorator as wtd
        timedfunc = wtd.timeout(timeoutpercall, use_signals=False)(func)
            # do not use signals because they are intercepted by R
        @functools.wraps(func)
        def newfunc(*args, **kw):
            for _ in range(maxretries):
                try:
                    return timedfunc(*args, **kw)
                except TimeoutError as exc:
                    print(f'###### {self._rfuncname}:', exc.__class__.__name__, *exc.args, '#######')
            return timedfunc(*args, **kw)
        return newfunc

    def __init_subclass__(cls, **kw):
        """ automatically modify subclasses """

        # if the subclass had an attribute `_methods` (a list of method
        # names), create automatically Python wrappers for those methods if
        # missing
        def implof(method):
            def impl(self, *args, **kw):
                if isinstance(self._robject, methods.RS4):
                    func = robjects.r['$'](self._robject, method)
                    out = func(*self._args2r(args), **self._kw2r(kw))
                else:
                    func = robjects.r(method)
                    out = func(self._robject, *self._args2r(args), **self._kw2r(kw))
                return self._r2py(out)
            return impl
        for method in getattr(cls, '_methods', []):
            if not hasattr(cls, method):
                setattr(cls, method, implof(method))

        # set the docstring from R help
        library, name = cls._rfuncname.split('::')
        package_help = robjects.help.Package(library)
        page = package_help.fetch(name)
        if cls.__doc__ is None:
            cls.__doc__ = ''
        cls.__doc__ += 'R documentation:\n' + page.to_docstring()
