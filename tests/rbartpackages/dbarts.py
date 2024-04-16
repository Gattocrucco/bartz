from rpy2 import robjects

from . import _base

class bart(_base.RObjectABC):
    """

    Python interface to dbarts::bart.

    The named numeric vector form of the `splitprobs` parameter must be
    specified as a dictionary in Python.

    """

    _rfuncname = 'dbarts::bart'
    _methods = ['predict', 'extract', 'fitted']
    _split_probs = 'splitprobs'

    def __init__(self, *args, **kw):

        split_probs = kw.get(self._split_probs)
        if isinstance(split_probs, dict):
            values = list(split_probs.values())
            names = list(split_probs.keys())
            split_probs = robjects.FloatVector(values)
            split_probs = robjects.r('setNames')(split_probs, names)
            kw[self._split_probs] = split_probs

        super().__init__(*args, **kw)

class bart2(bart):
    """

    Python interface to dbarts::bart2.

    The named numeric vector form of the `split_probs` parameter must be
    specified as a dictionary in Python.

    """

    _rfuncname = 'dbarts::bart2'
    _split_probs = 'split_probs'

    def __init__(self, formula, *args, **kw):
        formula = robjects.Formula(formula)
        super().__init__(formula, *args, **kw)

class rbart_vi(bart2):
    """

    Python interface to dbarts::rbart_vi.

    The named numeric vector form of the `split_probs` parameter must be
    specified as a dictionary in Python.

    """

    _rfuncname = 'dbarts::rbart_vi'

class dbarts(_base.RObjectABC):

    _rfuncname = 'dbarts::dbarts'
    _methods = (
        'run',
        'sampleTreesFromPrior',
        'sampleNodeParametersFromPrior',
        'copy',
        'show',
        'predict',
        'setControl',
        'setModel',
        'setData',
        'setResponse',
        'setOffset',
        'setSigma',
        'setPredictor',
        'setTestPredictor',
        'setTestPredictorAndOffset',
        'setTestOffset',
        'printTrees',
        'plotTree',
    )

class dbartsControl(_base.RObjectABC):

    _rfuncname = 'dbarts::dbartsControl'
