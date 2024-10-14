from rpy2 import robjects

from . import _base

class bartMachine(_base.RObjectABC):

   _rfuncname = 'bartMachine::bartMachine'
   _methods = 'predict', 'get_posterior', 'get_sigsqs'

   def __init__(self, *args, num_cores=None, megabytes=5000, **kw):
        robjects.r(f'options(java.parameters = "-Xmx{megabytes:d}m")')
        robjects.r('library(bartMachine)')
        if num_cores is not None:
            robjects.r(f'bartMachine::set_bart_machine_num_cores({int(num_cores)})')
        super().__init__(*args, **kw)
