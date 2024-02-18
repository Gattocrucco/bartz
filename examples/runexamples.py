# bartz/examples/runexamples.py
#
# Copyright (c) 2024, Giacomo Petrillo
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

"""Run the scripts given on the command line and saves the figures produced
in the same directory of each corresponding script."""

import sys
import warnings
import gc
import pathlib
import runpy

from matplotlib import pyplot as plt

warnings.filterwarnings('ignore', r'Matplotlib is currently using agg, which is a non-GUI backend, so cannot show the figure\.')
warnings.filterwarnings('ignore', r'FigureCanvasAgg is non-interactive, and thus cannot be shown')

for file in sys.argv[1:]:

    file = pathlib.Path(file)
    print(f'\nrunexamples.py: running {file}...')
    
    # reset working environment and run
    plt.close('all')
    runpy.run_path(str(file))
    gc.collect()
    
    # save figures
    nums = plt.get_fignums()
    directory = file.parent / 'plot'
    directory.mkdir(exist_ok=True)
    for num in nums:
        fig = plt.figure(num)
        suffix = f'-{num}' if num > 1 else ''
        out = directory / f'{file.stem}{suffix}.png'
        print(f'runexamples.py: write {out}')
        fig.savefig(out)
