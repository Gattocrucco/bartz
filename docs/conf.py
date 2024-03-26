# bartz/docs/conf.py
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

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sys
import inspect
import pathlib
import datetime
import os
import subprocess

import packaging

# -- Doc variant -------------------------------------------------------------

variant = os.environ.get('BARTZ_DOC_VARIANT', 'dev')
assert variant in ('dev', 'latest')

if variant == 'dev':

    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    modif = subprocess.run(['git', 'diff', '--quiet']).returncode
    modif_staged = subprocess.run(['git', 'diff', '--quiet', '--staged']).returncode
    uncommitted_stuff = modif or modif_staged
    version = f'{commit[:7]}{"+" if uncommitted_stuff else ""}'

elif variant == 'latest':

    # list git tags
    tags = subprocess.check_output(['git', 'tag'], text=True).splitlines()
    print(f'git tags: {tags}')
    
    # find final versions in tags
    versions = []
    for i, t in enumerate(tags):
        try:
            v = packaging.version.parse(t)
        except packaging.version.InvalidVersion:
            continue
        if v.is_prerelease or v.is_devrelease:
            continue
        versions.append((v, t))
    print(f'tags for releases: {versions}')

    # find latest versions
    versions.sort(key=lambda x: x[0])
    version, tag = versions[-1]
    
    # check it out and check it matches the version in the package
    subprocess.run(['git', 'checkout', tag], check=True)
    import bartz
    assert packaging.version.parse(bartz.__version__) == version
    
    version = str(version)
    uncommitted_stuff = False

import bartz

# -- Project information -----------------------------------------------------

project = f'bartz {version}'
author = 'Giacomo Petrillo'

now = datetime.datetime.now()
year = '2024'
if now.year > int(year):
    year += '-' + str(now.year)
copyright = year + ', ' + author

release = version

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.mathjax',
    'numpydoc', # process numpy format docstrings
    'sphinx.ext.intersphinx', # link to other documentations automatically
    'myst_parser', # markdown support
]

# decide whether to use viewcode or linkcode extension
ext = 'viewcode' # copy source code in static website
if not uncommitted_stuff:
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    commit_on_github = subprocess.check_output(['git', 'branch', '--remotes', '--contains', commit], text=True)
    if commit_on_github.strip():
        ext = 'linkcode' # links to code on github
extensions.append(f'sphinx.ext.{ext}')

myst_enable_extensions = [
    # "amsmath",
    "dollarmath",
]

# Add any paths that contain templates here, relative to this directory.
# templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'alabaster'

html_title = f'{project} documentation'

html_theme_options = dict(
    description = 'A jax implementation of BART',
    fixed_sidebar = True,
    github_button = True,
    github_type = 'star',
    github_repo = 'bartz',
    github_user = 'Gattocrucco',
    show_relbars = True,
)

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

master_doc = 'index'

# -- Other options -------------------------------------------------

autoclass_content = 'both' # concatenate the class and __init__ docstrings
autodoc_preserve_defaults = True # default arguments are printed as in source
                                 # instead of being evaluated
autodoc_default_options = {
    'member-order': 'bysource',
}

numpydoc_class_members_toctree = False
numpydoc_show_class_members = False

default_role = 'py:obj'

intersphinx_mapping = dict(
    scipy=('https://docs.scipy.org/doc/scipy', None),
    numpy=('https://numpy.org/doc/stable', None),
    jax=('https://jax.readthedocs.io/en/latest/', None),
)

viewcode_line_numbers = True # for 'viewcode' extension

def linkcode_resolve(domain, info):
    """
    Determine the URL corresponding to Python object, for extension linkcode

    Adapted from scipy/doc/release/conf.py
    """
    if domain != 'py':
        return None

    modname = info['module']
    fullname = info['fullname']

    submod = sys.modules.get(modname)
    if submod is None:
        return None

    obj = submod
    for part in fullname.split('.'):
        try:
            obj = getattr(obj, part)
        except Exception:
            return None

    # Use the original function object if it is wrapped.
    obj = getattr(obj, "__wrapped__", obj)
    
    try:
        fn = inspect.getsourcefile(obj)
    except Exception:
        fn = None
    if not fn:
        try:
            fn = inspect.getsourcefile(sys.modules[obj.__module__])
        except Exception:
            fn = None
    if not fn:
        return None

    try:
        source, lineno = inspect.getsourcelines(obj)
    except Exception:
        lineno = None

    if lineno:
        linespec = f'#L{lineno}-L{lineno + len(source) - 1}'
    else:
        linespec = ''

    prefix = 'https://github.com/Gattocrucco/bartz/blob'
    root = pathlib.Path(bartz.__file__).parent
    path = pathlib.Path(fn).relative_to(root).as_posix()
    return f'{prefix}/{commit}/src/bartz/{path}{linespec}'
