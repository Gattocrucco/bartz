# bartz/Makefile
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

# Makefile for running tests, prepare and upload a release.

COVERAGE_SUFFIX=

.PHONY: all
all:
	@echo "Available targets:"
	@echo "- tests: run unit tests, saving coverage information"
	@echo "- docs: build html documentation"
	@echo "- covreport: build html coverage report"
	@echo "- release: packages the python module, invokes tests and docs first"
	@echo "- upload: upload release to PyPI"
	@echo
	@echo "Release instructions:"
	@echo "- $$ poetry version <rule>"
	@echo "- describe release in docs/changelog.md"
	@echo "- $$ make release (repeat until it goes smoothly)"
	@echo "- push and check CI completes"
	@echo "- $$ make upload"

.PHONY: release
release: tests docs
	test ! -d dist || rm -r dist
	poetry build

.PHONY: copy-version
copy-version: src/bartz/_version.py
src/bartz/_version.py: pyproject.toml
	python -c 'import tomli, pathlib; version = tomli.load(open("pyproject.toml", "rb"))["tool"]["poetry"]["version"]; pathlib.Path("src/bartz/_version.py").write_text(f"__version__ = {version!r}\n")'

PY = MPLBACKEND=agg coverage run
TESTSPY = COVERAGE_FILE=.coverage.tests$(COVERAGE_SUFFIX) $(PY) --context=tests$(COVERAGE_SUFFIX)

.PHONY: tests
tests: copy-version
	$(TESTSPY) -m pytest tests

# I did not manage to make parallel pytest (pytest -n<processes>) work with
# coverage

.PHONY: docs-latest
docs-latest:
	BARTZ_DOC_VARIANT=latest make -C docs html
	git switch -
	test ! -d _site/docs || rm -r _site/docs
	mv docs/_build/html _site/docs

.PHONY: docs
docs:
	make -C docs html
	test ! -d _site/docs-dev || rm -r _site/docs-dev
	mv docs/_build/html _site/docs-dev

.PHONY: docs-all
docs-all: copy-version docs-latest docs
	@echo
	@echo "Now open _site/index.html"

.PHONY: covreport
covreport:
	coverage combine
	coverage html
	test ! -d _site/coverage || rm -r _site/coverage
	mv htmlcov _site/coverage
	@echo
	@echo "Now open _site/index.html"

.PHONY: version-tag
version-tag: copy-version
	git diff --quiet
	git diff --quiet --staged
	git fetch --tags
	git tag v$(shell python -c 'import bartz; print(bartz.__version__)')
	git push --tags

.PHONY: upload
upload: version-tag
	poetry publish
