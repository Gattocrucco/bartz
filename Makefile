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

RELEASE_TARGETS = tests docs
TARGETS = upload release $(RELEASE_TARGETS) covreport examples

.PHONY: all $(TARGETS)

all:
	@echo "available targets: $(TARGETS)"
	@echo "release = $(RELEASE_TARGETS) (in order) + build"
	@echo
	@echo "Release instructions:"
	@echo " 1) $$ poetry version <rule>"
	@echo " 2) describe release in docs/changelog.md"
	@echo " 3) commit, push and check CI completes"
	@echo " 4) $$ make release"
	@echo " 5) repeat 3 and 4 until everything goes smoothly"
	@echo " 6) $$ make upload"
	@echo " 7) publish the github release"

upload:
	poetry publish

release: $(RELEASE_TARGETS)
	test ! -d dist || rm -r dist
	poetry build

.PHONY: version
version: src/bartz/_version.py
src/bartz/_version.py: pyproject.toml
	python -c 'import tomli, pathlib; version = tomli.load(open("pyproject.toml", "rb"))["tool"]["poetry"]["version"]; pathlib.Path("src/bartz/_version.py").write_text(f"__version__ = {version!r}\n")'

PY = MPLBACKEND=agg coverage run
TESTSPY = COVERAGE_FILE=.coverage.tests$(COVERAGE_SUFFIX) $(PY) --context=tests$(COVERAGE_SUFFIX)
EXAMPLESPY = COVERAGE_FILE=.coverage.examples$(COVERAGE_SUFFIX) $(PY) --context=examples$(COVERAGE_SUFFIX)

tests: version
	$(TESTSPY) -m pytest tests

# I did not manage to make parallel pytest (pytest -n<processes>) work with
# coverage

EXAMPLES = $(wildcard examples/*.py)
.PHONY: $(EXAMPLES)
examples: $(EXAMPLES)
	$(EXAMPLESPY) examples/runexamples.py $(EXAMPLES)

.PHONY: docs-latest
docs-latest:
	BARTZ_DOC_VARIANT=latest make -C docs html
	git switch -
	rm -r _site/docs || true
	mv docs/_build/html _site/docs

.PHONY: docs-dev
docs-dev:
	BARTZ_DOC_VARIANT=dev make -C docs html
	rm -r _site/docs-dev || true
	mv docs/_build/html _site/docs-dev

docs: version docs-latest docs-dev
	@echo
	@echo "Now open _site/index.html"

covreport: version
	coverage combine
	coverage html
	rm -r _site/coverage || true
	mv htmlcov _site/coverage
	@echo
	@echo "Now open _site/index.html"
