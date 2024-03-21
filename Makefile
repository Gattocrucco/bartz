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
	@echo " 1) bump version in pyproject.toml and src/bartz/__init__.py"
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

PY = MPLBACKEND=agg coverage run
TESTSPY = COVERAGE_FILE=.coverage.tests$(COVERAGE_SUFFIX) $(PY) --context=tests$(COVERAGE_SUFFIX)
EXAMPLESPY = COVERAGE_FILE=.coverage.examples$(COVERAGE_SUFFIX) $(PY) --context=examples$(COVERAGE_SUFFIX)

tests:
	$(TESTSPY) -m pytest tests

# I did not manage to make parallel pytest (pytest -n<processes>) work with
# coverage

EXAMPLES = $(wildcard examples/*.py)
.PHONY: $(EXAMPLES)
examples: $(EXAMPLES)
	$(EXAMPLESPY) examples/runexamples.py $(EXAMPLES)

docs:
	make -C docs html
	echo `python -c 'import re, bartz; print(re.fullmatch(r"(\d+(\.\d+)*)(.dev\d+)?", bartz.__version__).group(1))'` > docs/_build/html/bartzversion.txt
	rm -r _site/docs || true
	mv docs/_build/html _site/docs
	@echo
	@echo "Now open _site/docs/index.html"

covreport:
	coverage combine
	coverage html
	rm -r _site/coverage || true
	mv htmlcov _site/coverage
	@echo
	@echo "Now open _site/coverage/index.html"
