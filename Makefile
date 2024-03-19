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

RELEASE_TARGETS = tests examples docscode docs
TARGETS = upload release $(RELEASE_TARGETS) covreport

.PHONY: all $(TARGETS)

all:
	@echo "available targets: $(TARGETS)"
	@echo "release = $(RELEASE_TARGETS) (in order) + build"
	@echo
	@echo "Release instructions:"
	@echo " 1) remove .devN suffix from version in src/bartz/__init__.py"
	@echo " 2) describe release in docs/changelog.md"
	@echo " 3) link versioned docs in docs/index.rst"
	@echo " 4) commit, push and check CI completes"
	@echo " 5) $$ make release"
	@echo " 6) repeat 4 and 5 until everything goes smoothly"
	@echo " 7) $$ make upload"
	@echo " 8) publish the github release"
	@echo " 9) bump version number and add .dev0 suffix"

upload:
	poetry publish

release: $(RELEASE_TARGETS)
	test ! -d dist || rm -r dist
	poetry build

PY = MPLBACKEND=agg coverage run
TESTSPY = COVERAGE_FILE=.coverage.tests$(COVERAGE_SUFFIX) $(PY) --context=tests$(COVERAGE_SUFFIX)
EXAMPLESPY = COVERAGE_FILE=.coverage.examples$(COVERAGE_SUFFIX) $(PY) --context=examples$(COVERAGE_SUFFIX)
DOCSPY = COVERAGE_FILE=.coverage.docs$(COVERAGE_SUFFIX) $(PY) --context=docs$(COVERAGE_SUFFIX)

tests:
	$(TESTSPY) -m pytest tests

# I did not manage to make parallel pytest (pytest -n<processes>) work with
# coverage

EXAMPLES = $(wildcard examples/*.py)
.PHONY: $(EXAMPLES)
examples: $(EXAMPLES)
	$(EXAMPLESPY) examples/runexamples.py $(EXAMPLES)

docscode:
	$(DOCSPY) docs/runcode.py docs/*.rst

docs:
	make -C docs html
	@echo
	@echo "Now open docs/_build/html/index.html"

covreport:
	coverage combine
	coverage html
	@echo
	@echo "Now open htmlcov/index.html"
