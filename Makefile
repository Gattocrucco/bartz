# bartz/Makefile
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

# Makefile for running tests, prepare and upload a release.

COVERAGE_SUFFIX =
OLD_PYTHON = 3.10
OLD_DATE = 2025-05-15

.PHONY: all
all:
	@echo "Available targets:"
	@echo "- setup: create R and Python environments for development"
	@echo "- tests: run unit tests, saving coverage information"
	@echo "- tests-old: run unit tests with oldest supported python and dependencies"
	@echo "- docs: build html documentation"
	@echo "- docs-latest: build html documentation for latest release"
	@echo "- covreport: build html coverage report"
	@echo "- release: packages the python module, invokes tests and docs first"
	@echo "- upload: upload release to PyPI"
	@echo "- upload-test: upload release to TestPyPI"
	@echo "- asv-all-tags: run benchmarks on all unbenchmarked tagged releases"
	@echo "- asv-publish: create html benchmark report"
	@echo "- asv-preview: create html report and start server"
	@echo "- asv-main: run benchmarks on main branch"
	@echo "- asv-quick: run quick benchmarks on current code, no saving"
	@echo "- ipython: start an ipython shell with stuff pre-imported"
	@echo "- ipython-old: start an ipython shell with oldest supported python and dependencies"
	@echo
	@echo "Release workflow:"
	@echo "- $$ uv version --bump major|minor|patch"
	@echo "- describe release in docs/changelog.md"
	@echo "- $$ make release (repeat until it goes smoothly)"
	@echo "- push and check CI completes (if it doesn't, go to previous step)"
	@echo "- $$ make upload"
	@echo "- publish github release (updates zenodo automatically)"
	@echo "- if the online docs are not up-to-date, press 'run workflow' on https://github.com/Gattocrucco/bartz/actions/workflows/tests.yml, and try to understand why 'make upload' didn't do it"


.PHONY: setup
setup:
	Rscript -e "renv::restore()"
	uv run --all-groups pre-commit install


################# TESTS #################

TESTS_COMMAND = python -m coverage run --data-file=.coverage.tests$(COVERAGE_SUFFIX) --context=tests$(COVERAGE_SUFFIX) -m pytest $(ARGS)
# I did not manage to make parallel pytest (pytest -n<processes>) work with
# coverage

UV_RUN_CI = uv run --group ci
UV_OPTS_OLD = --python $(OLD_PYTHON) --resolution lowest-direct --exclude-newer $(OLD_DATE)
UV_VARS_OLD = UV_PROJECT_ENVIRONMENT=.venv-old

.PHONY: tests
tests:
	$(UV_RUN_CI) $(TESTS_COMMAND)

.PHONY: tests-old
tests-old:
	$(UV_VARS_OLD) $(UV_RUN_CI) $(UV_OPTS_OLD) $(TESTS_COMMAND)


################# DOCS #################

.PHONY: docs
docs:
	$(UV_RUN_CI) make -C docs html
	test ! -d _site/docs-dev || rm -r _site/docs-dev
	mv docs/_build/html _site/docs-dev
	@echo
	@echo "Now open _site/index.html"

.PHONY: docs-latest
docs-latest:
	BARTZ_DOC_VARIANT=latest $(UV_RUN_CI) make -C docs html
	git switch - || git switch main
	test ! -d _site/docs || rm -r _site/docs
	mv docs/_build/html _site/docs
	@echo
	@echo "Now open _site/index.html"

.PHONY: covreport
covreport:
	$(UV_RUN_CI) coverage combine
	$(UV_RUN_CI) coverage html
	@echo
	@echo "Now open _site/index.html"


################# RELEASE #################

.PHONY: update-deps
update-deps:
	test ! -d .venv || rm -r .venv
	uv lock --upgrade

.PHONY: copy-version
copy-version: src/bartz/_version.py
src/bartz/_version.py: pyproject.toml
	uv run --group only-local python -c 'import tomli, pathlib; version = tomli.load(open("pyproject.toml", "rb"))["project"]["version"]; pathlib.Path("src/bartz/_version.py").write_text(f"__version__ = {version!r}\n")'

.PHONY: check-committed
check-committed:
	git diff --quiet
	git diff --quiet --staged

.PHONY: release
release: update-deps copy-version check-committed
	@$(MAKE) tests
	@$(MAKE) tests-old
	@$(MAKE) docs
	test ! -d dist || rm -r dist
	uv build

.PHONY: version-tag
version-tag: copy-version check-committed
	git fetch --tags
	git tag v$(shell uv run python -c 'import bartz; print(bartz.__version__)')
	git push --tags

.PHONY: upload
upload: version-tag
	@echo "Enter PyPI token:"
	@read -s UV_PUBLISH_TOKEN && \
	export UV_PUBLISH_TOKEN="$$UV_PUBLISH_TOKEN" && \
	uv publish
	@VERSION=$$(uv run python -c 'import bartz; print(bartz.__version__)') && \
	echo "Try to install bartz $$VERSION from PyPI" && \
	uv tool run --with "bartz==$$VERSION" python -c 'import bartz; print(bartz.__version__)'

.PHONY: upload-test
upload-test: check-committed
	@echo "Enter TestPyPI token:"
	@read -s UV_PUBLISH_TOKEN && \
	export UV_PUBLISH_TOKEN="$$UV_PUBLISH_TOKEN" && \
	uv publish --check-url https://test.pypi.org/simple/ --publish-url https://test.pypi.org/legacy/
	@VERSION=$$(uv run --group only-local python -c 'import tomli; print(tomli.load(open("pyproject.toml", "rb"))["project"]["version"])') && \
	echo "Try to install bartz $$VERSION from TestPyPI" && \
	uv tool run --index https://test.pypi.org/simple/ --index-strategy unsafe-best-match --with "bartz==$$VERSION" python -c 'import bartz; print(bartz.__version__)'


################# BENCHMARKS #################

ASV = $(UV_RUN_CI) python -m asv

.PHONY: asv-all-tags
asv-all-tags:
	git tag | $(ASV) run --skip-existing --show-stderr HASHFILE:- $(ARGS)

.PHONY: asv-publish
asv-publish:
	$(ASV) publish $(ARGS)

.PHONY: asv-preview
asv-preview: asv-publish
	$(ASV) preview $(ARGS)

.PHONY: asv-main
asv-main:
	$(ASV) run --show-stderr main^! $(ARGS)

.PHONY: asv-quick
asv-quick:
	$(ASV) run --python=same --quick --dry-run --show-stderr $(ARGS)


################# IPYTHON SHELL #################

.PHONY: ipython
ipython:
	IPYTHONDIR=config/ipython uv run --all-groups ipython $(ARGS)

.PHONY: ipython-old
ipython-old:
	IPYTHONDIR=config/ipython $(UV_VARS_OLD) uv run --all-groups $(UV_OPTS_OLD) ipython $(ARGS)
