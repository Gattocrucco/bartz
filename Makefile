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
	@echo "- lock: determine versions of dependencies and pin them"
	@echo "- setup: create a python environment and install everything"
	@echo "- lock-old, setup-old: the same with lowest supported versions"
	@echo "- tests: run unit tests, saving coverage information"
	@echo "- docs: build html documentation"
	@echo "- covreport: build html coverage report"
	@echo "- release: packages the python module, invokes tests and docs first"
	@echo "- upload: upload release to PyPI"
	@echo
	@echo "Release instructions:"
	@echo "- $$ uv version --bump major|minor|patch"
	@echo "- describe release in docs/changelog.md"
	@echo "- $$ make release (repeat until it goes smoothly)"
	@echo "- push and check CI completes (if it doesn't, go to previous step)"
	@echo "- $$ make upload"
	## make upload should also update the docs automatically because it pushes
	## a tag which triggers a workflow, however last time it didn't work. I made
	## a change to try to fix it, see if it works on the next release.
	@echo "- publish github release (updates zenodo automatically)"
	@echo "- press 'run workflow' on https://github.com/Gattocrucco/bartz/actions/workflows/tests.yml"

SETUP_MICROMAMBA = micromamba env create --file config/condaenv.yml --prefix ./.venv --yes
UV_RUN = uv run --no-sync
UV_SYNC = uv sync --frozen --inexact

.PHONY: lock
lock:
	uv lock
	mv uv.lock config/uv-highest.lock

.PHONY: setup
setup:
	$(SETUP_MICROMAMBA)
	cp config/uv-highest.lock uv.lock
	$(UV_SYNC) --all-groups
	$(UV_RUN) pre-commit install

.PHONY: setup-ci
setup-ci:
	$(SETUP_MICROMAMBA)
	cp config/uv-highest.lock uv.lock
	$(UV_SYNC) --group ci

.PHONY: lock-old
lock-old:
	uv lock --resolution lowest-direct --exclude-newer $(OLD_DATE)
	mv uv.lock config/uv-lowest-direct.lock

.PHONY: setup-old
setup-old:
	$(SETUP_MICROMAMBA) python=$(OLD_PYTHON)
	cp config/uv-lowest-direct.lock uv.lock
	$(UV_SYNC) --all-groups
	$(UV_RUN) pre-commit install

.PHONY: setup-ci-old
setup-ci-old:
	$(SETUP_MICROMAMBA) python=$(OLD_PYTHON)
	cp config/uv-lowest-direct.lock uv.lock
	$(UV_SYNC) --group ci

.PHONY: release
release: copy-version lock lock-old check-committed setup-old tests setup tests docs
	test ! -d dist || rm -r dist
	uv build

.PHONY: check-committed
check-committed:
	git diff --quiet
	git diff --quiet --staged

.PHONY: copy-version
copy-version: src/bartz/_version.py
src/bartz/_version.py: pyproject.toml
	$(UV_RUN) python -c 'import tomli, pathlib; version = tomli.load(open("pyproject.toml", "rb"))["project"]["version"]; pathlib.Path("src/bartz/_version.py").write_text(f"__version__ = {version!r}\n")'

.PHONY: tests
tests:
	$(UV_RUN) coverage run --data-file=.coverage.tests$(COVERAGE_SUFFIX) --context=tests$(COVERAGE_SUFFIX) -m pytest $(ARGS)

# I did not manage to make parallel pytest (pytest -n<processes>) work with
# coverage

.PHONY: docs-latest
docs-latest:
	BARTZ_DOC_VARIANT=latest $(UV_RUN) make -C docs html
	git switch - || git switch main
	test ! -d _site/docs || rm -r _site/docs
	mv docs/_build/html _site/docs

.PHONY: docs
docs:
	$(UV_RUN) make -C docs html
	test ! -d _site/docs-dev || rm -r _site/docs-dev
	mv docs/_build/html _site/docs-dev

.PHONY: docs-all
docs-all: copy-version docs-latest docs
	@echo
	@echo "Now open _site/index.html"

.PHONY: covreport
covreport:
	$(UV_RUN) coverage combine
	$(UV_RUN) coverage html
	@echo
	@echo "Now open _site/index.html"

.PHONY: version-tag
version-tag: copy-version check-committed
	git fetch --tags
	git tag v$(shell $(UV_RUN) python -c 'import bartz; print(bartz.__version__)')
	git push --tags

.PHONY: upload
upload: version-tag
	@echo "Enter PyPI token:"
	@read -s UV_PUBLISH_TOKEN && \
	export UV_PUBLISH_TOKEN="$$UV_PUBLISH_TOKEN" && \
	uv publish
	@VERSION=$$($(UV_RUN) python -c 'import bartz; print(bartz.__version__)') && \
	echo "Try to install bartz $$VERSION from PyPI" && \
	uv tool run --with "bartz==$$VERSION" python -c 'import bartz; print(bartz.__version__)'

.PHONY: upload-test
upload-test: check-committed
	@echo "Enter TestPyPI token:"
	@read -s UV_PUBLISH_TOKEN && \
	export UV_PUBLISH_TOKEN="$$UV_PUBLISH_TOKEN" && \
	uv publish --check-url https://test.pypi.org/simple/ --publish-url https://test.pypi.org/legacy/
	@VERSION=$$($(UV_RUN) python -c 'import tomli; print(tomli.load(open("pyproject.toml", "rb"))["project"]["version"])') && \
	echo "Try to install bartz $$VERSION from TestPyPI" && \
	uv tool run --index https://test.pypi.org/simple/ --index-strategy unsafe-best-match --with "bartz==$$VERSION" python -c 'import bartz; print(bartz.__version__)'

.PHONY: benchmark-tags
benchmark-tags:
	git tag | $(UV_RUN) asv run --skip-existing --show-stderr HASHFILE:- $(ARGS)

.PHONY: benchmark-site
benchmark-site:
	$(UV_RUN) asv publish $(ARGS)

.PHONY: benchmark-server
benchmark-server: benchmark-site
	$(UV_RUN) asv preview $(ARGS)

.PHONY: benchmark-current
benchmark-current: check-committed
	$(UV_RUN) asv run --show-stderr main^! $(ARGS)

.PHONY: python
python:
	$(UV_RUN) python $(ARGS)

.PHONY: ipython
ipython:
	IPYTHONDIR=config/ipython $(UV_RUN) ipython $(ARGS)

.PHONY: mypy
mypy:
	$(UV_RUN) mypy $(ARGS)
