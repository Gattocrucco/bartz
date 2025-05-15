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

COVERAGE_SUFFIX=

.PHONY: all
all:
	@echo "Available targets:"
	@echo "- setup: create a python environment and install everything"
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

SETUP_MICROMAMBA = micromamba env create --file condaenv.yml --prefix ./.venv --yes
UV_RUN = uv run --no-sync

.PHONY: lock
lock:
	uv lock
	mv uv.lock uv-highest.lock

.PHONY: setup
setup:
	$(SETUP_MICROMAMBA)
	cp uv-highest.lock uv.lock
	uv sync --locked --inexact --all-groups
	$(UV_RUN) pre-commit install

.PHONY: setup-ci
setup-ci:
	$(SETUP_MICROMAMBA)
	cp uv-highest.lock uv.lock
	uv sync --locked --inexact --group ci

.PHONY: lock-old
lock-old:
	uv lock --resolution lowest-direct
	mv uv.lock uv-lowest-direct.lock

.PHONY: setup-old
setup-old:
	$(SETUP_MICROMAMBA) python=3.10
	cp uv-lowest-direct.lock uv.lock
	uv sync --locked --inexact --all-groups --resolution lowest-direct
	$(UV_RUN) pre-commit install

.PHONY: setup-ci-old
setup-ci-old:
	$(SETUP_MICROMAMBA) python=3.10
	cp uv-lowest-direct.lock uv.lock
	uv sync --locked --inexact --group ci --resolution lowest-direct

.PHONY: release
release: copy-version tests docs
	git diff --quiet
	git diff --quiet --staged
	test ! -d dist || rm -r dist
	uv build

.PHONY: copy-version
copy-version: src/bartz/_version.py
src/bartz/_version.py: pyproject.toml
	$(UV_RUN) python -c 'import tomli, pathlib; version = tomli.load(open("pyproject.toml", "rb"))["project"]["version"]; pathlib.Path("src/bartz/_version.py").write_text(f"__version__ = {version!r}\n")'

.PHONY: tests
tests:
	COVERAGE_FILE=.coverage.tests$(COVERAGE_SUFFIX) $(UV_RUN) coverage run --context=tests$(COVERAGE_SUFFIX) -m pytest tests

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
	test ! -d _site/coverage || rm -r _site/coverage
	mv htmlcov _site/coverage
	@echo
	@echo "Now open _site/index.html"

.PHONY: version-tag
version-tag: copy-version
	git diff --quiet
	git diff --quiet --staged
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
upload-test:
	git diff --quiet
	git diff --quiet --staged
	@echo "Enter TestPyPI token:"
	@read -s UV_PUBLISH_TOKEN && \
	export UV_PUBLISH_TOKEN="$$UV_PUBLISH_TOKEN" && \
	uv publish --check-url https://test.pypi.org/simple/ --publish-url https://test.pypi.org/legacy/
	@VERSION=$$($(UV_RUN) python -c 'import tomli; print(tomli.load(open("pyproject.toml", "rb"))["project"]["version"])') && \
	echo "Try to install bartz $$VERSION from TestPyPI" && \
	uv tool run --index https://test.pypi.org/simple/ --index-strategy unsafe-best-match --with "bartz==$$VERSION" python -c 'import bartz; print(bartz.__version__)'

.PHONY: benchmark-tags
benchmark-tags:
	git tag | $(UV_RUN) asv run --skip-existing HASHFILE:-

.PHONY: benchmark-site
benchmark-site:
	$(UV_RUN) asv publish
