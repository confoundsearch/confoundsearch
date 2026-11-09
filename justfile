# ConfoundSearch: development and release tasks. Run `just` to list them. Needs just >= 1.17.
#
# Secrets never go in this file. Put API tokens in `.env` (it is gitignored; `just` loads it):
#   TEST_PYPI_TOKEN=pypi-...      # from https://test.pypi.org/manage/account/token/
#   PYPI_API_TOKEN=pypi-...       # from https://pypi.org/manage/account/token/
#   ANON_BLOCKLIST=blocklist.txt  # optional: a file of regexes (your name, institution) that must not appear

set dotenv-load := true
set shell := ["bash", "-euo", "pipefail", "-c"]

py := env("PYTHON", "python")

default:
    @just --list --unsorted

# ---- everyday ---------------------------------------------------------------

# Install the package for development, with the test and release tools, and the pre-push check.
setup: hooks
    {{py}} -m pip install -e ".[dev,release]"

# Make git run `just check` before every push (installs the tracked hook in .githooks/).
hooks:
    git config core.hooksPath .githooks
    chmod +x .githooks/pre-push
    @echo "pre-push hook installed: a push now runs the checks first (skip once with: git push --no-verify)"

# Format the code and fix what can be fixed.
fmt:
    ruff format .
    ruff check --fix .

# Lint and type-check, as CI does.
lint:
    ruff check .
    ruff format --check .
    mypy confoundsearch devops/tools devops/proxy examples

# Run the tests, with the 100% branch-coverage gate. Extra arguments go to pytest.
test *args:
    {{py}} -m pytest {{args}}

# Uses ~/.blocklist (names to refuse) if it exists, and the clone in /tmp/confoundsearch.
# Update the anonymous public review repo from orgsci: copy the files, show the change, ask, commit as Anonymous, push.
public dest="/tmp/confoundsearch":
    bash devops/tools/update_review.sh {{dest}}

# Copy the files into the review clone without committing or pushing, to look first.
public-preview dest="/tmp/confoundsearch":
    bash devops/tools/sync_anonymous.sh {{dest}}

# Everything CI runs, on this machine.
check: lint test

# The live tests: Semantic Scholar, plus OpenRouter / OpenAI / the skill in Claude Code when their keys and tools are present.
live:
    {{py}} -m pytest -m live --no-cov

# The reviewer's view: install the released package from PyPI into a fresh venv and do a short real run (needs OPENAI_API_KEY).
live-pypi version="":
    bash devops/tools/pypi_live.sh {{version}}

# Does the latest release on PyPI install and work on Python 3.8 to 3.13? Needs uv (pip install uv). Pass a version to pin it.
compat target="confoundsearch" *args:
    {{py}} devops/tools/python_matrix.py {{target}} {{args}}

# All the integration tests: the live pytest suite, the released package from PyPI, and every Python (needs OPENAI_API_KEY).
integration: live live-pypi compat

# `just keywords` embeds the OpenAlex keywords (about $0.84 and 20 minutes; needs OPENAI_API_KEY). With an index
# name it also pushes them to Pinecone (needs PINECONE_API_KEY). A trial of 1000: `just keywords "" 1000`.

# Embed every OpenAlex keyword into keyword_vectors/, and push to Pinecone if you name an index. Safe to re-run.
keywords index="" limit="":
    {{py}} devops/tools/keyword_vectors.py build --out keyword_vectors {{ if limit != "" { "--limit " + limit } else { "" } }}
    {{ if index != "" { py + " devops/tools/keyword_vectors.py push --out keyword_vectors --index " + index + if limit != "" { " --limit " + limit } else { "" } } else { "echo 'built keyword_vectors/; name an index to push it to Pinecone'" } }}

# Re-embed every keyword with its description, then DELETE every record in the Pinecone index and load the new ones
# (about $1.50 and an hour; needs OPENAI_API_KEY and PINECONE_API_KEY). The old index is only deleted once the new
# vectors are built, and it does not ask. Running it again after an interruption carries on, and deletes nothing.

# Start over: re-embed with descriptions and replace everything in the index.
rebuild index:
    {{py}} devops/tools/keyword_vectors.py build --out keyword_vectors
    {{py}} devops/tools/keyword_vectors.py push --out keyword_vectors --index {{index}} --fresh --yes

# Does the domain name you own reveal you, is its DNS in Route 53, and does the service answer on it? Needs the internet.
check-domain domain host="":
    {{py}} devops/proxy/check_domain.py {{domain}} {{ if host != "" { "--host " + host } else { "" } }}

# Build the zip for the keyword proxy (devops/proxy/README.md says how to deploy it).
proxy-build:
    {{py}} devops/proxy/build.py

# The nearest keywords to each construct in the known-confounds file, from the Pinecone index (needs OPENAI_API_KEY and
# PINECONE_API_KEY). Writes keyword_neighbors.md. Add `--min-works 500` etc. as extra arguments.

# Look at the keywords nearest each known construct, and how many of the hoped-for terms show up.
neighbors index *args:
    {{py}} devops/tools/keyword_vectors.py neighbors examples/known_confounds.json --out keyword_vectors --index {{index}} --expect examples/known_construct_terms.json --report keyword_neighbors.md {{args}}

# Create the conda environment from requirements.yml, or update it if it already exists.
env:
    conda env create --file requirements.yml || conda env update --file requirements.yml --prune

# ---- releasing --------------------------------------------------------------

# The version that would be released. Change it in confoundsearch/__init__.py, the one place it lives.
version:
    @{{py}} -c "import confoundsearch; print(confoundsearch.__version__)"

# Remove build output, Python's bytecode caches, and the empty folders a reorganization leaves behind.
clean:
    rm -rf dist build *.egg-info
    find . -name __pycache__ -type d -not -path "./.git/*" -prune -exec rm -rf {} +
    find confoundsearch -type d -empty -delete

# Build the wheel and the source distribution into dist/.
build: clean
    {{py}} -m build

# Everything that can be checked before uploading: build, twine's check, the release gate, a clean install.
verify: build
    {{py}} -m twine check dist/*
    {{py}} devops/tools/release_check.py --dist dist ${ANON_BLOCKLIST:+--blocklist "$ANON_BLOCKLIST"}
    {{py}} devops/tools/smoke_test.py dist/*.whl

# Everything checked before an upload: lint, tests, the build, the index, and (real PyPI) a clean checkout.
preflight index="pypi": check verify
    {{py}} devops/tools/release_check.py --dist dist --online --index {{index}} {{ if index == "pypi" { "--git" } else { "" } }}

# Upload to TestPyPI, the practice index (needs TEST_PYPI_TOKEN), then install it back from there to check it.
[confirm("Upload to TestPyPI?")]
testpypi: (preflight "testpypi")
    @test -n "${TEST_PYPI_TOKEN:-}" || { echo "set TEST_PYPI_TOKEN in .env" >&2; exit 1; }
    TWINE_USERNAME=__token__ TWINE_PASSWORD="$TEST_PYPI_TOKEN" {{py}} -m twine upload --repository-url https://test.pypi.org/legacy/ dist/*
    just verify-published testpypi

# Publishing is PUBLIC and PERMANENT: a version can never be uploaded again, and the uploader's PyPI account
# name appears on the project page.

# Upload to the real PyPI (needs PYPI_API_TOKEN), then tag, push the tag, and check it installs from PyPI.
[confirm("Upload confoundsearch to the REAL PyPI? This is public and cannot be undone.")]
publish: (preflight "pypi")
    @test -n "${PYPI_API_TOKEN:-}" || { echo "set PYPI_API_TOKEN in .env" >&2; exit 1; }
    TWINE_USERNAME=__token__ TWINE_PASSWORD="$PYPI_API_TOKEN" {{py}} -m twine upload dist/*
    just tag
    just verify-published
    @echo ""
    @echo "Released, tagged and checked."

# One command to release: `just release 0.1.2` sets the version, commits it, checks everything, uploads to PyPI,
# tags, verifies the install from PyPI, and pushes. With no version it releases the current one.

# Release to the real PyPI in one go: [version] sets and commits it first, then everything else follows.
[confirm("Release to the REAL PyPI? This is public and cannot be undone.")]
release version="":
    #!/usr/bin/env bash
    set -euo pipefail
    if [ -n "{{version}}" ]; then
        if ! git diff --quiet || ! git diff --cached --quiet; then
            echo "commit or stash your changes first, so the release commit holds only the version" >&2
            exit 1
        fi
        {{py}} devops/tools/set_version.py "{{version}}"
        git commit -q -am "Release {{version}}"
    fi
    just --yes publish
    git push --no-verify origin HEAD   # the checks just ran on this exact commit, so the hook would only repeat them
    echo "Released and pushed."

# Tag the released commit as v<version> and push the tag. Safe to repeat; `publish` runs it for you.
tag:
    #!/usr/bin/env bash
    set -euo pipefail
    v=$({{py}} -c "import confoundsearch; print(confoundsearch.__version__)")
    if git rev-parse -q --verify "refs/tags/v$v" >/dev/null; then
        if [ "$(git rev-parse "v$v^{commit}")" != "$(git rev-parse HEAD)" ]; then
            echo "tag v$v already exists at a different commit" >&2
            exit 1
        fi
        echo "tag v$v already exists at this commit"
    else
        git tag -a "v$v" -m "confoundsearch $v"
        echo "created tag v$v"
    fi
    git push origin "v$v" || echo "could not push the tag; push it yourself: git push origin v$v"

# Install the released version into a fresh environment and use it, retrying while the index catches up.
verify-published index="pypi":
    #!/usr/bin/env bash
    set -euo pipefail
    v=$({{py}} -c "import confoundsearch; print(confoundsearch.__version__)")
    if [ "{{index}}" = "testpypi" ]; then
        {{py}} devops/tools/smoke_test.py "confoundsearch==$v" --index-url https://test.pypi.org/simple/ --retries 12 --wait 15 --exact-file
    else
        {{py}} devops/tools/smoke_test.py "confoundsearch==$v" --retries 12 --wait 15 --exact-file
    fi
