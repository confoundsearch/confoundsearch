#!/usr/bin/env bash
# What a reviewer does, cheaply: install the released package into a fresh venv, run a short search on real
# papers with a small model, and verify the audit trail. Needs OPENAI_API_KEY; S2_KEY is optional.
#   devops/tools/pypi_live.sh [version]        e.g. devops/tools/pypi_live.sh 0.1.0   (default: the latest on PyPI)
set -euo pipefail
: "${OPENAI_API_KEY:?set OPENAI_API_KEY first (cloud environment variable, or your .env)}"
spec="confoundsearch${1:+==$1}"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
python3 -m venv "$work/venv"
"$work/venv/bin/pip" install --quiet "$spec"
cd "$work"  # nothing from the repository is on the path
"$work/venv/bin/confoundsearch" run --corpus semanticscholar --model "${OPENAI_MODEL:-gpt-4o-mini}" \
  --iv "IT investment" --dv "firm performance" --max-steps "${MAX_STEPS:-5}" \
  --out result.html --audit audit.jsonl
"$work/venv/bin/confoundsearch" audit audit.jsonl
echo "OK: $("$work/venv/bin/pip" show confoundsearch | grep '^Version') installed from PyPI, ran, audit verified"
