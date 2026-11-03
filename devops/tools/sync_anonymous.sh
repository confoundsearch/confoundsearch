#!/usr/bin/env bash
# Copy the files of a branch into the folder of the anonymous review repository. Files only, no git.
#
#   devops/tools/sync_anonymous.sh [DEST]          DEST defaults to /tmp/confoundsearch
#
#   SOURCE_REF  the branch (or tag) to copy     default: orgsci
#   EXCLUDE     folders or files left out of the public copy, space separated     default: examples
#   BLOCKLIST   file of patterns that reveal you, one regex per line (keep it outside this repository)
#
# DEST is a clone of the review repository (or an empty folder). Its files are made to match the branch
# exactly: new files appear, changed ones are replaced, deleted ones are removed. Its .git is left alone
# and nothing is committed or pushed; no history, author or session link of this repository comes along,
# because only the committed files are exported. The files are checked against BLOCKLIST first, and
# nothing is touched if any match. Then you commit and push in DEST yourself.
set -euo pipefail

dest="${1:-/tmp/confoundsearch}"
ref="${SOURCE_REF:-orgsci}"
blocklist="${BLOCKLIST:-}"
[ -n "$blocklist" ] || { [ ! -f "$HOME/.blocklist" ] || blocklist="$HOME/.blocklist"; }  # the usual place
[ -n "$blocklist" ] || echo "warning: no BLOCKLIST given, so nothing checks the files for names" >&2

git rev-parse --verify -q "$ref^{commit}" > /dev/null || { echo "no such branch or tag: $ref" >&2; exit 2; }
here="$(cd "$(dirname "$0")" && pwd)"
cd "$(git rev-parse --show-toplevel)"
if [ -e "$dest" ] && [ ! -d "$dest/.git" ] && [ -n "$(ls -A "$dest")" ]; then
  echo "$dest is neither a git clone nor empty; refusing to overwrite it" >&2
  exit 2
fi

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
SNAPSHOT_REF="$ref" SNAPSHOT_EXCLUDE="${EXCLUDE-examples}" "$here/anonymous_snapshot.sh" "$work/snapshot" ${blocklist:+"$blocklist"} > /dev/null

mkdir -p "$dest"
find "$dest" -mindepth 1 -maxdepth 1 ! -name .git -exec rm -rf {} +  # so files deleted here are deleted there
tar -C "$work/snapshot" --exclude=./.git -cf - . | tar -x -C "$dest"
count="$(cd "$work/snapshot" && git ls-files | wc -l | tr -d ' ')"

echo "Copied $count files from $ref to $dest (no git history or author information)."
if [ -d "$dest/.git" ]; then
  echo "Now commit and push, with a neutral identity so your own name is not recorded:"
  echo "  cd $dest && git add -A && git -c user.name=Anonymous -c user.email=anonymous@example.com commit -m Update && git push"
fi
