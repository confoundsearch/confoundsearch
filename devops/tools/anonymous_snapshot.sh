#!/usr/bin/env bash
# Make an anonymous, history-free copy of the committed files, for a blind-review repository.
#
#   devops/tools/anonymous_snapshot.sh DEST_DIR [BLOCKLIST_FILE]
#
# DEST_DIR gets the files of HEAD (committed files only) and one commit by "Anonymous", so none
# of this repository's history, authors or session links come along. If BLOCKLIST_FILE is given
# (one extended regex per line: your names, institution, email, usernames; keep that file out of
# the repository), the snapshot's contents AND its file names are searched for those patterns and
# the script fails if any match.
set -euo pipefail

dest="${1:?usage: anonymous_snapshot.sh DEST_DIR [BLOCKLIST_FILE]}"
blocklist="${2:-}"
[ -z "$blocklist" ] || [ -f "$blocklist" ] || { echo "blocklist file not found: $blocklist" >&2; exit 2; }
[ ! -e "$dest" ] || [ -z "$(ls -A "$dest")" ] || { echo "$dest exists and is not empty" >&2; exit 2; }

mkdir -p "$dest"
git archive "${SNAPSHOT_REF:-HEAD}" | tar -x -C "$dest"  # SNAPSHOT_REF: another branch or tag

for path in ${SNAPSHOT_EXCLUDE:-}; do rm -rf "${dest:?}/$path"; done  # left out before the check: they never ship

if [ -n "$blocklist" ]; then
  if grep -rInEf "$blocklist" "$dest"; then
    echo "identifying text found in the snapshot (see above); fix it and commit, then try again" >&2
    rm -rf "$dest"
    exit 1
  fi
  if (cd "$dest" && find . -type f | grep -Ef "$blocklist"); then
    echo "identifying text found in a file name (see above); rename the file and commit, then try again" >&2
    rm -rf "$dest"
    exit 1
  fi
fi

cd "$dest"
git init -q -b main
git add -A
git -c user.name=Anonymous -c user.email=anonymous@example.com -c commit.gpgsign=false commit -q -m "Anonymous submission"
echo "Anonymous snapshot ready in $dest ($(git ls-files | wc -l) files, 1 commit)."
