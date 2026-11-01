#!/usr/bin/env bash
# Update the anonymous review repository in one go: copy the files, commit as "Anonymous", push.
#
#   BLOCKLIST=~/.blocklist bash devops/tools/update_review.sh [DEST]      DEST defaults to /tmp/confoundsearch
#
# Each run starts from a fresh clone of the review repository in DEST (over ssh), so there is nothing stale
# to go wrong. The address is the review repository's (set REVIEW_REMOTE to use another). If
# ~/.ssh/confoundsearch_deploy (or $DEPLOY_KEY) exists it is used, and an https address becomes its ssh form,
# so no password is ever asked for. The files are then copied in, shown, and, after you say yes, committed
# as "Anonymous" and pushed. ASSUME_YES=1 skips the question.
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
dest="${1:-/tmp/confoundsearch}"
ref="${SOURCE_REF:-orgsci}"
if git remote get-url origin > /dev/null 2>&1; then
  if git fetch -q origin "$ref" 2> /dev/null; then
    [ "$(git rev-parse "$ref")" = "$(git rev-parse FETCH_HEAD)" ] || {
      echo "your $ref differs from origin's: run  git pull origin $ref  (or push) first, so the review copy matches what is shared" >&2
      exit 1
    }
  else
    echo "warning: could not check $ref against origin" >&2
  fi
fi

# Where to clone from.
remote="${REVIEW_REMOTE:-}"
if [ -z "$remote" ] && [ -d "$dest/.git" ]; then remote="$(git -C "$dest" remote get-url origin 2> /dev/null || true)"; fi
remote="${remote:-https://github.com/confoundsearch/confoundsearch}"  # the review repository; becomes ssh below
key="${DEPLOY_KEY:-$HOME/.ssh/confoundsearch_deploy}"
if [ -f "$key" ]; then  # the ssh form of an https address, and the repository's own key
  remote="$(printf '%s' "$remote" | sed 's#^https://\([^/]*\)/#git@\1:#')"
  export GIT_SSH_COMMAND="ssh -i $key -o IdentitiesOnly=yes"
fi

# Start fresh. Only ever removes an earlier clone, never this repository or a folder of other things.
if [ -e "$dest" ] && [ -n "$(ls -A "$dest")" ]; then
  [ -d "$dest/.git" ] || { echo "$dest is neither a git clone nor empty; refusing to replace it" >&2; exit 2; }
  [ "$(cd "$dest" && pwd -P)" != "$(git rev-parse --show-toplevel)" ] || { echo "$dest is this repository" >&2; exit 2; }
  rm -rf "${dest:?}"
fi
git clone -q "$remote" "$dest" 2> /dev/null || {
  echo "could not clone $remote (is the deploy key added to it, with write access?)" >&2
  exit 1
}
git -C "$dest" rev-parse -q --verify HEAD > /dev/null || git -C "$dest" checkout -q -B "${TARGET:-main}"  # empty repository
[ ! -f "$key" ] || git -C "$dest" config core.sshCommand "ssh -i $key -o IdentitiesOnly=yes"

bash "$here/sync_anonymous.sh" "$dest" > /dev/null

# How many files $ref has once the left-out ones are removed: what the remote should hold.
left_out="${EXCLUDE-examples}"
expected="$(git ls-tree -r --name-only "$ref" | awk -v skip="$left_out" '
  BEGIN { n = split(skip, s, " ") }
  { for (i = 1; i <= n; i++) if ($0 == s[i] || index($0, s[i] "/") == 1) next; print }' | wc -l | tr -d ' ')"

cd "$dest"
git add -A
count="$(git ls-files | wc -l | tr -d ' ')"

# Confirm that what is on the remote is exactly what was copied from $ref: clone it again from scratch and
# compare the whole tree (every file and its content), not only what is new.
verify() {
  local check want got
  check="$(mktemp -d)"
  git clone -q "$remote" "$check/repo" 2> /dev/null || { echo "could not clone $remote again to check it" >&2; rm -rf "$check"; exit 1; }
  want="$(git write-tree)"
  got="$(git -C "$check/repo" rev-parse 'HEAD^{tree}')"
  rm -rf "$check"
  [ "$count" = "$expected" ] || { echo "NOT verified: $ref has $expected files to publish but $count were copied" >&2; exit 1; }
  if [ "$want" = "$got" ]; then
    echo "Verified: the repository on the remote has all $count files, identical to $ref${left_out:+ (without: $left_out)}."
  else
    echo "NOT verified: the remote differs from what was copied from $ref. Run it again; if it persists, look at the remote." >&2
    exit 1
  fi
}

if git diff --cached --quiet; then
  echo "Nothing new to publish: $dest already matches."
  verify
  exit 0
fi
git diff --cached --stat | tail -n 15
if [ -z "${ASSUME_YES:-}" ]; then
  read -r -p "Push these changes to $(git remote get-url origin)? [y/N] " answer
  [ "$answer" = y ] || [ "$answer" = Y ] || { echo "Not pushed. The files are copied and staged in $dest." >&2; exit 1; }
fi
git -c user.name=Anonymous -c user.email=anonymous@example.com -c commit.gpgsign=false commit -q -m "Update"
git push -q -u origin HEAD
echo "Published."
verify
