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

cd "$dest"
git add -A
if git diff --cached --quiet; then
  echo "Nothing new to publish: $dest already matches."
  exit 0
fi
git diff --cached --stat | tail -n 15
if [ -z "${ASSUME_YES:-}" ]; then
  read -r -p "Push these changes to $(git remote get-url origin)? [y/N] " answer
  [ "$answer" = y ] || [ "$answer" = Y ] || { echo "Not pushed. The files are copied and staged in $dest." >&2; exit 1; }
fi
git -c user.name=Anonymous -c user.email=anonymous@example.com -c commit.gpgsign=false commit -q -m "Update"
git push -u origin HEAD
echo "Published."
