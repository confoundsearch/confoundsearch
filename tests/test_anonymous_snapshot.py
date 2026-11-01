from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
SCRIPT = ROOT / "devops" / "tools" / "anonymous_snapshot.sh"

pytestmark = pytest.mark.skipif(
    shutil.which("git") is None or not (ROOT / ".git").exists(), reason="needs a git checkout"
)


def run(*args, cwd=ROOT):
    return subprocess.run([str(SCRIPT), *map(str, args)], cwd=cwd, capture_output=True, text=True)


def git(dest, *args):
    return subprocess.run(["git", *args], cwd=dest, capture_output=True, text=True, check=True).stdout.strip()


def test_snapshot_has_one_anonymous_commit_and_the_committed_files(tmp_path):
    dest = tmp_path / "anon"
    result = run(dest)
    assert result.returncode == 0, result.stderr
    assert git(dest, "rev-list", "--count", "HEAD") == "1"
    assert git(dest, "log", "-1", "--format=%an <%ae>") == "Anonymous <anonymous@example.com>"
    assert git(dest, "log", "-1", "--format=%s") == "Anonymous submission"
    tracked = set(git(dest, "ls-files").splitlines())
    assert {"README.md", "pyproject.toml", "confoundsearch/__init__.py"} <= tracked
    assert not any(name.startswith(".git/") for name in tracked)


def test_snapshot_matches_head_exactly(tmp_path):
    dest = tmp_path / "anon"
    assert run(dest).returncode == 0
    head = set(git(ROOT, "ls-tree", "-r", "--name-only", "HEAD").splitlines())
    assert set(git(dest, "ls-files").splitlines()) == head
    assert (dest / "README.md").read_bytes() == (
        subprocess.run(["git", "show", "HEAD:README.md"], cwd=ROOT, capture_output=True).stdout
    )


def test_a_blocklist_hit_fails_and_leaves_nothing_behind(tmp_path):
    blocklist = tmp_path / "block.txt"
    blocklist.write_text("zzz-" + "not-present\nconfoundsearch" + " serve\n")
    dest = tmp_path / "anon"
    result = run(dest, blocklist)
    assert result.returncode == 1 and "identifying text found" in result.stderr
    assert not dest.exists()


def test_a_name_in_a_file_name_is_caught_too(tmp_path):
    """The contents can be clean while a file's name gives someone away."""
    name = "evals/lexical_relationship" + "_rating.html"  # split: this file must not match itself
    blocklist = tmp_path / "block.txt"
    blocklist.write_text(name.replace(".", "\\.") + "\n")
    dest = tmp_path / "anon"
    result = run(dest, blocklist)
    assert result.returncode == 1 and "in a file name" in result.stderr
    assert name in result.stdout and not dest.exists()


def test_a_clean_blocklist_passes(tmp_path):
    blocklist = tmp_path / "block.txt"
    blocklist.write_text("zzz-" + "never-in-the-repo" + "-zzz\n")  # split so this file does not match itself
    result = run(tmp_path / "anon", blocklist)
    assert result.returncode == 0 and "1 commit" in result.stdout


def test_refuses_a_missing_blocklist_or_a_nonempty_destination(tmp_path):
    assert run(tmp_path / "a", tmp_path / "missing.txt").returncode == 2
    full = tmp_path / "full"
    full.mkdir()
    (full / "x").write_text("x")
    assert run(full).returncode == 2


ALLOWED_GITHUB_OWNERS = {"astral-sh", "confoundsearch"}  # the ruff badge; the anonymous review account
ALLOWED_EMAIL = ("example.com", "example.org", "noreply.github.com", "noreply.anthropic.com")
SESSION = "claude" + ".ai/code"  # written in two parts so this file does not match itself
OWNER = re.compile(r"github\.com/([A-Za-z0-9_-]+)/")
EMAIL = re.compile(r"[\w.+-]+@([\w-]+(?:\.[\w-]+)+)")


def test_the_tracked_files_name_no_owner_email_or_session():
    """No tracked file names a repository owner, a real email address, or a session link."""
    listing = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True).stdout
    offenders = []
    for name in filter(None, listing.decode().split("\0")):
        if name.startswith("devops/badges/"):
            continue
        try:
            text = (ROOT / name).read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        offenders += [
            (name, f"github.com/{o}") for o in OWNER.findall(text) if o not in ALLOWED_GITHUB_OWNERS
        ]
        offenders += [(name, m) for m in EMAIL.findall(text) if not m.endswith(ALLOWED_EMAIL)]
        if SESSION in text:
            offenders.append((name, "session link"))
    assert offenders == []
