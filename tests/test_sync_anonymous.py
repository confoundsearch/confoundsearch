"""Copying the files of a branch into the review repository's folder, tried on small repositories."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
SCRIPT = ROOT / "devops" / "tools" / "sync_anonymous.sh"

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="needs git")


def git(cwd, *args):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()


@pytest.fixture
def source(tmp_path):
    """A private repository with a branch called orgsci, as ours is."""
    repo = tmp_path / "source"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "orgsci")
    (repo / "a.txt").write_text("one\n")
    (repo / "gone.txt").write_text("bye\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "private history by Somebody")
    return repo


@pytest.fixture
def dest(tmp_path):
    """A fresh clone of an empty review repository."""
    bare = tmp_path / "anon.git"
    git(tmp_path, "init", "-q", "--bare", "-b", "main", str(bare))
    git(tmp_path, "clone", "-q", str(bare), str(tmp_path / "clone"))
    return tmp_path / "clone"


def sync(source, *args, env=None):
    return subprocess.run(
        [str(SCRIPT), *map(str, args)],
        cwd=source,
        capture_output=True,
        text=True,
        env={**os.environ, **(env or {})},
    )


def files(folder):
    return {
        p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file() and ".git" not in p.parts
    }


def test_the_files_are_copied_and_nothing_of_git_comes_with_them(source, dest):
    result = sync(source, dest)
    assert result.returncode == 0, result.stderr
    assert files(dest) == {"a.txt", "gone.txt"} and (dest / "a.txt").read_text() == "one\n"
    assert git(dest, "branch", "--list") == "" and git(dest, "status", "--short").count("??") == 2
    assert "Somebody" not in "".join(p.read_text(errors="ignore") for p in dest.rglob("*") if p.is_file())
    assert (
        "Anonymous" in result.stdout and "Copied 2 files" in result.stdout
    )  # the commit command it suggests


def test_later_syncs_add_change_and_remove_files(source, dest):
    sync(source, dest)
    (source / "a.txt").write_text("two\n")
    (source / "new.txt").write_text("new\n")
    (source / "sub").mkdir()
    (source / "sub" / "deep.txt").write_text("deep\n")
    git(source, "rm", "-q", "gone.txt")
    git(source, "add", "-A")
    git(source, "commit", "-q", "-m", "second")
    assert sync(source, dest).returncode == 0
    assert files(dest) == {"a.txt", "new.txt", "sub/deep.txt"} and (dest / "a.txt").read_text() == "two\n"


def test_the_clones_own_git_folder_is_left_alone(source, dest):
    git(dest, "config", "user.name", "Anonymous")
    sync(source, dest)
    assert git(dest, "config", "user.name") == "Anonymous"


def test_only_committed_files_of_the_chosen_branch_are_copied(source, dest):
    (source / "untracked.txt").write_text("not committed\n")
    (source / "a.txt").write_text("edited but not committed\n")
    git(source, "checkout", "-q", "-b", "other")
    (source / "secret.txt").write_text("wip\n")
    git(source, "add", "secret.txt")
    git(source, "commit", "-q", "-m", "wip")
    git(source, "checkout", "-q", "orgsci")
    sync(source, dest)
    assert files(dest) == {"a.txt", "gone.txt"} and (dest / "a.txt").read_text() == "one\n"
    sync(source, dest, env={"SOURCE_REF": "other"})
    assert "secret.txt" in files(dest)


def test_an_empty_or_new_folder_is_fine_and_says_no_git_step_when_it_is_not_a_clone(source, tmp_path):
    fresh = tmp_path / "not-yet"
    result = sync(source, fresh)
    assert result.returncode == 0 and files(fresh) == {"a.txt", "gone.txt"} and "commit" not in result.stdout


def test_a_folder_of_other_things_is_never_wiped(source, tmp_path):
    other = tmp_path / "precious"
    other.mkdir()
    (other / "thesis.docx").write_text("mine\n")
    result = sync(source, other)
    assert result.returncode == 2 and "refusing" in result.stderr
    assert (other / "thesis.docx").exists()


def test_a_blocklist_hit_changes_nothing(source, dest, tmp_path):
    sync(source, dest)
    (source / "a.txt").write_text("written by Jane Q. Researcher\n")
    git(source, "commit", "-q", "-am", "oops")
    block = tmp_path / "block.txt"
    block.write_text("Jane Q\\. Researcher\n")
    result = sync(source, dest, env={"BLOCKLIST": str(block)})
    assert result.returncode == 1 and "identifying text found" in result.stderr
    assert (dest / "a.txt").read_text() == "one\n"  # the earlier copy is untouched


def test_without_a_blocklist_it_warns(source, dest):
    assert "no BLOCKLIST" in sync(source, dest).stderr


def test_an_unknown_branch_is_refused(source, dest):
    result = sync(source, dest, env={"SOURCE_REF": "no-such-branch"})
    assert result.returncode == 2 and "no such branch" in result.stderr
    assert files(dest) == set()


UPDATE = ROOT / "devops" / "tools" / "update_review.sh"


def update(source, dest, *, answer=None, env=None):
    return subprocess.run(
        [str(UPDATE), str(dest)],
        cwd=source,
        capture_output=True,
        text=True,
        input=answer,
        env={**os.environ, **(env or {})},
    )


def pushed(dest):
    bare = git(dest, "remote", "get-url", "origin")
    return git(bare, "rev-list", "--count", "main") if git(bare, "branch", "--list") else "0"


def test_update_copies_commits_as_anonymous_and_pushes(source, dest):
    result = update(source, dest, env={"ASSUME_YES": "1"})
    assert result.returncode == 0, result.stderr
    bare = git(dest, "remote", "get-url", "origin")
    assert (
        git(bare, "log", "-1", "--format=%an <%ae> %cn %s")
        == "Anonymous <anonymous@example.com> Anonymous Update"
    )
    assert set(git(bare, "ls-tree", "-r", "--name-only", "main").splitlines()) == {"a.txt", "gone.txt"}


def test_update_ignores_a_signing_setting_so_the_commit_is_unsigned(source, dest):
    signing = {
        "GIT_CONFIG_COUNT": "2",
        "GIT_CONFIG_KEY_0": "commit.gpgsign",
        "GIT_CONFIG_VALUE_0": "true",
        "GIT_CONFIG_KEY_1": "user.signingkey",
        "GIT_CONFIG_VALUE_1": "does-not-exist",
    }
    assert update(source, dest, env={"ASSUME_YES": "1", **signing}).returncode == 0


def test_update_asks_first_and_a_no_pushes_nothing(source, dest):
    result = update(source, dest, answer="n\n")
    assert result.returncode == 1 and "Not pushed" in result.stderr
    assert pushed(dest) == "0"
    assert update(source, dest, answer="y\n").returncode == 0 and pushed(dest) == "1"


def test_update_with_nothing_new_pushes_nothing(source, dest):
    update(source, dest, env={"ASSUME_YES": "1"})
    result = update(source, dest, env={"ASSUME_YES": "1"})
    assert result.returncode == 0 and "Nothing new" in result.stdout and pushed(dest) == "1"
    assert "Verified" in result.stdout  # still checked against the remote, even with nothing to push


def test_update_says_when_it_cannot_clone(source, tmp_path):
    result = update(
        source, tmp_path / "plain", env={"ASSUME_YES": "1", "REVIEW_REMOTE": str(tmp_path / "missing.git")}
    )
    assert result.returncode == 1 and "could not clone" in result.stderr and "deploy key" in result.stderr


def test_the_review_repository_is_the_default_address():
    """Read from the script, not tried: running it with no address would reach for the real repository."""
    script = UPDATE.read_text()
    assert 'remote="${remote:-https://github.com/confoundsearch/confoundsearch}"' in script


def test_update_refuses_when_the_branch_is_not_what_origin_has(source, dest, tmp_path):
    origin = tmp_path / "origin.git"
    git(tmp_path, "init", "-q", "--bare", "-b", "orgsci", str(origin))
    git(source, "remote", "add", "origin", str(origin))
    git(source, "push", "-q", "origin", "orgsci")
    assert update(source, dest, env={"ASSUME_YES": "1"}).returncode == 0  # in step with origin
    (source / "a.txt").write_text("changed but not pushed\n")
    git(source, "commit", "-q", "-am", "local only")
    result = update(source, dest, env={"ASSUME_YES": "1"})
    assert result.returncode == 1 and "git pull origin orgsci" in result.stderr


def test_update_carries_on_with_a_warning_when_origin_cannot_be_reached(source, dest, tmp_path):
    git(source, "remote", "add", "origin", str(tmp_path / "nowhere.git"))
    result = update(source, dest, env={"ASSUME_YES": "1"})
    assert result.returncode == 0 and "could not check" in result.stderr


def test_the_usual_blocklist_is_used_when_none_is_named(source, dest, tmp_path):
    (source / "a.txt").write_text("by Jane Q. Researcher\n")
    git(source, "commit", "-q", "-am", "oops")
    home = tmp_path / "home"
    home.mkdir()
    (home / ".blocklist").write_text("Jane Q\\. Researcher\n")
    result = update(source, dest, env={"ASSUME_YES": "1", "HOME": str(home)})
    assert result.returncode == 1 and "identifying text found" in result.stderr


@pytest.fixture
def with_examples(source):
    (source / "examples").mkdir()
    (source / "examples" / "demo.txt").write_text("private demo\n")
    (source / "keep").mkdir()
    (source / "keep" / "k.txt").write_text("k\n")
    git(source, "add", "-A")
    git(source, "commit", "-q", "-m", "examples")
    return source


def test_examples_stay_out_of_the_public_copy_by_default(with_examples, dest):
    (dest / "examples").mkdir()
    (dest / "examples" / "old.txt").write_text("from an earlier sync\n")
    assert sync(with_examples, dest).returncode == 0
    assert files(dest) == {"a.txt", "gone.txt", "keep/k.txt"}  # and an earlier copy is removed


def test_what_is_left_out_can_be_chosen(with_examples, dest):
    sync(with_examples, dest, env={"EXCLUDE": "keep gone.txt"})
    assert files(dest) == {"a.txt", "examples/demo.txt"}
    sync(with_examples, dest, env={"EXCLUDE": ""})
    assert "examples/demo.txt" in files(dest) and "keep/k.txt" in files(dest)


def test_left_out_files_are_not_checked_against_the_blocklist(with_examples, dest, tmp_path):
    (with_examples / "examples" / "demo.txt").write_text("by Jane Q. Researcher\n")
    git(with_examples, "commit", "-q", "-am", "name in an excluded file")
    block = tmp_path / "block.txt"
    block.write_text("Jane Q\\. Researcher\n")
    assert sync(with_examples, dest, env={"BLOCKLIST": str(block)}).returncode == 0
    assert sync(with_examples, dest, env={"BLOCKLIST": str(block), "EXCLUDE": ""}).returncode == 1


def test_a_fresh_clone_is_told_to_use_the_deploy_key_if_there_is_one(source, dest, tmp_path):
    home = tmp_path / "home"
    (home / ".ssh").mkdir(parents=True)
    key = home / ".ssh" / "confoundsearch_deploy"
    key.write_text("not a real key\n")
    assert update(source, dest, env={"ASSUME_YES": "1", "HOME": str(home)}).returncode == 0
    assert git(dest, "config", "core.sshCommand") == f"ssh -i {key} -o IdentitiesOnly=yes"


def test_without_a_key_the_clone_config_is_left_alone(source, dest, tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    update(source, dest, env={"ASSUME_YES": "1", "HOME": str(home)})
    assert subprocess.run(["git", "config", "core.sshCommand"], cwd=dest, capture_output=True).returncode == 1


def fake_ssh(tmp_path):
    """An `ssh` that records how it was called and then fails, so nothing leaves the machine."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    ssh = bin_dir / "ssh"
    ssh.write_text(f'#!/bin/sh\necho "$@" >> {tmp_path / "ssh-calls.txt"}\nexit 1\n')
    ssh.chmod(0o755)
    return bin_dir


def test_an_https_remote_is_cloned_over_ssh_with_the_key(source, dest, tmp_path):
    home = tmp_path / "home"
    (home / ".ssh").mkdir(parents=True)
    key = home / ".ssh" / "confoundsearch_deploy"
    key.write_text("x\n")
    git(dest, "remote", "set-url", "origin", "https://host.example.org/owner/repo.git")
    path = f"{fake_ssh(tmp_path)}{os.pathsep}{os.environ['PATH']}"
    result = update(source, dest, env={"HOME": str(home), "PATH": path, "ASSUME_YES": "1"})
    assert result.returncode == 1 and "could not clone git@host.example.org:owner/repo.git" in result.stderr
    call = (tmp_path / "ssh-calls.txt").read_text()
    assert f"-i {key}" in call and "IdentitiesOnly=yes" in call and "git@host.example.org" in call


def test_an_https_remote_is_left_alone_without_a_key(source, dest, tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    git(dest, "remote", "set-url", "origin", "https://host.example.org/owner/repo.git")
    path = f"{fake_ssh(tmp_path)}{os.pathsep}{os.environ['PATH']}"
    result = update(source, dest, env={"HOME": str(home), "PATH": path, "ASSUME_YES": "1"})
    assert (
        result.returncode == 1 and "could not clone https://host.example.org/owner/repo.git" in result.stderr
    )


def test_the_remote_can_be_named_when_there_is_no_clone_yet(source, tmp_path):
    bare = tmp_path / "fresh.git"
    git(tmp_path, "init", "-q", "--bare", "-b", "main", str(bare))
    result = update(source, tmp_path / "new-folder", env={"ASSUME_YES": "1", "REVIEW_REMOTE": str(bare)})
    assert result.returncode == 0, result.stderr
    assert git(bare, "rev-list", "--count", "main") == "1"


def test_it_never_replaces_a_folder_that_is_not_a_clone_or_this_repository(source, tmp_path):
    other = tmp_path / "precious"
    other.mkdir()
    (other / "thesis.txt").write_text("mine\n")
    result = update(source, other, env={"ASSUME_YES": "1", "REVIEW_REMOTE": str(tmp_path / "x.git")})
    assert result.returncode == 2 and "refusing" in result.stderr and (other / "thesis.txt").exists()
    result = update(source, source, env={"ASSUME_YES": "1", "REVIEW_REMOTE": str(tmp_path / "x.git")})
    assert result.returncode == 2 and "this repository" in result.stderr and (source / "a.txt").exists()


def test_the_check_counts_the_files_that_should_be_there_without_the_left_out_ones(with_examples, dest):
    result = update(with_examples, dest, env={"ASSUME_YES": "1"})
    assert result.returncode == 0, result.stderr
    assert "all 3 files" in result.stdout and "without: examples" in result.stdout
    result = update(with_examples, dest, env={"ASSUME_YES": "1", "EXCLUDE": ""})
    assert "all 4 files" in result.stdout and "without" not in result.stdout
