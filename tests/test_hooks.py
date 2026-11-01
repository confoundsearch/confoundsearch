"""The pre-push hook is what keeps a failing change from leaving the machine, so it is tested end to end."""

from __future__ import annotations

import os
import re
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
HOOK = ROOT / ".githooks" / "pre-push"
JUSTFILE = (ROOT / "justfile").read_text(encoding="utf-8")
ZERO = "0" * 40
SHA = "1" * 40
git_available = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")


def fake_just(folder: Path, *, passes: bool = True) -> Path:
    """A stand-in `just` that records how it was called and then succeeds or fails."""
    folder.mkdir(exist_ok=True)
    script = folder / "just"
    script.write_text(f'#!/bin/sh\necho "$@" >> "{folder}/calls"\nexit {0 if passes else 1}\n')
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return folder


def run_hook(stdin: str, path: str, cwd: Path = ROOT):
    env = {"PATH": path, "HOME": str(cwd)}
    return subprocess.run(["bash", str(HOOK)], input=stdin, capture_output=True, text=True, cwd=cwd, env=env)


def calls(folder: Path) -> list[str]:
    log = folder / "calls"
    return log.read_text().splitlines() if log.exists() else []


def push_line(local_ref="refs/heads/orgsci", local_sha=SHA):
    return f"{local_ref} {local_sha} refs/heads/orgsci {ZERO}\n"


def path_with(folder: Path) -> str:
    return f"{folder}:{os.environ['PATH']}"


class TestTheHookScript:
    def test_it_is_executable_and_tracked_as_executable(self):
        assert HOOK.stat().st_mode & stat.S_IXUSR
        if shutil.which("git") and (ROOT / ".git").exists():
            listing = subprocess.run(
                ["git", "ls-files", "-s", str(HOOK)], capture_output=True, text=True, cwd=ROOT
            )
            assert listing.stdout.startswith("100755"), "the hook must be committed with the executable bit"

    def test_a_branch_push_runs_the_full_checks(self, tmp_path):
        stub = fake_just(tmp_path / "bin")
        done = run_hook(push_line(), path_with(stub))
        assert done.returncode == 0 and calls(stub) == ["check"]
        assert "running lint and the full test suite" in done.stderr

    def test_failing_checks_stop_the_push_and_say_how_to_override(self, tmp_path):
        stub = fake_just(tmp_path / "bin", passes=False)
        done = run_hook(push_line(), path_with(stub))
        assert done.returncode == 1 and calls(stub) == ["check"]
        assert "nothing was pushed" in done.stderr and "git push --no-verify" in done.stderr

    @pytest.mark.parametrize(
        "stdin",
        [
            push_line("refs/tags/v0.1.0"),  # a tag is not new code
            push_line("(delete)", ZERO),  # deleting a branch
            push_line("(delete)", "0" * 64),  # the same, in a repository that uses SHA-256
            "",  # nothing to push
        ],
    )
    def test_pushes_of_no_new_code_skip_the_checks(self, tmp_path, stdin):
        stub = fake_just(tmp_path / "bin", passes=False)  # would fail if it ran
        assert run_hook(stdin, path_with(stub)).returncode == 0 and calls(stub) == []

    def test_a_branch_pushed_together_with_a_tag_is_still_checked(self, tmp_path):
        stub = fake_just(tmp_path / "bin")
        done = run_hook(push_line("refs/tags/v1") + push_line(), path_with(stub))
        assert done.returncode == 0 and calls(stub) == ["check"]

    def test_a_branch_pushed_together_with_a_deletion_is_still_checked(self, tmp_path):
        stub = fake_just(tmp_path / "bin", passes=False)
        done = run_hook(push_line("(delete)", ZERO) + push_line(), path_with(stub))
        assert done.returncode == 1 and calls(stub) == ["check"]

    def test_without_just_it_refuses_and_explains(self, tmp_path):
        done = run_hook(push_line(), "/usr/bin:/bin")
        assert done.returncode == 1 and "'just' is not installed" in done.stderr
        assert "requirements.yml" in done.stderr and "--no-verify" in done.stderr

    def test_the_checks_run_from_the_repository_root(self, tmp_path):
        stub = fake_just(tmp_path / "bin")
        run_hook(push_line(), path_with(stub), cwd=ROOT / "tests")
        assert calls(stub) == ["check"]


@git_available
class TestAgainstARealRepository:
    """Install the hook the way `just hooks` does and push to a real (local) remote."""

    def repo(self, tmp_path, stub_dir: Path):
        remote, work = tmp_path / "remote.git", tmp_path / "work"
        subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(remote)], check=True)
        subprocess.run(["git", "init", "-q", "-b", "main", str(work)], check=True)
        (work / ".githooks").mkdir()
        shutil.copy(HOOK, work / ".githooks" / "pre-push")
        recipe = re.search(r"^hooks:\n((?:    .+\n)+)", JUSTFILE, re.M).group(1)
        commands = [line.strip() for line in recipe.splitlines() if not line.strip().startswith("@")]
        subprocess.run(["bash", "-euo", "pipefail", "-c", "\n".join(commands)], cwd=work, check=True)
        for key, value in (("user.name", "T"), ("user.email", "t@example.org"), ("commit.gpgsign", "false")):
            subprocess.run(["git", "config", key, value], cwd=work, check=True)
        subprocess.run(["git", "remote", "add", "origin", str(remote)], cwd=work, check=True)
        (work / "f.txt").write_text("x")
        subprocess.run(["git", "add", "-A"], cwd=work, check=True)
        subprocess.run(["git", "commit", "-q", "-m", "c"], cwd=work, check=True)
        return remote, work, {"PATH": path_with(stub_dir), "HOME": str(tmp_path)}

    def push(self, work, env, *args):
        return subprocess.run(
            ["git", "push", *args, "origin", "main"], cwd=work, capture_output=True, text=True, env=env
        )

    def remote_has_main(self, remote):
        return (
            subprocess.run(
                ["git", "--git-dir", str(remote), "rev-parse", "--verify", "-q", "main"]
            ).returncode
            == 0
        )

    def test_the_recipe_points_git_at_the_tracked_hooks(self, tmp_path):
        _, work, _env = self.repo(tmp_path, fake_just(tmp_path / "bin"))
        config = subprocess.run(["git", "config", "core.hooksPath"], cwd=work, capture_output=True, text=True)
        assert config.stdout.strip() == ".githooks"

    def test_a_push_with_passing_checks_goes_through(self, tmp_path):
        stub = fake_just(tmp_path / "bin")
        remote, work, env = self.repo(tmp_path, stub)
        done = self.push(work, env)
        assert done.returncode == 0 and self.remote_has_main(remote) and calls(stub) == ["check"]

    def test_a_push_with_failing_checks_is_blocked_and_the_remote_is_untouched(self, tmp_path):
        stub = fake_just(tmp_path / "bin", passes=False)
        remote, work, env = self.repo(tmp_path, stub)
        done = self.push(work, env)
        assert done.returncode != 0 and not self.remote_has_main(remote)
        assert "the checks failed, so nothing was pushed" in done.stderr

    def test_no_verify_is_the_deliberate_way_around_it(self, tmp_path):
        stub = fake_just(tmp_path / "bin", passes=False)
        remote, work, env = self.repo(tmp_path, stub)
        done = self.push(work, env, "--no-verify")
        assert done.returncode == 0 and self.remote_has_main(remote) and calls(stub) == []


def test_ci_runs_on_every_push_and_pull_request_with_everything_installed():
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    assert re.search(r"^on:\n  push:\n  pull_request:", ci, re.M), (
        "CI must run on every push and pull request"
    )
    assert 'pip install -e ".[dev,release]"' in ci  # so nothing skips: yaml, just and the real builds all run
    assert "          pytest --tb=short" in ci and "fail-fast: false" in ci  # the tests run on every Python


def test_setup_installs_the_hook():
    assert re.search(r"^setup: hooks$", JUSTFILE, re.M)
