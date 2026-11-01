"""The justfile is how releases happen, so its safety properties are tested, not assumed."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
TEXT = (ROOT / "justfile").read_text(encoding="utf-8")
RECIPES = {
    "default", "setup", "fmt", "lint", "test", "check", "live",
    "env", "hooks", "release", "version", "clean", "build", "verify", "preflight", "testpypi", "publish",
    "tag", "verify-published", "keywords", "neighbors", "rebuild", "proxy-build", "check-domain",
    "public", "public-preview",
}  # fmt: skip
needs_just = pytest.mark.skipif(shutil.which("just") is None, reason="just is not installed")


def just(*args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["just", *args], cwd=ROOT, capture_output=True, text=True, env={**os.environ, **(env or {})}
    )


def test_every_recipe_the_project_relies_on_exists():
    defined = set(re.findall(r"^([a-z][\w-]*)(?: [^:\n]*)?:", TEXT, re.M))
    assert defined >= RECIPES, RECIPES - defined


def test_the_tools_it_calls_exist():
    for script in set(re.findall(r"devops/tools/(\w+\.py)", TEXT)):
        assert (ROOT / "devops" / "tools" / script).is_file(), script


def test_no_secret_is_written_in_the_file():
    assert not re.search(r"pypi-[A-Za-z0-9_-]{20,}", TEXT)  # a real token looks like this
    for name in ("TEST_PYPI_TOKEN", "PYPI_API_TOKEN"):
        assert f"${{{name}:-}}" in TEXT  # it is only ever read from the environment, never given a value
        assert not re.search(rf"^\s*{name}\s*[:=]\s*\S", TEXT, re.M)


def test_the_environment_file_that_holds_tokens_is_not_committed():
    ignored = (ROOT / ".gitignore").read_text().split()
    assert ".env" in ignored and "set dotenv-load := true" in TEXT


def test_uploading_always_asks_first():
    for recipe in ("testpypi", "publish"):
        assert re.search(rf'\[confirm\("[^"]+"\)\]\n{recipe}\b', TEXT), (
            f"{recipe} must be behind a confirmation"
        )
    assert "REAL PyPI" in TEXT and "cannot be undone" in TEXT


def test_only_two_lines_upload_and_only_one_goes_to_the_real_index():
    uploads = [line for line in TEXT.splitlines() if "twine upload" in line]
    assert len(uploads) == 2
    assert sum("test.pypi.org" in line for line in uploads) == 1


def test_the_default_recipe_only_lists_things():
    assert re.search(r"^default:\n    @just --list", TEXT, re.M)


@needs_just
class TestWithJustInstalled:
    def test_the_file_parses_and_lists_every_recipe(self):
        done = just("--summary")
        assert done.returncode == 0, done.stderr
        assert set(done.stdout.split()) >= RECIPES

    def test_every_recipe_has_a_one_line_description(self):
        listing = just("--list").stdout
        for recipe in RECIPES - {"default"}:
            assert re.search(rf"^\s+{recipe}\b.*# \S", listing, re.M), (
                f"{recipe} has no description in `just --list`"
            )

    def test_version_prints_the_package_version(self):
        from confoundsearch import __version__

        assert just("version").stdout.strip() == __version__

    def test_publishing_runs_every_check_before_it_uploads(self):
        done = just("--dry-run", "--yes", "publish", env={"PYPI_API_TOKEN": "not-a-real-token"})
        assert done.returncode == 0, done.stderr
        steps = [line for line in done.stderr.splitlines() if line and not line.startswith("#")]
        wanted = [
            "ruff check .", "mypy", "pytest", "python -m build", "twine check",
            "release_check.py --dist dist", "smoke_test.py",
            "release_check.py --dist dist --online --index pypi",
            "TWINE_USERNAME=__token__", "twine upload dist/*",
        ]  # fmt: skip
        positions = [next(i for i, step in enumerate(steps) if marker in step) for marker in wanted]
        assert positions == sorted(positions), "the checks must run before the upload, in this order"
        assert "test.pypi.org" not in done.stderr

    def test_testpypi_checks_against_testpypi_and_uploads_there(self):
        done = just("--dry-run", "--yes", "testpypi", env={"TEST_PYPI_TOKEN": "not-a-real-token"})
        assert done.returncode == 0, done.stderr
        assert "--online --index testpypi" in done.stderr and "https://test.pypi.org/legacy/" in done.stderr

    def test_the_token_is_referenced_by_name_not_pasted_into_the_command(self):
        done = just("--dry-run", "--yes", "publish", env={"PYPI_API_TOKEN": "sekret-value"})
        assert "sekret-value" not in done.stderr + done.stdout

    def test_without_the_yes_flag_it_asks_and_runs_nothing_on_no(self):
        done = subprocess.run(
            ["just", "publish"], cwd=ROOT, capture_output=True, text=True, input="n\n",
            env={"PATH": os.environ["PATH"], "PYPI_API_TOKEN": "x", "HOME": str(ROOT)},
        )  # fmt: skip
        assert done.returncode != 0 and "Upload confoundsearch to the REAL PyPI" in done.stderr + done.stdout
        assert "twine upload" not in done.stdout  # nothing ran

    def test_publish_refuses_without_a_token(self):
        # run only the token guard, as the recipe does, to prove it fails before any upload
        guard = next(line for line in TEXT.splitlines() if "PYPI_API_TOKEN:-" in line).strip().lstrip("@")
        done = subprocess.run(
            ["bash", "-c", guard], capture_output=True, text=True, env={"PATH": "/usr/bin:/bin"}
        )
        assert done.returncode == 1 and "set PYPI_API_TOKEN in .env" in done.stderr

    def test_only_the_tag_and_release_recipes_push_and_release_pushes_only_after_publishing(self):
        pushes = [line.strip() for line in TEXT.splitlines() if line.strip().startswith("git push")]
        assert len(pushes) == 2
        assert sum(p.startswith('git push origin "v$v"') for p in pushes) == 1  # the tag
        assert (
            sum(p.startswith("git push --no-verify origin HEAD") for p in pushes) == 1
        )  # the release commit
        release = TEXT[re.search(r'^release version=""', TEXT, re.M).start() :]
        assert release.index("just --yes publish") < release.index("git push --no-verify origin HEAD")

    def test_publishing_tags_and_verifies_after_the_upload_and_lists_what_is_left(self):
        done = just("--dry-run", "--yes", "publish", env={"PYPI_API_TOKEN": "not-a-real-token"})
        steps = [line for line in done.stderr.splitlines() if line and not line.startswith("#")]
        order = [next(i for i, step in enumerate(steps) if marker in step) for marker in (
            "twine upload dist/*", "just tag", "just verify-published", "Released, tagged and checked",
        )]  # fmt: skip
        assert order == sorted(order)
        assert (
            "--index pypi --git" in done.stderr
        )  # a dirty checkout or a taken tag stops it before the upload
        assert "Released, tagged and checked." in done.stderr

    def test_the_testpypi_practice_run_checks_the_install_back_from_testpypi(self):
        done = just("--dry-run", "--yes", "testpypi", env={"TEST_PYPI_TOKEN": "not-a-real-token"})
        steps = [line for line in done.stderr.splitlines() if line and not line.startswith("#")]
        assert "--git" not in done.stderr  # practice runs may be done from a dirty checkout
        assert steps.index(next(s for s in steps if "twine upload" in s)) < steps.index(
            "just verify-published testpypi"
        )

    def test_verify_published_installs_the_released_version_with_retries(self, tmp_path):
        stub = tmp_path / "python"
        stub.write_text('#!/bin/sh\nif [ "$1" = "-c" ]; then echo 9.9.9; exit 0; fi\necho "$@" >> "$LOG"\n')
        stub.chmod(0o755)
        log = tmp_path / "log"
        env = {"PYTHON": str(stub), "LOG": str(log)}
        assert just("verify-published", env=env).returncode == 0
        assert just("verify-published", "testpypi", env=env).returncode == 0
        pypi, testpypi = log.read_text().splitlines()
        assert pypi == "devops/tools/smoke_test.py confoundsearch==9.9.9 --retries 12 --wait 15"
        assert testpypi == (
            "devops/tools/smoke_test.py confoundsearch==9.9.9 "
            "--index-url https://test.pypi.org/simple/ --retries 12 --wait 15"
        )


def git(cwd: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "GIT_AUTHOR_NAME": "T", "GIT_AUTHOR_EMAIL": "t@example.org",
           "GIT_COMMITTER_NAME": "T", "GIT_COMMITTER_EMAIL": "t@example.org"}  # fmt: skip
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, env=env, check=check)


@needs_just
@pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
class TestTheTagRecipe:
    """Run `just tag` for real in throwaway repositories."""

    def project(self, tmp_path, *, remote=True):
        work = tmp_path / "work"
        work.mkdir()
        (work / "confoundsearch").mkdir()
        (work / "confoundsearch" / "__init__.py").write_text('__version__ = "9.9.9"\n')
        shutil.copy(ROOT / "justfile", work / "justfile")
        git(work, "init", "-q", "-b", "main")
        git(work, "add", "-A")
        git(work, "commit", "-q", "-m", "release")
        if remote:
            bare = tmp_path / "remote.git"
            git(tmp_path, "init", "-q", "--bare", str(bare))
            git(work, "remote", "add", "origin", str(bare))
        return work

    def tag(self, work):
        return subprocess.run(
            ["just", "tag"], cwd=work, capture_output=True, text=True, env={**os.environ, "PYTHON": "python3"}
        )

    def remote_tags(self, tmp_path):
        return git(tmp_path / "remote.git", "tag", "--list").stdout.split()

    def test_it_tags_the_release_after_the_version_and_pushes_the_tag(self, tmp_path):
        work = self.project(tmp_path)
        done = self.tag(work)
        assert done.returncode == 0 and "created tag v9.9.9" in done.stdout
        assert git(work, "tag", "--list").stdout.split() == ["v9.9.9"] and self.remote_tags(tmp_path) == [
            "v9.9.9"
        ]
        assert (
            git(work, "cat-file", "-t", "v9.9.9").stdout.strip() == "tag"
        )  # an annotated tag, with a message

    def test_running_it_again_changes_nothing_and_does_not_fail(self, tmp_path):
        work = self.project(tmp_path)
        self.tag(work)
        done = self.tag(work)
        assert done.returncode == 0 and "already exists at this commit" in done.stdout
        assert self.remote_tags(tmp_path) == ["v9.9.9"]

    def test_a_tag_on_a_different_commit_is_refused(self, tmp_path):
        work = self.project(tmp_path)
        self.tag(work)
        (work / "later.txt").write_text("x")
        git(work, "add", "-A")
        git(work, "commit", "-q", "-m", "later")
        done = self.tag(work)
        assert done.returncode == 1 and "already exists at a different commit" in done.stderr

    def test_a_failed_push_is_reported_but_does_not_undo_the_release(self, tmp_path):
        work = self.project(tmp_path, remote=False)  # no remote to push to
        done = self.tag(work)
        assert done.returncode == 0 and "created tag v9.9.9" in done.stdout
        assert "could not push the tag; push it yourself: git push origin v9.9.9" in done.stdout
        assert git(work, "tag", "--list").stdout.split() == ["v9.9.9"]


def test_clean_removes_stale_caches_and_leftover_folders_but_keeps_real_files(tmp_path):
    """Run the recipe's own commands on a tree that looks like the aftermath of the reorganization."""
    lines = re.search(r"^clean:\n((?:    .+\n)+)", TEXT, re.M).group(1).splitlines()
    tree = {
        "confoundsearch/actions/__pycache__/merge.cpython-312.pyc": b"\0",  # the old layout's leftover
        "confoundsearch/logic/__pycache__/graph.cpython-312.pyc": b"\0",
        "confoundsearch/logic/__init__.py": b"",
        "confoundsearch/logic/graph.py": b"x = 1\n",
        "confoundsearch/interface/skill/SKILL.md": b"---\n",
        "dist/old.whl": b"\0",
        "tests/__pycache__/t.pyc": b"\0",
        "tests/test_a.py": b"",
    }
    for name, data in tree.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    done = subprocess.run(
        ["bash", "-euo", "pipefail", "-c", "\n".join(lines)], cwd=tmp_path, capture_output=True, text=True
    )
    assert done.returncode == 0, done.stderr
    assert not (tmp_path / "confoundsearch" / "actions").exists()  # the leftover is gone
    assert not list(tmp_path.rglob("__pycache__")) and not (tmp_path / "dist").exists()
    kept = {p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*") if p.is_file()}
    assert kept == {
        "confoundsearch/logic/__init__.py", "confoundsearch/logic/graph.py",
        "confoundsearch/interface/skill/SKILL.md", "tests/test_a.py",
    }  # fmt: skip


def recipe_text(name: str) -> str:
    """A recipe from the real justfile, with its attribute and doc comment, up to the next top-level item."""
    lines, out, inside = TEXT.splitlines(), [], False
    for i, line in enumerate(lines):
        if re.match(rf"^{name}\b[^:]*:", line):
            out = [lines[i - 1]] if lines[i - 1].startswith("[") else []
            inside = True
        if inside:
            if (
                out
                and line
                and not line.startswith((" ", "\t"))
                and not line.startswith(name)
                and not line.startswith("[")
            ):
                break
            out.append(line)
    return "\n".join(out).rstrip() + "\n"


@needs_just
@pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
class TestTheReleaseRecipe:
    """Run the real `release` recipe in throwaway repos with a stand-in `publish`: nothing is uploaded."""

    def project(self, tmp_path, *, publish_passes=True):
        work, bare = tmp_path / "work", tmp_path / "remote.git"
        (work / "confoundsearch").mkdir(parents=True)
        (work / "devops" / "tools").mkdir(parents=True)
        (work / "confoundsearch" / "__init__.py").write_text('__version__ = "1.0.0"\n')
        shutil.copy(
            ROOT / "devops" / "tools" / "set_version.py", work / "devops" / "tools" / "set_version.py"
        )
        (work / "justfile").write_text(
            'set shell := ["bash", "-euo", "pipefail", "-c"]\n'
            'py := env("PYTHON", "python")\n\n'
            f'publish:\n    @echo "publish at $(git log -1 --format=%s)" >> "{tmp_path}/events"\n'
            f"    @exit {0 if publish_passes else 1}\n\n" + recipe_text("release")
        )
        git(work, "init", "-q", "-b", "main")
        git(work, "add", "-A")
        git(work, "commit", "-q", "-m", "start")
        git(tmp_path, "init", "-q", "--bare", str(bare))
        git(work, "remote", "add", "origin", str(bare))
        git(work, "push", "-q", "origin", "main")
        return work, bare

    def release(self, work, *args, answer=None):
        command = ["just", *([] if answer else ["--yes"]), "release", *args]
        return subprocess.run(
            command, cwd=work, capture_output=True, text=True, input=answer,
            env={**os.environ, "PYTHON": "python3"},
        )  # fmt: skip

    def version(self, work):
        return (work / "confoundsearch" / "__init__.py").read_text().strip()

    def remote_subject(self, bare):
        return git(bare, "log", "-1", "--format=%s", "main").stdout.strip()

    def test_a_version_argument_bumps_commits_publishes_then_pushes(self, tmp_path):
        work, bare = self.project(tmp_path)
        done = self.release(work, "1.0.1")
        assert done.returncode == 0, done.stderr
        assert self.version(work) == '__version__ = "1.0.1"'
        assert git(work, "log", "-1", "--format=%s").stdout.strip() == "Release 1.0.1"
        assert (
            tmp_path / "events"
        ).read_text().strip() == "publish at Release 1.0.1"  # published the bumped commit
        assert self.remote_subject(bare) == "Release 1.0.1" and "Released and pushed." in done.stdout

    def test_without_a_version_it_releases_the_current_one_and_makes_no_commit(self, tmp_path):
        work, _bare = self.project(tmp_path)
        before = git(work, "rev-parse", "HEAD").stdout
        done = self.release(work)
        assert done.returncode == 0 and git(work, "rev-parse", "HEAD").stdout == before
        assert self.version(work) == '__version__ = "1.0.0"'
        assert (tmp_path / "events").read_text().strip() == "publish at start"

    def test_uncommitted_changes_stop_it_before_anything_happens(self, tmp_path):
        work, _bare = self.project(tmp_path)
        (work / "notes.txt").write_text("x")
        git(work, "add", "-A")
        done = self.release(work, "1.0.1")
        assert done.returncode == 1 and "commit or stash your changes first" in done.stderr
        assert self.version(work) == '__version__ = "1.0.0"' and not (tmp_path / "events").exists()

    @pytest.mark.parametrize(
        "bad,reason", [("1.0.0", "not newer"), ("0.9", "not newer"), ("v2", "not a valid version")]
    )
    def test_a_bad_version_stops_it_without_a_commit_or_a_publish(self, tmp_path, bad, reason):
        work, _bare = self.project(tmp_path)
        done = self.release(work, bad)
        assert done.returncode == 1 and reason in done.stderr
        assert (
            git(work, "log", "-1", "--format=%s").stdout.strip() == "start"
            and not (tmp_path / "events").exists()
        )

    def test_if_publishing_fails_nothing_is_pushed(self, tmp_path):
        work, bare = self.project(tmp_path, publish_passes=False)
        done = self.release(work, "1.0.1")
        assert done.returncode != 0 and "Released and pushed." not in done.stdout
        assert self.remote_subject(bare) == "start"  # the release commit stays local until a retry succeeds
        assert git(work, "log", "-1", "--format=%s").stdout.strip() == "Release 1.0.1"

    def test_a_retry_after_a_failure_needs_no_version_argument(self, tmp_path):
        work, bare = self.project(tmp_path, publish_passes=False)
        self.release(work, "1.0.1")
        (work / "justfile").write_text((work / "justfile").read_text().replace("@exit 1", "@exit 0"))
        git(work, "commit", "-q", "-am", "fix publish")
        done = self.release(work)
        assert done.returncode == 0 and self.remote_subject(bare) == "fix publish"

    def test_without_the_yes_flag_it_asks_and_does_nothing_on_no(self, tmp_path):
        work, _bare = self.project(tmp_path)
        done = self.release(work, "1.0.1", answer="n\n")
        assert done.returncode != 0 and "Release to the REAL PyPI" in done.stdout + done.stderr
        assert self.version(work) == '__version__ = "1.0.0"' and not (tmp_path / "events").exists()


@needs_just
def test_keywords_builds_and_only_pushes_when_given_an_index():
    plain = just("--dry-run", "keywords")
    assert plain.returncode == 0
    assert "keyword_vectors.py build --out keyword_vectors" in plain.stderr
    assert "keyword_vectors.py push" not in plain.stderr
    full = just("--dry-run", "keywords", "kw", "1000")
    assert "build --out keyword_vectors --limit 1000" in full.stderr
    assert "push --out keyword_vectors --index kw --limit 1000" in full.stderr


@needs_just
def test_neighbors_passes_the_index_and_extra_arguments():
    out = just("--dry-run", "neighbors", "myindex", "--min-works", "500").stderr
    assert "keyword_vectors.py neighbors examples/known_confounds.json" in out
    assert "--index myindex" in out and "--min-works 500" in out and "--report keyword_neighbors.md" in out


@needs_just
def test_rebuild_builds_then_clears_and_pushes():
    out = just("--dry-run", "rebuild", "myindex").stderr
    build = out.index("keyword_vectors.py build --out keyword_vectors")
    push = out.index(
        "keyword_vectors.py push --out keyword_vectors --index myindex --fresh --yes"
    )  # no prompt
    assert build < push  # the new vectors exist before anything is deleted
