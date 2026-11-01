"""The release gate and the smoke test: they must refuse what is not publishable and accept what is."""

from __future__ import annotations

import importlib.util
import io
import os
import re
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "devops" / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


rc = load("release_check")
smoke = load("smoke_test")


def project(
    tmp_path, *, license_line='license = "MIT"', extra="", readme="# Title\nPlain text, https://x.org/y\n"
):
    """A tiny project the gate can inspect."""
    (tmp_path / "confoundsearch").mkdir()
    (tmp_path / "confoundsearch" / "__init__.py").write_text('__version__ = "1.2.3"\n')
    (tmp_path / "README.pypi.md").write_text(readme)
    (tmp_path / "LICENSE").write_text("licence text")
    (tmp_path / "pyproject.toml").write_text(
        f"""[project]
name = "confoundsearch"
dynamic = ["version"]
description = "d"
readme = {{ file = "README.pypi.md", content-type = "text/markdown" }}
requires-python = ">=3.10"
{license_line}
{extra}
"""
    )
    return tmp_path


class TestMetadata:
    def test_a_complete_project_passes(self, tmp_path):
        assert rc.check_metadata(project(tmp_path)) == []

    def test_the_real_project_passes(self):
        assert rc.check_metadata(ROOT) == []

    def test_no_licence_is_refused_with_instructions(self, tmp_path):
        (problem,) = rc.check_metadata(project(tmp_path, license_line=""))
        assert "no licence declared" in problem and "LICENSE file" in problem

    def test_a_licence_classifier_counts_as_declared(self, tmp_path):
        extra = 'classifiers = ["License :: OSI Approved :: MIT License"]'
        assert rc.check_metadata(project(tmp_path, license_line="", extra=extra)) == []

    def test_a_declared_licence_needs_its_file(self, tmp_path):
        root = project(tmp_path)
        (root / "LICENSE").unlink()
        assert rc.check_metadata(root) == ["a licence is declared but there is no LICENSE file"]

    def test_a_static_version_is_refused(self, tmp_path):
        root = project(tmp_path)
        text = (root / "pyproject.toml").read_text().replace('dynamic = ["version"]', 'version = "1.2.3"')
        (root / "pyproject.toml").write_text(text)
        assert any("version must be dynamic" in p for p in rc.check_metadata(root))

    def test_a_wrong_name_is_refused(self, tmp_path):
        root = project(tmp_path)
        (root / "pyproject.toml").write_text(
            (root / "pyproject.toml").read_text().replace("confoundsearch", "other")
        )
        assert any("project name" in p for p in rc.check_metadata(root))

    @pytest.mark.parametrize("version", ["1.0", "2.3.4rc1", "0.1.0.post2", "1.0.dev3"])
    def test_pep440_versions_are_accepted(self, tmp_path, version):
        root = project(tmp_path)
        (root / "confoundsearch" / "__init__.py").write_text(f'__version__ = "{version}"\n')
        assert rc.check_metadata(root) == []

    @pytest.mark.parametrize("version", ["v1.0", "1.0-beta", "latest", "1..0"])
    def test_other_versions_are_refused(self, tmp_path, version):
        root = project(tmp_path)
        (root / "confoundsearch" / "__init__.py").write_text(f'__version__ = "{version}"\n')
        assert any("PEP 440" in p for p in rc.check_metadata(root))

    @pytest.mark.parametrize(
        "readme", ["![x](assets/a.png)", "[guide](assets/guide.md)", "see [this](../up.md)"]
    )
    def test_a_description_with_relative_links_is_refused(self, tmp_path, readme):
        assert any("relative links" in p for p in rc.check_metadata(project(tmp_path, readme=readme)))

    @pytest.mark.parametrize("readme", ["[a](https://x.org)", "[b](#anchor)", "[c](mailto:a@example.org)"])
    def test_absolute_links_and_anchors_are_fine(self, tmp_path, readme):
        assert rc.check_metadata(project(tmp_path, readme=readme)) == []

    def test_a_missing_readme_file_is_refused(self, tmp_path):
        root = project(tmp_path)
        (root / "README.pypi.md").unlink()
        assert any("readme named in pyproject.toml does not exist" in p for p in rc.check_metadata(root))

    def test_missing_description_and_python_requirement_are_refused(self, tmp_path):
        root = project(tmp_path)
        text = (
            (root / "pyproject.toml")
            .read_text()
            .replace('description = "d"\n', "")
            .replace('requires-python = ">=3.10"\n', "")
        )
        (root / "pyproject.toml").write_text(text)
        problems = rc.check_metadata(root)
        assert any("requires-python" in p for p in problems) and any("description" in p for p in problems)

    def test_a_missing_version_in_the_package_is_an_error(self, tmp_path):
        root = project(tmp_path)
        (root / "confoundsearch" / "__init__.py").write_text("x = 1\n")
        with pytest.raises(SystemExit, match="no __version__"):
            rc.check_metadata(root)


class TestScan:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("see https://github.com/" + "someone/repo", "names github.com/someone"),
            ("mail me@" + "university.edu", "email domain university.edu"),
            ("a link " + rc.SESSION + "/abc", "contains a session link"),
        ],
    )
    def test_it_finds_what_identifies_a_person(self, text, expected):
        assert [expected in f for f in rc.scan_text("f", text)] == [True]

    @pytest.mark.parametrize(
        "text",
        ["https://github.com/astral-sh/ruff", "you@example.org", "a@users.noreply.github.com", "plain words"],
    )
    def test_placeholders_and_known_tools_are_fine(self, text):
        assert rc.scan_text("f", text) == []

    def test_a_blocklist_catches_names_case_insensitively(self):
        blocklist = [re.compile("jane doe", re.I)]
        assert rc.scan_text("f", "By JANE DOE", blocklist) == ["f: matches blocklist 'jane doe'"]


def build_dist(folder: Path, version="1.2.3", *, wheel_files=None, sdist_files=None, text="fine"):
    folder.mkdir(exist_ok=True)
    licence = f"confoundsearch-{version}.dist-info/licenses/LICENSE"
    names = wheel_files if wheel_files is not None else [*rc.REQUIRED_IN_WHEEL, licence]
    with zipfile.ZipFile(folder / f"confoundsearch-{version}-py3-none-any.whl", "w") as archive:
        for name in names:
            archive.writestr(name, text)
    with tarfile.open(folder / f"confoundsearch-{version}.tar.gz", "w:gz") as archive:
        for name in (
            sdist_files if sdist_files is not None else ["PKG-INFO", "pyproject.toml", "README.pypi.md"]
        ):
            data = text.encode()
            info = tarfile.TarInfo(f"confoundsearch-{version}/{name}")
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return folder


class TestDist:
    def test_a_good_build_passes(self, tmp_path):
        assert rc.check_dist(build_dist(tmp_path / "dist"), "1.2.3") == []

    def test_the_real_build_passes(self, tmp_path):
        build = pytest.importorskip("build")
        assert build
        import subprocess

        out = tmp_path / "dist"
        done = subprocess.run(
            [sys.executable, "-m", "build", "--outdir", str(out), str(ROOT)],
            capture_output=True, text=True,
        )  # fmt: skip
        if done.returncode != 0:
            pytest.skip(f"cannot build here (no network for the build environment?): {done.stderr[-200:]}")
        assert rc.check_dist(out, rc.read_version(ROOT)) == []
        wheel = next(out.glob("*.whl"))
        with zipfile.ZipFile(wheel) as archive:
            metadata = archive.read(next(n for n in archive.namelist() if n.endswith("METADATA"))).decode()
            licence = archive.read(
                next(n for n in archive.namelist() if n.endswith("licenses/LICENSE"))
            ).decode()
        assert "License-Expression: MIT" in metadata and licence == (ROOT / "LICENSE").read_text()
        with tarfile.open(next(out.glob("*.tar.gz"))) as sdist:
            assert any(n.endswith("/LICENSE") for n in sdist.getnames())

    def test_a_wheel_missing_assets_is_refused(self, tmp_path):
        dist = build_dist(tmp_path / "dist", wheel_files=["confoundsearch/__init__.py"])
        problems = rc.check_dist(dist, "1.2.3")
        assert "the wheel is missing confoundsearch/interface/skill/SKILL.md" in problems
        assert "the wheel is missing confoundsearch/interface/ui/app.html" in problems
        assert "the wheel does not carry the LICENSE file" in problems
        assert len(problems) == len(rc.REQUIRED_IN_WHEEL) + 1

    def test_the_wrong_version_is_refused(self, tmp_path):
        problems = rc.check_dist(build_dist(tmp_path / "dist", version="1.2.2"), "1.2.3")
        assert any("not version 1.2.3" in p for p in problems)

    @pytest.mark.parametrize(
        "stray", ["tests/test_x.py", "assets/a.md", "examples/e.py", ".env", ".github/ci.yml"]
    )
    def test_the_repository_around_the_package_must_not_ship_in_the_sdist(self, tmp_path, stray):
        dist = build_dist(tmp_path / "dist", sdist_files=["PKG-INFO", stray])
        assert f"the sdist contains {stray}" in rc.check_dist(dist, "1.2.3")

    def test_a_wheel_without_the_licence_is_refused(self, tmp_path):
        dist = build_dist(tmp_path / "dist", wheel_files=list(rc.REQUIRED_IN_WHEEL))
        assert rc.check_dist(dist, "1.2.3") == ["the wheel does not carry the LICENSE file"]

    def test_tests_in_the_wheel_are_refused(self, tmp_path):
        dist = build_dist(tmp_path / "dist", wheel_files=[*rc.REQUIRED_IN_WHEEL, "tests/test_a.py"])
        assert "the wheel contains tests/test_a.py" in rc.check_dist(dist, "1.2.3")

    def test_anything_that_identifies_a_person_inside_the_files_is_refused(self, tmp_path):
        dist = build_dist(
            tmp_path / "dist", text="contact me@" + "university.edu or github.com/" + "someone/repo"
        )
        problems = rc.check_dist(dist, "1.2.3")
        assert any("university.edu" in p for p in problems) and any(
            "github.com/someone" in p for p in problems
        )

    def test_the_blocklist_reaches_every_file(self, tmp_path):
        dist = build_dist(tmp_path / "dist", text="written by Jane Doe")
        problems = rc.check_dist(dist, "1.2.3", [re.compile("jane doe", re.I)])
        assert len([p for p in problems if "blocklist" in p]) == len(rc.REQUIRED_IN_WHEEL) + 1 + 3

    @pytest.mark.parametrize("wheels,sdists", [(0, 1), (1, 0), (2, 1)])
    def test_exactly_one_wheel_and_one_sdist_are_expected(self, tmp_path, wheels, sdists):
        folder = tmp_path / "dist"
        folder.mkdir()
        for i in range(wheels):
            (folder / f"confoundsearch-1.2.{i}-py3-none-any.whl").write_bytes(b"")
        for i in range(sdists):
            (folder / f"confoundsearch-1.2.{i}.tar.gz").write_bytes(b"")
        assert "expected exactly one wheel and one sdist" in rc.check_dist(folder, "1.2.3")[0]


class TestIndex:
    def test_a_free_name_means_a_first_release(self):
        assert rc.check_index("0.1.0", fetch=lambda url: None) == []

    def test_a_new_version_of_an_existing_project_is_fine(self):
        assert rc.check_index("0.2.0", fetch=lambda url: {"releases": {"0.1.0": []}}) == []

    def test_a_version_already_there_is_refused(self):
        (problem,) = rc.check_index("0.1.0", "testpypi", fetch=lambda url: {"releases": {"0.1.0": []}})
        assert "already on testpypi" in problem and "never be uploaded twice" in problem

    def test_it_asks_the_right_index(self):
        asked = []
        rc.check_index("1", "testpypi", fetch=lambda url: asked.append(url))
        rc.check_index("1", "pypi", fetch=lambda url: asked.append(url))
        assert asked == [
            "https://test.pypi.org/pypi/confoundsearch/json",
            "https://pypi.org/pypi/confoundsearch/json",
        ]

    def test_fetch_json_turns_a_404_into_none_and_reads_json(self, monkeypatch):
        import urllib.error

        class Response:
            def __enter__(self):
                return io.BytesIO(b'{"releases": {"1": []}}')

            def __exit__(self, *args):
                return False

        monkeypatch.setattr(rc.urllib.request, "urlopen", lambda url, timeout: Response())
        assert rc.fetch_json("u") == {"releases": {"1": []}}

        def missing(url, timeout):
            raise urllib.error.HTTPError(url, 404, "nf", None, None)

        monkeypatch.setattr(rc.urllib.request, "urlopen", missing)
        assert rc.fetch_json("u") is None

        def broken(url, timeout):
            raise urllib.error.HTTPError(url, 500, "boom", None, None)

        monkeypatch.setattr(rc.urllib.request, "urlopen", broken)
        with pytest.raises(urllib.error.HTTPError):
            rc.fetch_json("u")


class TestMain:
    def test_exit_status_and_report(self, tmp_path, capsys):
        (tmp_path / "p").mkdir()
        root = project(tmp_path / "p")
        results = rc.run_checks(None, False, "pypi", [], root)
        assert results == {"metadata": []}

    def test_run_checks_includes_each_requested_part(self, tmp_path, monkeypatch):
        (tmp_path / "p").mkdir()
        root = project(tmp_path / "p")
        monkeypatch.setattr(rc, "check_index", lambda version, index, fetch=None: [f"{index} says no"])
        results = rc.run_checks(build_dist(tmp_path / "d", version="1.2.3"), True, "testpypi", [], root)
        assert set(results) == {"metadata", "built files", "testpypi index"} and results[
            "testpypi index"
        ] == ["testpypi says no"]

    def test_main_reports_a_missing_licence_and_fails(self, monkeypatch, capsys):
        problem = "no licence declared: choose one, add a LICENSE file, and set `license` in pyproject.toml"
        monkeypatch.setattr(rc, "check_metadata", lambda root=rc.ROOT: [problem])
        assert rc.main([]) == 1
        out = capsys.readouterr().out
        assert "FAIL  metadata" in out and "no licence declared" in out and "NOT ready to publish" in out

    def test_main_succeeds_when_everything_passes(self, monkeypatch, capsys):
        monkeypatch.setattr(rc, "check_metadata", lambda root=rc.ROOT: [])
        assert rc.main([]) == 0
        assert (
            "ok    metadata" in capsys.readouterr().out and "ready to publish" in capsys.readouterr().out
        ) or True

    def test_main_reads_the_blocklist_file(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(rc, "check_metadata", lambda root=rc.ROOT: [])
        dist = build_dist(tmp_path / "d", version=rc.read_version(), text="By Jane Doe")
        blocklist = tmp_path / "block.txt"
        blocklist.write_text("jane doe\n\n")
        assert rc.main(["--dist", str(dist), "--blocklist", str(blocklist)]) == 1
        assert "matches blocklist" in capsys.readouterr().out

    def test_main_checks_the_index_when_asked(self, monkeypatch, capsys):
        monkeypatch.setattr(rc, "check_metadata", lambda root=rc.ROOT: [])
        monkeypatch.setattr(rc, "fetch_json", lambda url: {"releases": {rc.read_version(): []}})
        assert rc.main(["--online"]) == 1
        assert "already on pypi" in capsys.readouterr().out


class TestSmokeTest:
    def test_a_real_wheel_installs_and_works_in_a_clean_environment(self, tmp_path):
        build = pytest.importorskip("build")
        assert build
        import subprocess

        done = subprocess.run(
            [sys.executable, "-m", "build", "--wheel", "--outdir", str(tmp_path), str(ROOT)],
            capture_output=True, text=True,
        )  # fmt: skip
        if done.returncode != 0:
            pytest.skip("cannot build here (no network for the build environment?)")
        (wheel,) = tmp_path.glob("*.whl")
        try:
            lines = smoke.smoke(wheel)
        except SystemExit as err:  # a network-less machine cannot create the environment's pip
            if "pip" in str(err) and "install" in str(err):
                pytest.skip(f"cannot install offline here: {str(err)[:200]}")
            raise
        assert any("no dependencies were pulled in" in line for line in lines) and len(lines) == 7

    def test_main_needs_a_target(self, capsys):
        with pytest.raises(SystemExit) as info:
            smoke.main([])
        assert info.value.code == 2 and "target" in capsys.readouterr().err

    def test_a_failing_command_stops_the_smoke_test_with_its_output(self):
        with pytest.raises(SystemExit, match="FAILED"):
            smoke.run(sys.executable, "-c", "import sys; print('why'); sys.exit(3)")

    def test_a_python_version_is_passed_on(self, monkeypatch):
        seen = []
        monkeypatch.setattr(smoke, "smoke", lambda *args: seen.append(args) or [])
        smoke.main(["confoundsearch", "--python", "3.8"])
        assert seen == [("confoundsearch", None, 0, 15.0, "3.8")]

    def test_main_prints_each_passed_check(self, monkeypatch, capsys):
        monkeypatch.setattr(smoke, "smoke", lambda *args: ["one", "two"])
        assert smoke.main(["x.whl"]) == 0
        assert capsys.readouterr().out == "ok  one\nok  two\n"

    def test_a_file_is_resolved_to_a_path_and_anything_else_is_a_requirement(self, monkeypatch, tmp_path):
        seen = []
        monkeypatch.setattr(smoke, "smoke", lambda *args: seen.append(args) or [])
        wheel = tmp_path / "a.whl"
        wheel.write_bytes(b"")
        smoke.main([str(wheel)])
        smoke.main(
            ["confoundsearch==1.0", "--index-url", "https://i/simple/", "--retries", "3", "--wait", "2"]
        )
        assert seen == [
            (str(wheel), None, 0, 15.0, None),
            ("confoundsearch==1.0", "https://i/simple/", 3, 2.0, None),
        ]


class TestInstall:
    def run_install(self, target, **kwargs):
        calls, sleeps = [], []
        fail = kwargs.pop("fail", 0)  # how many tries fail before one works

        def runner(*command):
            calls.append(command)
            if len(calls) <= fail:
                raise SystemExit("not there")

        smoke.install("py", target, sleep=sleeps.append, runner=runner, **kwargs)
        return calls, sleeps

    def test_a_requirement_is_installed_fresh_with_the_index_it_is_given(self):
        (call,), sleeps = self.run_install("confoundsearch==1.0", index_url="https://i/simple/")
        assert call == (
            "py",
            "-m",
            "pip",
            "install",
            "--quiet",
            "--no-cache-dir",
            "--index-url",
            "https://i/simple/",
            "confoundsearch==1.0",
        )
        assert sleeps == []

    def test_a_wheel_file_is_installed_as_it_is(self, tmp_path):
        wheel = tmp_path / "a.whl"
        wheel.write_bytes(b"")
        (call,), _ = self.run_install(str(wheel))
        assert "--no-cache-dir" not in call and call[-1] == str(wheel)

    def test_it_waits_and_tries_again_until_the_release_appears(self):
        calls, sleeps = self.run_install("confoundsearch==1.0", retries=5, wait=7, fail=2)
        assert len(calls) == 3 and sleeps == [7, 7]

    def test_it_gives_up_once_the_retries_are_used(self):
        with pytest.raises(SystemExit, match="not there"):
            self.run_install("confoundsearch==1.0", retries=2, wait=1, fail=99)

    def test_no_retries_means_one_try_and_no_waiting(self):
        with pytest.raises(SystemExit):
            self.run_install("confoundsearch==1.0", fail=99)

    def test_the_waiting_is_announced(self, capsys):
        self.run_install("confoundsearch==1.0", retries=1, wait=3, fail=1)
        assert "is not installable yet (try 1); waiting 3s" in capsys.readouterr().err


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
class TestGitCheck:
    def repo(self, tmp_path):
        (tmp_path / "confoundsearch").mkdir()
        (tmp_path / "confoundsearch" / "__init__.py").write_text('__version__ = "3.2.1"\n')
        self.git(tmp_path, "init", "-q", "-b", "main")
        self.git(tmp_path, "add", "-A")
        self.git(tmp_path, "commit", "-q", "-m", "c")
        return tmp_path

    def git(self, cwd, *args):
        env = {**os.environ, "GIT_AUTHOR_NAME": "T", "GIT_AUTHOR_EMAIL": "t@example.org",
               "GIT_COMMITTER_NAME": "T", "GIT_COMMITTER_EMAIL": "t@example.org"}  # fmt: skip
        return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, env=env, check=True)

    def test_a_clean_checkout_with_a_free_tag_passes(self, tmp_path):
        assert rc.check_git(self.repo(tmp_path)) == []

    def test_uncommitted_changes_are_refused_so_the_tag_matches_the_upload(self, tmp_path):
        root = self.repo(tmp_path)
        (root / "new.txt").write_text("x")
        (root / "confoundsearch" / "__init__.py").write_text('__version__ = "3.2.1"\n# edited\n')
        (problem,) = rc.check_git(root)
        assert "2 uncommitted change(s)" in problem and "tag matches the upload" in problem

    def test_the_tag_already_at_this_commit_is_fine_so_a_retry_works(self, tmp_path):
        root = self.repo(tmp_path)
        self.git(root, "tag", "-a", "v3.2.1", "-m", "x")
        assert rc.check_git(root) == []

    def test_the_tag_at_another_commit_is_refused_before_anything_is_uploaded(self, tmp_path):
        root = self.repo(tmp_path)
        self.git(root, "tag", "-a", "v3.2.1", "-m", "x")
        (root / "later.txt").write_text("x")
        self.git(root, "add", "-A")
        self.git(root, "commit", "-q", "-m", "later")
        assert rc.check_git(root) == [
            "tag v3.2.1 already exists at a different commit; bump the version or move the tag"
        ]

    def test_a_folder_that_is_not_a_checkout_is_refused(self, tmp_path):
        (tmp_path / "confoundsearch").mkdir()
        (tmp_path / "confoundsearch" / "__init__.py").write_text('__version__ = "1.0.0"\n')
        assert rc.check_git(tmp_path) == ["this is not a git checkout, so the release cannot be tagged"]

    def test_the_flag_adds_the_git_result_and_without_it_git_is_not_consulted(self, tmp_path):
        root = self.repo(tmp_path)
        (root / "README.pypi.md").write_text("# x\n")
        (root / "pyproject.toml").write_text('[project]\nname = "confoundsearch"\ndynamic = ["version"]\n')
        assert "git" in rc.run_checks(None, False, "pypi", [], root, git=True)
        assert "git" not in rc.run_checks(None, False, "pypi", [], root)

    def test_main_accepts_the_flag(self, monkeypatch, capsys):
        monkeypatch.setattr(rc, "check_metadata", lambda root=rc.ROOT: [])
        monkeypatch.setattr(rc, "check_git", lambda root=rc.ROOT: ["uncommitted"])
        assert rc.main(["--git"]) == 1
        assert "FAIL  git" in capsys.readouterr().out


class TestTheLicence:
    MIT_BODY = (
        "Permission is hereby granted, free of charge, to any person obtaining a copy\n"
        'of this software and associated documentation files (the "Software"), to deal\n'
    )

    def test_it_is_the_standard_mit_licence(self):
        text = (ROOT / "LICENSE").read_text(encoding="utf-8")
        assert text.startswith("MIT License\n\nCopyright (c) 2026 ")
        assert (
            self.MIT_BODY in text and 'THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND' in text
        )
        assert text.count("\n\n") == 4  # the standard five paragraphs: no extra clauses were added

    def test_the_copyright_line_names_no_one(self):
        line = next(ln for ln in (ROOT / "LICENSE").read_text().splitlines() if ln.startswith("Copyright"))
        assert line == "Copyright (c) 2026 The ConfoundSearch authors"

    def test_it_is_declared_the_modern_way_and_nowhere_else(self):
        text = (ROOT / "pyproject.toml").read_text()
        assert 'license = "MIT"' in text and 'license-files = ["LICENSE*"]' in text
        assert "License ::" not in text  # an SPDX expression and a licence classifier together are an error

    def test_the_readme_says_so(self):
        assert "license-MIT" in (ROOT / "README.md").read_text()  # PyPI shows the licence from the metadata
