"""devops/tools/python_matrix.py: one clean environment per Python, the same checks in each."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def load(name: str):
    path = ROOT / "devops" / "tools" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


smoke_test = load("smoke_test")
matrix = load("python_matrix")


class TestTheMatrix:
    def test_it_tries_every_python_and_says_why_each_failure_failed(self):
        calls = []

        def smoke(target, index_url, retries, wait, python):
            calls.append((target, python))
            if python == "3.8":
                raise SystemExit("FAILED: pip install\nERROR: No matching distribution found for x\n")
            if python == "3.9":
                raise AssertionError("the package should need nothing")
            if python == "3.10":
                raise SystemExit("")
            return ["ok"]

        results = matrix.matrix("x", ("3.8", "3.9", "3.10", "3.11"), smoke=smoke)
        assert [c[1] for c in calls] == ["3.8", "3.9", "3.10", "3.11"]  # a failure does not stop the rest
        assert results == {
            "3.8": "ERROR: No matching distribution found for x",
            "3.9": "the package should need nothing",
            "3.10": "failed",
            "3.11": "ok",
        }

    def test_the_options_reach_each_check(self):
        seen = []
        matrix.matrix("t", ("3.9",), "https://index", 3, 1.5, smoke=lambda *args: seen.append(args) or [])
        assert seen == [("t", "https://index", 3, 1.5, "3.9")]

    def test_the_report_marks_each_python(self):
        text = matrix.report("confoundsearch", {"3.8": "ok", "3.9": "no distribution"})
        assert "confoundsearch on each Python:" in text
        assert "  ok    Python 3.8" in text and "  FAIL  Python 3.9: no distribution" in text

    def test_it_covers_every_python_the_package_claims(self):
        pyproject = (ROOT / "pyproject.toml").read_text()
        for version in matrix.PYTHONS:
            assert f"Programming Language :: Python :: {version}" in pyproject
        assert 'requires-python = ">=3.8"' in pyproject and matrix.PYTHONS[0] == "3.8"


class TestTheCommand:
    def test_it_exits_zero_only_if_every_python_passed(self, capsys):
        good = {"3.8": "ok", "3.9": "ok"}
        assert matrix.main(["--pythons", "3.8", "3.9"], run_matrix=lambda *a: good) == 0
        assert matrix.main(["--pythons", "3.8"], run_matrix=lambda *a: {"3.8": "no"}) == 1
        assert "FAIL  Python 3.8: no" in capsys.readouterr().out

    def test_it_defaults_to_the_latest_release_on_every_python(self):
        seen = []
        matrix.main([], run_matrix=lambda *args: seen.append(args) or {})
        assert seen == [("confoundsearch", matrix.PYTHONS, None, 0, 15.0)]

    def test_a_built_wheel_is_passed_by_its_full_path(self, tmp_path):
        wheel = tmp_path / "x-1-py3-none-any.whl"
        wheel.write_text("")
        seen = []
        matrix.main([str(wheel)], run_matrix=lambda *args: seen.append(args[0]) or {})
        assert seen == [str(wheel.resolve())]

    def test_run_as_a_script_it_exits_with_the_status(self, monkeypatch):
        import runpy

        monkeypatch.setattr(sys, "argv", ["python_matrix.py", "--help"])
        with pytest.raises(SystemExit) as stop:
            runpy.run_path(str(ROOT / "devops" / "tools" / "python_matrix.py"), run_name="__main__")
        assert stop.value.code == 0


class TestTheEnvironment:
    def test_another_python_comes_from_uv(self, tmp_path):
        commands = []
        bin_dir = smoke_test.make_env(tmp_path / "env", "3.8", runner=lambda *c: commands.append(c) or "")
        assert commands == [("uv", "venv", "--seed", "--quiet", "--python", "3.8", str(tmp_path / "env"))]
        assert bin_dir == tmp_path / "env" / ("Scripts" if sys.platform == "win32" else "bin")

    def test_this_python_is_used_when_none_is_asked_for(self, tmp_path, monkeypatch):
        made = []

        class Builder:
            def __init__(self, with_pip):
                made.append(with_pip)

            def create(self, env):
                made.append(env)

        monkeypatch.setattr(smoke_test.venv, "EnvBuilder", Builder)
        smoke_test.make_env(tmp_path / "env", None)
        assert made == [True, tmp_path / "env"]
