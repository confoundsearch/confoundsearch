"""devops/tools/set_version.py: the one command that changes the version."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
spec = importlib.util.spec_from_file_location("set_version", ROOT / "devops" / "tools" / "set_version.py")
sv = importlib.util.module_from_spec(spec)
sys.modules["set_version"] = sv
spec.loader.exec_module(sv)

BODY = '"""Docstring."""\n\nfrom x import y\n\n__version__ = "{v}"\n__all__ = ["a"]\n'


def init(tmp_path, version="0.1.0"):
    path = tmp_path / "__init__.py"
    path.write_text(BODY.format(v=version))
    return path


@pytest.mark.parametrize(
    "old,new",
    [("0.1.0", "0.1.1"), ("0.1.0", "0.2.0"), ("0.9.9", "1.0.0"), ("1.2.3", "1.2.10"), ("0.1", "0.1.1"),
     ("0.1.0", "0.2.0rc1"), ("1.0", "1.0.1.post1"), ("0.1.0", "0.1.1.dev3")],
)  # fmt: skip
def test_a_newer_valid_version_is_written_and_nothing_else_changes(tmp_path, old, new):
    path = init(tmp_path, old)
    assert sv.set_version(new, path) == f"{old} -> {new}"
    assert path.read_text() == BODY.format(v=new)


@pytest.mark.parametrize("new", ["0.1.0", "0.1", "0.0.9", "0.0.99", "0.1.0.0"])
def test_a_version_that_is_not_newer_is_refused_and_the_file_is_untouched(tmp_path, new):
    path = init(tmp_path)
    with pytest.raises(SystemExit, match="not newer"):
        sv.set_version(new, path)
    assert path.read_text() == BODY.format(v="0.1.0")


@pytest.mark.parametrize("new", ["v1.0", "1.0-beta", "latest", "1..0", "", "1.0.", "01.x"])
def test_an_invalid_version_is_refused(tmp_path, new):
    path = init(tmp_path)
    with pytest.raises(SystemExit, match="not a valid version"):
        sv.set_version(new, path)
    assert path.read_text() == BODY.format(v="0.1.0")


def test_a_file_without_a_version_is_an_error(tmp_path):
    path = tmp_path / "__init__.py"
    path.write_text("x = 1\n")
    with pytest.raises(SystemExit, match="no __version__"):
        sv.set_version("1.0.0", path)


def test_only_the_first_version_line_is_changed(tmp_path):
    path = tmp_path / "__init__.py"
    path.write_text('__version__ = "0.1.0"\n# __version__ = "0.1.0"\n')
    sv.set_version("0.1.1", path)
    assert path.read_text() == '__version__ = "0.1.1"\n# __version__ = "0.1.0"\n'


def test_main_prints_the_change_and_needs_exactly_one_argument(tmp_path, monkeypatch, capsys):
    path = init(tmp_path)
    monkeypatch.setattr(sv, "INIT", path)
    monkeypatch.setattr(sv, "set_version", lambda new, init=path: sv.__dict__["_real"](new, init))
    sv.__dict__["_real"] = lambda new, init: f"0.1.0 -> {new}"
    assert sv.main(["0.1.1"]) == 0 and capsys.readouterr().out == "0.1.0 -> 0.1.1\n"
    assert sv.main([]) == 2 and sv.main(["a", "b"]) == 2
    assert "Set the package version" in capsys.readouterr().out


def test_the_real_package_version_is_valid_and_is_what_the_tool_edits():
    text = sv.INIT.read_text()
    match = sv.CURRENT.search(text)
    assert match and sv.VALID.match(match.group(1))
    assert sv.INIT == ROOT / "confoundsearch" / "__init__.py"
