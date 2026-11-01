"""requirements.yml must describe the same environment as pyproject.toml, and be valid conda."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib  # type: ignore[no-redef,import-not-found,unused-ignore]

ROOT = Path(__file__).parent.parent
TEXT = (ROOT / "requirements.yml").read_text(encoding="utf-8")
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]

# PyPI name -> conda-forge name, where they differ (checked against the conda-forge index)
CONDA_NAME = {"build": "python-build", "rust-just": "just"}
# Tools the environment needs that are not Python dependencies of the package
EXTRA_TOOLS = {"python", "pip"}
# Needed only on Python 3.10, which the environment (Python 3.12) is not
ONLY_ON_OLD_PYTHON = {"tomli"}


def parse(text: str) -> dict[str, object]:
    """Just enough YAML for this file: scalars, lists, and one nested `pip:` list."""
    parsed: dict[str, object] = {}
    key, pip = "", None
    for raw in text.splitlines():
        line = re.sub(r"(^|\s)#.*$", "", raw).rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        item = line.strip()
        if indent == 0 and ":" in item:
            key, _, value = item.partition(":")
            parsed[key] = value.strip() or []
            pip = None
        elif item.startswith("- pip:"):
            pip = []
            parsed.setdefault("dependencies", []).append({"pip": pip})  # type: ignore[union-attr]
        elif item.startswith("- ") and pip is not None and indent > 2:
            pip.append(item[2:].strip())
        elif item.startswith("- "):
            pip = None
            parsed[key].append(item[2:].strip())  # type: ignore[union-attr]
    return parsed


ENV = parse(TEXT)
DEPENDENCIES = [d for d in ENV["dependencies"] if isinstance(d, str)]  # type: ignore[union-attr]
PIP = next(d["pip"] for d in ENV["dependencies"] if isinstance(d, dict))  # type: ignore[union-attr,index]


def split(requirement: str) -> tuple[str, str]:
    """('pytest', '>=8') from 'pytest>=8'; ('python', '=3.12') from 'python=3.12'."""
    match = re.match(r"^([A-Za-z0-9_.-]+)\s*(.*)$", requirement.split(";")[0].strip())
    assert match, requirement
    return match.group(1).lower(), match.group(2).replace(" ", "")


def extras() -> dict[str, str]:
    """{conda name: version spec} for everything in the dev and release extras."""
    found = {}
    for extra in ("dev", "release"):
        for requirement in PYPROJECT["optional-dependencies"][extra]:
            name, spec = split(requirement)
            if name not in ONLY_ON_OLD_PYTHON:
                found[CONDA_NAME.get(name, name)] = spec
    return found


def conda() -> dict[str, str]:
    return dict(split(d) for d in DEPENDENCIES)


def test_it_is_valid_yaml():
    yaml = pytest.importorskip("yaml")
    loaded = yaml.safe_load(TEXT)
    assert loaded["name"] == "confoundsearch" and loaded["channels"] == ["conda-forge"]
    assert {"pip": ["-e ."]} in loaded["dependencies"]
    assert [
        d for d in loaded["dependencies"] if isinstance(d, str)
    ] == DEPENDENCIES  # the small parser agrees


def test_the_environment_is_named_for_the_project_and_uses_only_conda_forge():
    assert ENV["name"] == "confoundsearch" and ENV["channels"] == [
        "conda-forge"
    ]  # not the licensed "defaults"


def test_every_tool_in_the_dev_and_release_extras_is_in_the_environment_at_the_same_minimum():
    wanted, have = extras(), conda()
    assert wanted, "the extras were not read"
    for name, spec in wanted.items():
        assert name in have, f"{name} is in pyproject.toml's extras but not in requirements.yml"
        assert have[name] == spec, (
            f"{name}: requirements.yml says {have[name]!r}, pyproject.toml says {spec!r}"
        )


def test_nothing_stale_is_left_in_the_environment():
    extra = set(conda()) - set(extras()) - EXTRA_TOOLS
    assert not extra, f"in requirements.yml but not in pyproject.toml's extras: {sorted(extra)}"


def test_the_package_has_no_runtime_dependencies_so_none_are_listed():
    assert PYPROJECT["dependencies"] == []
    runtime = set(conda()) - set(extras()) - EXTRA_TOOLS
    assert not runtime


def test_the_package_itself_is_installed_editable_from_the_repository():
    assert PIP == ["-e ."]


def test_python_is_a_version_the_package_supports_and_ci_tests():
    name, spec = split(next(d for d in DEPENDENCIES if d.startswith("python")))
    version = spec[1:] if spec.startswith("=") else spec
    assert name == "python" and re.fullmatch(r"3\.\d+", version)
    assert f"Programming Language :: Python :: {version}" in PYPROJECT["classifiers"]
    low = tuple(int(x) for x in PYPROJECT["requires-python"][len(">=") :].split("."))
    assert tuple(int(x) for x in version.split(".")) >= low
    assert f'"{version}"' in (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    assert not ONLY_ON_OLD_PYTHON or tuple(int(x) for x in version.split(".")) >= (
        3,
        11,
    )  # tomli is not needed


def test_just_is_at_least_the_version_the_justfile_says_it_needs():
    justfile = (ROOT / "justfile").read_text()
    needed = re.search(r"Needs just >= (\d+(?:\.\d+)*)", justfile).group(1)
    assert conda()["just"] == f">={needed}"


def test_the_tomli_fallback_is_declared_for_python_3_10_in_both_extras():
    for extra in ("dev", "release"):
        assert any(
            r.startswith("tomli") and "python_version < '3.11'" in r
            for r in PYPROJECT["optional-dependencies"][extra]
        )


def test_the_conda_name_differences_are_documented_in_the_file():
    assert "python-build" in TEXT and "PyPI's `build` is `python-build`" in TEXT


def test_the_commands_in_the_header_are_the_real_ones():
    assert (
        "conda env create -f requirements.yml" in TEXT
        and "conda env update -f requirements.yml --prune" in TEXT
    )
    assert "conda env create --file requirements.yml" in (ROOT / "justfile").read_text()


class TestTheParser:
    def test_comments_blank_lines_and_nesting(self):
        parsed = parse(
            "name: x\n# c\nchannels:\n  - a  # trailing\n\ndependencies:\n  - p>=1\n  - pip:\n      - -e .\n"
        )
        assert parsed == {"name": "x", "channels": ["a"], "dependencies": ["p>=1", {"pip": ["-e ."]}]}

    @pytest.mark.parametrize(
        "requirement,expected",
        [("pytest>=8", ("pytest", ">=8")), ("python=3.12", ("python", "=3.12")), ("pip", ("pip", "")),
         ("tomli>=2; python_version < '3.11'", ("tomli", ">=2")), ("PyTest >= 8", ("pytest", ">=8"))],
    )  # fmt: skip
    def test_split(self, requirement, expected):
        assert split(requirement) == expected
