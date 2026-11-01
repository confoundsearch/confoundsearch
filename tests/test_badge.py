from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "coverage_badge", Path(__file__).parent.parent / "devops" / "tools" / "coverage_badge.py"
)
badge = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(badge)


def write_xml(path, lines=(10, 10), branches=(4, 4)):
    path.write_text(
        f'<coverage lines-valid="{lines[0]}" lines-covered="{lines[1]}" '
        f'branches-valid="{branches[0]}" branches-covered="{branches[1]}"/>'
    )
    return path


@pytest.mark.parametrize(
    "lines,branches,expected",
    [((10, 10), (4, 4), 100.0), ((10, 5), (10, 5), 50.0), ((90, 81), (10, 9), 90.0), ((0, 0), (0, 0), 100.0)],
)
def test_percent_counts_lines_and_branches_together(tmp_path, lines, branches, expected):
    assert badge.percent(write_xml(tmp_path / "c.xml", lines, branches)) == expected


@pytest.mark.parametrize(
    "value,expected",
    [
        (100, "#4c1"),
        (95, "#4c1"),
        (94.9, "#97ca00"),
        (90, "#97ca00"),
        (85, "#a4a61d"),
        (75, "#dfb317"),
        (60, "#fe7d37"),
        (10, "#e05d44"),
    ],
)
def test_color_thresholds(value, expected):
    assert badge.color(value) == expected


def test_svg_shows_the_percentage():
    svg = badge.badge_svg(97.5)
    assert svg.startswith("<svg") and "97.5%" in svg and "coverage" in svg
    assert "100%" in badge.badge_svg(100.0)


def test_main_writes_the_file_and_creates_directories(tmp_path):
    out = tmp_path / "nested" / "coverage.svg"
    assert badge.main(["x", str(write_xml(tmp_path / "c.xml")), str(out)]) == 0
    assert "100%" in out.read_text()
