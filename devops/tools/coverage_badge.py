"""Write a coverage badge (SVG) from coverage.xml. Standard library only.

    python devops/tools/coverage_badge.py coverage.xml devops/badges/coverage.svg

CI runs this after the tests and commits the result, so the README badge needs
no third-party service or token.
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

FONT = "Verdana,Geneva,DejaVu Sans,sans-serif"
SHADOW = 'fill="#010101" fill-opacity=".3"'
COLORS = [(95.0, "#4c1"), (90.0, "#97ca00"), (80.0, "#a4a61d"), (70.0, "#dfb317"), (50.0, "#fe7d37")]


def percent(coverage_xml: Path) -> float:
    """Total line-and-branch coverage, as coverage.py reports it."""
    root = ET.parse(coverage_xml).getroot()
    lines, covered = int(root.attrib["lines-valid"]), int(root.attrib["lines-covered"])
    branches, taken = int(root.attrib["branches-valid"]), int(root.attrib["branches-covered"])
    total = lines + branches
    return 100.0 if total == 0 else round(100.0 * (covered + taken) / total, 1)


def color(value: float) -> str:
    return next((hex_ for floor, hex_ in COLORS if value >= floor), "#e05d44")


def badge_svg(value: float) -> str:
    text = f"{value:g}%"
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="99" height="20" role="img" '
        f'aria-label="coverage: {text}">'
        '<linearGradient id="s" x2="0" y2="100%"><stop offset="0" stop-color="#bbb" stop-opacity=".1"/>'
        '<stop offset="1" stop-opacity=".1"/></linearGradient>'
        '<clipPath id="r"><rect width="99" height="20" rx="3" fill="#fff"/></clipPath>'
        '<g clip-path="url(#r)"><rect width="62" height="20" fill="#555"/>'
        f'<rect x="62" width="37" height="20" fill="{color(value)}"/>'
        '<rect width="99" height="20" fill="url(#s)"/></g>'
        f'<g fill="#fff" text-anchor="middle" font-family="{FONT}" font-size="11">'
        f'<text x="31" y="15" {SHADOW}>coverage</text><text x="31" y="14">coverage</text>'
        f'<text x="80.5" y="15" {SHADOW}>{text}</text><text x="80.5" y="14">{text}</text>'
        "</g></svg>\n"
    )


def main(argv: list[str]) -> int:
    source, target = Path(argv[1]), Path(argv[2])
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(badge_svg(percent(source)), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
