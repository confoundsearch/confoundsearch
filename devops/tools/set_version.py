"""Set the package version, the one place it lives.

    python devops/tools/set_version.py 0.1.1

Rewrites `__version__` in confoundsearch/__init__.py. Refuses a version that is not valid, or is not newer
than the current one. Standard library only.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INIT = ROOT / "confoundsearch" / "__init__.py"
VALID = re.compile(r"^\d+(\.\d+)*((a|b|rc)\d+)?(\.post\d+)?(\.dev\d+)?$")
CURRENT = re.compile(r'^__version__ = "([^"]+)"', re.M)


def release_numbers(version: str) -> tuple[int, ...]:
    """(0, 1, 1) from '0.1.1rc2': the numbers of the release, for comparing."""
    return tuple(int(part) for part in re.match(r"\d+(?:\.\d+)*", version).group(0).split("."))  # type: ignore[union-attr]


def padded(version: str, other: str) -> tuple[int, ...]:
    """The release numbers of `version`, zero-padded to the longer of the two, so that 0.1 equals 0.1.0."""
    numbers, width = release_numbers(version), max(len(release_numbers(version)), len(release_numbers(other)))
    return numbers + (0,) * (width - len(numbers))


def set_version(new: str, init: Path = INIT) -> str:
    """Write `new` into `init`. Returns a one-line summary, or raises SystemExit with the reason it cannot."""
    text = init.read_text(encoding="utf-8")
    match = CURRENT.search(text)
    if not match:
        raise SystemExit(f"no __version__ found in {init}")
    old = match.group(1)
    if not VALID.match(new):
        raise SystemExit(f"{new!r} is not a valid version (for example 0.2.0, 1.0.0rc1)")
    if padded(new, old) <= padded(old, new):
        raise SystemExit(f"{new} is not newer than the current version {old}")
    init.write_text(CURRENT.sub(f'__version__ = "{new}"', text, count=1), encoding="utf-8")
    return f"{old} -> {new}"


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if len(args) != 1:
        print(__doc__)
        return 2
    print(set_version(args[0]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
