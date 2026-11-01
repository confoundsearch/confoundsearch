"""Check that a release installs and works on every Python the package claims to support.

    python devops/tools/python_matrix.py                         # the latest release on PyPI, 3.8 to 3.13
    python devops/tools/python_matrix.py confoundsearch==0.2.0   # a particular release
    python devops/tools/python_matrix.py dist/*.whl --pythons 3.8 3.12

Each Python gets its own clean environment (uv fetches interpreters you do not have) and the same checks as
`smoke_test.py`: install, import, the command, both skills, and drawing a page. Exit status 0 means every
Python passed. It needs network access and `uv` (`pip install uv`).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))

import smoke_test  # noqa: E402

PYTHONS = ("3.8", "3.9", "3.10", "3.11", "3.12", "3.13")


def matrix(
    target: str,
    pythons: tuple[str, ...] = PYTHONS,
    index_url: str | None = None,
    retries: int = 0,
    wait: float = 15.0,
    smoke: Callable[..., list[str]] = smoke_test.smoke,
) -> dict[str, str]:
    """{python: "ok" or the reason it failed}, trying every Python even after one fails."""
    results = {}
    for python in pythons:
        try:
            smoke(target, index_url, retries, wait, python)
            results[python] = "ok"
        except (SystemExit, AssertionError) as err:
            lines = [line for line in str(err).splitlines() if line.strip()]
            results[python] = lines[-1] if lines else "failed"
    return results


def report(target: str, results: dict[str, str]) -> str:
    lines = [f"{target} on each Python:"]
    lines += [
        f"  {'ok  ' if r == 'ok' else 'FAIL'}  Python {p}" + ("" if r == "ok" else f": {r}")
        for p, r in results.items()
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None, run_matrix: Callable[..., dict[str, str]] = matrix) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("target", nargs="?", default="confoundsearch", help="a requirement or a built wheel")
    parser.add_argument("--pythons", nargs="+", default=list(PYTHONS), help="Python versions to try")
    parser.add_argument("--index-url", help="install from this package index instead of PyPI")
    parser.add_argument("--retries", type=int, default=0, help="try again if it is not installable yet")
    parser.add_argument("--wait", type=float, default=15.0, help="seconds between tries")
    args = parser.parse_args(argv)
    target = str(Path(args.target).resolve()) if Path(args.target).is_file() else args.target
    started = time.time()
    results = run_matrix(target, tuple(args.pythons), args.index_url, args.retries, args.wait)
    print(report(args.target, results))
    print(f"({time.time() - started:.0f}s)")
    return 0 if all(r == "ok" for r in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
