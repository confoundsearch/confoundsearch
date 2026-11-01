"""Build the Lambda package: handler.py plus the confoundsearch package, in one zip.

    python devops/proxy/build.py            # writes devops/proxy/build/lambda.zip

The package has no dependencies of its own, so the zip is just our files. (boto3, which the handler uses to
read the key from AWS, is already part of the Lambda Python runtime.) The zip is the same every time it is
built from the same files, so a re-deploy of unchanged code changes nothing.
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PACKAGE = ROOT / "confoundsearch"
SKIP_PARTS = {"__pycache__"}
SKIP_SUFFIXES = {".pyc", ".pyo"}
FIXED_TIME = (2026, 1, 1, 0, 0, 0)  # so that the zip does not change when only the clock does


def files(package: Path = PACKAGE, handler: Path = HERE / "handler.py") -> list[tuple[Path, str]]:
    """(source file, name inside the zip), sorted so that the order never varies."""
    found = [(handler, "handler.py")]
    for path in sorted(package.rglob("*")):
        relative = path.relative_to(package.parent)
        if path.is_file() and not SKIP_PARTS & set(relative.parts) and path.suffix not in SKIP_SUFFIXES:
            found.append((path, relative.as_posix()))
    return found


def build(
    out: Path = HERE / "build" / "lambda.zip", package: Path = PACKAGE, handler: Path = HERE / "handler.py"
) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as archive:
        for source, name in files(package, handler):
            info = zipfile.ZipInfo(name, FIXED_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, source.read_bytes())
    return out


def main(argv: list[str] | None = None) -> int:
    out = Path(argv[0]) if argv else HERE / "build" / "lambda.zip"
    built = build(out)
    print(f"wrote {built} ({built.stat().st_size / 1e3:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
