"""Read settings from a `.env` file, so API keys need not be typed into a terminal."""

from __future__ import annotations

import os
from pathlib import Path


def without_export(line: str) -> str:
    """`export KEY=1` -> `KEY=1` (str.removeprefix needs Python 3.9)."""
    return line[len("export ") :] if line.startswith("export ") else line


def load_dotenv(path: Path) -> None:
    """Put KEY=VALUE lines from `path` into the environment. Never overrides a variable already set."""
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = without_export(raw.strip())
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ.setdefault(key.strip(), value)


def update_dotenv(path: Path, values: dict[str, str]) -> None:
    """Set KEY=VALUE lines in `path`, keeping every other line. The file is made private to its owner."""
    pending = dict(values)
    lines = path.read_text(encoding="utf-8").splitlines() if path.is_file() else []
    for number, raw in enumerate(lines):
        key = without_export(raw.strip()).partition("=")[0].strip()
        if key in pending:
            lines[number] = f"{key}={pending.pop(key)}"
    lines += [f"{key}={value}" for key, value in pending.items()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    path.chmod(0o600)


def keep_out_of_git(path: Path) -> bool:
    """Make sure git ignores `path`, if it sits in a git folder. True if a line was added."""
    folder = path.parent
    ignore = folder / ".gitignore"
    if not ignore.is_file() and not (folder / ".git").exists():
        return False
    lines = ignore.read_text(encoding="utf-8").splitlines() if ignore.is_file() else []
    if path.name in (line.strip() for line in lines):
        return False
    ignore.write_text("\n".join([*lines, path.name]) + "\n", encoding="utf-8")
    return True
