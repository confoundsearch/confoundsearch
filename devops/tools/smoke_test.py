"""Install a built wheel, or a released version, into a brand-new environment and use it, as a user would.

    python devops/tools/smoke_test.py dist/confoundsearch-0.1.0-py3-none-any.whl     # a built wheel
    python devops/tools/smoke_test.py confoundsearch==0.1.0 --retries 12 --wait 15   # a release, from PyPI
    python devops/tools/smoke_test.py confoundsearch==0.1.0 --index-url https://test.pypi.org/simple/

A target that is not an existing file is installed from the package index, with retries, because a new
release takes a little while to appear. Checks that the command is installed, that the skill and the page
asset were packaged, that the public API imports, and that nothing from the repository (tests, assets,
examples) is needed. Exit status 0 means it works.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
import venv
from collections.abc import Callable
from pathlib import Path

PROBE = """
import pathlib, confoundsearch as c
here = pathlib.Path(c.__file__).parent / 'interface'
assert (here / 'ui' / 'app.html').read_bytes()
for skill in ('SKILL.md', 'UI_SKILL.md'):
    assert (here / 'skill' / skill).read_text().startswith('---')
names = ('run_episode', 'Environment', 'ElasticsearchCorpus', 'OpenAICompatibleClient', 'save_export')
for name in names + ('AuditLog',):
    assert hasattr(c, name), name
print(c.__version__)
"""


def run(*command: str, cwd: str | None = None) -> str:
    done = subprocess.run(command, capture_output=True, text=True, cwd=cwd)
    if done.returncode != 0:
        raise SystemExit(f"FAILED: {' '.join(command)}\n{done.stdout}{done.stderr}")
    return done.stdout.strip()


def install(
    python: str,
    target: str,
    index_url: str | None = None,
    retries: int = 0,
    wait: float = 15.0,
    sleep: Callable[[float], None] = time.sleep,
    runner: Callable[..., str] = run,
) -> None:
    """pip install `target`, trying again up to `retries` times: a fresh release may not be visible yet."""
    fresh = not Path(target).is_file()  # from an index, not a file: do not trust pip's cached index pages
    command = [python, "-m", "pip", "install", "--quiet", *(["--no-cache-dir"] if fresh else [])]
    command += ["--index-url", index_url] if index_url else []
    for attempt in range(retries + 1):
        try:
            runner(*command, target)
            return
        except SystemExit:
            if attempt == retries:
                raise
            print(f"{target} is not installable yet (try {attempt + 1}); waiting {wait:g}s", file=sys.stderr)
            sleep(wait)


KNOWN = {
    "iv": "X",
    "dv": "Y",
    "controls": [
        {
            "name": "Age",
            "citations_to_IV": [{"title": "A paper", "doi": "10.1/a", "quotation": "Quoted."}],
            "citations_to_DV": [{"title": "Another", "quotation": "Also quoted."}],
        }
    ],
}


def make_env(env: Path, python_version: str | None = None, runner: Callable[..., str] = run) -> Path:
    """A fresh virtual environment: this interpreter's, or with `python_version` that one (uv fetches it)."""
    if python_version:
        runner("uv", "venv", "--seed", "--quiet", "--python", python_version, str(env))
    else:
        venv.EnvBuilder(with_pip=True).create(env)
    return env / ("Scripts" if sys.platform == "win32" else "bin")


def smoke(
    target: str,
    index_url: str | None = None,
    retries: int = 0,
    wait: float = 15.0,
    python_version: str | None = None,
) -> list[str]:
    """Run the checks; returns a line per check passed. Raises SystemExit on the first failure."""
    lines = []
    with tempfile.TemporaryDirectory() as tmp:
        bin_dir = make_env(Path(tmp) / "venv", python_version)
        python, command = str(bin_dir / "python"), str(bin_dir / "confoundsearch")
        install(python, target, index_url, retries, wait)
        lines.append("installed into a clean environment")
        elsewhere = str(Path(tmp))  # not the repository, so nothing can be picked up from it
        version = run(python, "-c", PROBE, cwd=elsewhere)
        lines.append(f"imports, packaged assets and public API ok (version {version})")
        assert version in run(command, "--version", cwd=elsewhere)
        lines.append("the confoundsearch command runs")
        skills = Path(tmp) / "skills"
        run(command, "install-skill", "--dir", str(skills), cwd=elsewhere)
        for name in ("confoundsearch", "confoundsearchui"):
            assert (skills / name / "SKILL.md").stat().st_size > 0
        lines.append("install-skill works (both skills)")
        known = Path(tmp) / "known.json"
        known.write_text(json.dumps(KNOWN), encoding="utf-8")
        run(command, "show", str(known), cwd=elsewhere)
        assert "Candidate Confounds" in (Path(tmp) / "known.html").read_text(encoding="utf-8")
        lines.append("show draws the page")
        run(command, "audit", "--help", cwd=elsewhere)
        lines.append("audit command is available")
        shown = run(
            python, "-m", "pip", "show", "confoundsearch"
        )  # what this package needs, not the seed tools
        needs = [
            line.partition(":")[2].strip() for line in shown.splitlines() if line.startswith("Requires:")
        ]
        assert needs == [""], f"the package should need nothing, but it requires: {needs}"
        lines.append("no dependencies were pulled in")
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("target", help="a built wheel, or a requirement such as confoundsearch==0.1.0")
    parser.add_argument("--index-url", help="install from this package index instead of PyPI")
    parser.add_argument(
        "--retries", type=int, default=0, help="try again this many times if it is not installable yet"
    )
    parser.add_argument("--wait", type=float, default=15.0, help="seconds between tries")
    parser.add_argument("--python", help="test with this Python version, e.g. 3.8 (needs uv to fetch it)")
    args = parser.parse_args(argv)
    target = str(Path(args.target).resolve()) if Path(args.target).is_file() else args.target
    for line in smoke(target, args.index_url, args.retries, args.wait, args.python):
        print(f"ok  {line}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
