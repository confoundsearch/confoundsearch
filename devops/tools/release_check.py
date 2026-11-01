"""Is this ready to publish? Checks the metadata, the built files, and (optionally) the package index.

    python devops/tools/release_check.py                       # metadata only
    python devops/tools/release_check.py --dist dist           # plus the built wheel and sdist
    python devops/tools/release_check.py --dist dist --online  # plus: is this version already on PyPI?
    python devops/tools/release_check.py --online --index testpypi
    python devops/tools/release_check.py --dist dist --online --git   # before an upload: clean checkout

Exit status 0 means every check passed. Standard library only (tomli on Python 3.10).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tarfile
import urllib.error
import urllib.request
import zipfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Callable, Dict, Optional

try:
    import tomllib  # type: ignore[import-not-found,unused-ignore]
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib  # type: ignore[import-not-found,no-redef,unused-ignore]

ROOT = Path(__file__).resolve().parents[2]
NAME = "confoundsearch"
INDEXES = {"pypi": "https://pypi.org", "testpypi": "https://test.pypi.org"}
VERSION = re.compile(r"^\d+(\.\d+)*((a|b|rc)\d+)?(\.post\d+)?(\.dev\d+)?$")
RELATIVE_LINK = re.compile(r"\]\((?!https?://|mailto:|#)[^)\s]+\)")
OWNER = re.compile(r"github\.com/([A-Za-z0-9_-]+)/")
EMAIL = re.compile(r"[\w.+-]+@([\w-]+(?:\.[\w-]+)+)")
ALLOWED_OWNERS = {"astral-sh"}
ALLOWED_EMAIL_DOMAINS = ("example.com", "example.org", "noreply.github.com", "noreply.anthropic.com")
SESSION = "claude" + ".ai/code"
LAYERS = ("data", "logic", "model", "interface")
REQUIRED_IN_WHEEL = (
    f"{NAME}/py.typed",
    f"{NAME}/interface/skill/SKILL.md",
    f"{NAME}/interface/skill/UI_SKILL.md",
    f"{NAME}/interface/ui/app.html",
    *(f"{NAME}/{layer}/__init__.py" for layer in LAYERS),
)
FORBIDDEN_IN_SDIST = ("tests/", "assets/", "examples/", "evals/", ".github/", ".env", ".git/")

Fetch = Callable[[str], Optional[Dict[str, Any]]]


def read_version(root: Path = ROOT) -> str:
    text = (root / NAME / "__init__.py").read_text(encoding="utf-8")
    match = re.search(r'^__version__ = "([^"]+)"', text, re.M)
    if not match:
        raise SystemExit(f"no __version__ in {root / NAME / '__init__.py'}")
    return match.group(1)


def check_metadata(root: Path = ROOT) -> list[str]:
    """Problems with pyproject.toml and the files it points at."""
    problems = []
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    if project.get("name") != NAME:
        problems.append(f"project name is {project.get('name')!r}, expected {NAME!r}")
    if "version" in project or "version" not in project.get("dynamic", []):
        problems.append("version must be dynamic (read from __version__), so there is one place to bump it")
    version = read_version(root)
    if not VERSION.match(version):
        problems.append(f"__version__ {version!r} is not a valid PEP 440 version")
    if not project.get("license") and not any(
        c.startswith("License ::") for c in project.get("classifiers", [])
    ):
        problems.append(
            "no licence declared: choose one, add a LICENSE file, and set `license` in pyproject.toml"
        )
    elif not list(root.glob("LICENSE*")):
        problems.append("a licence is declared but there is no LICENSE file")
    readme = project.get("readme")
    readme_path = root / (readme["file"] if isinstance(readme, dict) else str(readme))
    if not readme or not readme_path.is_file():
        problems.append("the readme named in pyproject.toml does not exist")
    else:
        text = readme_path.read_text(encoding="utf-8")
        if RELATIVE_LINK.search(text):
            problems.append(
                f"{readme_path.name} has relative links or images, which break on the package page"
            )
    if not project.get("requires-python"):
        problems.append("requires-python is not set")
    if not project.get("description"):
        problems.append("description is not set")
    return problems


def scan_text(label: str, text: str, blocklist: Iterable[re.Pattern[str]] = ()) -> list[str]:
    """Anything in `text` that would identify a person: an owner, a real email, a session link, a name."""
    found = [f"{label}: names github.com/{o}" for o in OWNER.findall(text) if o not in ALLOWED_OWNERS]
    found += [
        f"{label}: email domain {d}" for d in EMAIL.findall(text) if not d.endswith(ALLOWED_EMAIL_DOMAINS)
    ]
    if SESSION in text:
        found.append(f"{label}: contains a session link")
    found += [f"{label}: matches blocklist {p.pattern!r}" for p in blocklist if p.search(text)]
    return found


def _members(path: Path) -> Iterable[tuple[str, str]]:
    """(name, text) for every text file inside a wheel or sdist."""
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            for name in archive.namelist():
                yield name, archive.read(name).decode("utf-8", "replace")
    else:
        with tarfile.open(path) as archive:
            for member in archive.getmembers():
                if member.isfile() and (handle := archive.extractfile(member)):
                    yield member.name, handle.read().decode("utf-8", "replace")


def check_dist(dist: Path, version: str, blocklist: Iterable[re.Pattern[str]] = ()) -> list[str]:
    """Problems with the built wheel and sdist."""
    wheels, sdists = sorted(dist.glob("*.whl")), sorted(dist.glob("*.tar.gz"))
    problems = []
    if len(wheels) != 1 or len(sdists) != 1:
        return [f"expected exactly one wheel and one sdist in {dist}, found {len(wheels)} and {len(sdists)}"]
    wheel, sdist = wheels[0], sdists[0]
    if f"-{version}-" not in wheel.name or f"-{version}.tar.gz" not in sdist.name:
        problems.append(f"the files in {dist} are not version {version}: {wheel.name}, {sdist.name}")
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
    problems += [f"the wheel is missing {name}" for name in REQUIRED_IN_WHEEL if name not in names]
    if f"{NAME}-{version}.dist-info/licenses/LICENSE" not in names:
        problems.append("the wheel does not carry the LICENSE file")
    problems += [
        f"the wheel contains {n}" for n in sorted(names) if n.startswith(("tests/", "assets/", "examples/"))
    ]
    for name, _ in _members(sdist):
        relative = name.split("/", 1)[1] if "/" in name else name
        problems += [
            f"the sdist contains {relative}" for bad in FORBIDDEN_IN_SDIST if relative.startswith(bad)
        ]
    for built in (wheel, sdist):
        for name, text in _members(built):
            problems += scan_text(f"{built.name}:{name}", text, blocklist)
    return problems


def check_git(root: Path = ROOT) -> list[str]:
    """The release must be tagged afterwards, so the checkout must match what is uploaded."""

    def git(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True)

    status = git("status", "--porcelain")
    if status.returncode != 0:
        return ["this is not a git checkout, so the release cannot be tagged"]
    problems = []
    changed = [line for line in status.stdout.splitlines() if line.strip()]
    if changed:
        problems.append(
            f"{len(changed)} uncommitted change(s): commit them first, so the tag matches the upload"
        )
    tag = f"v{read_version(root)}"
    existing = git("rev-parse", "-q", "--verify", f"refs/tags/{tag}^{{commit}}")
    if existing.returncode == 0 and existing.stdout.strip() != git("rev-parse", "HEAD").stdout.strip():
        problems.append(f"tag {tag} already exists at a different commit; bump the version or move the tag")
    return problems


def fetch_json(url: str) -> dict[str, Any] | None:
    """The JSON at `url`, or None on a 404. Other failures raise."""
    try:
        with urllib.request.urlopen(url, timeout=30) as response:  # noqa: S310
            data: dict[str, Any] = json.load(response)
            return data
    except urllib.error.HTTPError as err:
        if err.code == 404:
            return None
        raise


def check_index(version: str, index: str = "pypi", fetch: Fetch | None = None) -> list[str]:
    """Problems with publishing `version` to the index: that version is already there."""
    data = (fetch or fetch_json)(f"{INDEXES[index]}/pypi/{NAME}/json")
    if data is None:
        return []  # the name is free: this will be the first release
    released = list(data.get("releases", {}))
    if version in released:
        return [f"version {version} is already on {index}; a version can never be uploaded twice, so bump it"]
    return []


def run_checks(
    dist: Path | None,
    online: bool,
    index: str,
    blocklist: list[re.Pattern[str]],
    root: Path = ROOT,
    git: bool = False,
) -> dict[str, list[str]]:
    results: dict[str, list[str]] = {"metadata": check_metadata(root)}
    version = read_version(root)
    if git:
        results["git"] = check_git(root)
    if dist is not None:
        results["built files"] = check_dist(dist, version, blocklist)
    if online:
        results[f"{index} index"] = check_index(version, index)
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dist", type=Path, help="check the wheel and sdist in this folder")
    parser.add_argument(
        "--online", action="store_true", help="ask the package index whether this version exists"
    )
    parser.add_argument("--index", choices=INDEXES, default="pypi")
    parser.add_argument(
        "--git", action="store_true", help="the checkout must be clean and its release tag free"
    )
    parser.add_argument(
        "--blocklist", type=Path, help="file of regexes (names, institution) that must not appear"
    )
    args = parser.parse_args(argv)
    patterns = []
    if args.blocklist:
        patterns = [
            re.compile(line, re.I) for line in args.blocklist.read_text().splitlines() if line.strip()
        ]
    results = run_checks(args.dist, args.online, args.index, patterns, git=args.git)
    failed = False
    for name, problems in results.items():
        print(f"{'FAIL' if problems else 'ok  '}  {name}")
        for problem in problems:
            print(f"        - {problem}")
        failed = failed or bool(problems)
    print(f"\n{NAME} {read_version()}: " + ("NOT ready to publish" if failed else "ready to publish"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
