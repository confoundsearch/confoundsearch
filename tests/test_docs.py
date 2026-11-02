"""The README stays short (about 100 lines) and every link in it must lead somewhere."""

from __future__ import annotations

import importlib.util
import re
import shlex
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
README = ROOT / "README.md"
LINK = re.compile(r"\]\(([^)\s]+)\)")


def local_links(path: Path):
    for target in LINK.findall(path.read_text(encoding="utf-8")):
        if not target.startswith(("http://", "https://", "mailto:", "#")):
            yield target.split("#")[0], target


@pytest.mark.parametrize("path", [README], ids=lambda p: str(p.relative_to(ROOT)))
def test_local_links_and_images_exist(path):
    for file_part, target in local_links(path):
        assert (path.parent / file_part).exists(), f"{path.relative_to(ROOT)} links to missing {target}"


def test_anchors_point_at_real_headings():
    for path in [README]:
        for target in LINK.findall(path.read_text(encoding="utf-8")):
            if "#" in target and not target.startswith("http"):
                file_part, anchor = target.split("#", 1)
                doc = path if not file_part else path.parent / file_part
                headings = re.findall(r"^#+\s+(.*)$", doc.read_text(encoding="utf-8"), re.M)
                slugs = {re.sub(r"[^a-z0-9 -]", "", h.lower()).replace(" ", "-") for h in headings}
                assert anchor in slugs, f"{path.relative_to(ROOT)}: no heading for #{anchor}"


def test_readme_stays_within_one_page():
    lines = README.read_text(encoding="utf-8").splitlines()
    assert len(lines) <= 110, f"README is {len(lines)} lines; keep it short"


def test_readme_has_the_expected_sections():
    headings = re.findall(r"^## (.*)$", README.read_text(encoding="utf-8"), re.M)
    assert headings == [
        "Quick start",
        "Keys",
        "Data directory",
        "What's in this code base?",
        "Other key directories",
    ]


def test_readme_states_what_it_is_up_front():
    text = README.read_text(encoding="utf-8")
    assert "Organization Science" in text.split("## ")[0]


def test_the_readme_example_image_is_a_real_picture():
    image = ROOT / "assets" / "example.png"
    assert image.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n" and image.stat().st_size > 10_000


def load_make_example():
    spec = importlib.util.spec_from_file_location("make_example_docs", ROOT / "examples" / "make_example.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["make_example_docs"] = module
    spec.loader.exec_module(module)
    return module


PYPI = ROOT / "README.pypi.md"


def test_the_package_page_is_minimal_and_has_no_links_that_would_break_on_pypi():
    """PyPI shows it: a title, one line on what it is, and a pointer to the submission documents."""
    text = PYPI.read_text(encoding="utf-8")
    assert not re.findall(
        r"\]\((?!https?://|mailto:|#)[^)\s]+\)", text
    )  # relative links do not resolve there
    assert "Organization Science" in text and "See our submission documents for more." in text
    assert len(text.splitlines()) <= 6 and "```" not in text


def test_the_quick_start_commands_are_real():
    """The README quick start: demo first, then the appendix figure's setup, install-skill, run --out, review --in."""
    from confoundsearch.interface import cli

    text = README.read_text(encoding="utf-8")
    section = text[text.index("## Quick start") : text.index("## Keys")]
    block = "\n".join(re.findall(r"```bash\n(.*?)```", section, re.S))  # the quick start and the full setup
    commands = [line.split("#")[0].strip() for line in block.replace("\\\n", " ").splitlines()]
    parser = cli.build_parser()
    seen = [
        parser.parse_args(shlex.split(command)[1:])
        for command in commands
        if command.startswith("confoundsearch ")
    ]
    assert [a.command for a in seen] == ["demo", "setup", "install-skill", "run", "review"]
    run, review = seen[3], seen[4]
    assert (
        str(run.out) == "results/org.html" and str(review.input) == "results/org.html"
    )  # review pulls in run's output
    assert (run.iv, run.dv) == ("Organizational structure", "Firm performance")
    assert '/confoundsearch iv="Organizational structure" dv="Firm performance"' in text  # step 5A
    assert {"confoundsearch", "confoundsearchui"} <= set(cli.SKILLS)


def test_the_readme_points_only_at_files_the_public_copy_has():
    """The public review copy leaves out `examples/` (and what the sync's EXCLUDE says), so the one README
    must not rely on it."""
    private = ("examples/",)
    text = README.read_text(encoding="utf-8")
    assert not [t for _, t in local_links(README) if t.startswith(private)]
    assert not any(f"`{p}" in text or f"({p}" in text for p in private)
