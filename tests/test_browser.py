"""The exported page in a real browser (Chromium via Node's Playwright). Skipped where they are missing."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SEVEN = {
    "iv": "Organizational structure",
    "dv": "Firm performance",
    "controls": [
        {
            "name": name,
            "citations_to_IV": [{"title": f"{name} and structure", "doi": "10.1/a", "quotation": "Quoted."}],
            "citations_to_DV": [{"title": f"{name} and performance", "doi": "10.1/b", "quotation": "Also."}],
        }
        for name in ("Alpha", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot", "Golf")
    ],
}


def node_modules() -> str | None:
    if not shutil.which("node") or not shutil.which("npm"):
        return None
    root = subprocess.run(["npm", "root", "-g"], capture_output=True, text=True).stdout.strip()
    return root if (Path(root) / "playwright").exists() else None


@pytest.mark.skipif(node_modules() is None, reason="needs node with playwright")
def test_choosing_an_edge_a_confound_or_nothing_keeps_the_diagram_cards_and_evidence_in_agreement(tmp_path):
    from confoundsearch.interface.known import page_from_json

    PAGE = tmp_path / "seven.html"  # seven confounds: five are drawn in the diagram, all have cards
    PAGE.write_text(page_from_json(SEVEN), encoding="utf-8")
    done = subprocess.run(
        ["node", str(ROOT / "tests" / "browser" / "click_cards.js"), str(PAGE)],
        capture_output=True,
        text=True,
        env={**os.environ, "NODE_PATH": node_modules() or ""},
        timeout=120,
    )
    assert done.returncode == 0, done.stderr
    seen = json.loads(done.stdout)
    assert seen["errors"] == [] and seen["cards"] == 7 and seen["selectedAtStart"] == 0
    # one arrow chosen: it alone is strong, its count is a pill, every other arrow has faded
    assert seen["edgeSelected"] == 1 and seen["fadedEdges"] is True
    assert seen["pillShown"] == ["inline"] and seen["legActive"] == 1
    # no numbers on the arrows until one is chosen; then only that arrow's count, or both for a card
    assert (
        seen["countsAtStart"] == 0
        and seen["countsOnEdge"] == 1
        and seen["countsOnBoth"] == 2
        and seen["countsCleared"] == 0
    )
    # the evidence names the edge, and the pill, the ring and the side bar are the confound's own colour
    assert seen["header"].replace("\n", " ").startswith(seen["name"]) and seen["header"].endswith(
        "Firm performance"
    )
    assert seen["pillOutline"] == seen["expectedRgb"] == seen["panelBar"]
    assert seen["ringColor"] == seen["color"]
    # "also" switches to the other edge; a card shows both; a card past the fifth still has evidence
    assert seen["otherHeader"].endswith("Organizational structure")
    assert seen["sections"] == 2 and seen["edgesBoth"] == 2 and seen["ringsBoth"] == 2
    assert seen["lateEvidence"] == seen["lateName"] == "Golf" and seen["lateEdges"] == 0
    assert seen["nodesStillThere"] == 5  # choosing never hides a node
    assert seen["clearedSelection"] == 0 and seen["clearedPanel"].startswith("Click an arrow")
