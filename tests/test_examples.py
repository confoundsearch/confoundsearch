from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from confoundsearch import InMemoryCorpus

ROOT = Path(__file__).parent.parent
DEMO = ROOT / "examples" / "passages.jsonl"
SPEC = importlib.util.spec_from_file_location("make_example", ROOT / "examples" / "make_example.py")
make_example = importlib.util.module_from_spec(SPEC)
sys.modules["make_example"] = make_example
SPEC.loader.exec_module(make_example)


def test_the_demo_passages_load_and_are_searchable():
    corpus = InMemoryCorpus.from_jsonl(DEMO)
    assert len(corpus.passages) == 8
    assert len({p.id for p in corpus.passages}) == 8
    assert all(p.title.startswith("(demo)") for p in corpus.passages)
    assert {p.id for p in corpus.search("interest rates", 5)} == {"demo-5", "demo-6"}


def test_the_committed_example_audit_trail_is_exactly_what_the_script_produces(tmp_path):
    make_example.build(tmp_path / "example.html")
    rebuilt = (tmp_path / "example.audit.jsonl").read_text(encoding="utf-8")
    committed = (ROOT / "examples" / "example_audit.jsonl").read_text(encoding="utf-8")
    assert rebuilt == committed, "examples/example_audit.jsonl is stale; run: python examples/make_example.py"


def test_the_example_audit_trail_is_intact_complete_and_honest_about_its_model():
    from confoundsearch import AuditTrail

    trail = AuditTrail.load(ROOT / "examples" / "example_audit.jsonl")
    assert trail.intact and trail.complete
    models = {e.data["model"] for e in trail.of_kind("model_call")}
    assert models == {"scripted-stand-in"}  # the log says it was not a real model
    assert {r.confound for r in trail.evidence if r.verdict == "supported"} == {
        "firm age",
        "firm size",
        "interest rates",
    }
    assert all(r.retrieved_by for r in trail.evidence)  # every judged passage traces back to a search
    assert len(trail.of_kind("model_call")) < 3 * len(
        trail.of_kind("action")
    )  # about one or two calls per step


def prompt(label: str, target: str, passage: str) -> str:
    return f'Claim: "{label}" (also written: x) affects "{target}".\n\nPassage:\n{passage}'


@pytest.mark.parametrize(
    "text,expected",
    [
        (prompt("firm size", "IT investment", "Larger firms adopt information technology."), "yes"),
        (prompt("firm size", "firm performance", "Larger firms adopt information technology."), "no"),
        (prompt("firm age", "firm performance", "Older firms show higher profitability."), "yes"),
        (prompt("mystery", "firm performance", "Mystery drives performance."), "yes"),
        ("no claim in this prompt", "no"),
    ],
)
def test_the_scripted_judge(text, expected):
    assert make_example.ScriptedModel().judge(text) == expected


def test_the_scripted_model_proposes_json_and_judges_otherwise():
    model = make_example.ScriptedModel()
    assert "queries" in model.chat([{"role": "user", "content": "x. Reply with JSON only:"}])
    assert model.chat([{"role": "user", "content": "no claim here"}]) == "no"
