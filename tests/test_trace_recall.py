"""The recall trace labels each gold link with the first stage where a run lost it."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "evals" / "trace_recall.py"
spec = importlib.util.spec_from_file_location("trace_recall", SCRIPT)
assert spec is not None and spec.loader is not None
trace_recall = importlib.util.module_from_spec(spec)
sys.modules["trace_recall"] = trace_recall  # its dataclasses look their module up
spec.loader.exec_module(trace_recall)

KATHURIA = "A Strategic Value Appropriation Path for Cloud Computing"
SIZE_PAPER = "The effect of firm size on profitability"


def event(kind: str, step: int = 0, **data: object) -> str:
    return json.dumps({"seq": 0, "step": step, "time": "", "kind": kind, "data": data, "hash": ""})


def audit(tmp_path: Path, *, retrieved: bool = True, cited: bool = True, verdict: str = "supported") -> Path:
    size = {
        "type": "AddConfound",
        "label": "Organization size",
        "strings": ["firm size", "organization size"],
    }
    lines = [
        event(
            "run_started",
            independent_variable="Cloud computing capability",
            dependent_variable="Firm performance",
        ),
        event("action", 1, action=size),
        event("action", 2, action={"type": "Retrieve", "query": "firm size firm performance", "k": 3}),
        event(
            "action", 3, action={"type": "Retrieve", "query": "industry turbulence firm performance", "k": 3}
        ),
    ]
    if retrieved:
        passage = {"id": "p1", "title": SIZE_PAPER.upper(), "url": "", "text": "Size raises profits."}
        lines.append(event("retrieved", 2, query="firm size firm performance", new_passages=[passage]))
        if cited:
            lines.append(
                event("judged", 9, confound="Organization size", target="Y", verdict=verdict, passage=passage)
            )
    path = tmp_path / "run.audit.jsonl"
    path.write_text("\n".join(lines) + "\n")
    return path


class Judge:
    def __init__(self, match: str | None = "Organization size", claim: bool | None = True) -> None:
        self.match, self.claim = match, claim

    def same_construct(self, u: str, confounds: dict) -> str | None:
        return self.match

    def claim_in_abstract(self, u: str, v: str, abstract: str) -> bool | None:
        return self.claim


ROW = {"u": "Firm size", "v": "Firm performance", "doi": "10.1/x", "source_paper_title": KATHURIA}
PAPER = trace_recall.GoldPaper("10.1/x", SIZE_PAPER, "Firm size raises profitability.", "openalex")


def traced(tmp_path: Path, judge: Judge | None = None, paper=PAPER, row=ROW, **run) -> object:
    return trace_recall.trace_row(
        row, trace_recall.Run.load(audit(tmp_path, **run)), paper, judge or Judge(), "kathuria2018"
    )


def test_a_paper_on_the_arrow_is_found_whatever_the_case_of_its_title(tmp_path):
    t = traced(tmp_path)
    assert (t.stage, t.retrieved_by, t.cited_for) == (
        "ok",
        "firm size firm performance",
        ["Organization size -> Y"],
    )
    assert (t.queries_for_confound, t.link_supported_passages) == (1, 1)  # only the query using its phrasing


@pytest.mark.parametrize(
    ("judge", "run", "paper", "row", "expected"),
    [
        (Judge(match=None), {}, PAPER, ROW, "1"),  # no confound of the run is the same construct
        (Judge(), {"retrieved": False}, PAPER, ROW, "3"),
        (Judge(claim=False), {"retrieved": False}, PAPER, ROW, "0"),  # the abstract does not carry the claim
        (Judge(), {"retrieved": False}, None, {**ROW, "doi": None}, "0"),  # no DOI to look for
        (Judge(), {"cited": False}, PAPER, ROW, "4"),
        (Judge(), {"verdict": "rejected"}, PAPER, ROW, "5"),
    ],
)
def test_each_row_is_labelled_with_the_first_stage_that_lost_it(tmp_path, judge, run, paper, row, expected):
    assert traced(tmp_path, judge, paper, row, **run).stage == expected


def test_a_confound_never_searched_for_is_lost_at_stage_two(tmp_path):
    run = trace_recall.Run.load(audit(tmp_path))
    run.confounds["Organization size"] = ["head count"]  # no query uses this phrasing
    assert trace_recall.trace_row(ROW, run, PAPER, Judge()).stage == "2"


def test_the_oracle_reports_the_rank_of_the_gold_paper_for_the_gold_words(tmp_path):
    class Hit:
        def __init__(self, title: str) -> None:
            self.title = title

    t = traced(tmp_path, retrieved=False)
    found = trace_recall.oracle(t, lambda query, k: [Hit("other"), Hit(SIZE_PAPER)][:k], k=3)
    assert found == {
        "query": "Firm size Firm performance",
        "rank_in_top_20": 2,
        "within_run_depth": True,
        "indexed": True,
    }
    missing = trace_recall.oracle(t, lambda query, k: [Hit("other")], k=3)
    assert (missing["rank_in_top_20"], missing["within_run_depth"], missing["indexed"]) == (
        None,
        False,
        False,
    )


def test_saved_judgments_are_not_asked_again(tmp_path):
    asked = []

    def ask(prompt: str) -> str:
        asked.append(prompt)
        return 'Sure: {"match": "Organization size"}'

    judge = trace_recall.Judgments(tmp_path / "j.json", ask)
    confounds = {"Organization size": ["organization size", "firm size"]}
    assert judge.same_construct("Firm size", confounds) == "Organization size"
    again = trace_recall.Judgments(tmp_path / "j.json", None)  # no model at all: answered from the file
    assert again.same_construct("Firm size", confounds) == "Organization size"
    assert len(asked) == 1


def test_the_table_counts_each_stage(tmp_path):
    table = trace_recall.markdown([traced(tmp_path), traced(tmp_path, verdict="rejected")])
    assert "on the arrow: 1, rejected: 1" in table or "rejected: 1, on the arrow: 1" in table


def test_every_gold_row_has_a_question_and_its_v_is_one_side_of_it():
    asked = trace_recall.questions()
    gold = [
        json.loads(line)
        for line in trace_recall.GOLD.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert {row["source_paper_title"] for row in gold} == set(asked)
    for row in gold:
        q = asked[row["source_paper_title"]]
        assert row["v"].casefold() in (q["iv"].casefold(), q["dv"].casefold()), row
    assert len({q["name"] for q in asked.values()}) == len(asked)  # names are file names: one each
