"""The pilot's analysis script runs on the pilot's data and gives the numbers we report."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

EVALS = Path(__file__).resolve().parent.parent / "evals" / "pilot"
PILOT = EVALS / "relevance"
LEXICAL = EVALS / "lexical"


def run_kappa() -> list[str]:
    done = subprocess.run(
        [sys.executable, "kappa.py"], cwd=PILOT, capture_output=True, text=True, check=True
    )  # it reads passage_annotation.jsonl from the folder it is run in
    return done.stdout.splitlines()


def test_the_script_gives_the_pilots_agreement_numbers():
    assert run_kappa() == [
        "passage-level kappa: 0.48",
        "p(same sentence): 65%",
        "mean sentences/passage: 10.0",
        "sentence-level kappa: 0.61",
    ]


def test_the_annotations_are_well_formed():
    rows = [json.loads(line) for line in (PILOT / "passage_annotation.jsonl").read_text().splitlines()]
    assert len(rows) == 31 and len({r["item_id"] for r in rows}) == 31
    verdicts = {"supports", "contradicts", "no_bearing"}
    for row in rows:
        assert {row["rater_1"], row["rater_2"]} <= verdicts and row["n_sentences"] > 0
        for rater, sentence in (("rater_1", "sentence_1"), ("rater_2", "sentence_2")):
            picked = row[sentence]
            # a rater picks a sentence exactly when they say the passage supports the claim, and it exists
            assert (picked is not None) == (row[rater] == "supports")
            assert picked is None or 0 <= picked < row["n_sentences"]


def run_lexical() -> list[str]:
    done = subprocess.run(
        [sys.executable, "kappa_lex.py"], cwd=LEXICAL, capture_output=True, text=True, check=True
    )  # it reads pairs.jsonl from the folder it is run in
    return done.stdout.splitlines()


def test_the_lexical_script_gives_the_agreement_between_humans_and_the_two_models():
    assert run_lexical() == [
        "human-human kappa: 0.68",
        "human-Qwen kappa: 0.30",
        "human-Astra kappa: 0.43",
    ]


def test_the_lexical_pairs_are_well_formed():
    rows = [json.loads(line) for line in (LEXICAL / "pairs.jsonl").read_text().splitlines()]
    relations = {"same", "x_kind_of_y", "y_kind_of_x", "none"}
    assert len(rows) == 25
    for row in rows:
        assert row["x"].strip() and row["y"].strip()
        assert {row["rater_1"], row["rater_2"], row["qwen"], row["astra"]} <= relations
