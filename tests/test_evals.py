"""The pilot's analysis script runs on the pilot's data and gives the numbers we report."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

EVALS = Path(__file__).resolve().parent.parent / "evals" / "pilot"
PILOT = EVALS / "relevance"
LEXICAL = EVALS / "lexical"


def run_iaa() -> list[str]:
    done = subprocess.run(
        [sys.executable, "iaa.py"], cwd=PILOT, capture_output=True, text=True, check=True
    )  # it reads passage_sentence_ratings.jsonl from the folder it is run in
    return done.stdout.splitlines()


def test_the_script_gives_the_appendix_tables_of_agreement():
    """Table 1 (passage level) and Table 2 (sentence level); a star marks fewer than 10 items."""
    assert run_iaa() == [
        "Passage-level kappa:",
        "  rater_1 vs rater_2: 0.36 (n=50)",
        "  rater_1 vs rater_3: 0.33 (n=103)",
        "  rater_2 vs rater_3: 0.14* (n=6)",
        "",
        "Sentence-level kappa (both said supports):",
        "  rater_1 vs rater_2: 0.61 (n=18)",
        "  rater_1 vs rater_3: 0.67 (n=32)",
        "  rater_2 vs rater_3: 0.61* (n=3)",
        "",
        "Same sentence, all pairs: 37/53 = 70%",
        "Mean sentences per passage, those pairs: 7.4",
    ]


def test_the_ratings_are_well_formed():
    rows = [json.loads(line) for line in (PILOT / "passage_sentence_ratings.jsonl").read_text().splitlines()]
    assert len(rows) == 147 and len({r["item_id"] for r in rows}) == 147
    verdicts = {"supports", "contradicts", "no_bearing"}
    raters = ("rater_1", "rater_2", "rater_3")
    counts = {2: 0, 3: 0}
    for row in rows:
        assert row["n_sentences"] > 0
        given = [r for r in raters if row[f"{r}_q1"] is not None]
        counts[len(given)] += 1  # each item was annotated exactly twice, six of them a third time
        for rater in raters:
            verdict, sentence = row[f"{rater}_q1"], row[f"{rater}_q2"]
            assert verdict is None or verdict in verdicts
            # a rater who gave no answer gave no sentence, and one who found no bearing picked none
            assert (sentence is None) == (verdict in (None, "no_bearing"))
    assert counts == {2: 141, 3: 6}


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
