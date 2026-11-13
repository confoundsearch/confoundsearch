"""Does re-ranking mined candidates by "affects X as well", then adding them to the asked list, recall more?

Reads the saved results of proposal_recall.py (evals/recall/proposals.jsonl): no searching. For each
question the model scores every mined candidate (constructs papers say affect Y) for how plausibly it also
affects X, which is what makes it a confound rather than just a cause of Y. Candidates are then ordered by
that score, ties by how many papers named them. Three lists are scored against the gold confounds:

    asked       the pipeline's 8 today
    reranked    the mined list in the new order
    combined    the asked 8, then the reranked list (what a pipeline would search, at a total of N)

    python evals/rerank_recall.py

Writes evals/recall/rerank.md and rerank.jsonl.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Sequence

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from proposal_recall import OUT, RUNS, Ask, Saved, json_in, jsonl, matches, retrying  # noqa: E402

SCORE_BATCH = 100  # candidates per scoring call
SIZES = (8, 12, 16, 20, 28, 40)  # list lengths to compare at


def score_prompt(x: str, y: str, candidates: Sequence[str]) -> str:
    listed = "\n".join(f"{i}. {c}" for i, c in enumerate(candidates))
    return (
        f'A study asks whether "{x}" (X) affects "{y}" (Y). A confound is a construct that affects both X '
        "and Y. Papers report that each construct below affects Y. Score each one for how plausibly it also "
        "affects X, and so confounds the study:\n"
        "3 = very plausible: a standard control or a well-known cause of X\n"
        "2 = plausible\n"
        "1 = possible but a stretch\n"
        "0 = not plausible, or it is a form of X or Y itself, or an outcome of X\n\n"
        f"{listed}\n\n"
        'Reply with JSON lines only, one per construct: {"i": <number>, "score": <0-3>}'
    )


def rerank(ask: Ask, x: str, y: str, mined: Sequence[Sequence]) -> list[str]:
    """The mined candidates, most plausible confounds first; ties keep the papers-named order."""
    names = [name for name, _ in mined]
    scores = [0] * len(names)
    for start in range(0, len(names), SCORE_BATCH):
        batch = names[start : start + SCORE_BATCH]
        for line in ask(score_prompt(x, y, batch)).splitlines():
            item = json_in(line)
            if isinstance(item, dict) and isinstance(item.get("i"), int) and 0 <= item["i"] < len(batch):
                score = item.get("score")
                scores[start + item["i"]] = score if isinstance(score, int) else 0
    order = sorted(range(len(names)), key=lambda i: (-scores[i], i))
    return [names[i] for i in order]


def first(positions: Sequence[int]) -> int | None:
    return min(positions) if positions else None


def main() -> int:
    from confoundsearch.data.demo import RECORDED_MODEL
    from confoundsearch.model.llm import OpenAICompatibleClient, temperature_for

    client = OpenAICompatibleClient(RECORDED_MODEL, temperature=temperature_for(RECORDED_MODEL), timeout=180)
    ask = Saved(
        RUNS / "proposal_replies.json", retrying(lambda p: client.chat([{"role": "user", "content": p}]))
    )
    rows = []
    for r in jsonl(OUT / "proposals.jsonl"):
        names = [name for name, _ in r["mined"]]
        order = rerank(ask, r["iv"], r["dv"], r["mined"])
        position = {name: i for i, name in enumerate(order)}
        asked_keys = {" ".join(a.casefold().split()) for a in r["asked"]}
        extra = [name for name in order if name not in asked_keys]  # what combining adds after the 8
        extra_position = {name: i for i, name in enumerate(extra)}
        for g in r["gold"]:
            hits = matches(ask, g["u"], names)  # the same question as before: answered from saved replies
            rows.append(
                {
                    "name": r["name"],
                    "u": g["u"],
                    "asked": g["asked_rank"],
                    "mined": first(hits),
                    "reranked": first([position[names[i]] for i in hits]),
                    "extra": first([extra_position[names[i]] for i in hits if names[i] in extra_position]),
                    "top": order[:12],
                }
            )
        print(f"{r['name']}: reranked top 5: {', '.join(order[:5])}", flush=True)

    def at(key: str, size: int) -> int:
        return sum(1 for row in rows if row[key] is not None and row[key] < size)

    def combined(size: int) -> int:
        room = size - 8
        return sum(
            1
            for row in rows
            if row["asked"] is not None or (row["extra"] is not None and row["extra"] < room)
        )

    n = len(rows)
    head = "| list | " + " | ".join(f"{s}" for s in SIZES) + " |"
    lines = [
        "# Proposal recall: re-ranked and combined",
        "",
        f"{n} gold confounds. Each cell: how many are in the first N of that list.",
        "",
        head,
        "|---" * (len(SIZES) + 1) + "|",
        "| asked (only 8 exist) | " + " | ".join(str(at("asked", s)) for s in SIZES) + " |",
        "| mined, by papers naming it | " + " | ".join(str(at("mined", s)) for s in SIZES) + " |",
        "| mined, re-ranked by affects X | " + " | ".join(str(at("reranked", s)) for s in SIZES) + " |",
        "| combined: asked 8 + re-ranked | " + " | ".join(str(combined(s)) for s in SIZES) + " |",
        "",
        "## Each gold confound",
        "",
        "| question | gold | asked | mined rank | re-ranked rank |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        rank = lambda v: "—" if v is None else f"#{v + 1}"  # noqa: E731
        lines.append(
            f"| {row['name']} | {row['u']} | {'yes' if row['asked'] is not None else 'no'} | "
            f"{rank(row['mined'])} | {rank(row['reranked'])} |"
        )
    text = "\n".join(lines) + "\n"
    (OUT / "rerank.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    (OUT / "rerank.md").write_text(text)
    print(text.split("## Each")[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
