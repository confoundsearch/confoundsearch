"""Which way of proposing confounds recalls the known ones? Proposal only: no full runs.

Stage 1 of the recall trace (no confound of the run is the gold construct) is where most gold links are lost.
This compares two proposers on every question in existingquestions.jsonl:

    asked   the pipeline's proposer today: the model is asked for the confounds of X and Y (LLMConfounds)
    mined   search deep for Y itself (its name and a couple of synonyms), then have the model read each
            abstract and pull out every construct the paper says affects Y (controls and demographics too).
            Candidates are ranked by how many papers name them.

A gold confound counts as recalled if the same-construct judge finds it in the list. Since the mined list is
long, it is scored at several depths (top 8, 20, 50, all), so a longer list does not win just by being long.

    python evals/proposal_recall.py               # all questions (searches and model replies are saved)
    python evals/proposal_recall.py --only lai2016

Writes evals/recall/proposals.md and proposals.jsonl.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Sequence

HERE = Path(__file__).resolve().parent
GOLD = HERE / "existingconfounds.jsonl"
QUESTIONS = HERE / "existingquestions.jsonl"
OUT = HERE / "recall"
RUNS = OUT / "runs"
DEPTHS = (8, 20, 50, None)  # None: the whole list
BATCH = 30  # abstracts per extraction call

Ask = Callable[[str], str]


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def json_in(reply: str, opening: str = "{") -> Any:
    closing = "}" if opening == "{" else "]"
    start, end = reply.find(opening), reply.rfind(closing)
    try:
        return json.loads(reply[start : end + 1])
    except ValueError:
        return None


def retrying(ask: Ask, tries: int = 4, wait: float = 10.0) -> Ask:
    """A dropped connection to the model is retried (10 s, 20 s, 40 s) rather than ending a long run."""

    def asking(prompt: str) -> str:
        from confoundsearch.errors import RemoteError

        for attempt in range(tries):
            try:
                return ask(prompt)
            except RemoteError:
                if attempt == tries - 1:
                    raise
                time.sleep(wait * 2**attempt)
        raise AssertionError("unreachable")

    return asking


class Saved:
    """Model replies kept on disk by prompt, so a rerun asks nothing twice."""

    def __init__(self, path: Path, ask: Ask) -> None:
        self.path, self.ask = path, ask
        self.replies: dict[str, str] = json.loads(path.read_text()) if path.exists() else {}

    def __call__(self, prompt: str) -> str:
        key = hashlib.sha256(prompt.encode()).hexdigest()
        if key not in self.replies:
            self.replies[key] = self.ask(prompt)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.replies))
        return self.replies[key]


# -- the mined proposer ------------------------------------------------------------------------------------


def extraction_prompt(y: str, papers: Sequence[Any]) -> str:
    lines = [
        f'The outcome of interest is "{y}". Below are papers (an id, a title and an abstract).',
        f'For each paper, list every construct that the paper reports, tests or argues affects "{y}" (or a',
        "close form or measure of it): predictors, antecedents, determinants, moderators, and also control",
        "variables or respondent characteristics (such as age, gender, tenure, firm size, industry) when the",
        "text says they matter for the outcome. Each construct is a short noun phrase of 1 to 4 words, as the",  # noqa: E501
        "literature names it. Skip papers that are not about this outcome.",
        "",
        "Reply with JSON lines only, one per paper that names any:",
        '{"id": "<paper id>", "constructs": ["...", "..."]}',
        "",
    ]
    for paper in papers:
        lines += [f"id: {paper.id}", f"title: {paper.title}", f"abstract: {paper.text[:1200]}", ""]
    return "\n".join(lines)


def extract(ask: Ask, y: str, papers: Sequence[Any]) -> dict[str, set[str]]:
    """Construct (normalized) -> the ids of the papers that name it as affecting y."""
    found: dict[str, set[str]] = defaultdict(set)
    known = {paper.id for paper in papers}
    for start in range(0, len(papers), BATCH):
        reply = ask(extraction_prompt(y, papers[start : start + BATCH]))
        for line in reply.splitlines():
            item = json_in(line)
            if not isinstance(item, dict) or item.get("id") not in known:
                continue
            for construct in item.get("constructs") or []:
                if isinstance(construct, str) and construct.strip():
                    found[" ".join(construct.casefold().split())].add(item["id"])
    return found


def ranked(found: dict[str, set[str]], exclude: Sequence[str]) -> list[tuple[str, int]]:
    """Most-named first; the question's own variables left out."""
    skip = {" ".join(e.casefold().split()) for e in exclude}
    rows = [(c, len(ids)) for c, ids in found.items() if c not in skip]
    return sorted(rows, key=lambda row: (-row[1], row[0]))


def mine(ask: Ask, search: Callable[[str, int], Sequence[Any]], synonyms: Sequence[str], x: str, y: str,
         depth: int) -> tuple[list[tuple[str, int]], int]:  # fmt: skip
    """The candidates, most-named first, and how many papers were read."""
    papers: dict[str, Any] = {}
    for query in [y, *synonyms]:
        for paper in search(query, depth):
            papers.setdefault(paper.id, paper)
    ordered = sorted(papers.values(), key=lambda p: p.id)  # a fixed order, so saved replies match
    return ranked(extract(ask, y, ordered), [x, y, *synonyms]), len(ordered)


# -- scoring ----------------------------------------------------------------------------------------------


def matches(ask: Ask, u: str, candidates: Sequence[str]) -> list[int]:
    """The positions (0-based) of the candidates that are the same construct as u."""
    if not candidates:
        return []
    listed = "\n".join(f"{i}. {c}" for i, c in enumerate(candidates))
    prompt = (
        f'A known confound in the literature is "{u}". Here is a numbered list of candidate constructs:\n'
        f"{listed}\n\n"
        f'Which candidates are the same construct as "{u}" (the same idea, possibly in other words, or a '
        "direct measure of it)? A broader or merely related construct does not count.\n"
        'Reply with JSON only: {"matches": [<numbers>]} (an empty list if none)'
    )
    answer = json_in(ask(prompt)) or {}
    picked = answer.get("matches") if isinstance(answer, dict) else None
    return sorted({n for n in picked or [] if isinstance(n, int) and 0 <= n < len(candidates)})


def recall_at(ranks: Sequence[int | None], depth: int | None) -> int:
    return sum(1 for r in ranks if r is not None and (depth is None or r < depth))


def report(results: Sequence[dict]) -> str:
    gold = [g for r in results for g in r["gold"]]
    n = len(gold)

    def line(label: str, ranks: list[int | None]) -> str:
        cells = [f"{recall_at(ranks, d)}/{n}" for d in DEPTHS]
        return f"| {label} | " + " | ".join(cells) + " |"

    lines = [
        "# Proposal recall: asked vs mined",
        "",
        f"{len(results)} questions, {n} distinct gold confounds (a question's confound counted once).",
        "",
        "| proposer | top 8 | top 20 | top 50 | all |",
        "|---|---|---|---|---|",
        line("asked (today: 8 from the model)", [g["asked_rank"] for g in gold]),
        line("mined (papers on Y, read by the model)", [g["mined_rank"] for g in gold]),
        "",
        "Mean list length: mined "
        f"{sum(len(r['mined']) for r in results) / max(len(results), 1):.0f}; asked "
        f"{sum(len(r['asked']) for r in results) / max(len(results), 1):.0f}.",
        "",
    ]
    for r in results:
        lines += [f"## {r['name']}: {r['iv']} → {r['dv']}", ""]
        lines.append(f"Asked: {', '.join(r['asked'])}")
        top = ", ".join(f"{c} ({k})" for c, k in r["mined"][:20])
        lines += [f"Mined, top 20 of {len(r['mined'])} from {r['papers']} papers: {top}", ""]
        for g in r["gold"]:
            asked = "yes" if g["asked_rank"] is not None else "no"
            mined = (
                f"#{g['mined_rank'] + 1} ({r['mined'][g['mined_rank']][0]})"
                if g["mined_rank"] is not None
                else "no"
            )
            lines.append(f"- **{g['u']}**: asked {asked}; mined {mined}")
        lines.append("")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--only", action="append", metavar="NAME", help="just this question (repeatable)")
    parser.add_argument("--depth", type=int, default=50, help="papers per search on Y (default 50)")
    parser.add_argument("--synonyms", type=int, default=2, help="synonyms of Y also searched (default 2)")
    args = parser.parse_args(argv)

    from confoundsearch.data.demo import RECORDED_MODEL
    from confoundsearch.interface.cli import build_corpus, build_parser
    from confoundsearch.model.llm import LLMConfounds, LLMSynonyms, OpenAICompatibleClient, temperature_for

    client = OpenAICompatibleClient(RECORDED_MODEL, temperature=temperature_for(RECORDED_MODEL), timeout=180)
    ask = Saved(
        RUNS / "proposal_replies.json", retrying(lambda p: client.chat([{"role": "user", "content": p}]))
    )

    class Replying:  # the pipeline's own prompts, answered through the same saved replies
        def chat(self, messages: Sequence[dict]) -> str:
            return ask(messages[-1]["content"])

    cli = build_parser().parse_args(
        ["run", "--iv", "x", "--dv", "y", "--out", os.devnull, "--cache-dir", str(RUNS / "cache")]
    )
    corpus = build_corpus(cli)
    gold = jsonl(GOLD)
    done_file = (
        RUNS / f"proposals.{args.depth}.{args.synonyms}.partial.jsonl"
    )  # one line per finished question
    done = {r["name"]: r for r in jsonl(done_file)} if done_file.exists() else {}
    results = []
    for q in jsonl(QUESTIONS):
        if args.only and q["name"] not in args.only:
            continue
        if q["name"] in done:  # finished before a restart: kept, not asked again
            results.append(done[q["name"]])
            continue
        x, y = q["iv"], q["dv"]
        asked = LLMConfounds(Replying(), 8)(x, y)
        synonyms = LLMSynonyms(Replying(), 3)(y)[: args.synonyms]
        mined, papers = mine(ask, corpus.search, synonyms, x, y, args.depth)
        names = [c for c, _ in mined]
        us = list(dict.fromkeys(g["u"] for g in gold if g["source_paper_title"] == q["source_paper_title"]))
        scored = []
        for u in us:
            in_asked = matches(ask, u, asked)
            in_mined = matches(ask, u, names)
            scored.append(
                {
                    "u": u,
                    "asked_rank": in_asked[0] if in_asked else None,
                    "mined_rank": in_mined[0] if in_mined else None,
                }
            )
        results.append(
            {**q, "asked": asked, "synonyms": synonyms, "mined": mined, "papers": papers, "gold": scored}
        )
        done_file.parent.mkdir(parents=True, exist_ok=True)
        with done_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(results[-1], ensure_ascii=False) + "\n")
        print(f"{q['name']}: asked {sum(g['asked_rank'] is not None for g in scored)}/{len(us)}, "
              f"mined {sum(g['mined_rank'] is not None for g in scored)}/{len(us)} "
              f"({len(mined)} candidates from {papers} papers)", flush=True)  # fmt: skip

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "proposals.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in results))
    (OUT / "proposals.md").write_text(report(results))
    print(report(results).split("\n## ")[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
