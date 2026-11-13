"""Where does each known confound link get lost? A per-row trace of the MDP against existingconfounds.jsonl.

The goal is not a recall score but a list of fixes for the operations. Each gold row (u affects v, with the
paper that says so) is followed through a real run and labelled with the first stage where it was lost:

    0 unreachable     no DOI, or the paper's abstract (all the system sees) does not carry the claim
    1 not proposed    no confound the run asked for is the same construct as u
    2 not searched    no search used any phrasing of that confound
    3 not retrieved   searches ran, but the paper never came back
    4 not cited       retrieved, but the synthesis did not cite it for this link
    5 rejected        cited, but the judge rejected it
    ok                on the arrow

Stage 3 gets an oracle search: the same sources, asked "u v" in the gold's own words. If that finds the
paper, the run's wording is what missed it; if not, the paper is beyond plain search of these sources.

    python evals/trace_recall.py                # run both questions (saved; reruns are free), then trace
    python evals/trace_recall.py --trace-only   # trace the saved runs again
    python evals/trace_recall.py --only lai2016 # one question (repeatable)

The questions (each gold paper's X and Y) are in existingquestions.jsonl.

Runs, audit logs and the search and model caches go in evals/recall/runs/ (not tracked: they hold paper
abstracts and full prompts). The trace goes to evals/recall/trace.md and trace.jsonl.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

HERE = Path(__file__).resolve().parent
GOLD = HERE / "existingconfounds.jsonl"
OUT = HERE / "recall"
RUNS = OUT / "runs"

QUESTIONS = HERE / "existingquestions.jsonl"  # each gold paper's own question: name, title, iv, dv


def questions(path: Path = QUESTIONS) -> dict[str, dict]:
    """Source paper title -> its question: {name, iv, dv, ...}."""
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {row["source_paper_title"]: row for row in rows}


STAGES = {
    "0": "unreachable",
    "1": "not proposed",
    "2": "not searched",
    "3": "not retrieved",
    "4": "not cited",
    "5": "rejected",
    "ok": "on the arrow",
}
SYNTHESIS_TEXT = 900  # characters of each paper the synthesis reads (LLMSynthesis._messages)

Ask = Callable[[str], str]  # a prompt -> the model's reply


def title_key(title: str) -> str:
    from confoundsearch.data.multi import title_key as key

    return key(title)


# -- the gold papers -----------------------------------------------------------------------------------


@dataclass
class GoldPaper:
    doi: str
    title: str = ""
    abstract: str = ""
    found_by: str = ""  # which service gave the title and abstract


def _get_json(url: str, headers: dict[str, str] | None = None) -> Any:
    request = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310 (fixed https hosts)
        return json.load(response)


def resolve(doi: str) -> GoldPaper:
    """The title and abstract for a DOI: OpenAlex first, then Semantic Scholar for a missing abstract."""
    from confoundsearch.data.sources import abstract_from_inverted_index

    paper = GoldPaper(doi)
    query = {"select": "title,abstract_inverted_index"}
    if os.environ.get("OPENALEX_API_KEY"):
        query["api_key"] = os.environ["OPENALEX_API_KEY"]
    try:
        work = _get_json(f"https://api.openalex.org/works/doi:{doi}?{urllib.parse.urlencode(query)}")
        paper.title = work.get("title") or ""
        if work.get("abstract_inverted_index"):
            paper.abstract, paper.found_by = (
                abstract_from_inverted_index(work["abstract_inverted_index"]),
                "openalex",
            )
    except OSError:
        pass
    if not paper.abstract:
        headers = {"x-api-key": os.environ["S2_KEY"]} if os.environ.get("S2_KEY") else {}
        try:
            s2 = _get_json(
                f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}?fields=title,abstract,tldr",
                headers,
            )
            paper.title = paper.title or s2.get("title") or ""
            text = s2.get("abstract") or (s2.get("tldr") or {}).get("text") or ""
            if text:
                paper.abstract, paper.found_by = text, "semanticscholar"
        except OSError:
            pass
    return paper


# -- a run, read back from its audit log ------------------------------------------------------------------


@dataclass
class Run:
    x: str
    y: str
    confounds: dict[str, list[str]] = field(default_factory=dict)  # label -> its phrasings
    queries: list[str] = field(default_factory=list)
    retrieved: dict[str, dict] = field(default_factory=dict)  # title key -> {passage, query, step}
    judged: list[dict] = field(default_factory=list)  # {confound, target, verdict, key}

    @classmethod
    def load(cls, path: Path) -> Run:
        events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        start = next(e for e in events if e["kind"] == "run_started")["data"]
        run = cls(start["independent_variable"], start["dependent_variable"])
        for event in events:
            data = event["data"]
            if event["kind"] == "action" and data["action"].get("type") == "AddConfound":
                run.confounds[data["action"]["label"]] = sorted(data["action"]["strings"])
            elif event["kind"] == "action" and data["action"].get("type") == "Retrieve":
                run.queries.append(data["action"]["query"])
            elif event["kind"] == "retrieved":
                for passage in data["new_passages"]:
                    entry = {"passage": passage, "query": data["query"], "step": event["step"]}
                    run.retrieved.setdefault(title_key(passage.get("title") or ""), entry)
            elif event["kind"] == "judged":
                run.judged.append(
                    {
                        "confound": data["confound"],
                        "target": data["target"],
                        "verdict": data["verdict"],
                        "key": title_key(data["passage"].get("title") or ""),
                    }
                )
        return run

    def side(self, v: str) -> str:
        return "X" if v.casefold() == self.x.casefold() else "Y"

    def searched_with(self, confound: str) -> list[str]:
        """The queries that use any phrasing of this confound."""
        phrasings = [p.casefold() for p in self.confounds.get(confound, [])]
        return [q for q in self.queries if any(p in q.casefold() for p in phrasings)]

    def link_support(self, confound: str, side: str) -> int:
        return sum(
            1
            for j in self.judged
            if (j["confound"], j["target"], j["verdict"]) == (confound, side, "supported")
        )


# -- the model's two judgments, saved so a rerun asks nothing twice ---------------------------------------


class Judgments:
    def __init__(self, path: Path, ask: Ask | None) -> None:
        self.path, self.ask = path, ask
        self.saved: dict[str, Any] = json.loads(path.read_text()) if path.exists() else {}

    def _asked(self, key: str, prompt: str) -> Any:
        if key not in self.saved:
            if self.ask is None:
                return None
            reply = self.ask(prompt)
            start, end = reply.find("{"), reply.rfind("}")
            try:
                self.saved[key] = json.loads(reply[start : end + 1])
            except ValueError:
                self.saved[key] = {}
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.saved, indent=1, sort_keys=True))
        return self.saved[key]

    def same_construct(self, u: str, confounds: dict[str, list[str]]) -> str | None:
        """The run's confound that names the same construct as u, if any."""
        listed = "\n".join(
            f"- {label} (also: {', '.join(p for p in ps if p != label.casefold())})"
            for label, ps in confounds.items()
        )
        prompt = (
            f'A known confound in the literature is "{u}". '
            f"A search tool proposed these confounds:\n{listed}\n\n"
            f'Which one, if any, is the same construct as "{u}" '
            "(the same idea, possibly in other words or as "
            "a direct measure of it)? A broader or merely related construct does not count.\n"
            'Reply with JSON only: {"match": "<the label exactly as listed>"} or {"match": null}'
        )
        answer = self._asked(f"match|{u}|{json.dumps(confounds, sort_keys=True)}", prompt) or {}
        match = answer.get("match")
        return match if match in confounds else None

    def claim_in_abstract(self, u: str, v: str, abstract: str) -> bool | None:
        prompt = (
            f'Does this abstract state, report or argue that "{u}" affects or is associated with "{v}" '
            "(in these or other words)? Judge only from the abstract.\n\n"
            f"{abstract}\n\n"
            'Reply with JSON only: {"supports": true or false, "quote": "<the sentence, if any>"}'
        )
        answer = self._asked(f"claim|{u}|{v}|{abstract[:200]}", prompt)
        return None if answer is None else bool(answer.get("supports"))


# -- the trace ----------------------------------------------------------------------------------------------


@dataclass
class Trace:
    question: str
    u: str
    v: str
    doi: str | None
    gold_title: str = ""
    abstract_found: bool = False
    claim_in_abstract: bool | None = None
    matched_confound: str | None = None
    queries_for_confound: int = 0
    retrieved_by: str = ""
    retrieved_at_step: int = 0
    cited_for: list[str] = field(default_factory=list)  # every link the synthesis cited this paper for
    verdict: str = ""
    link_supported_passages: int = 0  # any paper backing the same link: link recall, not paper recall
    oracle: dict[str, Any] = field(default_factory=dict)
    stage: str = ""
    note: str = ""


def trace_row(row: dict, run: Run, paper: GoldPaper | None, judge: Judgments, name: str = "") -> Trace:
    out = Trace(name, row["u"], row["v"], row["doi"])
    side = run.side(row["v"])
    if paper is not None:
        out.gold_title = paper.title
        out.abstract_found = bool(paper.abstract)
        if paper.abstract:
            out.claim_in_abstract = judge.claim_in_abstract(row["u"], row["v"], paper.abstract)
    out.matched_confound = judge.same_construct(row["u"], run.confounds)
    if out.matched_confound:
        out.queries_for_confound = len(run.searched_with(out.matched_confound))
        out.link_supported_passages = run.link_support(out.matched_confound, side)
    key = title_key(out.gold_title) if out.gold_title else None
    hit = run.retrieved.get(key) if key else None
    if hit:
        out.retrieved_by, out.retrieved_at_step = hit["query"], hit["step"]
        out.cited_for = sorted({f"{j['confound']} -> {j['target']}" for j in run.judged if j["key"] == key})
        mine = [
            j
            for j in run.judged
            if j["key"] == key and (j["confound"], j["target"]) == (out.matched_confound, side)
        ]
        out.verdict = mine[0]["verdict"] if mine else ""
        if len(hit["passage"].get("text") or "") > SYNTHESIS_TEXT:
            length = len(hit["passage"]["text"])
            out.note = f"the synthesis read only the first {SYNTHESIS_TEXT} of {length} characters"
    out.stage = stage(out)
    return out


def stage(t: Trace) -> str:
    """The first stage where the row was lost. Stages 1-2 do not need the paper; 3-5 do."""
    if not t.matched_confound:
        return "1"
    if not t.queries_for_confound:
        return "2"
    if not t.doi or not t.gold_title:
        return "0"
    if not t.retrieved_by:
        return "0" if t.claim_in_abstract is False or not t.abstract_found else "3"
    if not t.verdict:
        return "4"
    return "ok" if t.verdict == "supported" else "5"


def oracle(t: Trace, search: Callable[[str, int], Sequence[Any]], k: int) -> dict[str, Any]:
    """Would plain search of the same sources find the paper, asked in the gold's own words?"""
    key = title_key(t.gold_title)
    wide = [title_key(p.title) for p in search(f"{t.u} {t.v}", 20)]
    by_title = [title_key(p.title) for p in search(t.gold_title, 5)]
    rank = wide.index(key) + 1 if key in wide else None
    return {
        "query": f"{t.u} {t.v}",
        "rank_in_top_20": rank,
        "within_run_depth": rank is not None and rank <= k,  # the run takes k results per search
        "indexed": key in by_title,  # a search by its own title finds it
    }


# -- running and writing ------------------------------------------------------------------------------------


def run_question(name: str, x: str, y: str, budget: float) -> Path:
    audit = RUNS / f"{name}.audit.jsonl"
    if not audit.exists():
        RUNS.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [sys.executable, "-m", "confoundsearch", "run", "--iv", x, "--dv", y,
             "--out", str(RUNS / f"{name}.html"), "--audit", str(audit),
             "--max-budget", str(budget), "--no-highlights"],
            check=True,
            env={**os.environ, "CONFOUNDSEARCH_CACHE": str(RUNS / "cache")},
        )  # fmt: skip
    return audit


def markdown(traces: Sequence[Trace]) -> str:
    lines = ["# Where each known confound link is lost", ""]
    counts: dict[str, int] = {}
    for t in traces:
        counts[t.stage] = counts.get(t.stage, 0) + 1
    lines += [", ".join(f"{STAGES[s]}: {counts[s]}" for s in STAGES if s in counts), ""]
    lines += [
        "| run | u → v | stage | run's confound | searches | paper retrieved by | cited for | oracle "
        "| note |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for t in traces:
        o = t.oracle
        oracle_text = (
            ""
            if not o
            else f"rank {o['rank_in_top_20'] or '>20'}; {'indexed' if o['indexed'] else 'not found by title'}"
        )
        lines.append(
            f"| {t.question} | {t.u} → {t.v} | **{STAGES[t.stage]}** | {t.matched_confound or '—'} | "
            f"{t.queries_for_confound} | {t.retrieved_by or '—'} | {'; '.join(t.cited_for) or '—'} | "
            f"{oracle_text} | {t.note} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--trace-only", action="store_true", help="read the saved runs; run nothing")
    parser.add_argument("--budget", type=float, default=3.0, help="USD per run (default 3)")
    parser.add_argument("--no-oracle", action="store_true", help="skip the oracle searches")
    parser.add_argument("--only", action="append", metavar="NAME", help="just this question (repeatable)")
    args = parser.parse_args(argv)

    from confoundsearch.data.demo import RECORDED_MODEL
    from confoundsearch.interface.cli import build_corpus, build_parser
    from confoundsearch.model.llm import OpenAICompatibleClient, temperature_for

    asked = {t: q for t, q in questions().items() if not args.only or q["name"] in args.only}
    rows = [json.loads(line) for line in GOLD.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [r for r in rows if r["source_paper_title"] in asked]
    runs = {}
    for title, q in asked.items():
        audit = RUNS / f"{q['name']}.audit.jsonl"
        if args.trace_only and not audit.exists():
            continue
        runs[title] = Run.load(
            audit if args.trace_only else run_question(q["name"], q["iv"], q["dv"], args.budget)
        )
    rows = [r for r in rows if r["source_paper_title"] in runs]

    papers_file = OUT / "gold_papers.json"
    papers = (
        {d: GoldPaper(**p) for d, p in json.loads(papers_file.read_text()).items()}
        if papers_file.exists()
        else {}
    )
    for doi in {r["doi"] for r in rows if r["doi"]} - set(papers):
        papers[doi] = resolve(doi)
    OUT.mkdir(parents=True, exist_ok=True)
    papers_file.write_text(json.dumps({d: asdict(p) for d, p in sorted(papers.items())}, indent=1))

    client = OpenAICompatibleClient(RECORDED_MODEL, temperature=temperature_for(RECORDED_MODEL))
    judge = Judgments(
        OUT / "judgments.json", lambda prompt: client.chat([{"role": "user", "content": prompt}])
    )
    traces = [
        trace_row(
            r,
            runs[r["source_paper_title"]],
            papers.get(r["doi"] or ""),
            judge,
            asked[r["source_paper_title"]]["name"],
        )
        for r in rows
    ]

    if not args.no_oracle:
        cli = build_parser().parse_args(["run", "--iv", "x", "--dv", "y", "--out", os.devnull,
                                         "--cache-dir", str(RUNS / "cache")])  # fmt: skip
        corpus = build_corpus(cli)
        for t in traces:
            if t.stage == "3":
                t.oracle = oracle(t, corpus.search, k=3)

    (OUT / "trace.jsonl").write_text(
        "".join(json.dumps(asdict(t), ensure_ascii=False) + "\n" for t in traces)
    )
    (OUT / "trace.md").write_text(markdown(traces))
    print(markdown(traces))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
