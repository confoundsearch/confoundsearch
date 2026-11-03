"""Build the demo database: saved searches and papers that ship with the package for the example question.

    python devops/tools/build_demo.py                          # needs network; S2_KEY makes it faster
    python devops/tools/build_demo.py --models gpt-4.1-mini    # also record real runs (needs a model key)

On the example question (organizational structure and firm performance) a default run makes searches of the
form "<confound> <variable>" for each confound the language model names, three papers each. This tool saves
those searches in advance for a long list of likely confounds, from Semantic Scholar and OpenAlex, in the
format of the data directory (`confoundsearch.data.cache`), plus broader searches that give the offline demo
(`confoundsearch demo --offline`) its papers. With `--models` it also runs the real program once per model and
keeps those searches too, rewordings included. The result is `confoundsearch/data/demo.sqlite3`, each paper
once (same title: the fuller abstract wins; author email addresses removed). It is copied into a user's data
directory the first time they run the example question, so those searches answer instantly. Run it again to
refresh, and commit the result.
"""

from __future__ import annotations

import argparse
import contextlib
import itertools
import json
import os
import re
import shutil
import sqlite3
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

from confoundsearch.data.cache import PaperDatabase, call_key, database_path, paper_key, wait_for_writes
from confoundsearch.data.corpus import Corpus
from confoundsearch.data.demo import (
    CURATED_CONFOUNDS,
    CURATED_SEARCH,
    DEMO_FILE,
    DEMO_QUESTION,
    RECORDED_MODEL,
    DemoCorpus,
)
from confoundsearch.data.multi import title_key
from confoundsearch.data.sources import OpenAlexCorpus, SemanticScholarCorpus
from confoundsearch.errors import RemoteError
from confoundsearch.http import resilient, urllib_transport
from confoundsearch.logic.graph import Passage

CONFOUNDS = (
    "firm size",
    "firm age",
    "industry",
    "technology",
    "IT investment",
    "R&D intensity",
    "environmental uncertainty",
    "market competition",
    "human capital",
    "financial leverage",
    "diversification",
    "international scope",
    "management quality",
    "organizational culture",
    "leadership",
    "strategy",
    "ownership structure",
    "corporate governance",
    "firm resources",
    "organizational slack",
    "innovation",
    "market conditions",
    "regulation",
    "geographic location",
    "capital intensity",
    "firm growth",
    "employee size",
    "top management team",
    "business cycle",
    "customer base",
    "supply chain",
    "organizational complexity",
    "decentralization",
    "formalization",
    "ownership type",
    "profitability",
    "firm characteristics",
    "industry type",
    "economic environment",
    "organizational size",
)
KNOWN_STUDIES = (  # well-known published studies for the three curated confounds (see CURATED_CONFOUNDS)
    "Who Lives in the C-Suite? Organizational Structure and the Division of Labor in Top Management",
    "Delegation of Authority In Business Organizations: An Empirical Test",
    "Information Technology, Workplace Organization, and the Demand for Skilled Labor: Firm-Level Evidence",
    "Americans Do IT Better: US Multinationals and the Productivity Miracle",
    "Managing with Style: The Effect of Managers on Firm Policies",
    "The Spread of the Multidivisional Form Among Large Firms, 1919-1979",
    "The Flattening Firm and Product Market Competition: "
    "The Effect of Trade Liberalization on Corporate Hierarchies",
    "The Organization of Firms Across Countries",
    "Competition and Corporate Performance",
    "Trade Liberalization, Exit, and Productivity Improvements: Evidence from Chilean Plants",
)
DISTRACTOR_QUERIES = (  # topical searches whose top papers pad the example set, so retrieval is not trivial
    "organizational structure firm performance",
    "decentralization of decision making",
    "firm size and growth",
    "industry effects on profitability",
    "management quality",
    "innovation and organizational design",
    "organizational culture",
    "strategy and structure",
    "corporate governance",
    "human capital productivity",
)
FORMS = (
    "{iv} {c}",
    "{c} {dv}",
    "{c} affects {dv}",
)  # broader searches: they give the offline corpus its papers
RESULTS = 3  # what a default run asks for per search (DeterministicPolicy.results_per_query)
BROAD = 10
EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")


def scrub(text: str) -> str:
    """Abstracts sometimes carry an author's email address; those are not ours to ship."""
    return EMAIL.sub("", text)


def scrubbed(found: list[Passage]) -> list[Passage]:
    return [Passage(p.id, scrub(p.text), scrub(p.title), p.url) for p in found]


def exact_queries(iv: str, dv: str) -> list[str]:
    """The searches a default run makes first for each confound: the confound and a variable, as written."""
    return [f"{c} {variable}" for c in CONFOUNDS for variable in (iv, dv)]


def broad_queries(iv: str, dv: str) -> list[str]:
    found = [f"{iv} {dv}", f"{iv} affects {dv}", iv, dv]
    for confound in CONFOUNDS[:12]:
        found += [form.format(iv=iv, dv=dv, c=confound) for form in FORMS]
    return found


def fresh_work(log: Path) -> int:
    """How many searches and model calls a run had to make afresh, from its log: 0 means all were saved."""
    events = (json.loads(line)["event"] for line in log.read_text(encoding="utf-8").splitlines())
    return sum(1 for event in events if event in ("cache_miss", "model_cache_miss"))


def save_curated_confounds(work: Path) -> None:
    """Save the model's answer to "which confounds?" as the curated three, whatever the model would say.

    This is the curation: the key is made from the very prompt a run sends, so the run finds this answer.
    """
    from confoundsearch.model.llm import LLMConfounds, temperature_for

    sent: list[list[dict[str, str]]] = []

    class Ask:
        model, temperature = RECORDED_MODEL, temperature_for(RECORDED_MODEL)

        def chat(self, messages: Sequence[Mapping[str, str]]) -> str:
            sent.append([dict(m) for m in messages])
            return json.dumps({"confounds": list(CURATED_CONFOUNDS)})

    iv, dv = DEMO_QUESTION
    LLMConfounds(Ask(), len(CURATED_CONFOUNDS))(iv, dv)
    reply = json.dumps({"confounds": list(CURATED_CONFOUNDS)})
    PaperDatabase(database_path(work, DEMO_QUESTION)).save_call(
        call_key(RECORDED_MODEL, temperature_for(RECORDED_MODEL), sent[0]), RECORDED_MODEL, sent[0], reply
    )


def record_demo(work: Path, passes: int = 5, curated: bool = False) -> bool:
    """Record the example run's model answers in `work`, and prove they replay with no model.

    The run is repeated until a pass finds everything already saved, so the prompts it saved match the papers
    as they now stand (a later search can improve a paper's abstract, and a prompt carries the abstract).
    Then it is run once more with `--replay`, which has no model to ask and fails on any answer missing.
    With `curated`, the run searches the example set in `work` and asks about the three curated confounds.
    """
    from confoundsearch.interface import cli

    iv, dv = DEMO_QUESTION
    db_path = database_path(work, DEMO_QUESTION)
    corpus = ["--corpus", "demo"] if curated else ["--corpus", "semanticscholar,openalex"]
    confounds = len(CURATED_CONFOUNDS) if curated else 4
    if curated:
        save_curated_confounds(work)

    def run(name: str, *extra: str) -> tuple[int, Path]:
        log = work / f"{name}.log.jsonl"
        argv = ["run", *corpus, "--model", RECORDED_MODEL, "--iv", iv, "--dv", dv]
        argv += [*(CURATED_SEARCH if curated else ()), "--confounds", str(confounds)]
        argv += ["--cache-dir", str(work), "--max-budget", "0.5", *extra]
        return cli.main([*argv, "--out", str(work / f"{name}.html"), "--log", str(log)]), log

    real = getattr(cli, "DemoCorpus")  # noqa: B009
    setattr(cli, "DemoCorpus", lambda: DemoCorpus(db_path))  # noqa: B010  (read the file being built)
    try:
        for number in range(1, passes + 1):
            code, log = run(f"record-{number}")
            wait_for_writes()
            print(f"  recording pass {number}: {fresh_work(log)} new", file=sys.stderr)
            if code == 0 and fresh_work(log) == 0:
                break
        else:
            return False
        code, _ = run("replay", "--replay")
    finally:
        setattr(cli, "DemoCorpus", real)  # noqa: B010
    return code == 0


def curate(db_path: Path, source: Corpus, distractors: int = 30) -> int:
    """Put the known studies and some distractors into the example set the curated demo searches.

    The studies' abstracts come from `source` (OpenAlex holds them in full). The distractors are the top
    papers of a few topical searches over everything already saved, so the demo's search still has to choose.
    Returns the size of the example set.
    """
    found: list[Passage] = []
    for title in KNOWN_STUDIES:
        want = title_key(title)
        match = [p for p in source.search(title, 10) if title_key(p.title)[:50] == want[:50]]
        if not match:
            raise RemoteError(f"could not find the known study {title!r}")
        found.append(scrubbed(match[:1])[0])
    PaperDatabase(db_path).save("curated", "well-known studies for the curated example", len(found), found)
    known = {paper_key(p, "curated") for p in found}
    everything = DemoCorpus(db_path)  # no example set yet: this searches every saved paper
    padding: dict[str, None] = {}
    for query in DISTRACTOR_QUERIES:
        for p in everything.search(query, 6):
            if paper_key(p, "curated") not in known:
                padding[paper_key(p, "curated")] = None
    keys = sorted(known | set(list(padding)[:distractors]))
    with contextlib.closing(sqlite3.connect(db_path)) as db:
        db.execute("CREATE TABLE IF NOT EXISTS example_papers (key TEXT PRIMARY KEY)")
        db.execute("DELETE FROM example_papers")
        db.executemany("INSERT INTO example_papers VALUES (?)", [(k,) for k in keys])
        db.commit()
    return len(keys)


def clean(path: Path) -> tuple[int, int]:
    """Remove emails left in what runs saved, and make one plain file that opens read-only anywhere.

    Returns (searches, papers). It counts with a plain connection: opening the file the way the cache does
    would switch it back to write-ahead mode, which is not what should ship.
    """
    with contextlib.closing(sqlite3.connect(path)) as db:
        for key, text, title in db.execute("SELECT key, text, title FROM papers").fetchall():
            if EMAIL.search(text) or EMAIL.search(title):
                db.execute("UPDATE papers SET text=?, title=? WHERE key=?", (scrub(text), scrub(title), key))
        db.commit()
        db.execute("PRAGMA journal_mode=DELETE")
        db.execute("VACUUM")
        return (
            db.execute("SELECT COUNT(*) FROM searches").fetchone()[0],
            db.execute("SELECT COUNT(*) FROM papers").fetchone()[0],
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=DEMO_FILE)
    parser.add_argument("--models", nargs="*", default=[], help="also record a real run with each of these")
    parser.add_argument(
        "--reuse", type=Path, help="continue from a folder an earlier, interrupted build left"
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="also record the model's answers for the example run (needs a key for the default model), "
        "so `confoundsearch demo` replays it with no key",
    )
    parser.add_argument(
        "--curate",
        action="store_true",
        help="build the curated example (known studies, three named confounds) and record its run "
        "(needs a key for the default model)",
    )
    parser.add_argument(
        "--confounds",
        nargs="*",
        type=int,
        default=[8, 4],
        help="with --models: record runs asking for this many confounds (`run` asks for 8, `demo` for 4)",
    )
    args = parser.parse_args(argv)
    iv, dv = DEMO_QUESTION
    sources: dict[str, Corpus] = {
        "semanticscholar": SemanticScholarCorpus(
            os.environ.get("S2_KEY"), transport=resilient(urllib_transport, min_interval=1.1)
        ),
        "openalex": OpenAlexCorpus(
            os.environ.get("OPENALEX_API_KEY"),
            os.environ.get("OPENALEX_EMAIL"),
            transport=resilient(urllib_transport, min_interval=0.2),
        ),
    }
    work = args.reuse or Path(tempfile.mkdtemp(prefix="confoundsearch-demo-"))
    database = PaperDatabase(database_path(work, DEMO_QUESTION))
    asked = failed = 0
    plan = [*((q, RESULTS) for q in exact_queries(iv, dv)), *((q, BROAD) for q in broad_queries(iv, dv))]
    for query, k in [] if args.reuse else plan:  # a reused folder already holds these searches
        for name, source in sources.items():
            asked += 1
            try:
                database.save(name, query, k, scrubbed(source.search(query, k)))
            except RemoteError as err:
                failed += 1
                print(f"  {name}: {query!r} failed: {err}", file=sys.stderr)
        print(f"  {query}", file=sys.stderr)
    if failed > asked // 4:  # a build with many failures is not worth keeping
        print(f"{failed} of {asked} searches failed; nothing written", file=sys.stderr)
        shutil.rmtree(work, ignore_errors=True)
        return 1
    if args.models:
        from confoundsearch.interface import cli

        for number, (model, confounds) in enumerate(itertools.product(args.models, args.confounds), 1):
            print(f"  real run {number}: {model}, {confounds} confounds", file=sys.stderr)
            cli.main(
                [
                    *["run", "--corpus", "semanticscholar,openalex", "--model", model],
                    *["--iv", iv, "--dv", dv, "--cache-dir", str(work), "--max-budget", "0.5"],
                    *["--confounds", str(confounds), "--out", str(work / f"run-{number}.html")],
                ]
            )
        wait_for_writes()
    if args.curate:
        print(f"  curated example set: {curate(database.path, sources['openalex'])} papers", file=sys.stderr)
        if not record_demo(work, curated=True):
            print("the curated run does not replay on its own; nothing written", file=sys.stderr)
            shutil.rmtree(work, ignore_errors=True)
            return 1
    if args.record and not record_demo(work):
        print("the recorded run does not replay on its own; nothing written", file=sys.stderr)
        shutil.rmtree(work, ignore_errors=True)
        return 1
    shutil.copyfile(database.path, args.out)
    searches, papers = clean(args.out)
    shutil.rmtree(work, ignore_errors=True)
    print(f"{papers} papers, {searches} saved searches ({failed} of {asked} failed) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
