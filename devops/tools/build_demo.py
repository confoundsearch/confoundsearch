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
from pathlib import Path

from confoundsearch.data.cache import PaperDatabase, database_path, wait_for_writes
from confoundsearch.data.corpus import Corpus
from confoundsearch.data.demo import DEMO_FILE, DEMO_QUESTION, RECORDED_MODEL
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


def record_demo(work: Path, passes: int = 5) -> bool:
    """Record the example run's model answers in `work`, and prove they replay with no model.

    The run is repeated until a pass finds everything already saved, so the prompts it saved match the papers
    as they now stand (a later search can improve a paper's abstract, and a prompt carries the abstract).
    Then it is run once more with `--replay`, which has no model to ask and fails on any answer missing.
    """
    from confoundsearch.interface import cli

    iv, dv = DEMO_QUESTION

    def run(name: str, *extra: str) -> tuple[int, Path]:
        log = work / f"{name}.log.jsonl"
        argv = [
            "run",
            "--corpus",
            "semanticscholar,openalex",
            "--model",
            RECORDED_MODEL,
            "--iv",
            iv,
            "--dv",
            dv,
        ]
        argv += ["--confounds", "4", "--cache-dir", str(work), "--max-budget", "0.5", *extra]
        return cli.main([*argv, "--out", str(work / f"{name}.html"), "--log", str(log)]), log

    for number in range(1, passes + 1):
        code, log = run(f"record-{number}")
        wait_for_writes()
        print(f"  recording pass {number}: {fresh_work(log)} new", file=sys.stderr)
        if code == 0 and fresh_work(log) == 0:
            break
    else:
        return False
    code, _ = run("replay", "--replay")
    return code == 0


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
