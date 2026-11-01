"""Build the demo database: real papers about organizational structure and firm performance.

    python devops/tools/build_demo.py            # needs network; S2_KEY and OPENALEX_EMAIL make it faster

It searches Semantic Scholar and OpenAlex for the question and for a list of likely confounds, keeps each
paper once (de-duplicated on title, the fuller abstract winning), and writes them in the format of the saved
searches (`confoundsearch.data.cache`) to `confoundsearch/data/demo.sqlite3`, which ships with the package.
Run it again to refresh the file, and commit the result.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import re
import sqlite3
import sys
from pathlib import Path

from confoundsearch.data.cache import PaperDatabase
from confoundsearch.data.corpus import Corpus
from confoundsearch.data.demo import DEMO_FILE, DEMO_QUESTION
from confoundsearch.data.sources import OpenAlexCorpus, SemanticScholarCorpus
from confoundsearch.errors import RemoteError
from confoundsearch.http import resilient, urllib_transport
from confoundsearch.logic.graph import Passage

CONFOUNDS = (
    "firm size",
    "firm age",
    "IT investment",
    "industry",
    "R&D intensity",
    "environmental uncertainty",
    "human capital",
    "financial leverage",
    "diversification",
    "international scope",
    "management quality",
    "organizational culture",
)
FORMS = ("{iv} {c}", "{c} {dv}", "{c} affects {dv}")  # how a search for one confound is worded
PER_SEARCH = 10
EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")


def scrub(text: str) -> str:
    """Abstracts sometimes carry an author's email address; those are not ours to ship."""
    return EMAIL.sub("", text)


def scrubbed(found: list[Passage]) -> list[Passage]:
    return [Passage(p.id, scrub(p.text), scrub(p.title), p.url) for p in found]


def queries(iv: str, dv: str) -> list[str]:
    found = [f"{iv} {dv}", f"{iv} affects {dv}", iv, dv]
    for confound in CONFOUNDS:
        found += [form.format(iv=iv, dv=dv, c=confound) for form in FORMS]
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=DEMO_FILE)
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
    args.out.unlink(missing_ok=True)
    database = PaperDatabase(args.out)
    asked = failed = 0
    for query in queries(iv, dv):
        for name, source in sources.items():
            asked += 1
            try:
                database.save(name, query, PER_SEARCH, scrubbed(source.search(query, PER_SEARCH)))
            except RemoteError as err:
                failed += 1
                print(f"  {name}: {query!r} failed: {err}", file=sys.stderr)
        print(f"  {query}", file=sys.stderr)
    searches, papers = database.counts()
    with contextlib.closing(sqlite3.connect(args.out)) as db:  # one plain file that opens read-only anywhere
        db.execute("PRAGMA journal_mode=DELETE")
        db.execute("VACUUM")
    print(f"{papers} papers from {searches} searches ({failed} of {asked} failed) -> {args.out}")
    return 1 if failed > asked // 4 else 0  # a build with many failures is not worth keeping


if __name__ == "__main__":
    raise SystemExit(main())
