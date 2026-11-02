"""A small set of real papers that ships with the package, for trying it out with no network search.

The papers are about one question, organizational structure and firm performance, with the confounds people
usually name for it (firm size and age, IT investment, industry, and so on). `DemoCorpus` searches them in
memory, so a demo needs nothing but a language model:

    confoundsearch run --corpus demo --iv "Organizational structure" --dv "Firm performance" ...

The file is a saved-searches database (`confoundsearch.data.cache`) built by `devops/tools/build_demo.py`.
"""

from __future__ import annotations

import contextlib
import math
import shutil
import sqlite3
from pathlib import Path

from confoundsearch.data.cache import database_path
from confoundsearch.errors import CorpusError
from confoundsearch.logic.graph import Passage, normalize

DEMO_FILE = Path(__file__).with_name("demo.sqlite3")
DEMO_QUESTION = ("Organizational structure", "Firm performance")
RECORDED_MODEL = "gpt-4.1-mini"  # the model whose answers are saved for the example question
PREWARMED_NOTE = (
    "Note: this is the built-in example question. Its searches, and the answers of the default model\n"
    f"({RECORDED_MODEL}), are already saved (a pre-warmed cache) in your data directory, so those come back\n"
    "instantly and for free. That is to show what a fast run looks like. It is a stub: any other question,\n"
    "or another model, runs live and takes longer."
)
CURATED_SEARCH = ("--queries", "25", "--synonyms", "8")  # the curated run searches many wordings
CURATED_CONFOUNDS = (
    "IT intensity",
    "CEO style",
    "market competition",
)  # the confounds the curated demo asks about
STOPWORDS = frozenset(
    (  # noqa: SIM905  (one string is easier to read and extend)
        "a an and are as at be by for from has have how in is it its of on or that the their this to was "
        "were what when which with does do affect affects effect effects between"
    ).split()
)
TITLE_WEIGHT = 2  # a word in the title counts as many times as this


def words(text: str) -> list[str]:
    return [w for w in normalize(text).split() if w not in STOPWORDS]


class DemoCorpus:
    """Ranks the papers by how rare the query's words are among them, counting title words extra: a plain
    TF-IDF, so "of" and "firm" count for little and "centralization" or "ambidexterity" for a lot."""

    def __init__(self, path: Path = DEMO_FILE) -> None:
        self.path = Path(path)
        self._papers: list[tuple[Passage, dict[str, int]]] | None = None
        self._idf: dict[str, float] = {}

    def _load(self) -> list[tuple[Passage, dict[str, int]]]:
        if self._papers is None:
            if not self.path.is_file():
                raise CorpusError(f"the demo papers are not installed ({self.path} is missing)")
            db = sqlite3.connect(
                f"{self.path.resolve().as_uri()}?mode=ro", uri=True
            )  # installs may be read-only
            try:
                rows = []
                with contextlib.suppress(
                    sqlite3.OperationalError
                ):  # no example set in this file: use every paper
                    rows = db.execute(
                        "SELECT p.id, p.text, p.title, p.url FROM papers p "
                        "JOIN example_papers e ON e.key = p.key ORDER BY p.key"
                    ).fetchall()
                if not rows:
                    rows = db.execute("SELECT id, text, title, url FROM papers ORDER BY key").fetchall()
            except sqlite3.Error as err:
                raise CorpusError(f"the demo papers could not be read: {err}") from err
            finally:
                db.close()
            papers = []
            for row in rows:
                passage = Passage(*row)
                counts: dict[str, int] = {}
                for word in words(passage.text):
                    counts[word] = counts.get(word, 0) + 1
                for word in words(passage.title):
                    counts[word] = counts.get(word, 0) + TITLE_WEIGHT
                papers.append((passage, counts))
            containing: dict[str, int] = {}
            for _, counts in papers:
                for word in counts:
                    containing[word] = containing.get(word, 0) + 1
            self._idf = {w: math.log(1 + len(papers) / n) for w, n in containing.items()}
            self._papers = papers
        return self._papers

    def search(self, query: str, k: int) -> list[Passage]:
        papers = self._load()
        wanted = set(words(query))
        scored = []
        for passage, counts in papers:
            score = sum(self._idf[w] * (1 + math.log(counts[w])) for w in wanted if w in counts)
            if score > 0:
                scored.append((-score, passage.id, passage))
        scored.sort(key=lambda item: item[:2])
        return [passage for _, _, passage in scored[: max(k, 0)]]


def seed_demo_cache(folder: Path, question: tuple[str, str], source: Path = DEMO_FILE) -> str | None:
    """Pre-warm `folder` for the demo question: copy the shipped saved searches in if it has none yet.

    Returns None for any other question, "seeded" when it copied the file, "present" when the data directory
    already had one (its own saved searches are never overwritten).
    """
    target = database_path(folder, DEMO_QUESTION)
    if database_path(folder, question) != target or not source.is_file():
        return None
    if target.is_file():
        return "present"
    try:
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        shutil.copyfile(source, target)
    except OSError:  # a read-only or full disk: the run goes on, just without the head start
        return None
    return "seeded"
