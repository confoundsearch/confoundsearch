"""A small set of real papers that ships with the package, for trying it out with no network search.

The papers are about one question, organizational structure and firm performance, with the confounds people
usually name for it (firm size and age, IT investment, industry, and so on). `DemoCorpus` searches them in
memory, so a demo needs nothing but a language model:

    confoundsearch run --corpus demo --iv "Organizational structure" --dv "Firm performance" ...

The file is a saved-searches database (`confoundsearch.data.cache`) built by `devops/tools/build_demo.py`.
"""

from __future__ import annotations

import math
import sqlite3
from pathlib import Path

from confoundsearch.errors import CorpusError
from confoundsearch.logic.graph import Passage, normalize

DEMO_FILE = Path(__file__).with_name("demo.sqlite3")
DEMO_QUESTION = ("Organizational structure", "Firm performance")
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
