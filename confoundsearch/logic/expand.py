"""Synonym-aware search: ask the corpus several ways and merge the answers.

A search engine matches the words it is given. The papers on a confound rarely use the label an agent starts
from ("organizational structure" against "decentralization" or "corporate hierarchies"), so one phrasing
misses them. `ExpandedCorpus` wraps any corpus: it searches the query and its rewordings, then merges the
lists by reciprocal rank, so a paper that several phrasings return comes first.

    ExpandedCorpus(SemanticScholarCorpus(key), rewordings)   # rewordings(query) -> other ways to say it
"""

from __future__ import annotations

from collections.abc import Callable

from confoundsearch.data.corpus import Corpus
from confoundsearch.errors import RemoteError
from confoundsearch.logic.graph import Passage

RRF_OFFSET = (
    60  # the usual constant for reciprocal rank fusion: it keeps one first place from swamping the rest
)


class ExpandedCorpus:
    def __init__(self, corpus: Corpus, rewordings: Callable[[str], list[str]], limit: int = 4) -> None:
        self.corpus = corpus
        self.rewordings = rewordings
        self.limit = limit

    def queries(self, query: str) -> list[str]:
        """The query first, then up to `limit` distinct rewordings. If making them fails, just the query."""
        try:
            extra = self.rewordings(query)
        except RemoteError:
            extra = []
        queries = [query]
        for other in extra:
            if len(queries) > self.limit:
                break
            if other.strip() and other.strip().casefold() not in {q.casefold() for q in queries}:
                queries.append(other.strip())
        return queries

    def search(self, query: str, k: int) -> list[Passage]:
        if k <= 0 or not query.strip():
            return []
        scores: dict[str, float] = {}
        seen: dict[str, Passage] = {}
        for text in self.queries(query):
            for rank, passage in enumerate(self.corpus.search(text, k), 1):
                seen.setdefault(passage.id, passage)
                scores[passage.id] = scores.get(passage.id, 0.0) + 1.0 / (RRF_OFFSET + rank)
        order = sorted(scores, key=lambda pid: (-scores[pid], pid))
        return [seen[pid] for pid in order[:k]]
