"""Several stores of papers searched at once, as one corpus.

No single source holds every paper, and any one can be slow or refuse a request (Semantic Scholar rate
limits). `MultiCorpus` asks every store the same question in parallel, skips a store that fails, merges the
lists by reciprocal rank (a paper several stores rank highly comes first), and keeps each paper once: papers
with the same title, ignoring case and punctuation, are the same paper, and the fuller copy is kept.

    MultiCorpus({"semanticscholar": SemanticScholarCorpus(key), "openalex": OpenAlexCorpus(email)})
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

from confoundsearch.data.corpus import Corpus
from confoundsearch.errors import CorpusError, RemoteError
from confoundsearch.logic.graph import Passage

RRF_OFFSET = 60  # the usual constant for reciprocal rank fusion


def title_key(title: str) -> str:
    """A title without case, punctuation or spacing, so two spellings of it compare equal."""
    return re.sub(r"[^a-z0-9]+", " ", title.casefold()).strip()


def paper_id(key: str, source: str, passage: Passage) -> str:
    """The same paper gets the same id from any search or source, so a graph never holds it twice."""
    if not title_key(passage.title):
        return f"{source}:{passage.id}"
    return "p" + hashlib.sha1(key.encode()).hexdigest()[:10]  # noqa: S324 (an identifier, not security)


class MultiCorpus:
    def __init__(self, stores: Mapping[str, Corpus], log: Callable[..., None] | None = None) -> None:
        if not stores:
            raise ValueError("a MultiCorpus needs at least one store")
        self.stores, self.log = dict(stores), log

    def _ask(self, name: str, query: str, k: int) -> tuple[str, list[Passage] | RemoteError]:
        try:
            return name, self.stores[name].search(query, k)
        except RemoteError as err:
            if self.log:
                self.log(
                    "source_failed", source=name, query=query, error=f"{type(err).__name__}: {err}"[:300]
                )
            return name, err

    def search(self, query: str, k: int) -> list[Passage]:
        if k <= 0 or not query.strip():
            return []
        with ThreadPoolExecutor(max_workers=len(self.stores)) as pool:
            answers = list(pool.map(lambda name: self._ask(name, query, k), self.stores))
        failures = {name: a for name, a in answers if isinstance(a, RemoteError)}
        if len(failures) == len(self.stores):
            raise CorpusError("every source failed: " + "; ".join(f"{n}: {e}" for n, e in failures.items()))
        scores: dict[str, float] = {}
        kept: dict[str, Passage] = {}
        for name, found in answers:
            if isinstance(found, RemoteError):
                continue
            for rank, passage in enumerate(found, 1):
                key = title_key(passage.title) or f"{name}:{passage.id}"  # no title: nothing to match it by
                scores[key] = scores.get(key, 0.0) + 1.0 / (RRF_OFFSET + rank)
                if key not in kept or len(passage.text) > len(kept[key].text):
                    kept[key] = Passage(
                        paper_id(key, name, passage), passage.text, passage.title, passage.url
                    )
        if self.log:
            self.log(
                "search",
                query=query,
                answered={n: len(a) for n, a in answers if not isinstance(a, RemoteError)},
                failed=sorted(failures),
                merged=len(kept),
            )
        return [kept[key] for key in sorted(scores, key=lambda key: (-scores[key], key))[:k]]
