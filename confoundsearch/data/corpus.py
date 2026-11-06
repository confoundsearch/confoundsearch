"""The corpus D, behind a one-method protocol."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Protocol

from confoundsearch.logic.graph import Passage, normalize


class Corpus(Protocol):
    def search(self, query: str, k: int) -> list[Passage]:
        """Return up to k passages relevant to the query."""
        ...


class InMemoryCorpus:
    """Ranks passages by how many query words they contain. Deterministic."""

    def __init__(self, passages: Iterable[Passage]) -> None:
        self.passages = tuple(passages)

    @classmethod
    def from_jsonl(cls, path: str | Path) -> InMemoryCorpus:
        """One JSON object per line with `id` and `text`, and optionally `title`, `url`, `authors`, `venue`,
        `year` and `cited_by`."""
        rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
        return cls(
            Passage(
                str(r["id"]),
                r["text"],
                r.get("title", ""),
                r.get("url", ""),
                str(r.get("authors") or ""),
                str(r.get("venue") or ""),
                r.get("year"),
                r.get("cited_by"),
            )
            for r in rows
        )

    def search(self, query: str, k: int) -> list[Passage]:
        words = set(normalize(query).split())
        scored = [(len(words & set(normalize(p.text).split())), p) for p in self.passages]
        hits = [(n, p) for n, p in scored if n > 0]
        hits.sort(key=lambda hit: (-hit[0], hit[1].id))
        return [p for _, p in hits[: max(k, 0)]]
