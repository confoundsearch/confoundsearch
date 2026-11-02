"""Expand a construct into a long list of query terms: recall first, precision later.

    expander = KeywordExpander(synonyms, neighbors, per_term=100, limit=1000)
    expansion = expander.expand("firm size")      # or expander("firm size")
    expansion.texts()          # 'firm size', the model's synonyms, then ~1000 keywords near them

`interface.expander.build_expander` wires one to a model and the keyword service.

Two steps, each handed in as a plain function so that nothing here knows which model or database is behind it:

  1. `synonyms(construct)` asks a model for other names, measures and neighbouring ideas. It may return plain
     strings or (term, kind) pairs; every term is then fetched for (up to `max_seeds` of them, in order).
  2. `neighbors(term, k)` returns the k nearest keywords of the vector index for the construct and for each
     of those terms. A result needs `.id`, `.score` and `.metadata` (the keyword's `name` and paper `works`).

Optionally a third, `neighbor_synonyms(names)`, asks the model for other names of the best keywords found, and
those join the list too (kind "neighbor synonym"): one more hop out. Together this is the grand query
expansion.

The lists are merged by reciprocal rank: a keyword that many of the terms bring up comes first. A step that
fails for one term, or altogether, is recorded in `skipped` and the rest carries on.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Protocol

from confoundsearch.errors import RemoteError
from confoundsearch.logic.graph import normalize

RRF_OFFSET = 60  # the usual constant for reciprocal rank fusion


class Neighbor(Protocol):
    @property
    def id(self) -> str: ...
    @property
    def score(self) -> float: ...
    @property
    def metadata(self) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class Term:
    text: str
    # "construct"; "synonym", "variant", "hypernym" or "variant synonym" (from the model, and which of its
    # steps); "neighbor" (from the vector index); or "neighbor synonym" (the model, for those neighbors)
    kind: str
    score: float | None = None  # a neighbor's best similarity to any of the terms that brought it up
    works: int | None = None  # papers that use the keyword, for a neighbor
    found_via: tuple[str, ...] = ()  # the terms whose neighbours included it


@dataclass
class Expansion:
    construct: str
    terms: list[Term]
    skipped: dict[str, str] = field(default_factory=dict)  # what did not work, and why

    def texts(self) -> list[str]:
        """Just the terms, in order: the construct, the model's, then the nearest keywords."""
        return [t.text for t in self.terms]

    def counts(self) -> dict[str, int]:
        """How many terms of each kind, for the kinds that occur."""
        counts: dict[str, int] = {}
        for term in self.terms:
            counts[term.kind] = counts.get(term.kind, 0) + 1
        return counts


class KeywordExpander:
    """The master keyword expansion: a construct in, a long ranked list of query terms out.

    Made once with its two steps and its limits, then used for any number of constructs.
    """

    def __init__(
        self,
        synonyms: Callable[[str], Sequence[str | tuple[str, str]]],
        neighbors: Callable[[str, int], Sequence[Neighbor]],
        per_term: int = 100,
        limit: int = 1000,
        workers: int = 8,
        max_seeds: int = 150,
        neighbor_synonyms: Callable[[list[str]], Sequence[str]] | None = None,
        neighbor_top: int = 100,
    ) -> None:
        self.synonyms, self.neighbors = synonyms, neighbors
        self.neighbor_synonyms, self.neighbor_top = neighbor_synonyms, neighbor_top
        self.per_term, self.limit, self.workers, self.max_seeds = per_term, limit, workers, max_seeds

    def __call__(self, construct: str) -> Expansion:
        return self.expand(construct)

    def expand(self, construct: str) -> Expansion:
        construct = " ".join(construct.split())
        if not construct:
            raise ValueError("there is no construct to expand")
        skipped: dict[str, str] = {}
        try:
            named = list(self.synonyms(construct))
        except RemoteError as err:  # no model: the vector index alone still gives a list
            skipped["synonyms"] = str(err)
            named = []
        seeds = [construct]
        kinds = {construct: "construct"}
        seen = {normalize(construct)}
        for item in named:  # a plain string is a "synonym"; a pair says what kind of term it is
            text, kind = (item, "synonym") if isinstance(item, str) else item
            text = " ".join(text.split())
            if text and normalize(text) not in seen and len(seeds) < max(self.max_seeds, 1):
                seen.add(normalize(text))
                seeds.append(text)
                kinds[text] = kind

        def ask(seed: str) -> tuple[str, Sequence[Neighbor], str | None]:
            try:
                return seed, self.neighbors(seed, self.per_term), None
            except (RemoteError, LookupError) as err:  # service down, or not a keyword and cannot be embedded
                return seed, [], str(err)

        with ThreadPoolExecutor(max_workers=max(1, min(self.workers, len(seeds)))) as pool:
            answers = list(pool.map(ask, seeds))

        fused: dict[str, float] = {}
        best: dict[str, Neighbor] = {}
        via: dict[str, list[str]] = {}
        for seed, found, error in answers:
            if error is not None:
                skipped[seed] = error
                continue
            for position, match in enumerate(found, 1):
                name = str(match.metadata.get("name") or match.id)
                key = normalize(name)
                if key in seen:  # already a term of its own
                    continue
                fused[key] = fused.get(key, 0.0) + 1.0 / (RRF_OFFSET + position)
                if key not in best or match.score > best[key].score:
                    best[key] = match
                if seed not in via.setdefault(key, []):
                    via[key].append(seed)

        terms = [Term(text, kinds[text]) for text in seeds]
        ranked = sorted(
            fused, key=lambda k: (-fused[k], -best[k].score, str(best[k].metadata.get("name") or best[k].id))
        )
        for key in ranked:
            match = best[key]
            works = match.metadata.get("works")
            terms.append(
                Term(
                    str(match.metadata.get("name") or match.id),
                    "neighbor",
                    float(match.score),
                    int(works) if isinstance(works, (int, float)) else None,
                    tuple(via[key]),
                )
            )
        if self.neighbor_synonyms is not None and self.neighbor_top > 0 and ranked:
            names = [terms[len(seeds) + i].text for i in range(min(self.neighbor_top, len(ranked)))]
            try:
                more = list(self.neighbor_synonyms(names))
            except RemoteError as err:
                skipped["neighbor synonyms"] = str(err)
                more = []
            for text in more:
                text = " ".join(text.split())
                if text and normalize(text) not in seen and normalize(text) not in fused:
                    seen.add(normalize(text))
                    terms.append(Term(text, "neighbor synonym"))
        return Expansion(construct, terms[: max(self.limit, 1)], skipped)


def expand_terms(
    construct: str,
    synonyms: Callable[[str], Sequence[str | tuple[str, str]]],
    neighbors: Callable[[str, int], Sequence[Neighbor]],
    per_term: int = 100,
    limit: int = 1000,
    workers: int = 8,
    max_seeds: int = 150,
    neighbor_synonyms: Callable[[list[str]], Sequence[str]] | None = None,
    neighbor_top: int = 100,
) -> Expansion:
    """One expansion without keeping an expander: `KeywordExpander(...).expand(construct)`."""
    return KeywordExpander(
        synonyms, neighbors, per_term, limit, workers, max_seeds, neighbor_synonyms, neighbor_top
    ).expand(construct)
