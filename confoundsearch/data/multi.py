"""Several stores of papers searched at once, as one corpus.

No single source holds every paper, and any one can be slow or refuse a request (Semantic Scholar rate
limits). `MultiCorpus` asks every store the same question in parallel, skips a store that fails, merges the
lists by reciprocal rank (a paper several stores rank highly comes first), and keeps each paper once: papers
with the same title, ignoring case and punctuation, are the same paper, and the fuller copy is kept.

    MultiCorpus({"semanticscholar": SemanticScholarCorpus(key), "openalex": OpenAlexCorpus(email)})

A source that fails (an error, or a timeout) is not asked again straight away: asking a dead source costs its
whole timeout on every search. It is left alone for a cooldown that doubles with each failure in a row (5
seconds, then 10, 20, up to 5 minutes), and then asked again; one success clears the record. `health()` says,
for each source, when it last answered, for which query, and when it will next be tried. If every source is
cooling down, all are asked anyway: an answer from a doubtful source beats none.
"""

from __future__ import annotations

import contextvars
import hashlib
import re
import time
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable

from confoundsearch.data.corpus import Corpus
from confoundsearch.errors import CorpusError, RemoteError
from confoundsearch.logic.graph import Passage

RRF_OFFSET = 60  # the usual constant for reciprocal rank fusion
COOLDOWN = (5.0, 300.0)  # seconds a failed source is left alone: the first time, and the most


@dataclass
class SourceState:
    """What is known about one source: its failures in a row, when it was last tried, and its last success."""

    failures: int = 0
    skip_until: float = 0.0
    last_error: str = ""
    last_ok_at: float | None = None
    last_ok_query: str | None = None


def title_key(title: str) -> str:
    """A title without case, punctuation or spacing, so two spellings of it compare equal."""
    return re.sub(r"[^a-z0-9]+", " ", title.casefold()).strip()


def paper_id(key: str, source: str, passage: Passage) -> str:
    """The same paper gets the same id from any search or source, so a graph never holds it twice."""
    if not title_key(passage.title):
        return f"{source}:{passage.id}"
    return "p" + hashlib.sha1(key.encode()).hexdigest()[:10]  # noqa: S324 (a short key, not security)


class MultiCorpus:
    def __init__(
        self,
        stores: Mapping[str, Corpus],
        log: Callable[..., None] | None = None,
        clock: Callable[[], float] = time.monotonic,
        cooldown: tuple[float, float] = COOLDOWN,
    ) -> None:
        if not stores:
            raise ValueError("a MultiCorpus needs at least one store")
        self.stores, self.log, self.clock, self.cooldown = dict(stores), log, clock, cooldown
        self.state = {name: SourceState() for name in self.stores}

    def _ask(self, name: str, query: str, k: int) -> tuple[str, list[Passage] | RemoteError]:
        try:
            return name, self.stores[name].search(query, k)
        except RemoteError as err:
            if self.log:
                self.log(
                    "source_failed", source=name, query=query, error=f"{type(err).__name__}: {err}"[:300]
                )
            return name, err

    def health(self) -> dict[str, dict[str, Any]]:
        """Per source: answering or not, failures in a row, the last success, and when it is next tried."""
        now = self.clock()
        return {
            name: {
                "ok": st.failures == 0,
                "failures": st.failures,
                "last_error": st.last_error,
                "last_ok_query": st.last_ok_query,
                "seconds_since_ok": None if st.last_ok_at is None else round(now - st.last_ok_at, 1),
                "retry_in": round(max(0.0, st.skip_until - now), 1),
            }
            for name, st in self.state.items()
        }

    def _ask_all(self, names: list[str], query: str, k: int) -> list[tuple[str, list[Passage] | RemoteError]]:
        with ThreadPoolExecutor(max_workers=len(names)) as pool:
            # each search runs in a copy of this thread's context, so it knows which run it belongs to
            contexts = [contextvars.copy_context() for _ in names]
            return list(
                pool.map(lambda pair: pair[0].run(self._ask, pair[1], query, k), zip(contexts, names))
            )

    def _record(self, answers: list[tuple[str, list[Passage] | RemoteError]], query: str) -> None:
        """Remember who answered and who did not. Done here, after the threads, so nothing races."""
        first, longest = self.cooldown
        for name, found in answers:
            st = self.state[name]
            if isinstance(found, RemoteError):
                st.failures += 1
                st.last_error = f"{type(found).__name__}: {found}"[:300]
                st.skip_until = self.clock() + min(first * 2 ** (st.failures - 1), longest)
                continue
            if st.failures and self.log:
                self.log("source_recovered", source=name, after_failures=st.failures)
            st.failures, st.skip_until, st.last_error = 0, 0.0, ""
            st.last_ok_at, st.last_ok_query = self.clock(), query

    def search(self, query: str, k: int) -> list[Passage]:
        if k <= 0 or not query.strip():
            return []
        now = self.clock()
        asking = [n for n in self.stores if self.state[n].skip_until <= now]
        resting = [n for n in self.stores if n not in asking]
        if not asking:  # all are cooling down: ask them anyway rather than answer nothing
            asking, resting = list(self.stores), []
        for name in resting:
            if self.log:
                self.log("source_skipped", source=name, retry_in=round(self.state[name].skip_until - now, 1))
        answers = self._ask_all(asking, query, k)
        if resting and all(isinstance(found, RemoteError) for _, found in answers):
            answers += self._ask_all(resting, query, k)  # the rest is the only hope: try the resting ones too
        answers.sort(key=lambda pair: list(self.stores).index(pair[0]))
        self._record(answers, query)
        failures = {name: a for name, a in answers if isinstance(a, RemoteError)}
        if len(failures) == len(answers):
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
                skipped=sorted(n for n in resting if n not in dict(answers)),
                merged=len(kept),
            )
        return [kept[key] for key in sorted(scores, key=lambda key: (-scores[key], key))[:k]]
