"""The default policy: ask for confounds, get their synonyms, search for all of them, then process them.

No randomness and no value estimates: a fixed plan over the MDP's actions, so the same answers from the model
give the same run. In order:

1. Find the confounds to look for, from two independent sources, each with its own count:
   - ask: the model names the confounds of X and Y (`ask`, up to `max_confounds`; 0 skips it);
   - mine: search for Y itself (its name and synonyms, `mine_queries` searches of `mine_depth` papers),
     then the model reads those papers for what they say affects Y, best confounds first (`mine`, up to
     `mine_count` more; 0 or no `mine` skips it). Its searches are ordinary moves, so what they find is
     evidence like any other.
2. Get synonyms for each confound and for the two variables (`synonyms`), once per term.
3. For each confound, in order:
   - AddConfound, with its synonyms as the phrasings the graph treats as one idea;
   - for each variable: Retrieve the confound's phrasings paired with the variable's (`queries_per_side`
     pairs, the plainest first) and rank what came back by how many of those phrasings it uses. If anything
     mentions both the confound and the variable, AddEdge and AttachPassage the best `attach_per_side`. The
     judge decides which really support the link; an edge nobody supports is still reported, as unsupported.
4. Stop.

With `synthesize` the order changes: every confound is added and searched first, then (with `measures`,
`mechanisms` and `attributes`) a second round of searches uses the words papers use for what the constructs
are made of and how they act: the measures of each variable, the mechanisms by which each confound could act
on each variable, and the observable attributes of each confound (alone, and with the first few measures).
Then one synthesis reads all that was retrieved and says, by meaning rather than by exact words, which
passages support which links. Only those become edges and attached passages; the environment's judge is
asked about each (`TrustSynthesis` accepts them, so the synthesis has the last word). If the synthesis cannot
be reached, the word-for-word ranking above is used instead.

The model is reached through callables, so this layer never imports the model layer.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Callable, Generator, Sequence, Tuple

from confoundsearch.errors import RemoteError
from confoundsearch.logic.actions import Action, AddConfound, AddEdge, AttachPassage, Retrieve, Stop
from confoundsearch.logic.graph import Passage, Target, normalize
from confoundsearch.logic.state import State

Ask = Callable[[str, str], Sequence[str]]  # (x, y) -> the confounds worth looking for
Mine = Callable[[str, str, Sequence[Passage]], Sequence[str]]
# (x, y, papers found for y) -> the confounds those papers suggest, most plausible first
Synonyms = Callable[[str], Sequence[str]]  # term -> other ways to say it
Synthesize = Callable[[str, str, Sequence[str], Sequence[Passage]], Sequence[Tuple[str, Target, str]]]
# (x, y, confounds, passages) -> (confound, "X" or "Y", passage id) for each passage that supports a link
Measures = Callable[[str], Sequence[str]]  # variable -> the specific measures or forms papers name for it
Mechanisms = Callable[[str, str], Sequence[str]]  # (confound, variable) -> channels it could act through
Attributes = Callable[[str], Sequence[str]]  # confound -> the observable characteristics papers measure

Script = Generator[Action, State, None]
Steps = Generator[Action, State, State]  # a part of the plan; it hands back the state it ended in


def distinct(texts: Iterable[object], limit: int | None = None) -> list[str]:
    """The non-blank strings, each once (ignoring case and spacing), in order."""
    seen: set[str] = set()
    kept: list[str] = []
    for text in texts:
        if isinstance(text, str) and text.strip() and normalize(text) not in seen:
            seen.add(normalize(text))
            kept.append(text.strip())
    return kept[:limit]


class DeterministicPolicy:
    def __init__(
        self,
        ask: Ask,
        synonyms: Synonyms,
        max_confounds: int = 8,
        synonyms_per_term: int = 3,
        queries_per_side: int = 3,
        results_per_query: int = 3,
        attach_per_side: int = 4,
        synthesize: Synthesize | None = None,
        measures: Measures | None = None,
        mechanisms: Mechanisms | None = None,
        attributes: Attributes | None = None,
        attribute_measures: int = 4,
        mine: Mine | None = None,
        mine_count: int = 8,
        mine_queries: int = 3,
        mine_depth: int = 50,
    ) -> None:
        self.ask, self.synonyms, self.synthesize = ask, synonyms, synthesize
        self.measures, self.mechanisms = measures, mechanisms
        self.attributes, self.attribute_measures = attributes, attribute_measures
        self.max_confounds, self.synonyms_per_term = max_confounds, synonyms_per_term
        self.queries_per_side, self.results_per_query = queries_per_side, results_per_query
        self.attach_per_side = attach_per_side
        self.mine, self.mine_count, self.mine_queries, self.mine_depth = (
            mine,
            mine_count,
            mine_queries,
            mine_depth,
        )
        self._script: Script | None = None

    def select(self, state: State, actions: Sequence[Action]) -> Action:
        if self._script is None:
            self._script = self._plan(state)
            return next(self._script)
        return self._script.send(state)

    # -- the plan ------------------------------------------------------------

    def _plan(self, state: State) -> Script:
        x, y = state.graph.x, state.graph.y
        phrasings: dict[str, list[str]] = {}
        names = distinct(self._asked(x, y), self.max_confounds)
        if self.mine is not None and self.mine_count > 0:
            phrasings[y] = self._phrasings(y)
            for query in phrasings[y][: self.mine_queries]:
                state = yield Retrieve(query, self.mine_depth)
            found = sorted(state.observed, key=lambda p: p.id)  # only the mining searches so far
            asked = {normalize(n) for n in names}
            mined = [n for n in distinct(self._mined(x, y, found)) if normalize(n) not in asked]
            names += mined[: self.mine_count]
        names = [n for n in names if normalize(n) not in {normalize(x), normalize(y)}]
        for term in distinct([x, y, *names]):
            if term not in phrasings:
                phrasings[term] = self._phrasings(term)
        taken: set[str] = set()
        added: list[tuple[str, list[str]]] = []
        sides: tuple[tuple[Target, str], tuple[Target, str]] = (("X", x), ("Y", y))
        for name in names:
            if normalize(name) in taken:
                continue  # another confound already uses this very phrase
            own = [p for p in phrasings[name] if normalize(p) not in taken]  # the confound itself comes first
            taken |= {normalize(p) for p in own}
            state = yield AddConfound(name, frozenset(normalize(p) for p in own))
            added.append((name, own))
            for target, variable in sides:
                if self.synthesize is None:
                    state = yield from self._search_and_process(state, name, own, target, phrasings[variable])
                else:
                    state = yield from self._search(state, own, phrasings[variable])
        if self.synthesize is not None:
            state = yield from self._second_phase(state, x, y, added)
            state = yield from self._synthesize_and_attach(state, x, y, added, phrasings)
        while True:
            state = yield Stop()

    def _search_and_process(
        self, state: State, name: str, strings: list[str], target: Target, variable: Sequence[str]
    ) -> Steps:
        state = yield from self._search(state, strings, variable)
        found = self._ranked(state.observed, strings, variable)
        return (yield from self._attach(state, name, target, [passage.id for passage in found]))

    def _search(self, state: State, strings: list[str], variable: Sequence[str]) -> Steps:
        for query in self._queries(strings, variable):
            state = yield Retrieve(query, self.results_per_query)
        return state

    def _second_phase(self, state: State, x: str, y: str, added: list[tuple[str, list[str]]]) -> Steps:
        """More searches in papers' own words: a confound's mechanisms, and its name with each measure."""
        sides = (("X", x), ("Y", y))
        measured = {variable: self._asked_for(self.measures, variable) for _, variable in sides}
        for name, own in added:
            queries: list[str] = []
            attributes = self._asked_for(self.attributes, name)
            queries += attributes  # a paper may name the attribute it measured, not the confound
            for _, variable in sides:
                queries += self._asked_for(self.mechanisms, name, variable)
                queries += [f"{own[0]} {measure}" for measure in measured[variable]]
                for measure in measured[variable][: self.attribute_measures]:
                    queries += [f"{attribute} {measure}" for attribute in attributes]
            for query in distinct(queries):
                state = yield Retrieve(query, self.results_per_query)
        return state

    @staticmethod
    def _asked_for(call: Callable[..., Sequence[str]] | None, *args: str) -> list[str]:
        """What the model says, or nothing if there is no such step or the model cannot be reached."""
        if call is None:
            return []
        try:
            return [item for item in call(*args) if isinstance(item, str)]
        except RemoteError:
            return []

    def _attach(self, state: State, name: str, target: Target, ids: list[str]) -> Steps:
        """If anything was found: add the edge and cite the best `attach_per_side` passages for it."""
        if ids:
            state = yield AddEdge(name, target)
            for ident in ids[: self.attach_per_side]:
                state = yield AttachPassage(name, target, ident)
        return state

    def _synthesize_and_attach(
        self,
        state: State,
        x: str,
        y: str,
        added: list[tuple[str, list[str]]],
        phrasings: dict[str, list[str]],
    ) -> Steps:
        """One reading of everything retrieved, then the edges and passages it supports."""
        assert self.synthesize is not None
        try:
            supports = list(
                self.synthesize(x, y, [name for name, _ in added], sorted(state.observed, key=lambda p: p.id))
            )
        except RemoteError:
            supports = None  # the model could not be reached: fall back to the word-for-word ranking
        sides: tuple[tuple[Target, str], tuple[Target, str]] = (("X", x), ("Y", y))
        for name, own in added:
            for target, variable in sides:
                if supports is None:
                    ids = [p.id for p in self._ranked(state.observed, own, phrasings[variable])]
                else:
                    ids = [ident for c, side, ident in supports if (c, side) == (name, target)]
                state = yield from self._attach(state, name, target, ids)
        return state

    # -- the parts -----------------------------------------------------------

    def _asked(self, x: str, y: str) -> Sequence[str]:
        if self.max_confounds <= 0:
            return []  # asking is switched off
        try:
            return self.ask(x, y)
        except RemoteError:
            return []

    def _mined(self, x: str, y: str, papers: Sequence[Passage]) -> Sequence[str]:
        assert self.mine is not None
        try:
            return self.mine(x, y, papers)
        except RemoteError:
            return []

    def _phrasings(self, term: str) -> list[str]:
        """The term, then up to `synonyms_per_term` other ways to say it."""
        try:
            others = self.synonyms(term)
        except RemoteError:
            others = []
        return distinct([term, *others], self.synonyms_per_term + 1)

    def _queries(self, confound: Sequence[str], variable: Sequence[str]) -> list[str]:
        """Phrasing pairs, the plainest first: the two literal terms, then one reworded, then the other."""
        pairs = sorted(
            ((i, j) for i in range(len(confound)) for j in range(len(variable))), key=lambda p: (sum(p), p[0])
        )
        return [f"{confound[i]} {variable[j]}" for i, j in pairs[: self.queries_per_side]]

    @staticmethod
    def _ranked(
        observed: Iterable[Passage], confound: Sequence[str], variable: Sequence[str]
    ) -> list[Passage]:
        """Passages that mention both the confound and the variable, those using most phrasings first."""
        scored = []
        for passage in observed:
            words = _words(f"{passage.title} {passage.text}")
            about_confound = sum(1 for phrase in confound if _mentions(words, phrase))
            about_variable = sum(1 for phrase in variable if _mentions(words, phrase))
            if about_confound and about_variable:
                scored.append((-(about_confound + about_variable), passage.id, passage))
        return [passage for *_, passage in sorted(scored, key=lambda item: item[:2])]


def _stem(word: str) -> str:
    return word[:-1] if len(word) > 3 and word.endswith("s") else word  # firms = firm, rates = rate


def _words(text: str) -> set[str]:
    return {_stem(word) for word in re.findall(r"[a-z0-9]+", text.casefold())}


def _mentions(words: set[str], phrase: str) -> bool:
    """Whether every word of `phrase` is in `words`, whatever order and with or without a plural s."""
    needed = {_stem(word) for word in re.findall(r"[a-z0-9]+", phrase.casefold())}
    return bool(needed) and needed <= words
