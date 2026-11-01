"""The default policy: ask for confounds, get their synonyms, search for all of them, then process them.

No randomness and no value estimates: a fixed plan over the MDP's actions, so the same answers from the model
give the same run. In order:

1. Ask the model which confounds to look for (`ask`), once.
2. Get synonyms for each of them and for the two variables (`synonyms`), once per term.
3. For each confound, in order:
   - AddConfound, with its synonyms as the phrasings the graph treats as one idea;
   - for each variable: Retrieve the confound's phrasings paired with the variable's (`queries_per_side`
     pairs, the plainest first) and rank what came back by how many of those phrasings it uses. If anything
     mentions both the confound and the variable, AddEdge and AttachPassage the best `attach_per_side`. The
     judge decides which really support the link; an edge nobody supports is still reported, as unsupported.
4. Stop.

The model is reached through two callables, so this layer never imports the model layer.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Callable, Generator, Sequence

from confoundsearch.errors import RemoteError
from confoundsearch.logic.actions import Action, AddConfound, AddEdge, AttachPassage, Retrieve, Stop
from confoundsearch.logic.graph import Passage, Target, normalize
from confoundsearch.logic.state import State

Ask = Callable[[str, str], Sequence[str]]  # (x, y) -> the confounds worth looking for
Synonyms = Callable[[str], Sequence[str]]  # term -> other ways to say it

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
    ) -> None:
        self.ask, self.synonyms = ask, synonyms
        self.max_confounds, self.synonyms_per_term = max_confounds, synonyms_per_term
        self.queries_per_side, self.results_per_query = queries_per_side, results_per_query
        self.attach_per_side = attach_per_side
        self._script: Script | None = None

    def select(self, state: State, actions: Sequence[Action]) -> Action:
        if self._script is None:
            self._script = self._plan(state)
            return next(self._script)
        return self._script.send(state)

    # -- the plan ------------------------------------------------------------

    def _plan(self, state: State) -> Script:
        x, y = state.graph.x, state.graph.y
        names = distinct(self._asked(x, y), self.max_confounds)
        names = [n for n in names if normalize(n) not in {normalize(x), normalize(y)}]
        phrasings = {term: self._phrasings(term) for term in distinct([x, y, *names])}
        taken: set[str] = set()
        for name in names:
            if normalize(name) in taken:
                continue  # another confound already uses this very phrase
            own = [p for p in phrasings[name] if normalize(p) not in taken]  # the confound itself comes first
            taken |= {normalize(p) for p in own}
            state = yield AddConfound(name, frozenset(normalize(p) for p in own))
            sides: tuple[tuple[Target, str], tuple[Target, str]] = (("X", x), ("Y", y))
            for target, variable in sides:
                state = yield from self._search_and_process(state, name, own, target, phrasings[variable])
        while True:
            state = yield Stop()

    def _search_and_process(
        self, state: State, name: str, strings: list[str], target: Target, variable: Sequence[str]
    ) -> Steps:
        for query in self._queries(strings, variable):
            state = yield Retrieve(query, self.results_per_query)
        found = self._ranked(state.observed, strings, variable)
        if found:
            state = yield AddEdge(name, target)
            for passage in found[: self.attach_per_side]:
                state = yield AttachPassage(name, target, passage.id)
        return state

    # -- the parts -----------------------------------------------------------

    def _asked(self, x: str, y: str) -> Sequence[str]:
        try:
            return self.ask(x, y)
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
