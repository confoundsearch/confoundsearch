"""The transition function T(s' | s, a).

Each action carries its own effect (see `confoundsearch.logic.actions`). Graph edits
are deterministic. Uncertainty enters through the environment: what the corpus
returns for a query, and whether the judge accepts a passage as support for an
edge. Both are injected, so the MDP is testable offline.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from confoundsearch.data.corpus import Corpus
from confoundsearch.errors import InvalidAction
from confoundsearch.logic.graph import Passage, Vertex
from confoundsearch.logic.state import State

if TYPE_CHECKING:
    from confoundsearch.logic.actions import Action


class Judge(Protocol):
    def supports(self, passage: Passage, source: Vertex, target: str) -> bool:
        """Does the passage support the claim that `source` affects `target`?"""
        ...


class TrustSynthesis:
    """Accepts every passage it is asked about.

    With a synthesis, the policy cites only passages that the synthesis (one reading of everything retrieved,
    by meaning) found to support a link, so that reading is the judgment and no second opinion is asked.
    """

    def supports(self, passage: Passage, source: Vertex, target: str) -> bool:
        return True


class Environment:
    def __init__(self, corpus: Corpus, judge: Judge) -> None:
        self.corpus = corpus
        self.judge = judge

    def initial_state(self, x: str, y: str) -> State:
        return State.initial(x, y)

    def step(self, state: State, action: Action) -> State:
        """Apply `action` to `state`. Raises InvalidAction if it is not allowed."""
        return self._run(state, action, "apply")

    def preview(self, state: State, action: Action) -> State:
        """The state `action` would lead to if the outside world cooperated, without asking it.

        A search is assumed to change nothing visible and the judge is assumed to accept a citation. No
        corpus query and no model call is made, so this is what to use to compare candidate moves.
        """
        return self._run(state, action, "preview")

    def _run(self, state: State, action: Action, how: str) -> State:
        if state.done:
            raise InvalidAction("the episode has ended")
        method = getattr(action, how, None) or getattr(action, "apply", None)
        if method is None:
            raise InvalidAction(f"unknown action {action!r}")
        result: State = method(state, self)
        return result
