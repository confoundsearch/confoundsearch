"""Retrieve: search the corpus D and add the passages found to O_t.

The only action whose outcome the agent cannot predict: it does not know how
many passages will come back. The graph is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from confoundsearch.errors import InvalidAction
from confoundsearch.logic.state import State

if TYPE_CHECKING:
    from confoundsearch.logic.transition import Environment


@dataclass(frozen=True)
class Retrieve:
    query: str
    k: int = 5

    def apply(self, state: State, env: Environment) -> State:
        self.preview(state, env)
        return state.with_observed(frozenset(env.corpus.search(self.query, self.k)))

    def preview(self, state: State, env: Environment) -> State:
        """Before the search runs nothing visible changes: what it will find is not known yet."""
        if not self.query.strip():
            raise InvalidAction("query must be non-empty")
        return state
