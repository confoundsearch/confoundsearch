"""Stop: end the search. The resulting state is the final state s_T."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from confoundsearch.logic.state import State

if TYPE_CHECKING:
    from confoundsearch.logic.transition import Environment


@dataclass(frozen=True)
class Stop:
    def apply(self, state: State, env: Environment) -> State:
        return state.stopped()
