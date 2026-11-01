"""Merge: collapse two vertices into one, e.g. "firm size" and "head count".

Edges and evidence of both vertices are carried over to the merged vertex.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from confoundsearch.logic.state import State

if TYPE_CHECKING:
    from confoundsearch.logic.transition import Environment


@dataclass(frozen=True)
class Merge:
    a: str
    b: str
    label: str | None = None

    def apply(self, state: State, env: Environment) -> State:
        return state.with_graph(state.graph.merge(self.a, self.b, self.label))
