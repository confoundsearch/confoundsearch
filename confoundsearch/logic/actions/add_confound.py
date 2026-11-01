"""AddConfound: add a vertex, a labelled cluster of strings, to the graph."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from confoundsearch.logic.graph import Vertex
from confoundsearch.logic.state import State

if TYPE_CHECKING:
    from confoundsearch.logic.transition import Environment


@dataclass(frozen=True)
class AddConfound:
    label: str
    strings: frozenset[str]

    def apply(self, state: State, env: Environment) -> State:
        return state.with_graph(state.graph.add_confound(Vertex(self.label, self.strings)))
