"""AddEdge: claim that a confound affects X or Y.

A confound counts toward the reward once it has edges to both.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from confoundsearch.logic.graph import Target
from confoundsearch.logic.state import State

if TYPE_CHECKING:
    from confoundsearch.logic.transition import Environment


@dataclass(frozen=True)
class AddEdge:
    confound: str
    target: Target

    def apply(self, state: State, env: Environment) -> State:
        return state.with_graph(state.graph.add_edge(self.confound, self.target))
