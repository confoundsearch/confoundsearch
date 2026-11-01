"""AttachPassage: cite an observed passage as evidence for an edge.

The environment's judge decides whether the passage really supports the claim.
If it does, the passage joins P(u, v). If not, the graph is unchanged and the
pair is remembered in `State.rejected` so it is not offered again.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from confoundsearch.errors import InvalidAction
from confoundsearch.logic.graph import TARGETS, Edge, Passage, Target
from confoundsearch.logic.state import State

if TYPE_CHECKING:
    from confoundsearch.logic.transition import Environment


@dataclass(frozen=True)
class AttachPassage:
    confound: str
    target: Target
    passage_id: str

    def _check(self, state: State) -> tuple[Passage, Edge]:
        passage = next((p for p in state.observed if p.id == self.passage_id), None)
        if passage is None:
            raise InvalidAction(f"passage {self.passage_id!r} has not been observed")
        if self.target not in TARGETS:
            raise InvalidAction(f"edge target must be X or Y, got {self.target!r}")
        edge: Edge = (self.confound, self.target)
        if edge not in state.graph.edges:
            raise InvalidAction(f"no edge {edge} to attach evidence to")
        return passage, edge

    def apply(self, state: State, env: Environment) -> State:
        graph = state.graph
        passage, edge = self._check(state)
        if env.judge.supports(passage, graph.vertex(self.confound), graph.target_name(self.target)):
            return state.with_graph(graph.attach(self.confound, self.target, passage))
        return state.with_rejected(edge, passage)

    def preview(self, state: State, env: Environment) -> State:
        """The state if the judge accepts the passage, without asking it. Used to compare moves cheaply."""
        passage, _ = self._check(state)
        return state.with_graph(state.graph.attach(self.confound, self.target, passage))
