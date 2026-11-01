"""MDP states: the partial graph plus the evidence observed so far."""

from __future__ import annotations

from dataclasses import dataclass

from confoundsearch.logic.graph import Edge, Graph, Passage


@dataclass(frozen=True)
class State:
    """s_t = (g_t, O_t).

    graph     the partial graph g_t
    observed  passages gathered from the corpus so far (O_t)
    rejected  (edge, passage id) pairs the judge refused, so they are not retried
    done      True once the STOP action was taken (the final state s_T)
    """

    graph: Graph
    observed: frozenset[Passage] = frozenset()
    rejected: frozenset[tuple[Edge, str]] = frozenset()
    done: bool = False

    @classmethod
    def initial(cls, x: str, y: str) -> State:
        """s_0: only the study's variables, no evidence."""
        return cls(Graph(x, y))

    def with_graph(self, graph: Graph) -> State:
        return State(graph, self.observed, self.rejected, self.done)

    def with_observed(self, passages: frozenset[Passage]) -> State:
        return State(self.graph, self.observed | passages, self.rejected, self.done)

    def with_rejected(self, edge: Edge, passage: Passage) -> State:
        return State(self.graph, self.observed, self.rejected | {(edge, passage.id)}, self.done)

    def stopped(self) -> State:
        return State(self.graph, self.observed, self.rejected, True)
