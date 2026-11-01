"""Which closed-form actions are valid in a state.

Retrieve, AddConfound and Merge are open-ended (any query, any label, and
which confounds are really the same idea is a judgment), so they come from a
proposer such as `confoundsearch.model.llm.LLMProposer`. Merging every pair here
would let a random agent collapse unrelated confounds into one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from confoundsearch.logic.actions.add_edge import AddEdge
from confoundsearch.logic.actions.attach_passage import AttachPassage
from confoundsearch.logic.actions.stop import Stop
from confoundsearch.logic.graph import TARGETS
from confoundsearch.logic.state import State

if TYPE_CHECKING:
    from confoundsearch.logic.actions import Action


def enumerate_actions(state: State) -> list[Action]:
    """Every closed-form action valid in `state`, in a deterministic order."""
    if state.done:
        return []
    graph = state.graph
    actions: list[Action] = [Stop()]
    for label in graph.labels:
        for target in TARGETS:
            if not graph.has_edge(label, target):
                actions.append(AddEdge(label, target))
    for label, target in sorted(graph.edges):
        attached = graph.passages(label, target)
        for passage in sorted(state.observed, key=lambda p: p.id):
            if passage not in attached and ((label, target), passage.id) not in state.rejected:
                actions.append(AttachPassage(label, target, passage.id))
    return actions
