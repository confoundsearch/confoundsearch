from __future__ import annotations

from confoundsearch import (
    AddEdge,
    AttachPassage,
    Merge,
    Passage,
    Retrieve,
    State,
    Stop,
    Vertex,
    enumerate_actions,
)
from tests.conftest import AGE, P1, P2, SIZE


def state() -> State:
    return State.initial("IT", "perf")


def test_actions_are_value_objects():
    assert Retrieve("q") == Retrieve("q", 5)
    assert Merge("a", "b") == Merge("a", "b", None)
    assert len({AddEdge("a", "X"), AddEdge("a", "X")}) == 1


def test_initial_state_only_allows_stop():
    assert enumerate_actions(state()) == [Stop()]


def test_done_state_has_no_actions():
    assert enumerate_actions(state().stopped()) == []


def test_enumerates_edges_but_never_merges():
    s = state().with_graph(state().graph.add_confound(SIZE).add_confound(AGE))
    actions = enumerate_actions(s)
    assert not any(isinstance(a, Merge) for a in actions)  # merging is a judgment call, left to the proposer
    for label in ("firm age", "firm size"):
        for target in ("X", "Y"):
            assert AddEdge(label, target) in actions


def test_existing_edges_are_not_offered_again():
    g = state().graph.add_confound(SIZE).add_edge("firm size", "X")
    actions = enumerate_actions(state().with_graph(g))
    assert AddEdge("firm size", "X") not in actions
    assert AddEdge("firm size", "Y") in actions


def test_attach_offered_for_observed_unattached_passages_on_existing_edges():
    g = state().graph.add_confound(SIZE).add_edge("firm size", "X")
    s = state().with_graph(g).with_observed(frozenset({P1, P2}))
    attach = [a for a in enumerate_actions(s) if isinstance(a, AttachPassage)]
    assert attach == [AttachPassage("firm size", "X", "p1"), AttachPassage("firm size", "X", "p2")]


def test_attach_skips_attached_and_rejected():
    g = state().graph.add_confound(SIZE).add_edge("firm size", "X").attach("firm size", "X", P1)
    s = state().with_graph(g).with_observed(frozenset({P1, P2, Passage("p3", "x")}))
    s = s.with_rejected(("firm size", "X"), P2)
    attach = [a for a in enumerate_actions(s) if isinstance(a, AttachPassage)]
    assert attach == [AttachPassage("firm size", "X", "p3")]


def test_enumeration_is_deterministic():
    g = state().graph.add_confound(SIZE).add_confound(AGE).add_confound(Vertex("r", frozenset({"r"})))
    s = state().with_graph(g)
    assert enumerate_actions(s) == enumerate_actions(s)
