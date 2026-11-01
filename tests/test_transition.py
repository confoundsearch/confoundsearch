from __future__ import annotations

import pytest

from confoundsearch import (
    AddConfound,
    AddEdge,
    AttachPassage,
    Environment,
    InMemoryCorpus,
    InvalidAction,
    Merge,
    Retrieve,
    State,
    Stop,
)
from tests.conftest import DV, IV, P1, P2, P3, StubJudge


def built(env: Environment) -> State:
    s = env.initial_state(IV, DV)
    for a in (
        AddConfound("firm size", frozenset({"firm size", "head count"})),
        AddEdge("firm size", "X"),
        AddEdge("firm size", "Y"),
        Retrieve("larger firms"),
    ):
        s = env.step(s, a)
    return s


def test_initial_state(env):
    s = env.initial_state(IV, DV)
    assert s == State.initial(IV, DV)


def test_stop_ends_the_episode(env):
    s = env.step(env.initial_state(IV, DV), Stop())
    assert s.done
    with pytest.raises(InvalidAction, match="ended"):
        env.step(s, Retrieve("x"))


def test_retrieve_collects_passages_without_touching_graph(env):
    s0 = env.initial_state(IV, DV)
    s1 = env.step(s0, Retrieve("larger firms invest", k=1))
    assert s1.observed == {P1} and s1.graph == s0.graph


def test_retrieve_accumulates(env):
    s = env.step(env.initial_state(IV, DV), Retrieve("invest"))
    s = env.step(s, Retrieve("profits"))
    assert s.observed == {P1, P2}


def test_retrieve_with_no_hits_is_a_valid_noop(env):
    s0 = env.initial_state(IV, DV)
    assert env.step(s0, Retrieve("zebra")) == s0


def test_retrieve_rejects_blank_query(env):
    with pytest.raises(InvalidAction):
        env.step(env.initial_state(IV, DV), Retrieve("  "))


def test_add_confound_and_duplicate(env):
    s = env.step(env.initial_state(IV, DV), AddConfound("size", frozenset({"size"})))
    assert s.graph.labels == ("size",)
    with pytest.raises(InvalidAction):
        env.step(s, AddConfound("size", frozenset({"other"})))


def test_merge(env):
    s = env.step(env.initial_state(IV, DV), AddConfound("a", frozenset({"a"})))
    s = env.step(s, AddConfound("b", frozenset({"b"})))
    s = env.step(s, Merge("a", "b", "ab"))
    assert s.graph.labels == ("ab",) and s.graph.vertex("ab").strings == {"a", "b"}


def test_add_edge_requires_confound(env):
    with pytest.raises(InvalidAction):
        env.step(env.initial_state(IV, DV), AddEdge("ghost", "X"))


def test_attach_accepted_passage(env, judge):
    s = env.step(built(env), AttachPassage("firm size", "X", "p1"))
    assert s.graph.passages("firm size", "X") == {P1}
    assert judge.calls == [("p1", "firm size", IV)]


def test_judge_sees_the_right_target_name(env, judge):
    env.step(built(env), AttachPassage("firm size", "Y", "p2"))
    assert judge.calls == [("p2", "firm size", DV)]


def test_rejected_passage_leaves_graph_and_is_recorded():
    env = Environment(InMemoryCorpus([P1, P2, P3]), StubJudge(set()))
    s = env.step(env.initial_state(IV, DV), AddConfound("firm size", frozenset({"firm size"})))
    s = env.step(s, AddEdge("firm size", "X"))
    s = env.step(s, Retrieve("larger"))
    t = env.step(s, AttachPassage("firm size", "X", "p1"))
    assert t.graph == s.graph and (("firm size", "X"), "p1") in t.rejected


def test_attach_unobserved_passage(env):
    with pytest.raises(InvalidAction, match="not been observed"):
        env.step(built(env), AttachPassage("firm size", "X", "nope"))


def test_attach_without_edge(env):
    s = env.step(env.initial_state(IV, DV), AddConfound("a", frozenset({"a"})))
    s = env.step(s, Retrieve("larger"))
    with pytest.raises(InvalidAction, match="no edge"):
        env.step(s, AttachPassage("a", "X", "p1"))


def test_attach_bad_target(env):
    with pytest.raises(InvalidAction, match="X or Y"):
        env.step(built(env), AttachPassage("firm size", "Z", "p1"))  # type: ignore[arg-type]


def test_attach_twice_is_invalid(env):
    s = env.step(built(env), AttachPassage("firm size", "X", "p1"))
    with pytest.raises(InvalidAction, match="already attached"):
        env.step(s, AttachPassage("firm size", "X", "p1"))


def test_step_does_not_mutate_input_state(env):
    s = built(env)
    before = (s.graph, s.observed, s.rejected, s.done)
    env.step(s, AttachPassage("firm size", "X", "p1"))
    assert (s.graph, s.observed, s.rejected, s.done) == before


def test_unknown_action_object(env):
    with pytest.raises(InvalidAction, match="unknown"):
        env.step(env.initial_state(IV, DV), object())  # type: ignore[arg-type]
