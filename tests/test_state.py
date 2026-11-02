from __future__ import annotations

from confoundsearch import Passage, State
from tests.conftest import P1, P2, SIZE


def test_initial_state_has_only_variables_and_no_evidence():
    s = State.initial("IT", "perf")
    assert (s.graph.x, s.graph.y) == ("IT", "perf")
    assert s.graph.confounds == () and s.observed == frozenset() and not s.done


def test_updates_return_new_states():
    s = State.initial("IT", "perf")
    s2 = s.with_observed(frozenset({P1}))
    assert s.observed == frozenset() and s2.observed == {P1}
    assert s2.with_observed(frozenset({P2})).observed == {P1, P2}


def test_a_paper_seen_twice_with_different_text_is_kept_once_with_the_fuller_text():
    short, full = Passage("p", "short"), Passage("p", "the fuller abstract")
    state = State.initial("a", "b").with_observed(frozenset({short}))
    assert state.with_observed(frozenset({full})).observed == {full}  # a fuller copy replaces it
    assert state.with_observed(frozenset({full})).with_observed(frozenset({short})).observed == {full}
    assert state.with_observed(frozenset({Passage("p", "other")})).observed == {
        Passage("p", "short")
    }  # a tie keeps the first
    both = State.initial("a", "b").with_observed(frozenset({short, full}))
    assert both.observed == {full}  # even in one batch


def test_with_graph_preserves_the_rest():
    s = State.initial("a", "b").with_observed(frozenset({P1})).with_rejected((("x", "X")), P2)
    t = s.with_graph(s.graph.add_confound(SIZE))
    assert t.observed == s.observed and t.rejected == s.rejected and t.graph != s.graph


def test_rejected_records_edge_and_passage_id():
    s = State.initial("a", "b").with_rejected(("c", "Y"), P1)
    assert s.rejected == {(("c", "Y"), "p1")}


def test_stopped_sets_done_and_keeps_everything_else():
    s = State.initial("a", "b").with_observed(frozenset({P1}))
    t = s.stopped()
    assert t.done and not s.done and t.observed == s.observed and t.graph == s.graph


def test_other_updates_keep_done():
    t = State.initial("a", "b").stopped()
    assert t.with_observed(frozenset({P1})).done
    assert t.with_rejected(("c", "X"), P1).done
