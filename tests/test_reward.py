from __future__ import annotations

from math import log

import pytest
from hypothesis import given
from hypothesis import strategies as st

from confoundsearch import Graph, Passage, State, Vertex, phi, reward
from tests.conftest import AGE, P1, P2, P3, SIZE


def connected(*vertices: Vertex) -> Graph:
    g = Graph("x", "y")
    for v in vertices:
        g = g.add_confound(v).add_edge(v.label, "X").add_edge(v.label, "Y")
    return g


def test_empty_graph_is_worth_nothing():
    assert phi(Graph("x", "y")) == 0.0


def test_each_connected_confound_is_worth_one_without_evidence():
    assert phi(connected(SIZE)) == pytest.approx(1.0)
    assert phi(connected(SIZE, AGE)) == pytest.approx(2.0)


def test_half_connected_confound_counts_for_nothing():
    g = Graph("x", "y").add_confound(SIZE).add_edge("firm size", "X")
    assert phi(g) == 0.0


def test_evidence_adds_log_returns():
    g = connected(SIZE).attach("firm size", "X", P1).attach("firm size", "Y", P2)
    assert phi(g) == pytest.approx(1 + log(3))


def test_evidence_on_unconnected_confound_does_not_count():
    g = Graph("x", "y").add_confound(SIZE).add_edge("firm size", "X").attach("firm size", "X", P1)
    assert phi(g) == 0.0


def test_more_evidence_is_better_but_with_diminishing_returns():
    g0 = connected(SIZE)
    g1 = g0.attach("firm size", "X", P1)
    g2 = g1.attach("firm size", "X", P2)
    g3 = g2.attach("firm size", "X", P3)
    gains = [phi(b) - phi(a) for a, b in ((g0, g1), (g1, g2), (g2, g3))]
    assert all(g > 0 for g in gains) and gains == sorted(gains, reverse=True)


def test_a_new_connected_confound_beats_one_more_passage():
    base = connected(SIZE).attach("firm size", "X", P1)
    assert phi(connected(SIZE, AGE).attach("firm size", "X", P1)) - phi(base) > phi(
        base.attach("firm size", "X", P2)
    ) - phi(base)


def test_reward_is_the_phi_difference():
    s0 = State(connected(SIZE))
    s1 = State(connected(SIZE).attach("firm size", "X", P1))
    assert reward(s0, s1) == pytest.approx(phi(s1.graph) - phi(s0.graph)) == pytest.approx(log(2))
    assert reward(s1, s0) == pytest.approx(-log(2))


def test_reward_of_a_state_with_itself_is_zero():
    s = State(connected(SIZE))
    assert reward(s, s) == 0.0


def test_merging_synonyms_does_not_inflate_phi():
    g = connected(SIZE, AGE)
    assert phi(g.merge("firm size", "firm age")) < phi(g)


@given(st.integers(0, 5), st.integers(0, 5))
def test_phi_never_decreases_as_the_graph_grows(n_confounds, n_passages):
    g = Graph("x", "y")
    previous = phi(g)
    for i in range(n_confounds):
        label = f"c{i}"
        g = g.add_confound(Vertex(label, frozenset({f"s{i}"})))
        g = g.add_edge(label, "X").add_edge(label, "Y")
        assert phi(g) >= previous
        previous = phi(g)
        for j in range(n_passages):
            g = g.attach(label, "X", Passage(f"{label}-{j}", "t"))
            assert phi(g) > previous
            previous = phi(g)
