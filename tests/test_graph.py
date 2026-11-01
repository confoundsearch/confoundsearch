from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from confoundsearch import Graph, InvalidAction, Passage, Vertex
from confoundsearch.logic.graph import normalize
from tests.conftest import AGE, P1, P2, P3, SIZE


def base() -> Graph:
    return Graph("IT", "perf").add_confound(SIZE).add_confound(AGE)


class TestVertex:
    def test_normalizes_strings(self):
        v = Vertex("  size ", frozenset({"  Firm   SIZE ", "Head Count"}))
        assert v.label == "size"
        assert v.strings == {"firm size", "head count"}

    def test_requires_label(self):
        with pytest.raises(InvalidAction, match="label"):
            Vertex("  ", frozenset({"a"}))

    @pytest.mark.parametrize("strings", [frozenset(), frozenset({"", "   "})])
    def test_requires_a_string(self, strings):
        with pytest.raises(InvalidAction, match="at least one string"):
            Vertex("a", strings)

    @given(st.text())
    def test_normalize_is_idempotent(self, s):
        assert normalize(normalize(s)) == normalize(s)


class TestPassage:
    def test_requires_id(self):
        with pytest.raises(InvalidAction):
            Passage("", "text")

    def test_value_equality(self):
        assert Passage("a", "t") == Passage("a", "t")
        assert len({Passage("a", "t"), Passage("a", "t")}) == 1


class TestConstruction:
    def test_empty_graph(self):
        g = Graph("IT", "perf")
        assert g.confounds == () and g.edges == frozenset() and dict(g.support) == {}

    @pytest.mark.parametrize("x,y", [("", "y"), ("x", " "), ("Same", "same")])
    def test_rejects_bad_variables(self, x, y):
        with pytest.raises(InvalidAction):
            Graph(x, y)

    def test_rejects_duplicate_labels(self):
        with pytest.raises(InvalidAction, match="duplicate"):
            Graph("x", "y", (SIZE, Vertex("firm size", frozenset({"other"}))))

    def test_rejects_shared_strings(self):
        with pytest.raises(InvalidAction, match="two vertices"):
            Graph("x", "y", (SIZE, Vertex("headcount", frozenset({"head count"}))))

    def test_rejects_edge_from_unknown_confound(self):
        with pytest.raises(InvalidAction, match="unknown confound"):
            Graph("x", "y", (), frozenset({("ghost", "X")}))

    def test_rejects_bad_edge_target(self):
        with pytest.raises(InvalidAction, match="X or Y"):
            Graph("x", "y", (SIZE,), frozenset({("firm size", "Z")}))  # type: ignore[arg-type]

    def test_rejects_support_without_edge(self):
        with pytest.raises(InvalidAction, match="missing edge"):
            Graph("x", "y", (SIZE,), frozenset(), {("firm size", "X"): frozenset({P1})})

    def test_confounds_are_sorted_so_equality_ignores_order(self):
        assert Graph("x", "y", (SIZE, AGE)) == Graph("x", "y", (AGE, SIZE))

    def test_support_is_read_only(self):
        g = base().add_edge("firm size", "X").attach("firm size", "X", P1)
        with pytest.raises(TypeError):
            g.support[("firm size", "Y")] = frozenset()  # type: ignore[index]

    def test_empty_support_entries_are_dropped(self):
        g = Graph("x", "y", (SIZE,), frozenset({("firm size", "X")}), {("firm size", "X"): frozenset()})
        assert dict(g.support) == {}

    def test_hashable_and_consistent_with_equality(self):
        a = base().add_edge("firm size", "X").attach("firm size", "X", P1)
        b = base().add_edge("firm size", "X").attach("firm size", "X", P1)
        assert a == b and hash(a) == hash(b) and len({a, b}) == 1


class TestQueries:
    def test_labels_and_vertex_lookup(self):
        g = base()
        assert g.labels == ("firm age", "firm size")
        assert g.vertex("firm age") is AGE or g.vertex("firm age") == AGE

    def test_unknown_vertex(self):
        with pytest.raises(InvalidAction, match="no confound"):
            base().vertex("ghost")

    def test_target_name(self):
        g = Graph("IT", "perf")
        assert (g.target_name("X"), g.target_name("Y")) == ("IT", "perf")

    def test_connected_requires_both_edges(self):
        g = base().add_edge("firm size", "X")
        assert not g.is_connected("firm size") and g.connected() == ()
        g = g.add_edge("firm size", "Y")
        assert g.is_connected("firm size")
        assert [v.label for v in g.connected()] == ["firm size"]

    def test_passages_default_empty(self):
        assert base().passages("firm size", "X") == frozenset()

    def test_evidence_counts_both_edges(self):
        g = base().add_edge("firm size", "X").add_edge("firm size", "Y")
        g = g.attach("firm size", "X", P1).attach("firm size", "Y", P2).attach("firm size", "Y", P3)
        assert g.evidence("firm size") == 3 and g.evidence("firm age") == 0


class TestEdits:
    def test_edits_do_not_mutate(self):
        g = base()
        g.add_edge("firm size", "X")
        assert g.edges == frozenset()

    def test_add_confound_rejects_clash(self):
        with pytest.raises(InvalidAction):
            base().add_confound(Vertex("size2", frozenset({"head count"})))

    def test_add_edge_unknown_confound(self):
        with pytest.raises(InvalidAction):
            base().add_edge("ghost", "X")

    def test_add_edge_twice(self):
        with pytest.raises(InvalidAction, match="already exists"):
            base().add_edge("firm size", "X").add_edge("firm size", "X")

    def test_attach_needs_edge(self):
        with pytest.raises(InvalidAction, match="no edge"):
            base().attach("firm size", "X", P1)

    def test_attach_twice(self):
        g = base().add_edge("firm size", "X").attach("firm size", "X", P1)
        with pytest.raises(InvalidAction, match="already attached"):
            g.attach("firm size", "X", P1)

    def test_attach_accumulates(self):
        g = base().add_edge("firm size", "X").attach("firm size", "X", P1).attach("firm size", "X", P2)
        assert g.passages("firm size", "X") == {P1, P2}


class TestMerge:
    def graph(self) -> Graph:
        g = base().add_edge("firm size", "X").add_edge("firm age", "X").add_edge("firm age", "Y")
        return g.attach("firm size", "X", P1).attach("firm age", "X", P2).attach("firm age", "Y", P3)

    def test_unions_strings_edges_and_evidence(self):
        m = self.graph().merge("firm size", "firm age", "firm characteristics")
        assert m.labels == ("firm characteristics",)
        assert m.vertex("firm characteristics").strings == SIZE.strings | AGE.strings
        assert m.edges == {("firm characteristics", "X"), ("firm characteristics", "Y")}
        assert m.passages("firm characteristics", "X") == {P1, P2}
        assert m.passages("firm characteristics", "Y") == {P3}
        assert m.is_connected("firm characteristics")

    def test_default_label_is_first(self):
        assert self.graph().merge("firm size", "firm age").labels == ("firm size",)

    def test_leaves_other_vertices_alone(self):
        g = base().add_confound(Vertex("rates", frozenset({"interest rates"})))
        g = g.add_edge("rates", "X").merge("firm size", "firm age", "firm")
        assert g.labels == ("firm", "rates") and ("rates", "X") in g.edges

    def test_can_reuse_one_of_the_merged_labels(self):
        assert base().merge("firm size", "firm age", "firm age").labels == ("firm age",)

    def test_label_collision(self):
        g = base().add_confound(Vertex("rates", frozenset({"interest rates"})))
        with pytest.raises(InvalidAction, match="taken"):
            g.merge("firm size", "firm age", "rates")

    def test_self_merge(self):
        with pytest.raises(InvalidAction, match="itself"):
            base().merge("firm size", "firm size")

    def test_unknown_label(self):
        with pytest.raises(InvalidAction):
            base().merge("firm size", "ghost")

    def test_is_symmetric_up_to_label(self):
        g = self.graph()
        a, b = g.merge("firm size", "firm age", "m"), g.merge("firm age", "firm size", "m")
        assert a == b
