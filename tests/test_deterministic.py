"""The default policy: ask for confounds, get synonyms, search for all of them, then process them."""

from __future__ import annotations

import pytest

from confoundsearch import (
    AddConfound,
    AddEdge,
    AttachPassage,
    DeterministicPolicy,
    Environment,
    InMemoryCorpus,
    Passage,
    Proposer,
    RemoteError,
    Retrieve,
    State,
    Stop,
    run_episode,
)
from confoundsearch.logic.deterministic import distinct
from tests.conftest import StubJudge

X, Y = "IT investment", "firm performance"

PASSAGES = [
    Passage("s1", "Firm size and IT investment: larger firms spend more.", "Size and IT"),
    Passage("s2", "Firm size shapes firm performance and profitability.", "Size and profits"),
    Passage("s3", "Head count matters for IT investment decisions.", "Head count"),
    Passage("a1", "Firm age is linked to IT investment.", "Age and IT"),
    Passage("w1", "Weather and tourism.", "Weather"),
]


def ask_for(*names):
    calls = []

    def ask(x, y):
        calls.append((x, y))
        return list(names)

    ask.calls = calls
    return ask


def synonyms_of(table):
    calls = []

    def synonyms(term):
        calls.append(term)
        return table.get(term.lower(), [])

    synonyms.calls = calls
    return synonyms


def play(policy, passages=PASSAGES, accepted=None, steps=200):
    judge = StubJudge(set(accepted) if accepted is not None else {p.id for p in passages})
    env = Environment(InMemoryCorpus(passages), judge)
    return run_episode(env, policy, Proposer(), X, Y, max_steps=steps), judge


def kinds(trajectory):
    return [type(step.action).__name__ for step in trajectory.steps]


class TestThePlan:
    def policy(self, **options):
        return DeterministicPolicy(
            ask_for("firm size", "firm age"),
            synonyms_of({"firm size": ["head count"], "it investment": ["information technology"]}),
            **options,
        )

    def test_it_asks_once_gets_each_terms_synonyms_once_then_searches_and_processes_each_confound(self):
        ask, syn = ask_for("firm size", "firm age"), synonyms_of({"firm size": ["head count"]})
        trajectory, _ = play(DeterministicPolicy(ask, syn))
        assert ask.calls == [(X, Y)]
        assert sorted(syn.calls) == sorted([X, Y, "firm size", "firm age"])
        first = [type(s.action).__name__ for s in trajectory.steps[:8]]
        # the confound, its two phrasings searched against the variable, then the edge for what was found
        assert first[:4] == ["AddConfound", "Retrieve", "Retrieve", "AddEdge"]
        assert trajectory.steps[-1].action == Stop()
        assert {"AddConfound", "Retrieve", "AddEdge", "AttachPassage", "Stop"} == set(kinds(trajectory))

    def test_the_confounds_synonyms_become_the_phrasings_of_its_vertex(self):
        trajectory, _ = play(self.policy())
        added = [s.action for s in trajectory.steps if isinstance(s.action, AddConfound)]
        assert [a.label for a in added] == ["firm size", "firm age"]
        assert added[0].strings == frozenset({"firm size", "head count"})
        assert added[1].strings == frozenset({"firm age"})

    def test_the_plainest_phrasing_pairs_are_searched_first(self):
        trajectory, _ = play(self.policy(queries_per_side=4))
        queries = [s.action.query for s in trajectory.steps if isinstance(s.action, Retrieve)][:4]
        assert queries == [
            "firm size IT investment",
            "firm size information technology",
            "head count IT investment",
            "head count information technology",
        ]

    def test_an_edge_is_added_only_where_something_mentions_the_confound(self):
        trajectory, _ = play(self.policy())
        edges = [
            (s.action.confound, s.action.target) for s in trajectory.steps if isinstance(s.action, AddEdge)
        ]
        # "firm size" is found on both sides; "firm age" only with IT investment
        assert ("firm size", "X") in edges and ("firm size", "Y") in edges
        assert ("firm age", "X") in edges and ("firm age", "Y") not in edges

    def test_it_attaches_the_passages_that_use_most_of_the_phrasings_and_no_more_than_the_limit(self):
        trajectory, _ = play(self.policy(attach_per_side=1))
        attached = [s.action for s in trajectory.steps if isinstance(s.action, AttachPassage)]
        by_edge = {(a.confound, a.target): a.passage_id for a in attached}
        assert len(attached) == len(by_edge)  # one per edge
        assert by_edge[("firm size", "X")] == "s1"  # mentions "firm size" and "IT investment": the best fit

    def test_the_judge_decides_and_an_unsupported_edge_is_still_in_the_graph(self):
        trajectory, judge = play(self.policy(), accepted=set())
        assert judge.calls and trajectory.final.graph.passages("firm size", "X") == frozenset()
        assert trajectory.final.graph.has_edge("firm size", "X")  # reported as unsupported, not hidden

    def test_supported_passages_are_attached_to_their_edges(self):
        trajectory, _ = play(self.policy())
        graph = trajectory.final.graph
        assert {p.id for p in graph.passages("firm size", "X")} >= {"s1"}
        assert {p.id for p in graph.passages("firm size", "Y")} == {"s2"}

    def test_the_same_answers_give_the_same_run(self):
        first, _ = play(self.policy())
        second, _ = play(self.policy())
        assert [s.action for s in first.steps] == [s.action for s in second.steps]

    def test_it_ends_with_stop_and_keeps_stopping(self):
        policy = DeterministicPolicy(ask_for(), synonyms_of({}))
        state = State.initial(X, Y)
        assert policy.select(state, []) == Stop() and policy.select(state, []) == Stop()

    def test_its_settings_are_what_the_audit_log_records(self):
        trajectory, _ = play(self.policy(queries_per_side=2, attach_per_side=5))
        started = trajectory.audit_trail().of_kind("run_started")[0].data
        assert started["policy"] == "DeterministicPolicy"
        assert started["policy_settings"] == {
            "max_confounds": 8,
            "synonyms_per_term": 3,
            "queries_per_side": 2,
            "results_per_query": 3,
            "attach_per_side": 5,
        }


class TestWhatTheModelSays:
    def test_the_confound_list_is_cleaned(self):
        ask = ask_for(
            " firm size ", "Firm  Size", "", "IT investment", "firm performance", "firm age", 7, None
        )
        trajectory, _ = play(DeterministicPolicy(ask, synonyms_of({})))
        added = [s.action.label for s in trajectory.steps if isinstance(s.action, AddConfound)]
        assert added == [
            "firm size",
            "firm age",
        ]  # duplicates, blanks, non-text and the variables themselves are dropped

    def test_it_looks_for_no_more_than_the_limit(self):
        ask = ask_for(*[f"confound {i}" for i in range(20)])
        trajectory, _ = play(DeterministicPolicy(ask, synonyms_of({}), max_confounds=3))
        assert len([s for s in trajectory.steps if isinstance(s.action, AddConfound)]) == 3

    def test_a_phrase_two_confounds_share_belongs_to_the_first(self):
        syn = synonyms_of({"firm size": ["scale"], "firm scale": ["scale", "firm size"]})
        trajectory, _ = play(DeterministicPolicy(ask_for("firm size", "firm scale"), syn))
        added = {
            s.action.label: s.action.strings for s in trajectory.steps if isinstance(s.action, AddConfound)
        }
        assert added["firm size"] == frozenset({"firm size", "scale"})
        assert added["firm scale"] == frozenset({"firm scale"})

    def test_a_confound_that_is_only_another_name_for_an_earlier_one_is_skipped(self):
        syn = synonyms_of({"firm size": ["scale"], "scale": ["firm size"]})
        trajectory, _ = play(DeterministicPolicy(ask_for("firm size", "firm size "), syn))
        assert len([s for s in trajectory.steps if isinstance(s.action, AddConfound)]) == 1

    def test_a_confound_named_like_an_earlier_ones_synonym_is_skipped(self):
        syn = synonyms_of({"firm size": ["scale"]})
        trajectory, _ = play(DeterministicPolicy(ask_for("firm size", "scale"), syn))
        assert [s.action.label for s in trajectory.steps if isinstance(s.action, AddConfound)] == [
            "firm size"
        ]

    def test_if_the_model_cannot_be_reached_it_still_runs_on_what_it_has(self):
        def down_ask(x, y):
            raise RemoteError("the model is down")

        def down_synonyms(term):
            raise RemoteError("the model is down")

        trajectory, _ = play(DeterministicPolicy(down_ask, down_synonyms))
        assert kinds(trajectory) == ["Stop"]  # nothing to look for
        policy = DeterministicPolicy(ask_for("firm size"), down_synonyms)
        trajectory, _ = play(policy)
        added = [s.action for s in trajectory.steps if isinstance(s.action, AddConfound)]
        assert added[0].strings == frozenset({"firm size"})  # no synonyms, just the confound itself

    def test_it_uses_at_most_the_requested_number_of_synonyms_per_term(self):
        syn = synonyms_of({"firm size": ["a", "b", "c", "d"]})
        trajectory, _ = play(DeterministicPolicy(ask_for("firm size"), syn, synonyms_per_term=2))
        added = next(s.action for s in trajectory.steps if isinstance(s.action, AddConfound))
        assert added.strings == frozenset({"firm size", "a", "b"})


@pytest.mark.parametrize(
    ("texts", "limit", "expected"),
    [
        (["a", "A ", " b", "a"], None, ["a", "b"]),
        (["a", "b", "c"], 2, ["a", "b"]),
        (["", "  ", 3, None, "x"], None, ["x"]),
    ],
)
def test_distinct_keeps_each_string_once_in_order(texts, limit, expected):
    assert distinct(texts, limit) == expected
