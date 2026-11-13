"""The default policy: ask for confounds, get synonyms, search for all of them, then process them."""

from __future__ import annotations

from typing import ClassVar

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
            "mine_count": 8,
            "mine_queries": 3,
            "mine_depth": 50,
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


class TestSynthesis:
    """With a synthesis: search every confound first, then one reading of what came back picks the links."""

    def policy(self, supports, **options):
        calls = []

        def synthesize(x, y, confounds, passages):
            calls.append((x, y, list(confounds), [p.id for p in passages]))
            return supports

        policy = DeterministicPolicy(
            ask_for("firm size", "firm age"),
            synonyms_of({"firm size": ["head count"]}),
            synthesize=synthesize,
            **options,
        )
        return policy, calls

    def test_everything_is_searched_before_it_reads_once_and_then_the_links_are_made(self):
        policy, calls = self.policy([("firm size", "X", "s1"), ("firm age", "Y", "a1")])
        trajectory, _ = play(policy)
        order = kinds(trajectory)
        assert order.index("AddEdge") > max(i for i, k in enumerate(order) if k == "Retrieve")
        assert len(calls) == 1 and calls[0][:3] == (X, Y, ["firm size", "firm age"])
        assert "s2" in calls[0][3]  # it reads all that was retrieved, whether or not it uses the words
        graph = trajectory.final.graph
        assert {p.id for p in graph.passages("firm size", "X")} == {"s1"}
        assert {p.id for p in graph.passages("firm age", "Y")} == {"a1"}
        assert not graph.has_edge("firm size", "Y")  # no word-match edge: only what it supported

    def test_it_reads_the_passages_in_a_fixed_order_so_the_same_run_asks_the_same_question(self):
        policy, calls = self.policy([])
        play(policy)
        assert calls[0][3] == sorted(calls[0][3])

    def test_a_passage_that_uses_other_words_can_be_the_evidence(self):
        policy, _ = self.policy([("firm age", "X", "s2")])  # s2 is about firm size, not age
        trajectory, _ = play(policy)
        assert {p.id for p in trajectory.final.graph.passages("firm age", "X")} == {"s2"}

    def test_it_attaches_no_more_than_the_limit_per_side_and_the_judge_still_decides(self):
        supports = [("firm size", "X", "s1"), ("firm size", "X", "s3"), ("firm size", "X", "a1")]
        policy, _ = self.policy(supports, attach_per_side=2)
        trajectory, _ = play(policy, accepted={"s1"})
        attached = [s.action.passage_id for s in trajectory.steps if isinstance(s.action, AttachPassage)]
        assert attached == ["s1", "s3"]
        assert {p.id for p in trajectory.final.graph.passages("firm size", "X")} == {"s1"}

    def test_nothing_supported_means_no_edges(self):
        policy, _ = self.policy([])
        trajectory, _ = play(policy)
        assert "AddEdge" not in kinds(trajectory)

    def test_if_the_model_cannot_be_reached_it_falls_back_to_the_words(self):
        def broken(*args):
            raise RemoteError("down")

        policy = DeterministicPolicy(
            ask_for("firm size", "firm age"), synonyms_of({"firm size": ["head count"]}), synthesize=broken
        )
        trajectory, _ = play(policy)
        graph = trajectory.final.graph
        assert {p.id for p in graph.passages("firm size", "X")} >= {"s1"}
        assert graph.has_edge("firm age", "X") and not graph.has_edge("firm age", "Y")


class TestSecondPhase:
    """With measures and mechanisms: more searches in papers' own words, before the synthesis reads all."""

    def policy(self, measures=None, mechanisms=None, attributes=None, **options):
        seen = []

        def synthesize(x, y, confounds, passages):
            seen.append([p.id for p in passages])
            return []

        policy = DeterministicPolicy(
            ask_for("firm size"),
            synonyms_of({}),
            synthesize=synthesize,
            measures=measures,
            mechanisms=mechanisms,
            attributes=attributes,
            **options,
        )
        return policy, seen

    def queries(self, trajectory):
        return [s.action.query for s in trajectory.steps if isinstance(s.action, Retrieve)]

    def test_it_searches_each_mechanism_alone_and_the_confound_with_each_measure(self):
        measures = lambda term: ["head count", "span"] if term == X else ["profit"]  # noqa: E731
        mechanisms = lambda confound, variable: [f"{variable} channel"]  # noqa: E731
        policy, seen = self.policy(measures, mechanisms)
        trajectory, _ = play(policy)
        queries = self.queries(trajectory)
        first = queries[:2]  # the plain round: the one pairing for each side
        assert all(q.startswith("firm size") for q in first)
        assert queries[2:] == [
            f"{X} channel",
            "firm size head count",
            "firm size span",
            f"{Y} channel",
            "firm size profit",
        ]
        assert len(seen) == 1  # and still one reading, after all of them

    def test_it_searches_the_attributes_alone_and_with_the_first_measures_of_each_variable(self):
        measures = lambda term: ["span", "levels", "forms"]  # noqa: E731
        attributes = lambda confound: ["tenure", "background"]  # noqa: E731
        policy, _ = self.policy(measures, lambda c, v: [], attributes=attributes, attribute_measures=2)
        trajectory, _ = play(policy)
        queries = self.queries(trajectory)[2:]
        assert queries[:2] == ["tenure", "background"]  # alone, once
        crossed = [q for q in queries if q.split()[0] in ("tenure", "background") and len(q.split()) > 1]
        assert sorted(set(crossed)) == sorted(
            f"{a} {m}" for a in ("tenure", "background") for m in ("span", "levels")
        )  # only the first two measures, for each variable's list
        assert "tenure forms" not in queries

    def test_the_same_search_is_not_made_twice(self):
        policy, _ = self.policy(lambda term: ["head count", "Head Count"], lambda c, v: [])
        trajectory, _ = play(policy)
        assert self.queries(trajectory).count("firm size head count") == 1

    def test_without_either_step_nothing_more_is_searched(self):
        policy, _ = self.policy()
        trajectory, _ = play(policy)
        assert len(self.queries(trajectory)) == 2

    def test_if_the_model_cannot_be_reached_the_plain_searches_are_kept(self):
        def broken(*args):
            raise RemoteError("down")

        policy, _ = self.policy(broken, broken)
        trajectory, _ = play(policy)
        assert len(self.queries(trajectory)) == 2 and trajectory.steps[-1].action == Stop()

    def test_what_is_not_text_is_ignored(self):
        policy, _ = self.policy(lambda term: [3, None, "span"], lambda c, v: [])
        trajectory, _ = play(policy)
        assert self.queries(trajectory)[2:] == ["firm size span"]

    def test_it_only_happens_with_the_synthesis(self):
        policy = DeterministicPolicy(
            ask_for("firm size"), synonyms_of({}), measures=lambda t: ["span"], mechanisms=lambda c, v: ["x"]
        )
        trajectory, _ = play(policy)
        assert len(self.queries(trajectory)) == 2


class TestMining:
    """Mining is its own source of confounds, beside asking: searched for Y, read by `mine`, own count."""

    PAPERS: ClassVar[list[Passage]] = [
        *PASSAGES,
        Passage("y1", "Gender and age shape firm performance.", "Who performs"),
        Passage("y2", "Firm performance and industry.", "Industry and performance"),
    ]

    def miner(self, *names, fail=False):
        calls = []

        def mine(x, y, papers):
            calls.append((x, y, [p.id for p in papers]))
            if fail:
                raise RemoteError("down")
            return list(names)

        mine.calls = calls
        return mine

    def policy(self, mine, **options):
        return DeterministicPolicy(
            ask_for("firm size"),
            synonyms_of({"firm performance": ["profitability", "firm results"]}),
            mine=mine,
            **options,
        )

    def test_it_searches_for_y_first_then_adds_the_mined_confounds_after_the_asked_ones(self):
        mine = self.miner("Firm size", "industry", "gender", "IT investment", "age")
        trajectory, _ = play(self.policy(mine, mine_count=2, mine_queries=2, mine_depth=7), self.PAPERS)
        searches = [s.action for s in trajectory.steps if isinstance(s.action, Retrieve)]
        assert [(r.query, r.k) for r in searches[:2]] == [(Y, 7), ("profitability", 7)]  # Y, then a synonym
        assert mine.calls[0][:2] == (X, Y) and set(mine.calls[0][2]) <= {p.id for p in self.PAPERS}
        added = [s.action.label for s in trajectory.steps if isinstance(s.action, AddConfound)]
        assert added == ["firm size", "industry", "gender"]  # firm size was asked already: 2 new ones mined

    def test_mining_can_be_the_only_source(self):
        ask = ask_for("firm size")
        trajectory, _ = play(
            DeterministicPolicy(ask, synonyms_of({}), max_confounds=0, mine=self.miner("industry")),
            self.PAPERS,
        )
        assert ask.calls == []  # asking is off
        assert [s.action.label for s in trajectory.steps if isinstance(s.action, AddConfound)] == ["industry"]

    @pytest.mark.parametrize("options", [{"mine_count": 0}, {}])
    def test_with_a_count_of_zero_or_no_miner_it_does_not_mine(self, options):
        mine = self.miner("industry") if options else None
        trajectory, _ = play(self.policy(mine, **options), self.PAPERS)
        assert mine is None or mine.calls == []
        assert [s.action.label for s in trajectory.steps if isinstance(s.action, AddConfound)] == [
            "firm size"
        ]

    def test_a_miner_that_cannot_be_reached_adds_nothing(self):
        trajectory, _ = play(self.policy(self.miner("industry", fail=True)), self.PAPERS)
        assert [s.action.label for s in trajectory.steps if isinstance(s.action, AddConfound)] == [
            "firm size"
        ]

    def test_y_is_asked_for_synonyms_once_even_when_mining_used_them(self):
        syn = synonyms_of({"firm performance": ["profitability"]})
        play(DeterministicPolicy(ask_for("firm size"), syn, mine=self.miner()), self.PAPERS)
        assert syn.calls.count(Y) == 1
