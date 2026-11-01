"""`preview` compares candidate moves without touching the outside world."""

from __future__ import annotations

import pytest

from confoundsearch import (
    AddConfound,
    AddEdge,
    AttachPassage,
    Environment,
    EpsilonGreedy,
    InMemoryCorpus,
    InvalidAction,
    LookaheadQ,
    Merge,
    Retrieve,
    Stop,
    reward,
)
from confoundsearch.logic.episode import run_episode
from tests.conftest import DV, IV, P1, P2, P3, P4, StubJudge
from tests.test_episode import SCRIPT, Scripted


class SpyCorpus(InMemoryCorpus):
    def __init__(self, passages):
        super().__init__(passages)
        self.queries = []

    def search(self, query, k):
        self.queries.append(query)
        return super().search(query, k)


@pytest.fixture
def spies():
    corpus, judge = SpyCorpus([P1, P2, P3, P4]), StubJudge()
    return Environment(corpus, judge), corpus, judge


def built(env):
    state = env.initial_state(IV, DV)
    for action in (
        AddConfound("firm size", frozenset({"firm size"})),
        AddEdge("firm size", "X"),
        AddEdge("firm size", "Y"),
        Retrieve("larger firms"),
    ):
        state = env.step(state, action)
    return state


class TestPreview:
    def test_a_search_previews_as_no_change_and_runs_no_query(self, spies):
        env, corpus, _ = spies
        state = env.initial_state(IV, DV)
        assert env.preview(state, Retrieve("larger firms")) == state and corpus.queries == []

    def test_a_blank_search_is_still_invalid(self, spies):
        env, _, _ = spies
        with pytest.raises(InvalidAction, match="non-empty"):
            env.preview(env.initial_state(IV, DV), Retrieve("  "))

    def test_a_citation_previews_as_accepted_and_asks_no_judge(self, spies):
        env, _, judge = spies
        state = built(env)
        preview = env.preview(state, AttachPassage("firm size", "X", "p1"))
        assert P1 in preview.graph.passages("firm size", "X") and judge.calls == []
        assert reward(state, preview) > 0

    def test_a_citation_that_could_not_be_made_is_invalid_in_a_preview_too(self, spies):
        env, _, _ = spies
        state = built(env)
        for bad in (
            AttachPassage("firm size", "X", "nope"),
            AttachPassage("ghost", "X", "p1"),
            AttachPassage("firm size", "Z", "p1"),  # type: ignore[arg-type]
        ):
            with pytest.raises(InvalidAction):
                env.preview(state, bad)
        attached = env.step(state, AttachPassage("firm size", "X", "p1"))
        with pytest.raises(InvalidAction, match="already attached"):
            env.preview(attached, AttachPassage("firm size", "X", "p1"))

    @pytest.mark.parametrize("action", [Stop(), AddConfound("a", frozenset({"a"}))])
    def test_graph_edits_preview_exactly_as_they_run(self, spies, action):
        env, _, _ = spies
        state = env.initial_state(IV, DV)
        assert env.preview(state, action) == env.step(state, action)

    def test_merge_and_edge_previews_match_too(self, spies):
        env, _, _ = spies
        state = env.step(env.initial_state(IV, DV), AddConfound("a", frozenset({"a"})))
        state = env.step(state, AddConfound("b", frozenset({"b"})))
        for action in (AddEdge("a", "X"), Merge("a", "b")):
            assert env.preview(state, action) == env.step(state, action)

    def test_an_ended_episode_cannot_be_previewed(self, spies):
        env, _, _ = spies
        with pytest.raises(InvalidAction, match="ended"):
            env.preview(env.step(env.initial_state(IV, DV), Stop()), Retrieve("x"))

    def test_an_unknown_action_cannot_be_previewed(self, spies):
        env, _, _ = spies
        with pytest.raises(InvalidAction, match="unknown"):
            env.preview(env.initial_state(IV, DV), object())  # type: ignore[arg-type]

    def test_a_preview_is_not_stored_anywhere(self, spies):
        env, _, _ = spies
        state = built(env)
        env.preview(state, AttachPassage("firm size", "X", "p1"))
        assert state.graph.passages("firm size", "X") == frozenset() and state.rejected == frozenset()


class TestLookaheadWithPreview:
    def test_scoring_every_candidate_makes_no_search_and_no_model_call(self, spies):
        env, corpus, judge = spies
        state = built(env)
        q = LookaheadQ(env.preview)
        candidates = [
            Retrieve("a"),
            Retrieve("b"),
            AttachPassage("firm size", "X", "p1"),
            AttachPassage("firm size", "Y", "p2"),
        ]
        before = len(corpus.queries)
        scores = [q(state, a) for a in candidates]
        assert len(corpus.queries) == before and judge.calls == []
        assert scores[2] > scores[0] and scores[3] > scores[1]  # a citation looks better than a search

    def test_a_run_makes_exactly_one_judgment_per_citation_it_chose_and_one_search_per_search(self):
        corpus, judge = SpyCorpus([P1, P2, P3, P4]), StubJudge()
        env = Environment(corpus, judge)

        class Open:
            def propose(self, state):
                from confoundsearch import enumerate_actions

                actions = enumerate_actions(state)
                if not state.done:
                    actions += [Retrieve("larger firms"), Retrieve("profits")]
                    if not state.graph.confounds:
                        actions.append(AddConfound("firm size", frozenset({"firm size"})))
                return actions

        import random

        policy = EpsilonGreedy(LookaheadQ(env.preview), 0.2, random.Random(3), explore_stop=False)
        trajectory = run_episode(env, policy, Open(), IV, DV, max_steps=60)
        events = trajectory.audit.events
        assert len(judge.calls) == sum(e.kind == "judged" for e in events) > 0
        assert len(corpus.queries) == sum(e.kind == "retrieved" for e in events) > 0

    def test_it_still_steers_toward_the_move_that_pays(self, spies):
        env, _, _ = spies
        state = built(env)
        policy = EpsilonGreedy(LookaheadQ(env.preview), epsilon=0.0)
        choice = policy.select(state, [Retrieve("x"), AttachPassage("firm size", "X", "p1"), Stop()])
        assert choice == AttachPassage("firm size", "X", "p1")


def test_the_scripted_run_is_unchanged_by_all_this(spies):
    env, _, _ = spies
    agent = Scripted(*SCRIPT)
    assert run_episode(env, agent, agent, IV, DV).final.done
