from __future__ import annotations

import random
from collections import Counter

import pytest
from hypothesis import given
from hypothesis import strategies as st

from confoundsearch import (
    AddConfound,
    AddEdge,
    EpsilonGreedy,
    LookaheadQ,
    Retrieve,
    Softmax,
    State,
    Stop,
)

S = State.initial("x", "y")
A, B, C = Retrieve("a"), Retrieve("b"), Retrieve("c")
TABLE = {A: 1.0, B: 3.0, C: 2.0}


def q(state, action):
    return TABLE[action]


class TestEpsilonGreedy:
    def test_greedy_picks_the_best_action(self):
        assert EpsilonGreedy(q, epsilon=0.0).select(S, [A, B, C]) == B

    def test_epsilon_one_is_uniform(self):
        pol = EpsilonGreedy(q, epsilon=1.0, rng=random.Random(0))
        counts = Counter(pol.select(S, [A, B, C]) for _ in range(3000))
        assert all(850 < n < 1150 for n in counts.values())

    def test_explores_at_the_requested_rate(self):
        pol = EpsilonGreedy(q, epsilon=0.3, rng=random.Random(1))
        picks = [pol.select(S, [A, B, C]) for _ in range(6000)]
        # P(best) = (1 - e) + e/3 = 0.8
        assert 0.77 < picks.count(B) / len(picks) < 0.83

    def test_ties_are_broken_at_random(self):
        pol = EpsilonGreedy(lambda s, a: 1.0, epsilon=0.0, rng=random.Random(2))
        assert {pol.select(S, [A, B, C]) for _ in range(100)} == {A, B, C}

    def test_seeded_runs_are_reproducible(self):
        run = lambda: [  # noqa: E731
            EpsilonGreedy(q, 0.5, random.Random(7)).select(S, [A, B, C]) for _ in range(20)
        ]
        assert run() == run()

    def test_exploration_may_pick_stop_by_default(self):
        pol = EpsilonGreedy(lambda s, a: 0.0, epsilon=1.0, rng=random.Random(0))
        assert Stop() in {pol.select(S, [A, Stop()]) for _ in range(50)}

    def test_explore_stop_false_never_explores_into_stop(self):
        pol = EpsilonGreedy(lambda s, a: 0.0, epsilon=1.0, rng=random.Random(0), explore_stop=False)
        assert {pol.select(S, [A, B, Stop()]) for _ in range(200)} == {A, B}

    def test_explore_stop_false_still_stops_when_it_is_the_only_action(self):
        pol = EpsilonGreedy(q, epsilon=1.0, explore_stop=False)
        assert pol.select(S, [Stop()]) == Stop()

    def test_explore_stop_false_still_stops_greedily(self):
        pol = EpsilonGreedy(lambda s, a: 5.0 if isinstance(a, Stop) else 1.0, epsilon=0.0, explore_stop=False)
        assert pol.select(S, [A, Stop()]) == Stop()

    @pytest.mark.parametrize("eps", [-0.1, 1.1])
    def test_rejects_bad_epsilon(self, eps):
        with pytest.raises(ValueError, match="epsilon"):
            EpsilonGreedy(q, epsilon=eps)

    def test_rejects_empty_action_list(self):
        with pytest.raises(ValueError, match="no actions"):
            EpsilonGreedy(q).select(S, [])

    def test_default_rng(self):
        assert EpsilonGreedy(q, epsilon=0.0).select(S, [A, B]) == B

    @given(st.lists(st.floats(-50, 50), min_size=1, max_size=8), st.integers(0, 10**6))
    def test_always_returns_one_of_the_actions(self, values, seed):
        actions = [Retrieve(str(i)) for i in range(len(values))]
        table = dict(zip(actions, values))
        pol = EpsilonGreedy(lambda s, a: table[a], 0.2, random.Random(seed))
        assert pol.select(S, actions) in actions


class TestSoftmax:
    def test_probabilities_sum_to_one_and_follow_q(self):
        p = Softmax(q, temperature=1.0).probabilities(S, [A, B, C])
        assert sum(p) == pytest.approx(1.0) and p[1] > p[2] > p[0]

    def test_matches_the_formula(self):
        import math

        p = Softmax(q, temperature=2.0).probabilities(S, [A, B])
        z = math.exp(0.5) + math.exp(1.5)
        assert p == pytest.approx([math.exp(0.5) / z, math.exp(1.5) / z])

    def test_low_temperature_is_nearly_greedy(self):
        p = Softmax(q, temperature=0.01).probabilities(S, [A, B, C])
        assert p[1] > 0.999

    def test_high_temperature_is_nearly_uniform(self):
        p = Softmax(q, temperature=1e6).probabilities(S, [A, B, C])
        assert p == pytest.approx([1 / 3] * 3, abs=1e-3)

    def test_is_stable_for_huge_values(self):
        big = Softmax(lambda s, a: 1e6 * TABLE[a], temperature=1.0)
        p = big.probabilities(S, [A, B, C])
        assert sum(p) == pytest.approx(1.0) and p[1] == pytest.approx(1.0)

    def test_sampling_follows_the_distribution(self):
        pol = Softmax(q, temperature=1.0, rng=random.Random(3))
        expected = pol.probabilities(S, [A, B, C])
        counts = Counter(pol.select(S, [A, B, C]) for _ in range(6000))
        for action, p in zip([A, B, C], expected):
            assert counts[action] / 6000 == pytest.approx(p, abs=0.03)

    @pytest.mark.parametrize("t", [0.0, -1.0])
    def test_rejects_bad_temperature(self, t):
        with pytest.raises(ValueError, match="temperature"):
            Softmax(q, temperature=t)

    def test_rejects_empty_action_list(self):
        with pytest.raises(ValueError, match="no actions"):
            Softmax(q).select(S, [])

    @given(st.lists(st.floats(-1e3, 1e3), min_size=1, max_size=8), st.floats(0.01, 100))
    def test_probabilities_are_a_distribution(self, values, temperature):
        actions = [Retrieve(str(i)) for i in range(len(values))]
        table = dict(zip(actions, values))
        p = Softmax(lambda s, a: table[a], temperature).probabilities(S, actions)
        assert sum(p) == pytest.approx(1.0) and all(0.0 <= x <= 1.0 for x in p)


class TestLookaheadQ:
    def test_scores_actions_by_one_step_reward(self, env):
        s = env.initial_state("IT", "perf")
        s = env.step(s, AddConfound("size", frozenset({"size"})))
        s = env.step(s, AddEdge("size", "X"))
        lookahead = LookaheadQ(env.preview)
        assert lookahead(s, AddEdge("size", "Y")) == pytest.approx(1.0 + lookahead.optimism)  # connects it
        assert lookahead(s, Stop()) == 0.0
        assert lookahead(s, Retrieve("larger")) == pytest.approx(lookahead.optimism)

    def test_invalid_actions_score_zero(self, env):
        s = env.initial_state("IT", "perf")
        assert LookaheadQ(env.preview)(s, AddEdge("ghost", "X")) == 0.0

    def test_drives_a_greedy_policy_to_the_rewarding_action(self, env):
        s = env.initial_state("IT", "perf")
        s = env.step(s, AddConfound("size", frozenset({"size"})))
        s = env.step(s, AddEdge("size", "X"))
        pol = EpsilonGreedy(LookaheadQ(env.preview), epsilon=0.0)
        assert pol.select(s, [Stop(), AddEdge("size", "Y")]) == AddEdge("size", "Y")

    def test_optimism_keeps_the_agent_from_stopping_when_nothing_pays(self, env):
        s = env.initial_state("IT", "perf")
        pol = EpsilonGreedy(LookaheadQ(env.preview), epsilon=0.0, rng=random.Random(0))
        for _ in range(50):
            assert pol.select(s, [Stop(), Retrieve("larger"), Retrieve("firms")]) != Stop()

    def test_stop_is_still_chosen_when_it_is_the_only_option(self, env):
        pol = EpsilonGreedy(LookaheadQ(env.preview), epsilon=0.0)
        assert pol.select(env.initial_state("IT", "perf"), [Stop()]) == Stop()

    def test_optimism_can_be_turned_off(self, env):
        s = env.initial_state("IT", "perf")
        assert LookaheadQ(env.preview, optimism=0.0)(s, Retrieve("larger")) == 0.0
