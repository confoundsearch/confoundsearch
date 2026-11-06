from __future__ import annotations

import random

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from confoundsearch import (
    AddConfound,
    AddEdge,
    AttachPassage,
    Environment,
    EpsilonGreedy,
    InMemoryCorpus,
    Merge,
    Proposer,
    Retrieve,
    Softmax,
    Stop,
    Trajectory,
    phi,
    run_episode,
)
from confoundsearch.logic.episode import Step
from confoundsearch.logic.state import State
from tests.conftest import DV, IV, P1, P2, P3, P4, StubJudge


class Scripted:
    """A policy and proposer in one: replays a fixed list of actions."""

    def __init__(self, *actions):
        self.actions = list(actions)

    def propose(self, state):
        return [self.actions[0]] if self.actions and not state.done else []

    def select(self, state, actions):
        return self.actions.pop(0)


SCRIPT = (
    AddConfound("firm size", frozenset({"firm size"})),
    AddEdge("firm size", "X"),
    AddEdge("firm size", "Y"),
    Retrieve("larger firms"),
    AttachPassage("firm size", "X", "p1"),
    AttachPassage("firm size", "Y", "p2"),
    Stop(),
)


def run(env, script=SCRIPT, **kw):
    agent = Scripted(*script)
    return run_episode(env, agent, agent, IV, DV, **kw)


def test_scripted_episode_builds_the_expected_graph(env):
    t = run(env)
    g = t.final.graph
    assert t.final.done and g.labels == ("firm size",) and g.is_connected("firm size")
    assert g.passages("firm size", "X") == {P1} and g.passages("firm size", "Y") == {P2}


def test_rewards_are_step_by_step_phi_differences(env):
    t = run(env)
    assert [round(s.reward, 6) for s in t.steps][:3] == [0.0, 0.0, 1.0]  # connected on the third edit
    assert t.steps[3].reward == 0.0 and t.steps[-1].reward == 0.0
    assert all(s.reward > 0 for s in t.steps[4:6])


def test_rewards_telescope_to_final_phi(env):
    t = run(env)
    assert t.total_reward == pytest.approx(phi(t.final.graph)) == pytest.approx(t.score)


def test_discounted_return(env):
    t = run(env)
    assert t.discounted_return(1.0) == pytest.approx(t.total_reward)
    assert t.discounted_return(0.0) == pytest.approx(t.steps[0].reward)
    manual = sum(s.reward * 0.9**i for i, s in enumerate(t.steps))
    assert t.discounted_return(0.9) == pytest.approx(manual)


def test_steps_chain_together(env):
    t = run(env)
    assert t.steps[0].state == t.start
    for a, b in zip(t.steps, t.steps[1:]):
        assert a.next_state == b.state


def test_step_budget_stops_the_episode(env):
    t = run(env, max_steps=3)
    assert len(t.steps) == 3 and not t.final.done


def test_zero_budget_returns_the_initial_state(env):
    t = run(env, max_steps=0)
    assert t.steps == [] and t.final == t.start and t.total_reward == 0.0 and t.score == 0.0


def test_empty_trajectory_properties():
    t = Trajectory(State.initial("a", "b"))
    assert t.final == t.start and t.total_reward == 0.0 and t.discounted_return(0.9) == 0.0


def test_stops_when_there_is_nothing_to_do(env):
    class Mute:
        def propose(self, state):
            return []

    t = run_episode(env, EpsilonGreedy(lambda s, a: 0.0), Mute(), IV, DV)
    assert t.steps == []


def test_episode_ends_after_stop_even_if_proposer_offers_more(env):
    class Eager:
        def propose(self, state):
            return [Retrieve("x")]

    class StopFirst:
        def select(self, state, actions):
            return Stop()

    t = run_episode(env, StopFirst(), Eager(), IV, DV)
    assert len(t.steps) == 1 and t.final.done


def test_step_is_a_record():
    s = State.initial("a", "b")
    assert Step(s, Stop(), 0.0, s.stopped()).next_state.done


@given(st.integers(0, 10_000), st.floats(0.0, 1.0))
@settings(max_examples=40, deadline=None)
def test_random_agents_always_produce_valid_telescoping_episodes(seed, epsilon):
    from confoundsearch import LookaheadQ

    env = Environment(InMemoryCorpus([P1, P2, P3, P4]), StubJudge())
    rng = random.Random(seed)

    class Fixed:
        """Closed-form actions plus a couple of fixed open-ended ones."""

        def propose(self, state):
            extra = [] if state.done else [Retrieve("firms"), Retrieve("head count")]
            used = {t for v in state.graph.confounds for t in v.strings} | set(state.graph.labels)
            fresh = next(f"c{i}" for i in range(100) if f"c{i}" not in used)
            if not state.done and len(state.graph.confounds) < 3:
                extra.append(AddConfound(fresh, frozenset({fresh})))
            labels = state.graph.labels
            if not state.done and len(labels) >= 2:
                extra.append(Merge(labels[0], labels[1]))
            return [*Proposer().propose(state), *extra]

    policy = EpsilonGreedy(LookaheadQ(env.preview), epsilon, rng)
    t = run_episode(env, policy, Fixed(), IV, DV, max_steps=40)
    assert t.total_reward == pytest.approx(phi(t.final.graph), abs=1e-9)
    for step in t.steps:
        assert step.reward == pytest.approx(phi(step.next_state.graph) - phi(step.state.graph))


def test_softmax_agent_runs_end_to_end(env):
    from confoundsearch import LookaheadQ

    proposer = Proposer()
    policy = Softmax(LookaheadQ(env.preview), temperature=0.5, rng=random.Random(0))
    t = run_episode(env, policy, proposer, IV, DV, max_steps=10)
    assert t.final.done or len(t.steps) == 10


def test_on_step_sees_every_step_as_it_happens(env):
    seen = []
    agent = Scripted(*SCRIPT)
    t = run_episode(env, agent, agent, IV, DV, on_step=seen.append)
    assert seen == t.steps and len(seen) == len(SCRIPT)


def test_should_stop_is_polled_before_each_step(env):
    calls = []

    def should_stop():
        calls.append(len(calls))
        return len(calls) > 3

    agent = Scripted(*SCRIPT)
    t = run_episode(env, agent, agent, IV, DV, should_stop=should_stop)
    assert len(t.steps) == 3 and not t.final.done and calls == [0, 1, 2, 3]


def test_finalize_sees_the_final_state_before_the_log_is_closed(env):
    from confoundsearch.logic.audit import AuditLog

    log = AuditLog()
    seen = []
    t = run(env, audit=log, finalize=lambda state: seen.append((state, log.events[-1].kind)))
    assert len(seen) == 1 and seen[0][0] == t.final and seen[0][1] != "run_finished"
    assert log.events[-1].kind == "run_finished"
