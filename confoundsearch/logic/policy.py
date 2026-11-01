"""Policies pi(a_t | s_t): how the agent chooses among available actions.

Both baselines rank actions with a Q-function Q(s, a), the estimated reward
of taking `a` in `s`.

    EpsilonGreedy   best action with probability 1 - epsilon, uniformly random
                    action with probability epsilon
    Softmax         action sampled with probability proportional to
                    exp(Q / temperature)
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from typing import Callable, Protocol

from confoundsearch.errors import InvalidAction
from confoundsearch.logic.actions import Action, Stop
from confoundsearch.logic.reward import reward
from confoundsearch.logic.state import State

QFunction = Callable[[State, Action], float]


class Policy(Protocol):
    def select(self, state: State, actions: Sequence[Action]) -> Action: ...


def _require(actions: Sequence[Action]) -> None:
    if not actions:
        raise ValueError("no actions to choose from")


class EpsilonGreedy:
    """With probability epsilon explore with a random action, else take the best one.

    By default exploration may pick any action, Stop included, as in the paper.
    With `explore_stop=False` Stop is only taken greedily, so a run uses its
    whole step budget instead of ending on a random draw.
    """

    def __init__(
        self,
        q: QFunction,
        epsilon: float = 0.1,
        rng: random.Random | None = None,
        explore_stop: bool = True,
    ) -> None:
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError(f"epsilon must be in [0, 1], got {epsilon}")
        self.q = q
        self.epsilon = epsilon
        self.rng = rng or random.Random()
        self.explore_stop = explore_stop

    def select(self, state: State, actions: Sequence[Action]) -> Action:
        _require(actions)
        if self.rng.random() < self.epsilon:
            pool = actions if self.explore_stop else [a for a in actions if not isinstance(a, Stop)]
            return self.rng.choice(pool or actions)
        values = [self.q(state, a) for a in actions]
        best = max(values)
        return self.rng.choice([a for a, v in zip(actions, values) if v == best])


class Softmax:
    def __init__(self, q: QFunction, temperature: float = 1.0, rng: random.Random | None = None) -> None:
        if temperature <= 0.0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        self.q = q
        self.temperature = temperature
        self.rng = rng or random.Random()

    def probabilities(self, state: State, actions: Sequence[Action]) -> list[float]:
        _require(actions)
        values = [self.q(state, a) / self.temperature for a in actions]
        top = max(values)
        weights = [math.exp(v - top) for v in values]  # shift by max for stability
        total = sum(weights)
        return [w / total for w in weights]

    def select(self, state: State, actions: Sequence[Action]) -> Action:
        probs = self.probabilities(state, actions)
        return self.rng.choices(list(actions), weights=probs, k=1)[0]


class LookaheadQ:
    """Q(s, a) = one-step reward of the move, plus a small optimism bonus for any action but Stop.

    Stands in for a learned value function. Pass `env.preview`: it simulates the move without searching
    or calling the AI model (a citation is assumed accepted), so scoring candidates costs nothing. With
    `env.step` instead, every candidate would run a real search and a real model call on every step.
    It sees the one-step reward and nothing beyond it. Most actions pay
    nothing immediately (retrieving, adding an edge), so without the bonus
    Stop would tie with all of them and end the search at random. With it,
    Stop is only chosen when exploring is exhausted or the policy explores.
    Actions that fail score 0.
    """

    def __init__(self, step: Callable[[State, Action], State], optimism: float = 1e-3) -> None:
        self.step = step
        self.optimism = optimism

    def __call__(self, state: State, action: Action) -> float:
        try:
            value = reward(state, self.step(state, action))
        except InvalidAction:
            return 0.0
        return value if isinstance(action, Stop) else value + self.optimism
