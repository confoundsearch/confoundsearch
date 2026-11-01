"""Running the MDP: observe, choose, transition, collect reward, until STOP."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

from confoundsearch.logic.actions import Action, Stop
from confoundsearch.logic.audit import (
    AuditLog,
    AuditTrail,
    record_finish,
    record_start,
    record_step,
    recording,
)
from confoundsearch.logic.policy import Policy
from confoundsearch.logic.reward import phi, reward
from confoundsearch.logic.state import State
from confoundsearch.logic.transition import Environment


class ActionSource(Protocol):
    def propose(self, state: State) -> list[Action]: ...


@dataclass(frozen=True)
class Step:
    state: State
    action: Action
    reward: float
    next_state: State


@dataclass
class Trajectory:
    start: State
    steps: list[Step] = field(default_factory=list)
    audit: AuditLog = field(default_factory=AuditLog, repr=False)

    def audit_trail(self) -> AuditTrail:
        """The evidence the run collected and processed, read back from its log."""
        return AuditTrail.from_log(self.audit)

    @property
    def final(self) -> State:
        """s_T, or the last state reached if the step budget ran out first."""
        return self.steps[-1].next_state if self.steps else self.start

    @property
    def total_reward(self) -> float:
        return sum(s.reward for s in self.steps)

    def discounted_return(self, gamma: float) -> float:
        return sum(s.reward * gamma**t for t, s in enumerate(self.steps))

    @property
    def score(self) -> float:
        """Phi of the final graph. Equals total_reward, since rewards telescope."""
        return phi(self.final.graph)


def run_episode(
    env: Environment,
    policy: Policy,
    proposer: ActionSource,
    x: str,
    y: str,
    max_steps: int = 50,
    on_step: Callable[[Step], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
    audit: AuditLog | None = None,
) -> Trajectory:
    """Run one episode and keep its audit log.

    `on_step` sees each step as it happens; `should_stop` is polled before each one. The log is always kept
    (see `confoundsearch.logic.audit`); pass `audit=AuditLog(path)` to also stream it to a file as it goes.
    If the log cannot be written the run fails rather than carry on unrecorded.
    """
    log = audit if audit is not None else AuditLog()
    state = env.initial_state(x, y)
    trajectory = Trajectory(state, audit=log)
    reason = "step_budget"
    with recording(log):
        record_start(log, x, y, max_steps, policy)
        try:
            for number in range(1, max_steps + 1):
                log.step = number
                if should_stop is not None and should_stop():
                    reason = "stopped_by_user"
                    break
                actions = proposer.propose(state)
                if not actions:
                    reason = "no_actions"
                    break
                action = policy.select(state, actions)
                next_state = env.step(state, action)
                step = Step(state, action, reward(state, next_state), next_state)
                trajectory.steps.append(step)
                record_step(log, step)
                if on_step is not None:
                    on_step(step)
                state = next_state
                if isinstance(action, Stop):
                    reason = "stop_action"
                    break
        except BaseException as err:
            log.record("run_failed", error=f"{type(err).__name__}: {err}", steps=len(trajectory.steps))
            raise
        log.step = len(trajectory.steps)
        record_finish(log, reason, len(trajectory.steps), trajectory.final)
    return trajectory
