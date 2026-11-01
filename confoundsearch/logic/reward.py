"""The reward: R(s_t, a_t) = Phi(g_{t+1}) - Phi(g_t).

Phi(g) increases with (i) the number of distinct, connected confounds and
(ii) the amount of valid supporting evidence for each. Concretely

    Phi(g) = sum over connected confounds C of  1 + log(1 + e(C))

where e(C) is the number of attached passages across C's two edges. A confound
is connected when it has edges to both X and Y. Each confound is worth 1 and
its evidence adds diminishing returns, so many well-supported confounds beat
one heavily-cited one. Phi is never negative and never decreases when
evidence or confounds are added.
"""

from __future__ import annotations

from math import log

from confoundsearch.logic.graph import Graph
from confoundsearch.logic.state import State


def phi(graph: Graph) -> float:
    return sum(1.0 + log(1.0 + graph.evidence(v.label)) for v in graph.connected())


def reward(state: State, next_state: State) -> float:
    return phi(next_state.graph) - phi(state.graph)
