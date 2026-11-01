"""Actions: pure data describing what the agent can do, one module per action.

Evidence    retrieve.py        Retrieve        search the corpus for passages
Graph       add_confound.py    AddConfound     add a vertex
            merge.py           Merge           collapse two vertices into one
            add_edge.py        AddEdge         claim that a confound affects X or Y
            attach_passage.py  AttachPassage   cite an observed passage for an edge
Termination stop.py            Stop            end the search

Each action carries its own transition effect in `apply(state, env)`.
`enumerate_actions` lists the closed-form actions valid in a state (Stop, AddEdge,
AttachPassage); Retrieve, AddConfound and Merge are open-ended and come from a proposer.
"""

from __future__ import annotations

from typing import Union

from confoundsearch.logic.actions.add_confound import AddConfound
from confoundsearch.logic.actions.add_edge import AddEdge
from confoundsearch.logic.actions.attach_passage import AttachPassage
from confoundsearch.logic.actions.enumerate import enumerate_actions
from confoundsearch.logic.actions.merge import Merge
from confoundsearch.logic.actions.retrieve import Retrieve
from confoundsearch.logic.actions.stop import Stop

Action = Union[Retrieve, AddConfound, Merge, AddEdge, AttachPassage, Stop]
ACTIONS: tuple[type[Action], ...] = (Retrieve, AddConfound, Merge, AddEdge, AttachPassage, Stop)

__all__ = [
    "ACTIONS",
    "Action",
    "AddConfound",
    "AddEdge",
    "AttachPassage",
    "Merge",
    "Retrieve",
    "Stop",
    "enumerate_actions",
]
