"""The graph g = (V, E, P) from the paper.

V   X, Y and a set of confound vertices; each vertex is a cluster of strings.
E   edges (C_i, X) and (C_i, Y); each edge claims that the confound affects
    the target.
P   maps each edge to the passages that support the claim.

Graphs are immutable. Every edit returns a new graph.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal, Tuple

from confoundsearch.errors import InvalidAction

Target = Literal["X", "Y"]
TARGETS: tuple[Target, ...] = ("X", "Y")
Edge = Tuple[str, Target]


def normalize(text: str) -> str:
    """Canonical form of a string inside a vertex: casefolded, single-spaced."""
    return " ".join(text.casefold().split())


@dataclass(frozen=True)
class Passage:
    """A passage of the corpus D. `title` and `url` are optional and only used for display."""

    id: str
    text: str
    title: str = ""
    url: str = ""

    def __post_init__(self) -> None:
        if not self.id:
            raise InvalidAction("passage id must be non-empty")


@dataclass(frozen=True)
class Vertex:
    """A confound: a labelled cluster of strings ("firm size", "head count")."""

    label: str
    strings: frozenset[str]

    def __post_init__(self) -> None:
        label = self.label.strip()
        strings = frozenset(s for s in map(normalize, self.strings) if s)
        if not label:
            raise InvalidAction("vertex label must be non-empty")
        if not strings:
            raise InvalidAction(f"vertex {label!r} needs at least one string")
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "strings", strings)


@dataclass(frozen=True)
class Graph:
    """The partial graph g_t. Invariants are checked on construction."""

    x: str
    y: str
    confounds: tuple[Vertex, ...] = ()
    edges: frozenset[Edge] = frozenset()
    support: Mapping[Edge, frozenset[Passage]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.x.strip() or not self.y.strip():
            raise InvalidAction("X and Y must be non-empty")
        if normalize(self.x) == normalize(self.y):
            raise InvalidAction("X and Y must differ")
        confounds = tuple(sorted(self.confounds, key=lambda v: v.label))
        labels = [v.label for v in confounds]
        if len(labels) != len(set(labels)):
            raise InvalidAction(f"duplicate vertex labels: {labels}")
        seen: set[str] = set()
        for vertex in confounds:
            clash = seen & vertex.strings
            if clash:
                raise InvalidAction(f"strings {sorted(clash)} appear in two vertices")
            seen |= vertex.strings
        for label, target in self.edges:
            if label not in labels:
                raise InvalidAction(f"edge from unknown confound {label!r}")
            if target not in TARGETS:
                raise InvalidAction(f"edge target must be X or Y, got {target!r}")
        for edge in self.support:
            if edge not in self.edges:
                raise InvalidAction(f"support attached to missing edge {edge}")
        frozen = {e: frozenset(ps) for e, ps in self.support.items() if ps}
        object.__setattr__(self, "confounds", confounds)
        object.__setattr__(self, "support", MappingProxyType(frozen))

    def __hash__(self) -> int:
        return hash((self.x, self.y, self.confounds, self.edges, frozenset(self.support.items())))

    # -- queries -------------------------------------------------------------

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(v.label for v in self.confounds)

    def vertex(self, label: str) -> Vertex:
        for v in self.confounds:
            if v.label == label:
                return v
        raise InvalidAction(f"no confound labelled {label!r}")

    def target_name(self, target: Target) -> str:
        return self.x if target == "X" else self.y

    def has_edge(self, label: str, target: Target) -> bool:
        return (label, target) in self.edges

    def passages(self, label: str, target: Target) -> frozenset[Passage]:
        """P(u, v): passages supporting the edge, empty if there are none."""
        return self.support.get((label, target), frozenset())

    def is_connected(self, label: str) -> bool:
        return self.has_edge(label, "X") and self.has_edge(label, "Y")

    def connected(self) -> tuple[Vertex, ...]:
        """Confounds with an edge to both X and Y."""
        return tuple(v for v in self.confounds if self.is_connected(v.label))

    def evidence(self, label: str) -> int:
        """Number of passages supporting the confound across both of its edges."""
        return sum(len(self.passages(label, t)) for t in TARGETS)

    # -- edits ---------------------------------------------------------------

    def add_confound(self, vertex: Vertex) -> Graph:
        return Graph(self.x, self.y, (*self.confounds, vertex), self.edges, self.support)

    def add_edge(self, label: str, target: Target) -> Graph:
        self.vertex(label)
        if self.has_edge(label, target):
            raise InvalidAction(f"edge {(label, target)} already exists")
        return Graph(self.x, self.y, self.confounds, self.edges | {(label, target)}, self.support)

    def attach(self, label: str, target: Target, passage: Passage) -> Graph:
        edge: Edge = (label, target)
        if edge not in self.edges:
            raise InvalidAction(f"no edge {edge} to attach evidence to")
        if passage in self.passages(label, target):
            raise InvalidAction(f"passage {passage.id!r} already attached to {edge}")
        support = {**self.support, edge: self.passages(label, target) | {passage}}
        return Graph(self.x, self.y, self.confounds, self.edges, support)

    def merge(self, a: str, b: str, label: str | None = None) -> Graph:
        """Collapse confounds a and b into one vertex, carrying edges and evidence."""
        if a == b:
            raise InvalidAction("cannot merge a confound with itself")
        va, vb = self.vertex(a), self.vertex(b)
        new_label = (label if label is not None else a).strip()
        if new_label in set(self.labels) - {a, b}:
            raise InvalidAction(f"label {new_label!r} is already taken")
        merged = Vertex(new_label, va.strings | vb.strings)
        rename = {a: new_label, b: new_label}
        edges = frozenset((rename.get(c, c), t) for c, t in self.edges)
        support: dict[Edge, frozenset[Passage]] = {}
        for (c, t), ps in self.support.items():
            key = (rename.get(c, c), t)
            support[key] = support.get(key, frozenset()) | ps
        kept = tuple(v for v in self.confounds if v.label not in (a, b))
        return Graph(self.x, self.y, (*kept, merged), edges, support)
