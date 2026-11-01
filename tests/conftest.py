from __future__ import annotations

import pytest

from confoundsearch import Environment, InMemoryCorpus, Passage, Vertex


@pytest.fixture(scope="session")
def git_identity(tmp_path_factory):
    path = tmp_path_factory.mktemp("git") / "gitconfig"
    path.write_text("[user]\n\tname = Test\n\temail = test@example.org\n[init]\n\tdefaultBranch = main\n")
    return path


@pytest.fixture(autouse=True)
def hermetic_git(monkeypatch, git_identity):
    """Tests that make git commits must not depend on the machine's git settings (CI has no identity)."""
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(git_identity))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")


@pytest.fixture(autouse=True)
def private_cache(monkeypatch, tmp_path_factory):
    """No test may read or fill the saved searches in the real home folder."""
    monkeypatch.setenv("CONFOUNDSEARCH_CACHE", str(tmp_path_factory.mktemp("cache")))


IV = "IT investment"
DV = "firm performance"

P1 = Passage("p1", "larger firms invest more in IT")
P2 = Passage("p2", "larger firms earn higher profits")
P3 = Passage("p3", "head count drives performance")
P4 = Passage("p4", "unrelated text about weather")

SIZE = Vertex("firm size", frozenset({"firm size", "head count"}))
AGE = Vertex("firm age", frozenset({"firm age"}))


class StubJudge:
    """Accepts passages whose id is in `accepted`; records every call."""

    def __init__(self, accepted: set[str] | None = None) -> None:
        self.accepted = {"p1", "p2", "p3"} if accepted is None else accepted
        self.calls: list[tuple[str, str, str]] = []

    def supports(self, passage: Passage, source: Vertex, target: str) -> bool:
        self.calls.append((passage.id, source.label, target))
        return passage.id in self.accepted


@pytest.fixture
def corpus() -> InMemoryCorpus:
    return InMemoryCorpus([P1, P2, P3, P4])


@pytest.fixture
def judge() -> StubJudge:
    return StubJudge()


@pytest.fixture
def env(corpus: InMemoryCorpus, judge: StubJudge) -> Environment:
    return Environment(corpus, judge)
