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


# Every setting the program reads from the environment (a developer's `.env` sets these, and `just` loads it).
SETTINGS = (
    "CONFOUNDSEARCH_API",
    "CONFOUNDSEARCH_CACHE",
    "ES_API_KEY",
    "ES_URL",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "OPENALEX_API_KEY",
    "OPENALEX_EMAIL",
    "OPENROUTER_API_KEY",
    "PINECONE_API_KEY",
    "S2_KEY",
)


@pytest.fixture(autouse=True)
def clean_environment(request, monkeypatch):
    """A test starts with none of the program's settings, whatever the developer's shell or `.env` holds.

    Without this, a local `OPENAI_BASE_URL` or a real key changes what a test does (or spends money). The live
    tests (`pytest -m live`) exist to use real keys, so they keep them.
    """
    if request.node.get_closest_marker("live"):
        return
    for name in SETTINGS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def away_from_the_repo(monkeypatch, tmp_path_factory):
    """Run every test from an empty folder, so a developer's real `.env` (keys!) is never read by a test."""
    monkeypatch.chdir(tmp_path_factory.mktemp("cwd"))


@pytest.fixture(autouse=True)
def private_cache(monkeypatch, tmp_path_factory):
    """No test may read or fill the saved searches in the real home folder."""
    monkeypatch.setenv("CONFOUNDSEARCH_CACHE", str(tmp_path_factory.mktemp("cache")))


@pytest.fixture(autouse=True)
def no_built_in_service(monkeypatch):
    """No test may depend on, or reach, the real service address built into the code. A test that wants
    a built-in address sets one itself."""
    from confoundsearch.data import keyword_service

    monkeypatch.setattr(keyword_service, "DEFAULT_SERVICE_URL", "")
    monkeypatch.delenv("CONFOUNDSEARCH_API", raising=False)


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
