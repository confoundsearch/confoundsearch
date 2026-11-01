"""Live integration tests: the real Semantic Scholar API, and the real OpenRouter with your own key.

Skipped by default. Run them with:

    just live                                          # or: pytest -m live --no-cov
    OPENROUTER_API_KEY=sk-or-... just live             # to include the OpenRouter run (a fraction of a cent)
    OPENAI_API_KEY=sk-... OPENAI_MODEL=gpt-4o-mini just live   # the same run, straight to OpenAI
    S2_KEY=...  OPENROUTER_MODEL=openai/gpt-4.1-mini   # optional: a Semantic Scholar key; another model

Semantic Scholar's public API needs no key but is rate limited: if it says so, the test is skipped, not
failed. An unreachable service is a failure. The checks on the results are plain functions, tested offline
below, so the live tests only add the real service.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from confoundsearch import (
    AuditedClient,
    Environment,
    EpsilonGreedy,
    InMemoryCorpus,
    LLMJudge,
    LLMProposer,
    LookaheadQ,
    OpenAICompatibleClient,
    Passage,
    Proposer,
    SemanticScholarCorpus,
    run_episode,
)
from confoundsearch.errors import CorpusError
from confoundsearch.model.llm import DEFAULT_BASE_URL
from tests.test_openrouter import KEY, OpenRouterStub
from tests.test_sources import S2_BODY, Api

QUERY = "firm size information technology investment"


def check_papers(passages: list[Passage], limit: int) -> None:
    """What a search for real papers must give back: real, distinct, readable passages."""
    assert 1 <= len(passages) <= limit
    assert len({p.id for p in passages}) == len(passages)
    for passage in passages:
        assert passage.id and passage.title.strip() and len(passage.text) > 50  # an abstract, not a stub


def search_or_skip(corpus: SemanticScholarCorpus, query: str, k: int) -> list[Passage]:
    try:
        return corpus.search(query, k)
    except CorpusError as err:
        if "HTTP 429" in str(err):
            pytest.skip(
                "Semantic Scholar is rate limiting the public API; set S2_KEY for a quota of your own"
            )
        raise


class TestTheChecksThemselves:
    """Offline: the live tests are only worth running if these checks would catch a bad answer."""

    def papers(self, api, **changes):
        server = api({**S2_BODY, **changes})
        return SemanticScholarCorpus(base_url=server.url).search(QUERY, 5)

    @pytest.fixture
    def api(self):
        made = []

        def make(body):
            made.append(Api(body))
            return made[-1]

        yield make
        for item in made:
            item.close()

    def test_good_papers_pass(self, api):
        good = {
            "data": [
                {"paperId": f"p{i}", "corpusId": i, "title": f"T{i}", "abstract": "x" * 80} for i in range(3)
            ]
        }
        check_papers(self.papers(api, **good), 5)

    def test_no_results_fails(self, api):
        with pytest.raises(AssertionError):
            check_papers(self.papers(api, data=[]), 5)

    @pytest.mark.parametrize("bad", [{"title": ""}, {"abstract": "too short"}])
    def test_a_stub_instead_of_a_real_paper_fails(self, api, bad):
        paper = {"paperId": "a", "corpusId": 1, "title": "T", "abstract": "x" * 80, **bad}
        with pytest.raises(AssertionError):
            check_papers(self.papers(api, data=[paper]), 5)

    def test_more_than_asked_fails(self):
        many = [Passage(str(i), "x" * 80, "T") for i in range(4)]
        with pytest.raises(AssertionError):
            check_papers(many, 3)

    def test_duplicate_ids_fail(self):
        with pytest.raises(AssertionError):
            check_papers([Passage("a", "x" * 80, "T"), Passage("a", "y" * 80, "U")], 5)

    def test_rate_limiting_skips_instead_of_failing(self, api):
        server = Api({"message": "slow down"}, 429)
        try:
            with pytest.raises(pytest.skip.Exception):
                search_or_skip(SemanticScholarCorpus(base_url=server.url), QUERY, 3)
        finally:
            server.close()

    def test_other_errors_still_fail(self):
        server = Api({"error": "boom"}, 500)
        try:
            with pytest.raises(CorpusError, match="HTTP 500"):
                search_or_skip(SemanticScholarCorpus(base_url=server.url), QUERY, 3)
        finally:
            server.close()


@pytest.mark.live
class TestSemanticScholarPublicApi:
    def test_a_real_search_returns_real_papers(self):
        corpus = SemanticScholarCorpus(api_key=os.environ.get("S2_KEY"))
        passages = search_or_skip(corpus, QUERY, 3)
        check_papers(passages, 3)
        print()
        for passage in passages:
            print(f"  {passage.id}  {passage.title[:90]}")


def short_run(model_client, slug: str, secret: str, corpus=None) -> None:
    """A short run through `model_client`; the audit trail must be sound and must not hold `secret`."""
    model = AuditedClient(model_client)
    corpus = corpus or SemanticScholarCorpus(api_key=os.environ.get("S2_KEY"))
    env = Environment(corpus, LLMJudge(model))
    policy = EpsilonGreedy(LookaheadQ(env.preview), 0.1, explore_stop=False)
    try:
        run = run_episode(
            env, policy, Proposer(LLMProposer(model)), "IT investment", "firm performance", max_steps=8
        )
    except CorpusError as err:
        if "HTTP 429" in str(err):
            pytest.skip("Semantic Scholar is rate limiting the public API; set S2_KEY")
        raise
    trail = run.audit_trail()
    assert trail.complete and trail.intact
    assert {e.data["model"] for e in trail.of_kind("model_call")} == {slug}
    searches = trail.of_kind("retrieved")
    assert searches, "the agent never searched"
    found = [p for event in searches for p in event.data["new_passages"]]
    if found:
        check_papers([Passage(p["id"], p["text"], p["title"]) for p in found[:3]], 3)
    assert secret not in run.audit.to_jsonl()
    calls = len(trail.of_kind("model_call"))
    print(f"\n  {calls} model calls, {len(found)} papers retrieved, score {run.score:.2f}")


class TestTheShortRunItself:
    """Offline: the same helper, against the OpenRouter stand-in and a made-up corpus."""

    def test_it_passes_for_a_sound_run(self):
        stub = OpenRouterStub()
        try:
            client = OpenAICompatibleClient("m", base_url=stub.url, api_key=KEY)
            corpus = InMemoryCorpus([Passage("a", "x" * 80, "Title A")])
            short_run(client, "m", KEY, corpus)
        finally:
            stub.close()


@pytest.mark.live
@pytest.mark.skipif(not os.environ.get("OPENROUTER_API_KEY"), reason="set OPENROUTER_API_KEY to run")
class TestRealPapersAndARealModel:
    def test_a_short_run_searches_semantic_scholar_through_openrouter_and_leaves_a_sound_audit_trail(self):
        slug = os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini")
        client = OpenAICompatibleClient.openrouter(slug, timeout=90.0)
        short_run(client, slug, os.environ["OPENROUTER_API_KEY"])


@pytest.mark.live
@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="set OPENAI_API_KEY to run")
class TestRealPapersAndOpenAI:
    def test_a_short_run_searches_semantic_scholar_through_openai_and_leaves_a_sound_audit_trail(self):
        slug = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        client = OpenAICompatibleClient(slug, base_url=DEFAULT_BASE_URL, timeout=90.0)
        short_run(client, slug, os.environ["OPENAI_API_KEY"])


# ---- the skill, driven by a real Claude Code ----------------------------------------------------------

SKILL_PROMPT = (
    "Use the confoundsearch skill to look for confounds between IT investment (X) and firm performance (Y). "
    "I approve the cost. Use --corpus semanticscholar, --model {model} and --max-steps 5, and write "
    "confoundsearch-result.html in the current directory."
)


def stream_events(stream: str) -> list[dict]:
    """The JSON events in Claude Code's `--output-format stream-json` output; other lines are skipped."""
    events = []
    for line in stream.splitlines():
        try:
            events.append(json.loads(line))
        except ValueError:
            continue
    return events


def tool_calls(stream: str) -> list[tuple[str, dict]]:
    """Every (tool name, input) Claude Code used."""
    calls = []
    for item in stream_events(stream):
        content = item.get("message", {}).get("content") if item.get("type") == "assistant" else None
        for block in content if isinstance(content, list) else []:
            if block.get("type") == "tool_use":
                calls.append((block["name"], block.get("input", {})))
    return calls


def check_skill_run(stream: str, workdir: Path) -> None:
    """The skill was used, it ran the tool, the tool left its files, and Claude Code reported no error."""
    calls = tool_calls(stream)
    assert any(n == "Skill" and "confoundsearch" in str(i.get("skill", "")) for n, i in calls), "skill unused"
    assert any(n == "Bash" and "confoundsearch run" in str(i.get("command", "")) for n, i in calls), "no run"
    assert (workdir / "confoundsearch-result.html").stat().st_size > 0
    assert (workdir / "confoundsearch-result.audit.jsonl").stat().st_size > 0
    results = [e for e in stream_events(stream) if e.get("type") == "result"]
    assert results and not results[-1].get("is_error"), "Claude Code reported an error"


def event(**fields) -> str:
    return json.dumps(fields)


def assistant(*blocks: dict) -> str:
    return event(type="assistant", message={"content": list(blocks)})


def use(name: str, **inputs) -> dict:
    return {"type": "tool_use", "name": name, "input": inputs}


class TestTheSkillChecksThemselves:
    """Offline: what a good Claude Code run looks like, and what each check catches."""

    GOOD = "\n".join(
        [
            event(type="system"),
            "not json",
            assistant({"type": "text", "text": "hi"}, use("Skill", skill="confoundsearch")),
            event(type="assistant", message={"content": "plain text"}),
            assistant(use("Bash", command="confoundsearch run --json --model gpt-4o-mini --iv a --dv b")),
            event(type="result", is_error=False),
        ]
    )

    @pytest.fixture
    def workdir(self, tmp_path):
        (tmp_path / "confoundsearch-result.html").write_text("<html>")
        (tmp_path / "confoundsearch-result.audit.jsonl").write_text("{}")
        return tmp_path

    def test_a_good_run_passes(self, workdir):
        check_skill_run(self.GOOD, workdir)
        assert [n for n, _ in tool_calls(self.GOOD)] == ["Skill", "Bash"]

    def test_it_fails_if_the_skill_was_not_used(self, workdir):
        stream = self.GOOD.replace('"confoundsearch"}', '"other"}')
        with pytest.raises(AssertionError, match="skill unused"):
            check_skill_run(stream, workdir)

    def test_it_fails_if_the_tool_was_not_run(self, workdir):
        with pytest.raises(AssertionError, match="no run"):
            check_skill_run(self.GOOD.replace("confoundsearch run", "echo"), workdir)

    @pytest.mark.parametrize("name", ["confoundsearch-result.html", "confoundsearch-result.audit.jsonl"])
    def test_it_fails_if_a_file_is_missing(self, workdir, name):
        (workdir / name).unlink()
        with pytest.raises(FileNotFoundError):
            check_skill_run(self.GOOD, workdir)

    def test_it_fails_if_claude_code_reports_an_error_or_no_result(self, workdir):
        with pytest.raises(AssertionError, match="reported an error"):
            check_skill_run(self.GOOD.replace('"is_error": false', '"is_error": true'), workdir)
        with pytest.raises(AssertionError, match="reported an error"):
            check_skill_run("\n".join(self.GOOD.splitlines()[:-1]), workdir)


@pytest.mark.live
@pytest.mark.skipif(not shutil.which("claude"), reason="needs Claude Code (the `claude` command)")
@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="set OPENAI_API_KEY to run")
class TestTheSkillInClaudeCode:
    def test_claude_code_finds_the_skill_runs_it_and_the_audit_trail_verifies(self, tmp_path):
        model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        subprocess.run(
            [sys.executable, "-m", "confoundsearch", "install-skill", "--project"], cwd=tmp_path, check=True
        )
        done = subprocess.run(
            [
                "claude",
                "-p",
                SKILL_PROMPT.format(model=model),
                "--output-format",
                "stream-json",
                "--verbose",
                "--allowedTools",
                "Skill",
                "Bash",
                "--max-budget-usd",
                "1",
            ],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=900,
        )
        assert done.returncode == 0, done.stderr
        check_skill_run(done.stdout, tmp_path)
        audit = subprocess.run(
            [sys.executable, "-m", "confoundsearch", "audit", "confoundsearch-result.audit.jsonl"],
            cwd=tmp_path,
            capture_output=True,
            text=True,
        )
        assert audit.returncode == 0, audit.stdout + audit.stderr


@pytest.mark.live
@pytest.mark.skipif(
    not shutil.which("uv"), reason="needs uv, which fetches the other Pythons (pip install uv)"
)
class TestTheLatestReleaseOnEveryPython:
    def test_it_installs_and_works_on_python_3_8_to_3_13(self):
        import importlib.util

        path = Path(__file__).resolve().parent.parent / "devops" / "tools" / "python_matrix.py"
        spec = importlib.util.spec_from_file_location("python_matrix_live", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["python_matrix_live"] = module
        spec.loader.exec_module(module)
        results = module.matrix("confoundsearch")
        failed = {python: why for python, why in results.items() if why != "ok"}
        assert not failed, module.report("confoundsearch", results)
