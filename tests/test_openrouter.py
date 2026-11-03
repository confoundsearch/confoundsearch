"""ConfoundSearch with OpenRouter: you bring your own OPENROUTER_API_KEY.

Most of this file runs offline against a stand-in that follows OpenRouter's contract: the /api/v1 path,
Bearer authentication, slug model names, and its error shapes. The `live` tests at the bottom call the real
OpenRouter with your key and are skipped unless you opt in:

    OPENROUTER_API_KEY=sk-or-... pytest -m live --no-cov
    OPENROUTER_MODEL=openai/gpt-4.1-mini   # optional: which model to use (the default is a cheap one)
"""

from __future__ import annotations

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from confoundsearch import (
    AuditedClient,
    AuditLog,
    Environment,
    EpsilonGreedy,
    InMemoryCorpus,
    LLMError,
    LLMJudge,
    LLMProposer,
    LookaheadQ,
    OpenAICompatibleClient,
    Passage,
    Proposer,
    Vertex,
    run_episode,
)
from confoundsearch.interface import cli
from confoundsearch.model.llm import OPENROUTER_URL
from tests.test_docs import load_make_example

ROOT = Path(__file__).parent.parent
DEMO = ROOT / "examples" / "passages.jsonl"
KEY = "sk-or-v1-test-key-0123456789"
SLUG = "openai/gpt-4.1-mini"


class OpenRouterStub:
    """Answers like OpenRouter does, and remembers every request."""

    def __init__(self, key: str = KEY, answer=None, status: int = 200, body=None) -> None:
        self.requests: list[dict] = []
        self.answer = answer or load_make_example().ScriptedModel().chat
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers["Content-Length"])
                headers = {k.lower(): v for k, v in self.headers.items()}
                payload = json.loads(self.rfile.read(length))
                outer.requests.append({"path": self.path, "headers": headers, "body": payload})
                if headers.get("authorization") != f"Bearer {key}":
                    return self.reply(401, {"error": {"code": 401, "message": "No auth credentials found"}})
                if body is not None or status != 200:
                    return self.reply(status, body if body is not None else {"error": {"message": "x"}})
                text = outer.answer(payload["messages"])
                message = {"role": "assistant", "content": text}
                choice = {"finish_reason": "stop", "native_finish_reason": "stop", "message": message}
                usage = {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}
                self.reply(
                    200,
                    {
                        "id": "gen-123",
                        "provider": "OpenAI",
                        "model": payload["model"],
                        "object": "chat.completion",
                        "choices": [choice],
                        "usage": usage,
                    },
                )

            def reply(self, status, data):
                raw = data if isinstance(data, bytes) else json.dumps(data).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(raw)

            def log_message(self, *args):
                pass

        self.httpd = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_port}/api/v1"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def stub():
    made = []

    def make(**kwargs):
        made.append(OpenRouterStub(**kwargs))
        return made[-1]

    yield make
    for item in made:
        item.close()


def client(server, model=SLUG, key=KEY, **kwargs):
    return OpenAICompatibleClient(model, base_url=server.url, api_key=key, **kwargs)


class TestTheContract:
    def test_requests_follow_openrouters_shape(self, stub):
        server = stub()
        assert client(server).chat([{"role": "user", "content": "Reply with the word ok"}])
        (request,) = server.requests
        assert request["path"] == "/api/v1/chat/completions"
        assert request["headers"]["authorization"] == f"Bearer {KEY}"
        assert request["headers"]["content-type"] == "application/json"
        assert request["body"]["model"] == SLUG and request["body"]["messages"][0]["role"] == "user"
        assert request["body"]["temperature"] == 0.0

    @pytest.mark.parametrize(
        "slug",
        [
            "openai/gpt-4.1",
            "anthropic/claude-3.5-sonnet:beta",
            "meta-llama/llama-3.3-70b-instruct:free",
            "x/y",
        ],
    )
    def test_model_slugs_pass_through_untouched(self, stub, slug):
        server = stub()
        client(server, model=slug).chat([{"role": "user", "content": "hi"}])
        assert server.requests[0]["body"]["model"] == slug

    def test_optional_attribution_headers_are_sent_when_given(self, stub):
        server = stub()
        extra = {"HTTP-Referer": "https://example.org/app", "X-Title": "My study"}
        client(server, extra_headers=extra).chat([{"role": "user", "content": "hi"}])
        client(server).chat([{"role": "user", "content": "hi"}])
        assert server.requests[0]["headers"]["http-referer"] == "https://example.org/app"
        assert server.requests[0]["headers"]["x-title"] == "My study"
        assert "http-referer" not in server.requests[1]["headers"]  # nothing personal is sent by default

    def test_extra_headers_cannot_replace_the_key(self, stub):
        server = stub()
        client(server, extra_headers={"Authorization": "Bearer stolen"}).chat(
            [{"role": "user", "content": "hi"}]
        )
        assert server.requests[0]["headers"]["authorization"] == f"Bearer {KEY}"

    def test_a_wrong_key_is_a_clear_error_with_openrouters_message(self, stub):
        server = stub()
        with pytest.raises(LLMError, match=r"HTTP 401.*No auth credentials.*check the API key"):
            client(server, key="sk-or-wrong").chat([{"role": "user", "content": "hi"}])


class TestTheShortcut:
    def test_it_points_at_openrouter_with_the_key_you_pass(self):
        c = OpenAICompatibleClient.openrouter(SLUG, api_key=KEY)
        assert (c.base_url, c.api_key, c.model) == (OPENROUTER_URL, KEY, SLUG)
        assert OPENROUTER_URL == "https://openrouter.ai/api/v1"

    def test_it_reads_the_key_from_openrouter_api_key(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "from-env")
        monkeypatch.setenv("OPENAI_API_KEY", "not-this-one")
        assert OpenAICompatibleClient.openrouter(SLUG).api_key == "from-env"

    def test_it_never_falls_back_to_an_openai_key(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "an-openai-key")
        with pytest.raises(LLMError, match="OPENROUTER_API_KEY"):
            OpenAICompatibleClient.openrouter(SLUG)

    def test_the_url_and_other_settings_can_be_overridden(self, stub):
        server = stub()
        c = OpenAICompatibleClient.openrouter(SLUG, KEY, base_url=server.url, temperature=0.3, timeout=5.0)
        c.chat([{"role": "user", "content": "hi"}])
        assert server.requests[0]["body"]["temperature"] == 0.3 and c.timeout == 5.0


class TestFailuresTheWayOpenRouterReportsThem:
    def ask(self, server):
        return client(server).chat([{"role": "user", "content": "hi"}])

    def test_out_of_credits(self, stub):
        body = {
            "error": {
                "code": 402,
                "message": "Insufficient credits. Add more using https://openrouter.ai/credits",
            }
        }
        with pytest.raises(LLMError, match=r"HTTP 402.*Insufficient credits.*out of credits"):
            self.ask(stub(status=402, body=body))

    def test_rate_limited(self, stub):
        body = {"error": {"code": 429, "message": "Rate limit exceeded"}}
        with pytest.raises(LLMError, match=r"HTTP 429.*Rate limit exceeded.*lower --max-steps"):
            self.ask(stub(status=429, body=body))

    def test_a_provider_outage(self, stub):
        with pytest.raises(LLMError, match=r"HTTP 502.*Provider returned error"):
            self.ask(stub(status=502, body={"error": {"code": 502, "message": "Provider returned error"}}))

    def test_other_statuses_get_no_hint(self, stub):
        with pytest.raises(LLMError, match="HTTP 400") as info:
            self.ask(stub(status=400, body={"error": {"code": 400, "message": "bad model"}}))
        assert "check the API key" not in str(info.value)

    def test_an_error_inside_a_200_response_is_reported_not_called_unexpected(self, stub):
        body = {
            "error": {"code": 429, "message": "Provider is overloaded", "metadata": {"provider_name": "X"}}
        }
        with pytest.raises(
            LLMError, match=r"model provider returned an error: Provider is overloaded \(code 429\)"
        ):
            self.ask(stub(body=body))

    def test_a_plain_text_error_in_a_200_response(self, stub):
        with pytest.raises(LLMError, match="provider returned an error: upstream timed out"):
            self.ask(stub(body={"error": "upstream timed out"}))

    def test_an_error_object_without_a_message_or_code_is_still_shown(self, stub):
        with pytest.raises(LLMError, match=r"returned an error: \{.*weird"):
            self.ask(stub(body={"error": {"weird": True}}))

    @pytest.mark.parametrize(
        "body",
        [{"choices": []}, {"choices": [{"message": {}}]}, {"id": "gen-1"}, b"<html>Cloudflare 502</html>"],
    )
    def test_malformed_responses(self, stub, body):
        with pytest.raises(LLMError, match="unexpected response"):
            self.ask(stub(body=body))

    def test_a_null_reply_from_a_model_that_returned_nothing(self, stub):
        body = {"choices": [{"message": {"role": "assistant", "content": None, "reasoning": "..."}}]}
        with pytest.raises(LLMError, match="not text"):
            self.ask(stub(body=body))


class TestTheCommandLine:
    def args(self, *extra):
        return cli.build_parser().parse_args(["serve", "--model", SLUG, *extra])

    def test_the_flag_selects_openrouter_and_its_key(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", KEY)
        c = cli.build_client(self.args("--openrouter"))
        assert (c.base_url, c.api_key, c.model) == (OPENROUTER_URL, KEY, SLUG)

    def test_with_only_an_openrouter_key_it_uses_openrouter_without_the_flag(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", KEY)
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        c = cli.build_client(self.args())
        assert (c.base_url, c.api_key) == (OPENROUTER_URL, KEY)

    @pytest.mark.parametrize(
        "setting",
        [
            ("env", "OPENAI_API_KEY"),
            ("env", "OPENAI_BASE_URL"),
            ("flag", "--api-key"),
            ("flag", "--base-url"),
        ],
    )
    def test_any_other_choice_of_service_keeps_openai_style_servers(self, monkeypatch, setting):
        monkeypatch.setenv("OPENROUTER_API_KEY", KEY)
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        kind, name = setting
        extra = []
        if kind == "env":
            monkeypatch.setenv(name, "http://elsewhere.test/v1" if name.endswith("URL") else "sk-openai")
        else:
            extra = [name, "http://elsewhere.test/v1" if name == "--base-url" else "sk-openai"]
        assert cli.build_client(self.args(*extra)).base_url != OPENROUTER_URL

    def test_with_no_keys_at_all_it_is_openai_and_says_nothing_yet(self, monkeypatch):
        for name in ("OPENROUTER_API_KEY", "OPENAI_BASE_URL", "OPENAI_API_KEY"):
            monkeypatch.delenv(name, raising=False)
        c = cli.build_client(self.args())
        assert c.base_url == "https://api.openai.com/v1" and c.api_key is None

    def test_a_key_on_the_command_line_beats_the_environment(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "from-env")
        assert cli.build_client(self.args("--openrouter", "--api-key", "from-flag")).api_key == "from-flag"

    def test_a_missing_key_says_what_to_set(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        with pytest.raises(SystemExit, match="OPENROUTER_API_KEY"):
            cli.build_client(self.args("--openrouter"))

    def test_the_url_can_still_be_overridden(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", KEY)
        assert (
            cli.build_client(self.args("--openrouter", "--base-url", "http://proxy.test/v1")).base_url
            == "http://proxy.test/v1"
        )

    def test_the_key_is_read_from_a_dotenv_file(self, monkeypatch, tmp_path, env):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".env").write_text(f"OPENROUTER_API_KEY={KEY}\n")
        seen = []

        class Server:
            server_port = 1

            def serve_forever(self):
                raise KeyboardInterrupt

            def server_close(self):
                pass

        real = cli.build_backend
        monkeypatch.setattr(
            cli, "build_backend", lambda a: seen.append(real(a)) or cli.Backend(env, object(), None)
        )
        monkeypatch.setattr(cli, "make_server", lambda *a, **k: Server())
        cli.main(["serve", "--model", SLUG, "--openrouter", "--corpus-file", str(DEMO)])
        audited = seen[0].client
        assert isinstance(audited, AuditedClient) and audited.client.client.client.api_key == KEY


class TestAWholeRun:
    def test_the_agent_runs_end_to_end_through_openrouter(self, stub):
        server = stub()
        model = AuditedClient(OpenAICompatibleClient.openrouter(SLUG, KEY, base_url=server.url))
        env = Environment(InMemoryCorpus.from_jsonl(DEMO), LLMJudge(model))
        import random

        policy = EpsilonGreedy(LookaheadQ(env.preview), 0.2, random.Random(1), explore_stop=False)
        trajectory = run_episode(
            env, policy, Proposer(LLMProposer(model)), "IT investment", "firm performance", max_steps=80
        )
        trail = trajectory.audit_trail()
        assert trail.complete and trail.intact and trail.supporting("firm size")
        assert {e.data["model"] for e in trail.of_kind("model_call")} == {SLUG}
        assert len(server.requests) == len(trail.of_kind("model_call"))
        assert {r["headers"]["authorization"] for r in server.requests} == {f"Bearer {KEY}"}

    def test_the_command_line_run_uses_your_key_and_never_records_it(
        self, stub, tmp_path, monkeypatch, capsys
    ):
        server = stub()
        monkeypatch.setenv("OPENROUTER_API_KEY", KEY)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        out = tmp_path / "result.html"
        argv = ["run", "--json", "--openrouter", "--base-url", server.url, "--model", SLUG]
        argv += ["--corpus-file", str(DEMO), "--iv", "IT investment", "--dv", "firm performance"]
        argv += ["--max-steps", "80", "--seed", "1", "--no-synthesis", "--out", str(out)]
        assert cli.main(argv) == 0
        printed = capsys.readouterr().out
        result = json.loads(printed)
        assert result["models"] == [SLUG] and result["model_calls"] == len(server.requests) > 0
        assert any(c["supported_on_both_links"] for c in result["confounds"]) and result["audit"]["intact"]
        assert {r["headers"]["authorization"] for r in server.requests} == {f"Bearer {KEY}"}
        for artefact in (printed, out.read_text(), (tmp_path / "result.audit.jsonl").read_text()):
            assert KEY not in artefact  # the key is used to call OpenRouter and written nowhere

    def test_a_failure_mid_run_is_in_the_audit_trail_and_raised(self, stub, tmp_path):
        server = stub(status=402, body={"error": {"code": 402, "message": "Insufficient credits"}})
        model = AuditedClient(OpenAICompatibleClient.openrouter(SLUG, KEY, base_url=server.url))
        env = Environment(InMemoryCorpus.from_jsonl(DEMO), LLMJudge(model))
        policy = EpsilonGreedy(LookaheadQ(env.preview), 0.0)
        log = AuditLog()
        with pytest.raises(LLMError, match="out of credits"):
            run_episode(env, policy, Proposer(LLMProposer(model)), "a", "b", max_steps=5, audit=log)
        kinds = [e.kind for e in log.events]
        assert (
            "model_error" in kinds
            and kinds[-1] == "run_failed"
            and log.events[-1].data["error"].startswith("LLMError")
        )


needs_key = pytest.mark.skipif(
    not os.environ.get("OPENROUTER_API_KEY"), reason="set OPENROUTER_API_KEY to run"
)


@pytest.fixture
def real():
    slug = os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini")
    return OpenAICompatibleClient.openrouter(slug, timeout=90.0)


@pytest.mark.live
@needs_key
class TestLive:
    """Real calls, with your own key. Each costs a fraction of a cent."""

    def test_the_key_works_and_the_model_answers(self, real):
        reply = real.chat([{"role": "user", "content": "Reply with exactly one word: ok"}])
        assert isinstance(reply, str) and reply.strip()

    def test_the_judge_tells_support_from_no_support(self, real):
        judge = LLMJudge(real)
        size = Vertex("firm size", frozenset({"firm size", "head count"}))
        on_topic = Passage(
            "1", "Larger firms invest more in information technology because of scale economies."
        )
        off_topic = Passage("2", "The weather in the mountains was pleasant throughout the summer.")
        assert judge.supports(on_topic, size, "IT investment") is True
        assert judge.supports(off_topic, size, "IT investment") is False

    def test_a_short_real_run_finishes_with_a_sound_audit_trail(self, real):
        model = AuditedClient(real)
        env = Environment(InMemoryCorpus.from_jsonl(DEMO), LLMJudge(model))
        policy = EpsilonGreedy(LookaheadQ(env.preview), 0.1, explore_stop=False)
        trajectory = run_episode(
            env, policy, Proposer(LLMProposer(model)), "IT investment", "firm performance", max_steps=6
        )
        trail = trajectory.audit_trail()
        assert trail.complete and trail.intact and len(trail.of_kind("model_call")) >= 6
        assert {e.data["model"] for e in trail.of_kind("model_call")} == {real.model}
        assert os.environ["OPENROUTER_API_KEY"] not in trajectory.audit.to_jsonl()
