from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from confoundsearch import (
    AddConfound,
    LLMError,
    LLMJudge,
    LLMProposer,
    Merge,
    OpenAICompatibleClient,
    Passage,
    Proposer,
    Retrieve,
    State,
    Stop,
    Vertex,
)
from confoundsearch.errors import RemoteError
from confoundsearch.http import urllib_transport
from confoundsearch.model.llm import DEFAULT_BASE_URL


def completion(text: str) -> bytes:
    return json.dumps({"choices": [{"message": {"role": "assistant", "content": text}}]}).encode()


class FakeClient:
    def __init__(self, *replies: str) -> None:
        self.replies, self.seen = list(replies), []

    def chat(self, messages):
        self.seen.append(messages)
        return self.replies.pop(0)


class Server:
    """A local stand-in for an OpenAI-compatible endpoint."""

    def __init__(self, status: int = 200, body: bytes | None = None) -> None:
        self.requests: list[dict] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers["Content-Length"])
                outer.requests.append(
                    {
                        "path": self.path,
                        "headers": dict(self.headers),
                        "body": json.loads(self.rfile.read(length)),
                    }
                )
                self.send_response(status)
                self.end_headers()
                self.wfile.write(body if body is not None else completion("hello"))

            def log_message(self, *args):
                pass

        self.httpd = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_port}/v1"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def server():
    srv = Server()
    yield srv
    srv.close()


class TestClient:
    def test_speaks_the_openai_chat_protocol(self, server):
        client = OpenAICompatibleClient("my-model", base_url=server.url, api_key="sk-test", temperature=0.2)
        assert client.chat([{"role": "user", "content": "hi"}]) == "hello"
        (req,) = server.requests
        assert req["path"] == "/v1/chat/completions"
        assert req["headers"]["Authorization"] == "Bearer sk-test"
        assert req["body"] == {
            "model": "my-model",
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0.2,
        }

    def test_a_temperature_of_none_is_left_out_for_models_that_allow_only_their_default(self, server):
        OpenAICompatibleClient("m", base_url=server.url, temperature=None).chat(
            [{"role": "user", "content": "x"}]
        )
        assert "temperature" not in server.requests[0]["body"]

    def test_works_without_an_api_key(self, server, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        OpenAICompatibleClient("m", base_url=server.url).chat([{"role": "user", "content": "hi"}])
        assert "Authorization" not in server.requests[0]["headers"]

    def test_trailing_slash_in_base_url(self, server):
        OpenAICompatibleClient("m", base_url=server.url + "/").chat([{"role": "user", "content": "x"}])
        assert server.requests[0]["path"] == "/v1/chat/completions"

    def test_reads_environment(self, monkeypatch):
        monkeypatch.setenv("OPENAI_BASE_URL", "http://example.test/v1/")
        monkeypatch.setenv("OPENAI_API_KEY", "env-key")
        c = OpenAICompatibleClient("m")
        assert (c.base_url, c.api_key) == ("http://example.test/v1", "env-key")

    def test_defaults_to_openai(self, monkeypatch):
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        assert OpenAICompatibleClient("m").base_url == DEFAULT_BASE_URL

    def test_explicit_arguments_beat_environment(self, monkeypatch):
        monkeypatch.setenv("OPENAI_BASE_URL", "http://env.test")
        assert OpenAICompatibleClient("m", base_url="http://arg.test").base_url == "http://arg.test"

    def test_http_errors_become_llm_errors(self):
        srv = Server(status=401, body=b'{"error": "bad key"}')
        try:
            with pytest.raises(LLMError, match=r"HTTP 401.*bad key"):
                OpenAICompatibleClient("m", base_url=srv.url).chat([])
        finally:
            srv.close()

    def test_unreachable_server_becomes_llm_error(self):
        with pytest.raises(RemoteError, match="could not reach"):
            urllib_transport("http://127.0.0.1:1/v1/x", {}, b"{}", 1.0)

    @pytest.mark.parametrize(
        "raw",
        [b"not json", b"{}", b'{"choices": []}', b'{"choices": [{"message": {}}]}', b'{"choices": [1]}'],
    )
    def test_malformed_responses(self, raw):
        client = OpenAICompatibleClient("m", transport=lambda *a: raw)
        with pytest.raises(LLMError, match="unexpected response"):
            client.chat([])

    def test_non_text_content(self):
        raw = json.dumps({"choices": [{"message": {"content": None}}]}).encode()
        with pytest.raises(LLMError, match="not text"):
            OpenAICompatibleClient("m", transport=lambda *a: raw).chat([])

    def test_transport_is_injectable(self):
        seen = []
        client = OpenAICompatibleClient(
            "m",
            base_url="http://x/v1",
            api_key="k",
            timeout=3.0,
            transport=lambda url, headers, body, timeout: seen.append((url, timeout)) or completion("ok"),
        )
        assert client.chat([{"role": "user", "content": "q"}]) == "ok" and seen == [
            ("http://x/v1/chat/completions", 3.0)
        ]


class TestJudge:
    SOURCE = Vertex("firm size", frozenset({"firm size", "head count"}))
    PASSAGE = Passage("p1", "Big firms do better.")

    @pytest.mark.parametrize(
        "reply,expected",
        [
            ("Yes", True),
            ("yes, it does", True),
            ("  YES.", True),
            ("No", False),
            ("Not sure", False),
            ("", False),
        ],
    )
    def test_parses_yes_no(self, reply, expected):
        assert LLMJudge(FakeClient(reply)).supports(self.PASSAGE, self.SOURCE, "performance") is expected

    def test_prompt_contains_claim_and_passage(self):
        client = FakeClient("yes")
        LLMJudge(client).supports(self.PASSAGE, self.SOURCE, "performance")
        (messages,) = client.seen
        prompt = messages[0]["content"]
        assert "firm size" in prompt and "head count" in prompt and "performance" in prompt
        assert "Big firms do better." in prompt and messages[0]["role"] == "user"


class TestProposer:
    S = State.initial("IT", "perf")

    def reply(self, **data):
        return json.dumps(data)

    def test_proposes_queries_and_confounds(self):
        client = FakeClient(
            self.reply(queries=["firm size and IT"], confounds=[{"label": "size", "strings": ["Firm Size"]}])
        )
        actions = LLMProposer(client).propose(self.S)
        assert actions == [Retrieve("firm size and IT"), AddConfound("size", frozenset({"firm size"}))]

    def test_prompt_mentions_variables_and_known_confounds(self):
        client = FakeClient(self.reply(queries=[], confounds=[]))
        s = self.S.with_graph(self.S.graph.add_confound(Vertex("size", frozenset({"size"}))))
        LLMProposer(client).propose(s)
        prompt = client.seen[0][0]["content"]
        assert "IT" in prompt and "perf" in prompt and "size" in prompt

    def test_accepts_fenced_json(self):
        client = FakeClient("```json\n" + self.reply(queries=["q"], confounds=[]) + "\n```")
        assert LLMProposer(client).propose(self.S) == [Retrieve("q")]

    @pytest.mark.parametrize("reply", ["not json", "[1, 2]", '"text"', "null", '{"queries": 5}'])
    def test_garbage_yields_no_actions(self, reply):
        assert LLMProposer(FakeClient(reply)).propose(self.S) == []

    def test_respects_limits(self):
        client = FakeClient(
            self.reply(
                queries=["a", "b", "c"],
                confounds=[{"label": f"c{i}", "strings": [f"s{i}"]} for i in range(3)],
            )
        )
        actions = LLMProposer(client, max_queries=2, max_confounds=1).propose(self.S)
        assert actions == [Retrieve("a"), Retrieve("b"), AddConfound("c0", frozenset({"s0"}))]

    def test_drops_bad_queries(self):
        client = FakeClient(self.reply(queries=["ok", "", "  ", 3, None], confounds=[]))
        assert LLMProposer(client).propose(self.S) == [Retrieve("ok")]

    @pytest.mark.parametrize(
        "item",
        [
            "string",
            None,
            {"label": 1, "strings": ["a"]},
            {"label": "a"},
            {"label": "a", "strings": "a"},
            {"label": " ", "strings": ["a"]},
            {"label": "a", "strings": []},
            {"label": "a", "strings": ["", 2]},
        ],
    )
    def test_drops_malformed_confounds(self, item):
        assert LLMProposer(FakeClient(self.reply(queries=[], confounds=[item]))).propose(self.S) == []

    def test_drops_confounds_that_would_be_invalid(self):
        s = self.S.with_graph(self.S.graph.add_confound(Vertex("size", frozenset({"firm size"}))))
        client = FakeClient(
            self.reply(
                queries=[],
                confounds=[
                    {"label": "size", "strings": ["other"]},  # label taken
                    {"label": "scale", "strings": ["Firm  Size"]},  # string taken (after normalizing)
                    {"label": "age", "strings": ["age"]},
                    {"label": "age2", "strings": ["age"]},  # clashes with the proposal just before
                ],
            )
        )
        assert LLMProposer(client).propose(s) == [AddConfound("age", frozenset({"age"}))]

    def two_known(self):
        graph = self.S.graph.add_confound(Vertex("firm size", frozenset({"firm size"})))
        graph = graph.add_confound(Vertex("head count", frozenset({"head count"})))
        return self.S.with_graph(graph)

    def test_proposes_merges_of_known_confounds(self):
        merges = [{"a": "firm size", "b": "head count", "label": "size"}]
        actions = LLMProposer(FakeClient(self.reply(queries=[], confounds=[], merges=merges))).propose(
            self.two_known()
        )
        assert actions == [Merge("firm size", "head count", "size")]

    def test_a_merge_without_a_label_keeps_the_first(self):
        merges = [{"a": "firm size", "b": "head count"}]
        actions = LLMProposer(FakeClient(self.reply(merges=merges))).propose(self.two_known())
        assert actions == [Merge("firm size", "head count")]

    def test_the_prompt_asks_for_merges(self):
        client = FakeClient(self.reply(queries=[], confounds=[]))
        LLMProposer(client).propose(self.two_known())
        assert "same idea" in client.seen[0][0]["content"] and '"merges"' in client.seen[0][0]["content"]

    @pytest.mark.parametrize(
        "item",
        [
            "string",
            None,
            {"a": "firm size"},
            {"a": 1, "b": "head count"},
            {"a": "firm size", "b": "ghost"},
            {"a": "ghost", "b": "head count"},
            {"a": "firm size", "b": "firm size"},
            {"a": "firm size", "b": "head count", "label": 5},
            {"a": "firm size", "b": "head count", "label": " "},
            {"a": "firm size", "b": "head count", "label": "other"},
        ],
    )
    def test_drops_invalid_merges(self, item):
        state = self.two_known()
        graph = state.graph.add_confound(Vertex("other", frozenset({"other"})))
        actions = LLMProposer(FakeClient(self.reply(merges=[item]))).propose(state.with_graph(graph))
        assert actions == []

    def test_a_merge_may_reuse_one_of_its_own_labels(self):
        merges = [{"a": "firm size", "b": "head count", "label": "head count"}]
        actions = LLMProposer(FakeClient(self.reply(merges=merges))).propose(self.two_known())
        assert actions == [Merge("firm size", "head count", "head count")]

    def test_duplicate_merges_and_the_limit(self):
        state = self.two_known()
        graph = state.graph.add_confound(Vertex("a", frozenset({"a"}))).add_confound(
            Vertex("b", frozenset({"b"}))
        )
        state = state.with_graph(graph)
        merges = [
            {"a": "firm size", "b": "head count"}, {"a": "head count", "b": "firm size"},
            {"a": "a", "b": "b"}, {"a": "a", "b": "firm size"},
        ]  # fmt: skip
        actions = LLMProposer(FakeClient(self.reply(merges=merges)), max_merges=3).propose(state)
        assert actions == [Merge("firm size", "head count"), Merge("a", "b")]
        assert len(LLMProposer(FakeClient(self.reply(merges=merges)), max_merges=1).propose(state)) == 1

    def test_proposer_combines_closed_form_and_llm_actions(self):
        client = FakeClient(self.reply(queries=["q"], confounds=[]))
        actions = Proposer(LLMProposer(client)).propose(self.S)
        assert actions == [Stop(), Retrieve("q")]

    def test_proposer_without_llm(self):
        assert Proposer().propose(self.S) == [Stop()]

    def test_proposer_is_silent_when_done_and_does_not_call_the_llm(self):
        client = FakeClient()
        assert Proposer(LLMProposer(client)).propose(self.S.stopped()) == [] and client.seen == []
