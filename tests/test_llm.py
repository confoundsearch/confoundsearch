from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import ClassVar

import pytest

from confoundsearch import (
    AddConfound,
    LLMError,
    LLMHighlights,
    LLMJudge,
    LLMMine,
    LLMProposer,
    LLMSynthesis,
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
from confoundsearch.model.llm import DEFAULT_BASE_URL, temperature_for


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


class TestSynthesis:
    PASSAGES: ClassVar[list] = [
        Passage("p1", "Managers shape policies.", "Managing"),
        Passage("p2", "Rivals cut prices.", "Rivalry"),
    ]

    def line(self, **fields):
        return json.dumps({"confound": "CEO style", "side": "Y", "id": "p1", "quote": "q", **fields})

    def read(self, *replies, passages=None, batch=40):
        client = FakeClient(*replies)
        synthesis = LLMSynthesis(client, batch)
        return synthesis(
            "X", "Y", ["CEO style", "market competition"], self.PASSAGES if passages is None else passages
        ), client

    def test_it_reads_json_lines_and_keeps_the_candidates_own_spelling(self):
        reply = "\n".join([self.line(), self.line(confound="Market Competition", side="X", id="p2")])
        found, client = self.read(reply)
        assert found == [("CEO style", "Y", "p1"), ("market competition", "X", "p2")]
        prompt = client.seen[0][0]["content"]
        assert (
            "id: p1" in prompt and "title: Managing" in prompt and "CEO style, market competition" in prompt
        )

    def test_it_drops_anything_it_cannot_check(self):
        reply = "\n".join(
            [
                "```json",
                "not json",
                self.line(id="made-up"),  # a passage that was never retrieved
                self.line(confound="age"),  # a confound nobody proposed
                self.line(side="Z"),
                self.line(confound=7),
                json.dumps({"confound": "CEO style"}),  # missing fields
                json.dumps([1]),  # not an object
                self.line(),
                self.line(),  # said twice
                "```",
            ]
        )
        found, _ = self.read(reply)
        assert found == [("CEO style", "Y", "p1")]

    def test_a_long_list_is_read_in_batches_and_each_passage_is_offered_once(self):
        found, client = self.read(self.line(), self.line(confound="market competition", id="p2"), batch=1)
        assert found == [("CEO style", "Y", "p1"), ("market competition", "Y", "p2")]
        assert len(client.seen) == 2 and "id: p2" not in client.seen[0][0]["content"]

    def test_nothing_to_read_asks_nothing(self):
        found, client = self.read(passages=[], batch=1)
        assert found == [] and client.seen == []


@pytest.mark.parametrize(
    "model, expected",
    [
        ("gpt-4.1-mini", 0.0),
        ("openai/gpt-4.1", 0.0),
        ("gpt-6-astra", None),
        ("openai/gpt-5.4", None),
        (None, 0.0),
    ],
)
def test_the_newer_gpt_models_get_no_temperature(model, expected):
    assert temperature_for(model) == expected


def test_each_kind_of_model_call_says_what_it_is_for():
    from confoundsearch.model.expansion import LLMAttributes, LLMMeasures, LLMMechanisms
    from confoundsearch.model.llm import LLMConfounds, LLMRewordings, LLMSynonyms, current_purpose

    class Seeing:
        def __init__(self, reply):
            self.reply, self.purposes = reply, []

        def chat(self, messages):
            self.purposes.append(current_purpose())
            return self.reply

    terms = Seeing('{"terms": [], "confounds": [], "synonyms": [], "queries": []}')
    LLMConfounds(terms)("X", "Y")
    LLMSynonyms(terms)("Firm performance")
    LLMRewordings(terms)("a query")
    LLMMeasures(terms)("Firm performance")
    LLMMechanisms(terms)("CEO style", "Firm performance")
    LLMMechanisms(terms)("CEO style")
    LLMAttributes(terms)("CEO style")
    assert terms.purposes == [
        "asking the model which confounds of X and Y to look for",
        "asking the model for other names of 'Firm performance'",
        "asking the model for other ways to search for 'a query'",
        "asking the model how papers measure 'Firm performance'",
        "asking the model how CEO style could affect Firm performance",
        "asking the model how CEO style works",
        "asking the model which traits of 'CEO style' papers measure",
    ]
    judge = Seeing("yes")
    LLMJudge(judge).supports(Passage("p", "text"), Vertex("size", frozenset({"size"})), "Y")
    synthesis = Seeing("")
    LLMSynthesis(synthesis)("X", "Y", ["size"], [Passage("p", "text")])
    highlight = Seeing("")
    LLMHighlights(highlight)("size", "Y", [Passage("p", "text")])
    proposer = Seeing("{}")
    LLMProposer(proposer).propose(State.initial("X", "Y"))
    assert judge.purposes == ["asking the model whether a paper supports size \u2192 Y"]
    assert synthesis.purposes == ["asking the model which of 1 papers support each link"]
    assert highlight.purposes == ["asking the model for the sentence that best shows size affects Y"]
    assert proposer.purposes == ["asking the model for the next moves"]
    assert current_purpose() == ""  # and nothing is left set afterwards


class TestLLMMine:
    """Mining: what papers on the outcome say affects it, ordered by how plausibly each also affects X."""

    PAPERS: ClassVar[list[Passage]] = [
        Passage("a", "Older users adopt less.", "Age"),
        Passage("b", "Women and men differ.", "Gender"),
    ]

    def test_it_reads_the_papers_then_scores_the_candidates_against_x(self):
        reading = "\n".join(
            [
                '{"id": "a", "constructs": ["Age", "income"]}',
                '```json\n{"id": "b", "constructs": ["age", "Gender", "ERP adoption"]}',
                "not json",
                '["a list, not an object"]',
                '{"id": "elsewhere", "constructs": ["invented"]}',  # not a paper it was given
                '{"id": "a", "constructs": "not a list"}',
                '{"id": "a", "constructs": [3, " "]}',
            ]
        )
        # candidates in papers-named order: age (2 papers), gender, income
        scoring = (
            '{"i": 0, "score": 1}\n{"i": 1, "score": 3}\n{"i": 2, "score": "high"}\n{"i": 9, "score": 3}'
        )
        client = FakeClient(reading, scoring)
        mined = LLMMine(client)("IT investment", "ERP adoption", self.PAPERS)
        assert mined == ["gender", "age", "income"]  # by score, then by how many papers named it; Y left out
        read, scored = (m[0]["content"] for m in client.seen)
        assert '"ERP adoption"' in read and "id: a" in read and "abstract: Older users adopt less." in read
        assert '"IT investment" (X)' in scored and "0. age\n1. gender\n2. income" in scored

    def test_it_reads_and_scores_in_batches(self):
        papers = [Passage(str(i), "text", "title") for i in range(3)]
        reading = "\n".join(f'{{"id": "{i}", "constructs": ["c{i}"]}}' for i in range(3))
        client = FakeClient(
            reading, "", '{"i": 1, "score": 2}', '{"i": 0, "score": 3}'
        )  # 2 readings, 2 scorings
        mined = LLMMine(client, batch=2, score_batch=2)("x", "y", papers)
        assert len(client.seen) == 4
        assert mined == ["c2", "c1", "c0"]  # c2 scored 3 in the second scoring batch, c1 scored 2

    def test_nothing_read_means_nothing_mined(self):
        client = FakeClient("")
        assert LLMMine(client)("x", "y", self.PAPERS) == []
