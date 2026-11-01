"""OpenAIEmbedder: a phrase to a unit-length vector, with the caller's own OpenAI key."""

import json

import pytest

from confoundsearch import LLMError, OpenAIEmbedder
from confoundsearch.errors import HttpError


def transport(calls, vector=(3.0, 4.0), fail=None, body=None):
    def send(url, headers, data, timeout):
        calls.append((url, headers, json.loads(data)))
        if fail:
            raise fail
        return body if body is not None else json.dumps({"data": [{"embedding": list(vector)}]}).encode()

    return send


def test_it_sends_the_model_size_and_key_and_returns_a_unit_vector():
    calls = []
    embed = OpenAIEmbedder("sk-mine", "text-embedding-3-small", 2, transport=transport(calls))
    assert embed("firm size") == [0.6, 0.8]
    url, headers, payload = calls[0]
    assert url == "https://api.openai.com/v1/embeddings" and headers["Authorization"] == "Bearer sk-mine"
    assert payload == {"model": "text-embedding-3-small", "input": ["firm size"], "dimensions": 2}


def test_the_older_model_cannot_be_shortened_so_no_size_is_sent():
    calls = []
    OpenAIEmbedder("k", "text-embedding-ada-002", transport=transport(calls))("x")
    assert "dimensions" not in calls[0][2]


def test_the_key_and_address_come_from_the_environment(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "from-env")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:9/v1/")
    calls = []
    OpenAIEmbedder(transport=transport(calls))("x")
    assert (
        calls[0][0] == "http://localhost:9/v1/embeddings"
        and calls[0][1]["Authorization"] == "Bearer from-env"
    )


def test_without_a_key_it_says_what_to_do(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(LLMError, match="set OPENAI_API_KEY"):
        OpenAIEmbedder(transport=transport([]))("x")


def test_nothing_to_embed_costs_nothing():
    calls = []
    with pytest.raises(LLMError, match="nothing to embed"):
        OpenAIEmbedder("k", transport=transport(calls))("   ")
    assert calls == []


def test_a_zero_vector_does_not_divide_by_zero():
    assert OpenAIEmbedder("k", dims=2, transport=transport([], vector=(0.0, 0.0)))("x") == [0.0, 0.0]


def test_failures_are_llm_errors():
    with pytest.raises(LLMError, match="HTTP 401"):
        OpenAIEmbedder("k", transport=transport([], fail=HttpError("HTTP 401 from x: bad key", 401)))("x")
    with pytest.raises(LLMError, match="could not reach"):
        from confoundsearch.errors import RemoteError

        OpenAIEmbedder("k", transport=transport([], fail=RemoteError("could not reach x: down")))("x")


@pytest.mark.parametrize("body", [b"not json", b'{"data": []}', b'{"data": [{"embedding": "x"}]}', b"{}"])
def test_unexpected_replies(body):
    with pytest.raises(LLMError, match="unexpected response"):
        OpenAIEmbedder("k", transport=transport([], body=body))("x")


def test_by_default_it_retries_busy_replies():
    attempts = []

    def flaky(url, headers, data, timeout):
        attempts.append(1)
        if len(attempts) < 3:
            raise HttpError("HTTP 429 from x: slow down", 429, 0.0)
        return json.dumps({"data": [{"embedding": [1.0, 0.0]}]}).encode()

    import confoundsearch.model.embeddings as module

    embed = OpenAIEmbedder("k", dims=2, sleep=lambda s: None)
    embed.transport = module.resilient(flaky, attempts=4, sleep=lambda s: None)
    assert embed("x") == [1.0, 0.0] and len(attempts) == 3
