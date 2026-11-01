"""OpenAI-compatible chat client, plus the judge and proposer built on it.

Any server that speaks POST {base_url}/chat/completions works: OpenAI, vLLM,
Ollama, LM Studio, llama.cpp, OpenRouter, and so on. Only the standard library
is used.

    client = OpenAICompatibleClient(model="gpt-4.1")                  # OpenAI
    client = OpenAICompatibleClient("llama3", base_url="http://localhost:11434/v1")

`base_url` and `api_key` fall back to OPENAI_BASE_URL and OPENAI_API_KEY.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Sequence
from typing import Any, Mapping, Protocol

from confoundsearch.errors import LLMError, RemoteError
from confoundsearch.http import Transport, urllib_transport
from confoundsearch.logic.actions import Action, AddConfound, Merge, Retrieve, enumerate_actions
from confoundsearch.logic.graph import Passage, Vertex, normalize
from confoundsearch.logic.state import State
from confoundsearch.model.cost import Spend

Message = Mapping[str, str]
DEFAULT_BASE_URL = "https://api.openai.com/v1"
OPENROUTER_URL = "https://openrouter.ai/api/v1"
HINTS = {
    401: "check the API key",
    402: "the account is out of credits",
    429: "rate limited: wait a moment, or lower --max-steps",
}


class ChatClient(Protocol):
    def chat(self, messages: Sequence[Message]) -> str: ...


def _with_hint(message: str) -> str:
    match = re.search(r"HTTP (\d{3})", message)
    hint = HINTS.get(int(match.group(1))) if match else None
    return f"{message} ({hint})" if hint else message


def _error_text(error: Any) -> str:
    """The message from an OpenAI-style error object, or the error itself if it is just text."""
    if isinstance(error, dict):
        message = error.get("message") or json.dumps(error)
        return f"{message} (code {error['code']})" if error.get("code") is not None else str(message)
    return str(error)


class OpenAICompatibleClient:
    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float | None = 0.0,
        timeout: float = 60.0,
        transport: Transport = urllib_transport,
        extra_headers: Mapping[str, str] | None = None,
    ) -> None:
        self.model = model
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.temperature = temperature
        self.timeout = timeout
        self.transport = transport
        self.extra_headers = dict(extra_headers or {})
        self.spend = Spend(model)  # what the calls so far have cost, estimated

    @classmethod
    def openrouter(cls, model: str, api_key: str | None = None, **kwargs: Any) -> OpenAICompatibleClient:
        """A client for OpenRouter. `model` is an OpenRouter slug such as "openai/gpt-4.1".

        The key is yours: pass it, or set OPENROUTER_API_KEY.
        """
        key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise LLMError("OpenRouter needs an API key: pass api_key or set OPENROUTER_API_KEY")
        return cls(model, api_key=key, **{"base_url": OPENROUTER_URL, **kwargs})

    def chat(self, messages: Sequence[Message]) -> str:
        headers = {"Content-Type": "application/json", **self.extra_headers}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload: dict[str, Any] = {"model": self.model, "messages": list(messages)}
        if (
            self.temperature is not None
        ):  # None: leave it to the model, which some accept only at their default
            payload["temperature"] = self.temperature
        url = f"{self.base_url}/chat/completions"
        try:
            raw = self.transport(url, headers, json.dumps(payload).encode(), self.timeout)
        except RemoteError as err:
            raise LLMError(_with_hint(str(err))) from err
        try:
            data = json.loads(raw)
            if isinstance(data, dict) and data.get("error"):  # some gateways report failures with a 200
                raise LLMError(f"the model provider returned an error: {_error_text(data['error'])}")
            content = data["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as err:
            raise LLMError(f"unexpected response: {raw[:200]!r}") from err
        if not isinstance(content, str):
            raise LLMError(f"response content is not text: {content!r}")
        self.spend.add(data.get("usage"))
        return content


class LLMJudge:
    """Decides whether a passage supports the claim that a confound affects X or Y."""

    def __init__(self, client: ChatClient) -> None:
        self.client = client

    def supports(self, passage: Passage, source: Vertex, target: str) -> bool:
        names = ", ".join(sorted(source.strings))
        prompt = (
            f'Claim: "{source.label}" (also written: {names}) affects "{target}".\n\n'
            f"Passage:\n{passage.text}\n\n"
            "Does the passage state or directly support this claim? Answer yes or no."
        )
        answer = self.client.chat([{"role": "user", "content": prompt}])
        return answer.strip().casefold().startswith("yes")


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$")


def _strings(reply: str, key: str) -> list[str]:
    """The strings under `key` in a JSON reply (fenced or not), or nothing if the reply is not that."""
    try:
        items = json.loads(_FENCE.sub("", reply.strip())).get(key, [])
    except (ValueError, AttributeError, TypeError):
        return []
    return [item for item in items if isinstance(item, str)] if isinstance(items, list) else []


class LLMRewordings:
    """Other ways to write a search query, in the vocabulary the papers themselves use."""

    def __init__(self, client: ChatClient, count: int = 4) -> None:
        self.client = client
        self.count = count

    def __call__(self, query: str) -> list[str]:
        prompt = (
            f'A search for scientific papers used the query "{query}". Give {self.count} other queries that '
            "would find the same papers. Each is 2 to 4 plain words, no operators or quotes, in the field's "
            "own vocabulary: swap the abstract label for the specific mechanism, measure or setting a paper "
            'would name. Reply with JSON only: {"queries": ["..."]}'
        )
        return _strings(self.client.chat([{"role": "user", "content": prompt}]), "queries")


class LLMConfounds:
    """Asks which confounds to look for: constructs that plausibly influence both variables."""

    def __init__(self, client: ChatClient, count: int = 8) -> None:
        self.client = client
        self.count = count

    def __call__(self, x: str, y: str) -> list[str]:
        prompt = (
            f'A study asks how "{x}" (the independent variable) affects "{y}" (the dependent variable). '
            f"List the {self.count} most important confounds in social science research: constructs that "
            "influence both. Each is a short noun phrase of 1 to 4 words, as papers name it. "
            'Reply with JSON only: {"confounds": ["..."]}'
        )
        return _strings(self.client.chat([{"role": "user", "content": prompt}]), "confounds")


class LLMSynonyms:
    """Other ways the research literature names the same thing."""

    def __init__(self, client: ChatClient, count: int = 3) -> None:
        self.client = client
        self.count = count

    def __call__(self, term: str) -> list[str]:
        prompt = (
            f'List {self.count} synonyms or closely related terms for "{term}" as used in social science '
            "research, the way papers write them (1 to 4 words each). "
            'Reply with JSON only: {"synonyms": ["..."]}'
        )
        return _strings(self.client.chat([{"role": "user", "content": prompt}]), "synonyms")


class LLMProposer:
    """Proposes the open-ended actions: Retrieve queries, AddConfound vertices, and Merges."""

    def __init__(
        self, client: ChatClient, max_queries: int = 3, max_confounds: int = 3, max_merges: int = 2
    ) -> None:
        self.client = client
        self.max_queries = max_queries
        self.max_confounds = max_confounds
        self.max_merges = max_merges

    def propose(self, state: State) -> list[Action]:
        graph = state.graph
        known = ", ".join(graph.labels) or "none yet"
        prompt = (
            f'A study asks how "{graph.x}" affects "{graph.y}". Confounds found so far: {known}.\n'
            "Suggest search queries for a scientific corpus and new confounds, i.e. constructs "
            "that plausibly influence both. "
            "If two confounds found so far are the same idea in different words, list them under "
            '"merges"; otherwise leave it empty. Reply with JSON only:\n'
            '{"queries": ["..."], "confounds": [{"label": "...", "strings": ["..."]}], '
            '"merges": [{"a": "...", "b": "...", "label": "..."}]}'
        )
        reply = self.client.chat([{"role": "user", "content": prompt}])
        try:
            data = json.loads(_FENCE.sub("", reply.strip()))
            queries, confounds = list(data.get("queries", [])), list(data.get("confounds", []))
            merges = list(data.get("merges", []))
        except (ValueError, AttributeError, TypeError):
            return []
        actions: list[Action] = [
            Retrieve(q.strip()) for q in queries[: self.max_queries] if isinstance(q, str) and q.strip()
        ]
        taken = {s for v in graph.confounds for s in v.strings}
        for item in confounds[: self.max_confounds]:
            action = self._confound(item, set(graph.labels), taken)
            if action is not None:
                actions.append(action)
                taken |= action.strings
        pairs: set[frozenset[str]] = set()
        for item in merges[: self.max_merges]:
            merge = self._merge(item, set(graph.labels))
            if merge is not None and frozenset((merge.a, merge.b)) not in pairs:
                pairs.add(frozenset((merge.a, merge.b)))
                actions.append(merge)
        return actions

    @staticmethod
    def _merge(item: object, labels: set[str]) -> Merge | None:
        if not isinstance(item, dict):
            return None
        a, b, label = item.get("a"), item.get("b"), item.get("label")
        if not isinstance(a, str) or not isinstance(b, str) or a not in labels or b not in labels or a == b:
            return None
        if label is None:
            return Merge(a, b)
        if not isinstance(label, str) or not label.strip() or label.strip() in labels - {a, b}:
            return None
        return Merge(a, b, label.strip())

    @staticmethod
    def _confound(item: object, labels: set[str], taken: set[str]) -> AddConfound | None:
        if not isinstance(item, dict):
            return None
        label, strings = item.get("label"), item.get("strings")
        if not isinstance(label, str) or not isinstance(strings, list):
            return None
        cleaned = frozenset(normalize(s) for s in strings if isinstance(s, str) and normalize(s))
        if not label.strip() or not cleaned or label.strip() in labels or cleaned & taken:
            return None
        return AddConfound(label.strip(), cleaned)


class Proposer:
    """Every valid closed-form action, plus the LLM's open-ended suggestions if given one."""

    def __init__(self, llm: LLMProposer | None = None) -> None:
        self.llm = llm

    def propose(self, state: State) -> list[Action]:
        if state.done:
            return []
        extra = self.llm.propose(state) if self.llm else []
        return [*enumerate_actions(state), *extra]
