"""Turn a phrase into a vector with OpenAI, using your own key.

    embed = OpenAIEmbedder()                 # key from OPENAI_API_KEY
    embed("how big a firm is")               # -> 512 unit-length numbers

The model and size must be the ones the keyword index was built with (the service's manifest says which):
vectors from different models cannot be compared. `base_url` also lets it talk to any server that speaks
OpenAI's embeddings protocol.
"""

from __future__ import annotations

import json
import math
import os
import time
from collections.abc import Callable

from confoundsearch.errors import LLMError, RemoteError
from confoundsearch.http import Transport, resilient, urllib_transport

DEFAULT_BASE_URL = "https://api.openai.com/v1"


class OpenAIEmbedder:
    def __init__(
        self,
        api_key: str | None = None,
        model: str = "text-embedding-3-small",
        dims: int = 512,
        base_url: str | None = None,
        timeout: float = 30.0,
        transport: Transport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model
        self.dims = dims
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.timeout = timeout
        self.transport = transport or resilient(urllib_transport, attempts=4, sleep=sleep)

    def __call__(self, text: str) -> list[float]:
        if not self.api_key:
            raise LLMError("embedding needs an OpenAI key: pass api_key or set OPENAI_API_KEY")
        if not text.strip():
            raise LLMError("there is nothing to embed")
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
        payload: dict[str, object] = {"model": self.model, "input": [text]}
        if self.model.startswith("text-embedding-3"):
            payload["dimensions"] = self.dims  # the older ada model cannot be shortened
        try:
            raw = self.transport(
                f"{self.base_url}/embeddings", headers, json.dumps(payload).encode(), self.timeout
            )
        except RemoteError as err:
            raise LLMError(str(err)) from err
        try:
            vector = [float(x) for x in json.loads(raw)["data"][0]["embedding"]]
        except (ValueError, KeyError, IndexError, TypeError) as err:
            raise LLMError(f"unexpected response: {raw[:200]!r}") from err
        length = math.sqrt(sum(x * x for x in vector)) or 1.0
        return [x / length for x in vector]
