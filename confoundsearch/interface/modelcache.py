"""Remembering what the language model said, so the same question to it is never paid for twice.

`CachedClient` wraps a chat client. Each call is looked up by the model, its temperature and the exact
messages; a saved reply is returned at once and for free, and anything else goes to the model and is saved.
The replies live in the data directory beside the saved searches (`confoundsearch.data.cache`), one database
per question, so a new pair of variables starts afresh.

    client = CachedClient(OpenAICompatibleClient("gpt-4.1-mini"), cache_dir(), current_question)
    CachedClient(None, cache_dir(), current_question, model="gpt-4.1-mini")   # replay only: no model, no key

A replay is not a fresh judgement by the model, so the audit trail says which calls were replays
(`confoundsearch.logic.audit`). Calls that fail are not saved. The package ships the replies of one recorded
run of the example question, which is what lets `confoundsearch demo` work without any key.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Callable

from confoundsearch.data.cache import WRITES, PaperDatabase, call_key, database_path
from confoundsearch.errors import ReplayMiss
from confoundsearch.logic.audit import mark_replayed
from confoundsearch.model.cost import Spend
from confoundsearch.model.llm import ChatClient, temperature_for


class CachedClient:
    def __init__(
        self,
        client: ChatClient | None,
        folder: Path,
        scope: Callable[[], tuple[str, str] | None],
        log: Callable[..., None] | None = None,
        model: str | None = None,
    ) -> None:
        self.client, self.folder, self.scope, self.log = client, Path(folder), scope, log
        self.model = getattr(client, "model", None) or model
        # a replay has no client to ask, so it assumes what a recording of that model used
        self.temperature = getattr(client, "temperature", temperature_for(self.model))
        self.spend = getattr(client, "spend", None) or Spend(self.model or "")  # a replay costs nothing
        self.hits = self.misses = 0

    def chat(self, messages: Sequence[Mapping[str, str]]) -> str:
        database = PaperDatabase(database_path(self.folder, self.scope()))
        key = call_key(self.model, self.temperature, messages)
        saved = database.load_call(key)
        if saved is not None:
            self.hits += 1
            mark_replayed()
            if self.log:
                self.log("model_cache_hit", model=self.model)
            return saved
        if self.client is None:
            raise ReplayMiss(
                f"no saved reply for this question to {self.model or 'the model'}, and no model to ask: "
                "this replay only covers the recorded example run; set a model key (confoundsearch setup) "
                "to run it live"
            )
        reply = self.client.chat(messages)  # a failure raises here and is not saved
        self.misses += 1
        WRITES.submit(lambda: database.save_call(key, self.model, messages, reply))
        if self.log:
            self.log("model_cache_miss", model=self.model)
        return reply


def find_cached(client: object) -> CachedClient | None:
    """The CachedClient inside a stack of wrappers (audited, announced, ...), if there is one."""
    while client is not None:
        if isinstance(client, CachedClient):
            return client
        client = getattr(client, "client", None)
    return None
