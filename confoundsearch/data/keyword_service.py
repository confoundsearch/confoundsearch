"""The keyword service: a read-only vector store reached over HTTPS, through a proxy that holds the real keys.

    db = KeywordService("https://confoundsearch.org", embedder=OpenAIEmbedder())   # your own OpenAI key
    db.query("firm size")                      # a keyword: the service looks it up; no OpenAI call
    db.query("how big a firm is", k=10)        # not a keyword: embedded with your key, then searched

The proxy (devops/proxy/) sits in front of the vector database and holds the keys. `KeywordService` asks it
for neighbours in one call. `HttpVectorStore` is the lower-level view of the same service: a read-only
`VectorStore` (fetch and query), so that `KeywordIndex` and other code written to the interface work on it.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from confoundsearch.data.keyword_index import Embedder, KeywordNotFound, pool_size, without_echoes
from confoundsearch.data.vectors import Match, Metadata, VectorRecord, json_call
from confoundsearch.errors import CorpusError, VectorStoreError
from confoundsearch.http import Transport, urllib_transport
from confoundsearch.logic.graph import Passage

# The address of the ConfoundSearch service (devops/proxy), built into the code so that nothing has to be
# configured. It is the domain the project owns, so it stays the same whatever is deployed behind it. Searches
# made through it go to that server; `--api off` (or CONFOUNDSEARCH_API=off) switches it off, and `--api URL`
# points at another deployment.
DEFAULT_SERVICE_URL = "https://confoundsearch.org"
OFF_WORDS = frozenset({"", "off", "none", "no", "false", "0"})  # any of these as a setting switches it off
FETCH_LIMIT = 10  # ids per fetch request: the proxy refuses more
SERVICE_MAX_K = 100  # most neighbours the proxy returns in one answer
SERVICE_MAX_PAPERS = 50  # most papers the proxy returns in one answer
READ_ONLY = "the keyword service is read-only: it cannot create, write or delete"


def service_url(setting: str | None = None) -> str | None:
    """The service address to use: the setting if there is one, else $CONFOUNDSEARCH_API, else the built-in.

    `off` (or an empty setting) switches the service off: None. So does an empty built-in address.
    """
    chosen = setting if setting is not None else os.environ.get("CONFOUNDSEARCH_API")
    if chosen is None:
        chosen = DEFAULT_SERVICE_URL
    chosen = chosen.strip()
    return None if chosen.casefold() in OFF_WORDS else chosen


class HttpVectorStore:
    def __init__(
        self,
        base_url: str | None = None,
        timeout: float = 30.0,
        transport: Transport = urllib_transport,
        sleep: Callable[[float], None] = time.sleep,
        tries: int = 4,
    ) -> None:
        address = service_url(base_url)
        if address is None:
            raise VectorStoreError(
                "no service address: pass one or set CONFOUNDSEARCH_API (the service may have been switched "
                "off with `off`, or the built-in address removed)"
            )
        self.base_url = address.rstrip("/")
        self.timeout = timeout
        self.transport = transport
        self.sleep = sleep
        self.tries = tries

    def _call(self, path: str, payload: Mapping[str, Any] | None = None) -> Any:
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        return json_call(
            self.transport, f"{self.base_url}{path}", headers, payload, self.timeout, self.sleep, self.tries
        )

    def manifest(self) -> dict[str, Any]:
        """What the service serves: its version, the embedding model and size, where a local copy lives."""
        reply = self._call("/v1/manifest")
        if not isinstance(reply, dict):
            raise VectorStoreError(f"unexpected response: {str(reply)[:200]!r}")
        return reply

    def fetch(self, ids: Sequence[str]) -> list[VectorRecord]:
        found = []
        for start in range(0, len(ids), FETCH_LIMIT):
            reply = self._call("/v1/fetch", {"ids": list(ids[start : start + FETCH_LIMIT])})
            try:
                found += [
                    VectorRecord(str(r["id"]), r["values"], r.get("metadata") or {}) for r in reply["records"]
                ]
            except (KeyError, TypeError, AttributeError) as err:
                raise VectorStoreError(f"unexpected response: {str(reply)[:200]!r}") from err
        return found

    def query(self, vector: Sequence[float], k: int, filter: Metadata | None = None) -> list[Match]:
        if k <= 0:
            return []
        payload: dict[str, Any] = {"vector": [float(x) for x in vector], "k": k}
        if filter:
            payload["filter"] = dict(filter)
        reply = self._call("/v1/query", payload)
        try:
            return [Match(str(m["id"]), float(m["score"]), m.get("metadata") or {}) for m in reply["matches"]]
        except (KeyError, TypeError, ValueError, AttributeError) as err:
            raise VectorStoreError(f"unexpected response: {str(reply)[:200]!r}") from err

    def create_index(self, dimension: int, metric: str = "cosine") -> None:
        raise VectorStoreError(READ_ONLY)

    def upsert(self, records: Sequence[VectorRecord]) -> int:
        raise VectorStoreError(READ_ONLY)

    def update(
        self, id: str, values: Sequence[float] | None = None, metadata: Metadata | None = None
    ) -> None:
        raise VectorStoreError(READ_ONLY)

    def clear(self) -> None:
        raise VectorStoreError(READ_ONLY)


class KeywordService:
    """`db.query("firm size")` against the proxy. A keyword is looked up there and costs nothing.

    The service holds no OpenAI key. For any other phrase, give an `embedder` (such as `OpenAIEmbedder()`,
    which uses your own key): the phrase is embedded on your side and only the vector is sent.
    """

    def __init__(
        self,
        base_url: str | None = None,
        embedder: Embedder | None = None,
        timeout: float = 30.0,
        transport: Transport = urllib_transport,
        sleep: Callable[[float], None] = time.sleep,
        tries: int = 4,
        max_k: int = SERVICE_MAX_K,
    ) -> None:
        self.store = HttpVectorStore(base_url, timeout, transport, sleep, tries)
        self.embedder = embedder
        self.max_k = max_k  # the most neighbours the service returns in one answer (its manifest says)

    def query(
        self,
        name: str,
        k: int = 25,
        min_works: int = 0,
        distinct: bool = False,
        embedder: Embedder | None = None,
    ) -> list[Match]:
        """The k keywords nearest the name, nearest first, not counting the keyword itself."""
        if k <= 0:
            return []
        payload = {"name": name, "k": k, "min_works": min_works, "distinct": distinct}
        try:
            reply = self.store._call("/v1/neighbors", payload)
        except VectorStoreError as err:
            if "HTTP 404" not in str(err):
                raise
            embed = embedder or self.embedder
            if embed is None:
                raise KeywordNotFound(
                    f"no keyword named {name!r}; to search any phrase, give an embedder (an OpenAI key)"
                ) from err
            return self._by_phrase(name, embed, k, min_works, distinct)
        try:
            return [Match(str(m["id"]), float(m["score"]), m.get("metadata") or {}) for m in reply["matches"]]
        except (KeyError, TypeError, ValueError, AttributeError) as err:
            raise VectorStoreError(f"unexpected response: {str(reply)[:200]!r}") from err

    def _by_phrase(self, phrase: str, embed: Embedder, k: int, min_works: int, distinct: bool) -> list[Match]:
        flt = {"works": {"$gte": min_works}} if min_works else None
        wanted = min(pool_size(k), self.max_k) if distinct else k
        found = self.store.query(embed(phrase), wanted, flt)
        return without_echoes(found, phrase, k) if distinct else found[:k]

    def manifest(self) -> dict[str, Any]:
        return self.store.manifest()

    def learn_limits(self) -> int:
        """Take max_k from the service's manifest if it says (else keep the old value). Returns it."""
        try:
            said = self.manifest().get("max_k")
        except VectorStoreError:
            return self.max_k
        if isinstance(said, int) and not isinstance(said, bool) and said > 0:
            self.max_k = said
        return self.max_k


class HttpCorpus:
    """The papers (the proxy's Elasticsearch) as a `Corpus`: `search(query, k)` returns passages.

    It plugs in wherever a corpus does, including `MultiCorpus`, so the agent can search papers through the
    service without holding any Elasticsearch key.
    """

    def __init__(
        self,
        base_url: str | None = None,
        timeout: float = 30.0,
        transport: Transport = urllib_transport,
        sleep: Callable[[float], None] = time.sleep,
        tries: int = 4,
    ) -> None:
        self.store = HttpVectorStore(base_url, timeout, transport, sleep, tries)

    def search(self, query: str, k: int) -> list[Passage]:
        if k <= 0 or not query.strip():
            return []
        try:
            reply = self.store._call("/v1/papers", {"query": query, "k": min(k, SERVICE_MAX_PAPERS)})
        except VectorStoreError as err:
            raise CorpusError(str(err)) from err
        try:
            return [
                Passage(str(p["id"]), p["text"], p.get("title") or "", p.get("url") or "")
                for p in reply["passages"]
            ]
        except (KeyError, TypeError, AttributeError) as err:
            raise CorpusError(f"unexpected response: {str(reply)[:200]!r}") from err
