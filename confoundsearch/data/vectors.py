"""Vector databases behind one small interface: create an index, insert or update vectors, query.

The interface is `VectorStore`. Two implementations ship with it:

    PineconeVectorStore(api_key, "keywords")   # Pinecone's REST API; standard library only
    InMemoryVectorStore("keywords")            # exact search in memory: tests, small sets, a stand-in

Another database (Qdrant, pgvector, ...) is one more class with the same four methods. Filters use Pinecone's
syntax, `{"works": {"$gte": 300}}`, and the in-memory store understands the same operators, so code written
against one store runs against the other.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol
from urllib.parse import quote, unquote, urlencode

from confoundsearch.errors import HttpError, RemoteError, VectorStoreError
from confoundsearch.http import Transport, urllib_transport

Metadata = Mapping[str, Any]
METRICS = ("cosine", "dotproduct", "euclidean")


@dataclass(frozen=True)
class VectorRecord:
    id: str
    values: Sequence[float]
    metadata: Metadata = field(default_factory=dict)


@dataclass(frozen=True)
class Match:
    id: str
    score: (
        float  # higher is closer for cosine and dotproduct; for euclidean it is a distance, lower is closer
    )
    metadata: Metadata = field(default_factory=dict)


class VectorStore(Protocol):
    def create_index(self, dimension: int, metric: str = "cosine") -> None:
        """Make the index if it does not exist. Calling it again with the same settings changes nothing."""
        ...

    def upsert(self, records: Sequence[VectorRecord]) -> int:
        """Insert the records, replacing any with the same id. Returns how many were written."""
        ...

    def update(
        self, id: str, values: Sequence[float] | None = None, metadata: Metadata | None = None
    ) -> None:
        """Change one existing record: replace its vector and/or set fields of its metadata."""
        ...

    def fetch(self, ids: Sequence[str]) -> list[VectorRecord]:
        """The records with these ids, with their vectors. Ids that do not exist are left out."""
        ...

    def clear(self) -> None:
        """Delete every record (in the namespace, for stores that have them). The index itself stays."""
        ...

    def query(self, vector: Sequence[float], k: int, filter: Metadata | None = None) -> list[Match]:
        """The k records closest to the vector, closest first, optionally only those whose metadata match."""
        ...


# -- in memory -------------------------------------------------------------------------------------------

OPERATORS: dict[str, Callable[[Any, Any], bool]] = {
    "$eq": lambda value, wanted: value == wanted,
    "$ne": lambda value, wanted: value != wanted,
    "$gt": lambda value, wanted: value is not None and value > wanted,
    "$gte": lambda value, wanted: value is not None and value >= wanted,
    "$lt": lambda value, wanted: value is not None and value < wanted,
    "$lte": lambda value, wanted: value is not None and value <= wanted,
    "$in": lambda value, wanted: value in wanted,
    "$nin": lambda value, wanted: value not in wanted,
}


def matches(metadata: Metadata, condition: Metadata) -> bool:
    """Pinecone-style filter: every field must match; a plain value means equality."""
    for name, wanted in condition.items():
        tests = wanted if isinstance(wanted, Mapping) else {"$eq": wanted}
        for operator, operand in tests.items():
            if operator not in OPERATORS:
                raise VectorStoreError(f"unsupported filter operator {operator!r}")
            if not OPERATORS[operator](metadata.get(name), operand):
                return False
    return True


class InMemoryVectorStore:
    def __init__(self, index: str = "index") -> None:
        self.index = index
        self.dimension: int | None = None
        self.metric = "cosine"
        self.records: dict[str, tuple[tuple[float, ...], dict[str, Any]]] = {}

    def create_index(self, dimension: int, metric: str = "cosine") -> None:
        if metric not in ("cosine", "dotproduct"):
            raise VectorStoreError(f"the in-memory store supports cosine and dotproduct, not {metric!r}")
        if dimension <= 0:
            raise VectorStoreError(f"dimension must be positive, got {dimension}")
        if self.dimension is not None and (self.dimension, self.metric) != (dimension, metric):
            raise VectorStoreError(
                f"index {self.index!r} exists with {self.dimension} dimensions, {self.metric}"
            )
        self.dimension, self.metric = dimension, metric

    def _check(self, values: Sequence[float]) -> tuple[float, ...]:
        if self.dimension is None:
            raise VectorStoreError(f"index {self.index!r} does not exist: call create_index first")
        if len(values) != self.dimension:
            raise VectorStoreError(f"expected {self.dimension} dimensions, got {len(values)}")
        return tuple(float(x) for x in values)

    def upsert(self, records: Sequence[VectorRecord]) -> int:
        checked = [(r.id, self._check(r.values), dict(r.metadata)) for r in records]
        for id, values, metadata in checked:
            self.records[id] = (values, metadata)
        return len(checked)

    def update(
        self, id: str, values: Sequence[float] | None = None, metadata: Metadata | None = None
    ) -> None:
        if values is None and metadata is None:
            raise VectorStoreError("update needs new values or new metadata")
        if id not in self.records:
            raise VectorStoreError(f"no record with id {id!r}")
        old_values, old_metadata = self.records[id]
        new_values = old_values if values is None else self._check(values)
        self.records[id] = (new_values, {**old_metadata, **(metadata or {})})

    def clear(self) -> None:
        self.records.clear()

    def fetch(self, ids: Sequence[str]) -> list[VectorRecord]:
        return [
            VectorRecord(id, self.records[id][0], dict(self.records[id][1]))
            for id in ids
            if id in self.records
        ]

    def query(self, vector: Sequence[float], k: int, filter: Metadata | None = None) -> list[Match]:
        query = self._check(vector)
        if k <= 0:
            return []
        norm = math.sqrt(sum(x * x for x in query)) or 1.0
        scored = []
        for id, (values, metadata) in self.records.items():
            if filter and not matches(metadata, filter):
                continue
            dot = sum(a * b for a, b in zip(query, values))
            if self.metric == "cosine":
                dot /= norm * (math.sqrt(sum(x * x for x in values)) or 1.0)
            scored.append(Match(id, dot, dict(metadata)))
        scored.sort(key=lambda m: (-m.score, m.id))
        return scored[:k]


# -- Pinecone --------------------------------------------------------------------------------------------

BATCH = 100  # vectors per upsert request: Pinecone's advice, and well inside its 2 MB request limit
FETCH_BATCH = 100  # ids per fetch request: they travel in the URL, which has a length limit
MAX_ID_LENGTH = 512  # Pinecone's limit on an id, once it is written in ASCII
RETRY_STATUS = {429, 500, 502, 503, 504}


def plan_limit_reached(err: Exception) -> bool:
    """Is this a plan limit (say, a free plan's monthly writes)? Waiting a few seconds does not cure it."""
    text = str(err).casefold()
    return "upgrade your plan" in text or "limit for the current month" in text


class PineconeVectorStore:
    """One Pinecone serverless index, reached over REST. Create it with `create_index`, then use it."""

    def __init__(
        self,
        api_key: str,
        index: str,
        namespace: str = "",
        cloud: str = "aws",
        region: str = "us-east-1",
        host: str | None = None,
        control_url: str = "https://api.pinecone.io",
        api_version: str = "2025-04",
        timeout: float = 60.0,
        transport: Transport = urllib_transport,
        sleep: Callable[[float], None] = time.sleep,
        ready_timeout: float = 300.0,
    ) -> None:
        self.api_key = api_key
        self.index = index
        self.namespace = namespace
        self.cloud = cloud
        self.region = region
        self.host = host.removeprefix("https://") if host else None
        self.control_url = control_url.rstrip("/")
        self.api_version = api_version
        self.timeout = timeout
        self.transport = transport
        self.sleep = sleep
        self.ready_timeout = ready_timeout

    def _call(self, url: str, payload: Mapping[str, Any] | None, tries: int = 5) -> Any:
        """GET when there is no payload, else POST. Retries rate limits and server errors."""
        headers = {
            "Api-Key": self.api_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
            "X-Pinecone-API-Version": self.api_version,
        }
        body = None if payload is None else json.dumps(payload).encode()
        attempt = 0
        while True:
            try:
                raw = self.transport(url, headers, body, self.timeout)
                break
            except RemoteError as err:
                attempt += 1
                retryable = (
                    isinstance(err, HttpError) and err.status in RETRY_STATUS and not plan_limit_reached(err)
                )
                if not (retryable and attempt < tries):
                    raise VectorStoreError(str(err)) from err
                self.sleep(min(2.0 ** (attempt - 1), 30.0))
        try:
            return json.loads(raw) if raw.strip() else {}
        except ValueError as err:
            raise VectorStoreError(f"unexpected response: {raw[:200]!r}") from err

    def create_index(self, dimension: int, metric: str = "cosine") -> None:
        if metric not in METRICS:
            raise VectorStoreError(f"metric must be one of {METRICS}, got {metric!r}")
        spec = {"serverless": {"cloud": self.cloud, "region": self.region}}
        try:
            self._call(
                f"{self.control_url}/indexes",
                {"name": self.index, "dimension": dimension, "metric": metric, "spec": spec},
            )
        except VectorStoreError as err:
            if "HTTP 409" not in str(err):  # 409: the index is already there, which is fine
                raise
        waited = 0.0
        while True:
            info = self._call(f"{self.control_url}/indexes/{self.index}", None)
            if (info.get("status") or {}).get("ready") and info.get("host"):
                self.host = str(info["host"]).removeprefix("https://")
                if info.get("dimension") not in (None, dimension):
                    raise VectorStoreError(f"index {self.index!r} exists with {info['dimension']} dimensions")
                return
            if waited >= self.ready_timeout:
                raise VectorStoreError(f"index {self.index!r} was not ready after {self.ready_timeout:.0f}s")
            self.sleep(2.0)
            waited += 2.0

    def _url(self, path: str) -> str:
        if self.host is None:
            info = self._call(f"{self.control_url}/indexes/{self.index}", None)
            if not info.get("host"):
                raise VectorStoreError(f"index {self.index!r} has no host yet: call create_index first")
            self.host = str(info["host"]).removeprefix("https://")
        return f"https://{self.host}{path}"

    @staticmethod
    def _wire_id(id: str) -> str:
        """Pinecone ids must be ASCII, so anything else is percent-encoded (and decoded again in results)."""
        wire = quote(id, safe="")
        if len(wire) > MAX_ID_LENGTH:
            raise VectorStoreError(
                f"id {id[:40]!r}... is longer than {MAX_ID_LENGTH} characters once encoded"
            )
        return wire

    def _with_namespace(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.namespace:
            payload["namespace"] = self.namespace
        return payload

    def upsert(self, records: Sequence[VectorRecord]) -> int:
        written = 0
        for start in range(0, len(records), BATCH):
            vectors = [
                {
                    "id": self._wire_id(r.id),
                    "values": [float(x) for x in r.values],
                    "metadata": dict(r.metadata),
                }
                for r in records[start : start + BATCH]
            ]
            reply = self._call(self._url("/vectors/upsert"), self._with_namespace({"vectors": vectors}))
            written += int(reply.get("upsertedCount", len(vectors)))
        return written

    def update(
        self, id: str, values: Sequence[float] | None = None, metadata: Metadata | None = None
    ) -> None:
        if values is None and metadata is None:
            raise VectorStoreError("update needs new values or new metadata")
        payload: dict[str, Any] = {"id": self._wire_id(id)}
        if values is not None:
            payload["values"] = [float(x) for x in values]
        if metadata is not None:
            payload["setMetadata"] = dict(metadata)
        self._call(self._url("/vectors/update"), self._with_namespace(payload))

    def fetch(self, ids: Sequence[str]) -> list[VectorRecord]:
        found = []
        for start in range(0, len(ids), FETCH_BATCH):
            pairs = [("ids", self._wire_id(id)) for id in ids[start : start + FETCH_BATCH]]
            if self.namespace:
                pairs.append(("namespace", self.namespace))
            reply = self._call(self._url("/vectors/fetch?" + urlencode(pairs)), None)
            try:
                for item in (reply.get("vectors") or {}).values():
                    found.append(
                        VectorRecord(unquote(str(item["id"])), item["values"], item.get("metadata") or {})
                    )
            except (KeyError, TypeError, AttributeError) as err:
                raise VectorStoreError(f"unexpected response: {str(reply)[:200]!r}") from err
        return found

    def clear(self) -> None:
        try:
            self._call(self._url("/vectors/delete"), self._with_namespace({"deleteAll": True}))
        except VectorStoreError as err:
            if "HTTP 404" not in str(
                err
            ):  # 404: there is no such namespace yet, so there is nothing to delete
                raise

    def query(self, vector: Sequence[float], k: int, filter: Metadata | None = None) -> list[Match]:
        if k <= 0:
            return []
        payload: dict[str, Any] = {"vector": [float(x) for x in vector], "topK": k, "includeMetadata": True}
        if filter:
            payload["filter"] = dict(filter)
        reply = self._call(self._url("/query"), self._with_namespace(payload))
        try:
            return [
                Match(unquote(str(m["id"])), float(m["score"]), m.get("metadata") or {})
                for m in reply["matches"]
            ]
        except (KeyError, TypeError, ValueError) as err:
            raise VectorStoreError(f"unexpected response: {str(reply)[:200]!r}") from err
