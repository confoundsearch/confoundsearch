"""The keyword proxy: an AWS Lambda behind API Gateway that lets anyone read the keyword index.

It holds two secrets, a read-only Pinecone key and a read-only Elasticsearch key, so that the package does not
have to, and answers these requests and refuses everything else:

    GET  /v1/health     {"ok": true}
    GET  /v1/manifest   what is served: version, embedding model, dimensions, keyword count, local-copy URL
    POST /v1/neighbors  {"name": "firm size", "k": 25, "min_works": 500, "distinct": true}  -> keywords
    POST /v1/query      {"vector": [...], "k": 25, "filter": {"works": {"$gte": 500}}}  -> keywords
    POST /v1/fetch      {"ids": ["firm-size"]}                   -> the stored vectors (up to 10 ids)
    POST /v1/papers     {"query": "family firms decentralization", "k": 10}   -> papers (Elasticsearch)

/v1/neighbors takes the name of a keyword, which is looked up in the index. Anything else is a phrase the
service cannot turn into a vector, since it holds no model key: the caller embeds it with their own OpenAI key
(the manifest says which model and size) and sends the vector to /v1/query. Every input is checked before the
database is asked: a vector must have exactly the right length and only finite numbers, k is 1 to MAX_K, and
the only filter is a number comparison on "works". The database's own error messages are logged here and never
sent back. Settings are environment variables (see template.yaml).
"""

from __future__ import annotations

import base64
import importlib
import json
import math
import os
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(
    0, str(Path(__file__).resolve().parent)
)  # the Lambda zip puts confoundsearch/ beside this file

from confoundsearch.data.corpus import Corpus  # noqa: E402
from confoundsearch.data.es import ElasticsearchCorpus  # noqa: E402
from confoundsearch.data.keyword_index import KeywordIndex, KeywordNotFound  # noqa: E402
from confoundsearch.data.vectors import PineconeVectorStore, VectorStore  # noqa: E402
from confoundsearch.errors import RemoteError, VectorStoreError  # noqa: E402

VERSION = 1
MAX_BODY = 100_000  # bytes; a 512-number vector is about 10,000
MAX_NAME = 200  # characters: a keyword name is short
MAX_QUERY = 300  # characters of a paper search
MAX_PAPERS = 50  # papers in one answer: each carries an abstract
FETCH_LIMIT = 10
MAX_ID_LENGTH = 512
FILTER_OPERATORS = ("$gte", "$lte", "$gt", "$lt")
PUBLIC_METADATA = ("name", "works")  # the only metadata fields that leave the service


@dataclass(frozen=True)
class Config:
    dims: int = 512
    max_k: int = 100
    model: str = "text-embedding-3-small"
    count: int | None = None
    file_url: str | None = None
    papers: bool = False  # is paper search (Elasticsearch) switched on?


def config_from_env(env: dict[str, str] | None = None) -> Config:
    env = dict(os.environ) if env is None else env
    return Config(
        dims=int(env.get("EMBED_DIMS", "512")),
        max_k=int(env.get("MAX_K", "100")),
        model=env.get("EMBED_MODEL", "text-embedding-3-small"),
        count=int(env["KEYWORD_COUNT"]) if env.get("KEYWORD_COUNT") else None,
        file_url=env.get("LOCAL_COPY_URL") or None,
        papers=bool(env.get("ELASTIC_URL")),
    )


class BadRequest(Exception):
    """The caller asked for something the service does not do; the message is safe to send back."""

    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


def reply(status: int, body: dict[str, Any]) -> dict[str, Any]:
    return {
        "statusCode": status,
        "headers": {"content-type": "application/json", "cache-control": "no-store"},
        "body": json.dumps(body),
    }


# -- checking what the caller sent -------------------------------------------------------------------------


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def is_whole(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def parse_body(event: dict[str, Any]) -> dict[str, Any]:
    raw = event.get("body") or ""
    if event.get("isBase64Encoded"):
        try:
            raw = base64.b64decode(raw, validate=True).decode("utf-8")
        except ValueError as err:
            raise BadRequest("the body is not valid base64 text") from err
    if len(raw.encode("utf-8")) > MAX_BODY:
        raise BadRequest("the request is too large", 413)
    try:
        body = json.loads(raw)
    except ValueError as err:
        raise BadRequest("the body is not valid JSON") from err
    if not isinstance(body, dict):
        raise BadRequest("the body must be a JSON object")
    return body


def check_vector(value: Any, dims: int) -> list[float]:
    if not isinstance(value, list) or len(value) != dims:
        raise BadRequest(f"vector must be a list of exactly {dims} numbers")
    if not all(is_number(x) for x in value):
        raise BadRequest("vector must contain only finite numbers")
    return [float(x) for x in value]


def check_k(value: Any, max_k: int) -> int:
    if not is_whole(value) or not 1 <= value <= max_k:
        raise BadRequest(f"k must be a whole number from 1 to {max_k}")
    return int(value)


def check_filter(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) - {"works"}:
        raise BadRequest('the only filter allowed is on "works"')
    tests = value.get("works")
    if not isinstance(tests, dict) or not tests or set(tests) - set(FILTER_OPERATORS):
        raise BadRequest(f'"works" takes a comparison: {", ".join(FILTER_OPERATORS)}')
    if not all(is_number(x) for x in tests.values()):
        raise BadRequest('the "works" comparison must be a number')
    return {"works": dict(tests)}


def check_ids(value: Any) -> list[str]:
    if not isinstance(value, list) or not 1 <= len(value) <= FETCH_LIMIT:
        raise BadRequest(f"ids must be a list of 1 to {FETCH_LIMIT} keyword ids")
    if not all(isinstance(i, str) and 1 <= len(i) <= MAX_ID_LENGTH for i in value):
        raise BadRequest(f"each id must be text of 1 to {MAX_ID_LENGTH} characters")
    return list(value)


def check_name(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BadRequest("name must be some text")
    if len(value) > MAX_NAME:
        raise BadRequest(f"name must be at most {MAX_NAME} characters")
    return value.strip()


def check_query(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BadRequest("query must be some text")
    if len(value) > MAX_QUERY:
        raise BadRequest(f"query must be at most {MAX_QUERY} characters")
    return value.strip()


def check_flag(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise BadRequest(f"{name} must be true or false")
    return value


def public(metadata: Any) -> dict[str, Any]:
    return {k: metadata[k] for k in PUBLIC_METADATA if isinstance(metadata, dict) and k in metadata}


def matches_body(matches: Any) -> list[dict[str, Any]]:
    return [{"id": m.id, "score": float(m.score), "metadata": public(m.metadata)} for m in matches]


# -- the routes --------------------------------------------------------------------------------------------


def handle(
    event: dict[str, Any],
    get_store: Callable[[], VectorStore],
    config: Config,
    log: Callable[[str], None] = print,
    get_corpus: Callable[[], Corpus | None] = lambda: None,
) -> dict[str, Any]:
    """One API Gateway HTTP API (payload 2.0) request in, one response out. Never raises."""
    started = time.monotonic()
    http = (event.get("requestContext") or {}).get("http") or {}
    method, path = str(http.get("method", "")).upper(), str(event.get("rawPath") or http.get("path") or "")
    try:
        response = route(method, path, event, get_store, config, get_corpus)
    except BadRequest as err:
        response = reply(err.status, {"error": str(err)})
    except KeywordNotFound:
        response = reply(
            404,
            {"error": "no keyword has that name; embed the phrase yourself and send the vector to /v1/query"},
        )
    except RemoteError as err:  # the vector database or Elasticsearch failed
        log(json.dumps({"event": "database_error", "path": path, "detail": str(err)[:500]}))
        response = reply(502, {"error": "the database could not answer; try again shortly"})
    except Exception as err:  # noqa: BLE001 - a Lambda must answer, whatever went wrong
        log(json.dumps({"event": "unexpected_error", "path": path, "type": type(err).__name__}))
        response = reply(500, {"error": "something went wrong"})
    log(json.dumps({"event": "request", "method": method, "path": path, "status": response["statusCode"],
                    "ms": round((time.monotonic() - started) * 1000)}))  # fmt: skip
    return response


def route(
    method: str,
    path: str,
    event: dict[str, Any],
    get_store: Callable[[], VectorStore],
    config: Config,
    get_corpus: Callable[[], Corpus | None],
) -> dict[str, Any]:
    if path == "/v1/health" and method == "GET":
        return reply(200, {"ok": True})
    if path == "/v1/manifest" and method == "GET":
        return reply(
            200,
            {"version": VERSION, "model": config.model, "dims": config.dims,
             "count": config.count, "file": config.file_url, "papers": config.papers},
        )  # fmt: skip
    if path == "/v1/neighbors" and method == "POST":
        body = parse_body(event)
        name = check_name(body.get("name"))
        k = check_k(body.get("k", min(25, config.max_k)), config.max_k)
        min_works = body.get("min_works", 0)
        if not is_whole(min_works) or min_works < 0:
            raise BadRequest("min_works must be a whole number, 0 or more")
        distinct = check_flag(body.get("distinct", False), "distinct")
        found = KeywordIndex(get_store()).query(name, k, min_works, distinct=distinct)
        return reply(200, {"matches": matches_body(found)})
    if path == "/v1/papers" and method == "POST":
        body = parse_body(event)
        query = check_query(body.get("query"))
        limit = min(config.max_k, MAX_PAPERS)
        k = check_k(body.get("k", min(10, limit)), limit)
        corpus = get_corpus()
        if corpus is None:
            raise BadRequest("paper search is not switched on here", 404)
        papers = corpus.search(query, k)
        return reply(
            200,
            {"passages": [{"id": p.id, "title": p.title, "text": p.text, "url": p.url} for p in papers]},
        )
    if path == "/v1/fetch" and method == "POST":
        ids = check_ids(parse_body(event).get("ids"))
        records = get_store().fetch(ids)
        return reply(
            200,
            {"records": [{"id": r.id, "values": [float(x) for x in r.values], "metadata": public(r.metadata)}
                         for r in records]},
        )  # fmt: skip
    if path == "/v1/query" and method == "POST":
        body = parse_body(event)
        vector = check_vector(body.get("vector"), config.dims)
        k = check_k(body.get("k", min(25, config.max_k)), config.max_k)
        found = get_store().query(vector, k, check_filter(body.get("filter")))
        return reply(200, {"matches": matches_body(found)})
    if path in ("/v1/health", "/v1/manifest", "/v1/neighbors", "/v1/papers", "/v1/fetch", "/v1/query"):
        raise BadRequest("that method is not allowed here", 405)
    raise BadRequest("not found", 404)


# -- the Lambda entry point --------------------------------------------------------------------------------

_store: VectorStore | None = None
_corpus: Corpus | None = None
_corpus_ready = False


def read_secret(
    key_var: str, parameter_var: str, env: dict[str, str] | None = None, ssm: Any = None
) -> str | None:
    """A secret: from the environment if set (tests), else from AWS Systems Manager (encrypted), else None."""
    env = dict(os.environ) if env is None else env
    if env.get(key_var):
        return env[key_var]
    name = env.get(parameter_var)
    if not name:
        return None
    if ssm is None:
        ssm = importlib.import_module("boto3").client("ssm")  # boto3 is in the Lambda Python runtime
    return str(ssm.get_parameter(Name=name, WithDecryption=True)["Parameter"]["Value"])


def make_store(env: dict[str, str] | None = None, ssm: Any = None) -> VectorStore:
    env = dict(os.environ) if env is None else env
    key = read_secret("PINECONE_API_KEY", "PINECONE_KEY_PARAMETER", env, ssm)
    if key is None:
        raise VectorStoreError("no Pinecone key: set PINECONE_KEY_PARAMETER (or PINECONE_API_KEY for a test)")
    return PineconeVectorStore(
        key,
        env.get("PINECONE_INDEX", "keywords"),
        env.get("PINECONE_NAMESPACE", ""),
        host=env.get("PINECONE_HOST") or None,
    )


def make_corpus(env: dict[str, str] | None = None, ssm: Any = None) -> Corpus | None:
    """The papers (Elasticsearch), read-only. None if ELASTIC_URL is not set."""
    env = dict(os.environ) if env is None else env
    url = env.get("ELASTIC_URL")
    if not url:
        return None
    return ElasticsearchCorpus(
        url,
        env.get("ELASTIC_INDEX", "papers"),
        read_secret("ELASTIC_API_KEY", "ELASTIC_KEY_PARAMETER", env, ssm),
    )


def get_store() -> VectorStore:
    """One store per warm Lambda: the key is read from the secure store once, not on every request."""
    global _store
    if _store is None:
        _store = make_store()
    return _store


def get_corpus() -> Corpus | None:
    global _corpus, _corpus_ready
    if not _corpus_ready:
        _corpus, _corpus_ready = make_corpus(), True
    return _corpus


def lambda_handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    return handle(event, get_store, config_from_env(), print, get_corpus)
