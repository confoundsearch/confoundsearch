"""Embed every OpenAlex keyword and keep the vectors as one matrix, to find related terms by meaning.

OpenAlex publishes its ~1.9 million research keywords (with alternative names and a description) in a free
snapshot. This downloads them, embeds each (name, alternative names and description; `--no-description`
leaves the last out) with an OpenAI embedding model, and writes one matrix: a raw float16 file with one row
per keyword, in the order of keywords.jsonl. Standard library only to build.

    python devops/tools/keyword_vectors.py estimate                      # size, tokens, cost: no spend
    python devops/tools/keyword_vectors.py build --out kw --limit 20000  # a trial: the first 20,000 keywords
    python devops/tools/keyword_vectors.py build --out kw                # everything (needs OPENAI_API_KEY)
    python devops/tools/keyword_vectors.py query "firm performance" --out kw -k 20
    python devops/tools/keyword_vectors.py push --out kw --index keywords                   # to Pinecone
    python devops/tools/keyword_vectors.py query "firm performance" --index keywords --out kw  # Pinecone

`build` can be stopped and run again: finished shards are kept. It refuses a job estimated to cost more than
--max-usd (default 5). Prices are typed in below from memory: check them on OpenAI's pricing page.

`push` writes the matrix to a vector database (Pinecone, with PINECONE_API_KEY; `confoundsearch.data.vectors`
has the interface for others) and carries on where it stopped if interrupted. `query --index` asks that
database instead of the local file, and then needs no numpy.
"""

from __future__ import annotations

import argparse
import gzip
import importlib
import itertools
import json
import math
import os
import re
import struct
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from confoundsearch.data.keyword_index import KeywordIndex, pool_size, without_echoes  # noqa: E402
from confoundsearch.data.vectors import (  # noqa: E402
    PineconeVectorStore,
    VectorRecord,
    VectorStore,
    plan_limit_reached,
)
from confoundsearch.dotenv import load_dotenv  # noqa: E402
from confoundsearch.errors import VectorStoreError  # noqa: E402

SNAPSHOT = "https://openalex.s3.amazonaws.com"
EMBEDDINGS = "https://api.openai.com/v1/embeddings"
USD_PER_MILLION_TOKENS = {
    "text-embedding-3-small": 0.02,
    "text-embedding-3-large": 0.13,
    "text-embedding-ada-002": 0.10,
}
MAX_BATCH = 2048  # inputs per request, the API's limit
SHARD = 20_000  # keywords per shard file
MAX_ALTERNATIVES = 6
MAX_CHARS = 1000  # a text is cut here: no keyword needs more, and the API rejects very long inputs
RETRY_STATUS = {429, 500, 502, 503, 504}

Post = Callable[[str, dict[str, str], bytes], bytes]


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


# -- the snapshot ----------------------------------------------------------------------------------------


def http_get(url: str, timeout: float = 600.0) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310
        data: bytes = response.read()
        return data


def download_snapshot(folder: Path, get: Callable[[str], bytes] = http_get) -> tuple[list[Path], str]:
    """The keyword files of the latest snapshot, downloaded once. Returns (files, snapshot date)."""
    manifest = json.loads(get(f"{SNAPSHOT}/data/jsonl/keywords/manifest.json"))
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for entry in manifest["files"]:
        key = entry["url"].replace("s3://openalex/", "")
        path = folder / key.replace("/", "_")
        if not path.exists() or path.stat().st_size != entry["meta"]["content_length"]:
            log(f"downloading {key}")
            path.write_bytes(get(f"{SNAPSHOT}/{key}"))
        paths.append(path)
    return paths, str(manifest.get("date", ""))


def keyword_text(record: dict[str, Any], with_description: bool = False) -> str:
    """What gets embedded: the name, its alternative names, and optionally the description."""
    parts = [record["display_name"], *(record.get("display_name_alternatives") or [])[:MAX_ALTERNATIVES]]
    text = " | ".join(parts)
    if with_description and record.get("description"):
        text += ". " + record["description"]
    return text[:MAX_CHARS]


def read_keywords(paths: list[Path], with_description: bool = False) -> Iterator[dict[str, Any]]:
    for path in paths:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                yield {
                    "id": record["id"].rsplit("/", 1)[-1],
                    "name": record["display_name"],
                    "works": record.get("works_count") or 0,
                    "text": keyword_text(record, with_description),
                }


# -- cost ------------------------------------------------------------------------------------------------


def estimate(rows: list[dict[str, Any]], model: str, dims: int) -> dict[str, Any]:
    tokens = sum(len(row["text"]) for row in rows) / 4  # about 4 characters a token in English
    price = USD_PER_MILLION_TOKENS.get(model)
    return {
        "keywords": len(rows),
        "tokens": round(tokens),
        "usd": None if price is None else round(tokens / 1e6 * price, 2),
        "matrix_gb_float16": round(len(rows) * dims * 2 / 1e9, 2),
        "matrix_gb_float32": round(len(rows) * dims * 4 / 1e9, 2),
    }


# -- embedding -------------------------------------------------------------------------------------------


def urllib_post(url: str, headers: dict[str, str], body: bytes) -> bytes:
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310
            data: bytes = response.read()
            return data
    except urllib.error.HTTPError as err:
        raise HttpError(
            err.code, err.read().decode("utf-8", "replace"), err.headers.get("Retry-After")
        ) from err


class HttpError(Exception):
    def __init__(self, status: int, detail: str, retry_after: str | None = None) -> None:
        super().__init__(f"HTTP {status}: {detail[:300]}")
        self.status, self.retry_after = status, retry_after


def normalize(vector: list[float]) -> list[float]:
    length = math.sqrt(sum(x * x for x in vector)) or 1.0
    return [x / length for x in vector]


def embed(
    texts: list[str],
    model: str,
    dims: int,
    key: str,
    post: Post = urllib_post,
    sleep: Callable[[float], None] = time.sleep,
    tries: int = 8,
) -> list[list[float]]:
    """One request for up to MAX_BATCH texts, retried when rate limited. Vectors come back unit length."""
    payload: dict[str, Any] = {"model": model, "input": texts}
    if model.startswith("text-embedding-3"):
        payload["dimensions"] = dims  # the older ada model cannot be shortened
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}
    for attempt in range(tries):
        try:
            data = json.loads(post(EMBEDDINGS, headers, json.dumps(payload).encode()))
        except HttpError as err:
            if err.status not in RETRY_STATUS or attempt == tries - 1:
                raise
            sleep(float(err.retry_after) if err.retry_after else min(2.0**attempt, 60.0))
            continue
        ordered = sorted(data["data"], key=lambda item: item["index"])
        return [normalize(item["embedding"]) for item in ordered]
    raise AssertionError("unreachable")  # pragma: no cover


def pack(vectors: list[list[float]]) -> bytes:
    return b"".join(struct.pack(f"<{len(v)}e", *v) for v in vectors)


# -- build -----------------------------------------------------------------------------------------------


def finished(out: Path, wanted: dict[str, Any]) -> bool:
    """Is there already a complete matrix for exactly this model, size, snapshot and text?"""
    meta, matrix = out / "meta.json", out / "vectors.f16"
    if not (meta.exists() and matrix.exists()):
        return False
    same = bool(json.loads(meta.read_text(encoding="utf-8")) == wanted)
    return same and matrix.stat().st_size == wanted["count"] * wanted["dims"] * 2


def build(
    out: Path,
    model: str,
    dims: int,
    limit: int | None,
    with_description: bool,
    max_usd: float,
    key: str,
    post: Post = urllib_post,
    get: Callable[[str], bytes] = http_get,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    paths, date = download_snapshot(out / "snapshot", get)
    rows = list(itertools.islice(read_keywords(paths, with_description), limit))
    wanted = {"model": model, "dims": dims, "count": len(rows), "dtype": "float16", "snapshot": date,
              "with_description": with_description}  # fmt: skip
    if finished(out, wanted):
        log(f"already built: {len(rows)} keywords in {out}/vectors.f16; nothing to embed")
        return wanted
    cost = estimate(rows, model, dims)
    if cost["usd"] is not None and cost["usd"] > max_usd:
        raise SystemExit(f"error: estimated ${cost['usd']} is over --max-usd {max_usd}; raise it to go ahead")
    log(f"{cost['keywords']} keywords, about {cost['tokens']:,} tokens, about ${cost['usd']}")
    with (out / "keywords.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    shards = out / "shards"
    shards.mkdir(exist_ok=True)
    signature = {"model": model, "dims": dims, "snapshot": date, "with_description": with_description}
    marker = shards / "signature.json"
    if not marker.exists() or json.loads(marker.read_text(encoding="utf-8")) != signature:
        for stale in shards.glob("shard_*.f16"):  # made by a different model, size or snapshot: not reusable
            stale.unlink()
        marker.write_text(json.dumps(signature), encoding="utf-8")
    files = []
    for number, start in enumerate(range(0, len(rows), SHARD)):
        chunk = rows[start : start + SHARD]
        shard = shards / f"shard_{number:05d}.f16"
        files.append(shard)
        if shard.exists() and shard.stat().st_size == len(chunk) * dims * 2:
            continue  # done in an earlier run
        blob = b""
        for first in range(0, len(chunk), MAX_BATCH):
            texts = [row["text"] for row in chunk[first : first + MAX_BATCH]]
            blob += pack(embed(texts, model, dims, key, post, sleep))
        shard.write_bytes(blob)
        log(f"shard {number + 1}/{math.ceil(len(rows) / SHARD)} done")
    with (out / "vectors.f16").open("wb") as matrix:
        for shard in files:
            matrix.write(shard.read_bytes())
    for shard in files:
        shard.unlink()
    (out / "meta.json").write_text(json.dumps(wanted, indent=2), encoding="utf-8")
    return wanted


# -- push to a vector database ------------------------------------------------------------------------------

PUSH_CHUNK = 1000  # keywords between saves of the progress file
PUSH_REQUEST = 100  # keywords per request


def push(
    out: Path,
    store: VectorStore,
    limit: int | None = None,
    label: str = "",
    workers: int = 1,
    min_works: int = 0,
    clobber: bool = False,
    fresh: bool = False,
) -> dict[str, Any]:
    """Write the matrix to a vector store, carrying on from an earlier, interrupted push.

    With workers > 1 the requests of each chunk run side by side; progress is still saved a chunk at a time.
    With min_works, keywords used by fewer papers are skipped (they still count as passed over).
    clobber deletes every record first. fresh does too, unless this very matrix is already being pushed (an
    interrupted push): then it carries on, so running the same command again never wipes a half-done load.
    """
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    total = meta["count"] if limit is None else min(limit, meta["count"])
    dims = meta["dims"]
    store.create_index(dims, "cosine")
    progress = out / "pushed.json"
    state = json.loads(progress.read_text(encoding="utf-8")) if progress.exists() else {}
    same = (
        state.get("index") == label and state.get("dims") == dims and state.get("min_works", 0) == min_works
    )
    matrix_file = out / "vectors.f16"
    matrix_id = f"{matrix_file.stat().st_size}:{matrix_file.stat().st_mtime_ns}"  # changes when it is rebuilt
    if clobber or (fresh and not (same and state.get("matrix") == matrix_id)):
        store.clear()  # the caller has already asked the user; start from nothing
        progress.unlink(missing_ok=True)
        state, same = {}, False
    elif fresh:
        log("this matrix is already being pushed: carrying on, nothing is deleted")
    done = state["count"] if same else 0
    pushed = state.get("pushed", done) if same else 0
    row_bytes = dims * 2
    with (out / "keywords.jsonl").open(encoding="utf-8") as names, (out / "vectors.f16").open("rb") as matrix:
        matrix.seek(done * row_bytes)
        for _ in range(done):
            next(names)
        while done < total:
            records = []
            scanned = min(PUSH_CHUNK, total - done)
            for _ in range(scanned):
                row = json.loads(next(names))
                values = struct.unpack(f"<{dims}e", matrix.read(row_bytes))
                if row["works"] >= min_works:
                    records.append(
                        VectorRecord(row["id"], values, {"name": row["name"], "works": row["works"]})
                    )
            slices = [records[i : i + PUSH_REQUEST] for i in range(0, len(records), PUSH_REQUEST)]
            if workers > 1:
                with ThreadPoolExecutor(workers) as pool:
                    list(pool.map(store.upsert, slices))  # list(): raise the first failure, if any
            else:
                for piece in slices:
                    store.upsert(piece)
            done += scanned
            pushed += len(records)
            state = {
                "index": label,
                "dims": dims,
                "min_works": min_works,
                "count": done,
                "pushed": pushed,
                "matrix": matrix_id,
            }
            progress.write_text(json.dumps(state), encoding="utf-8")
            log(f"pushed {pushed} ({done}/{total} keywords passed over)")
    return {"index": label, "pushed": pushed, "passed_over": done, "dims": dims, "min_works": min_works}


# -- neighbours of the constructs in a known-confounds file --------------------------------------------------


def construct_queries(known: dict[str, Any]) -> list[tuple[str, str, str]]:
    """(kind, name, phrase): the two variables, and each confound named 'a / b' as 'a b', as a, and as b."""
    out = [("variable", known[side], known[side]) for side in ("iv", "dv")]
    for control in known["controls"]:
        name = control["name"]
        plain = " ".join(re.sub(r"\(.*?\)|\u2014.*", "", name).replace("/", " ").split())
        parts = [p.strip() for p in re.sub(r"\(.*?\)|\u2014.*", "", name).split("/") if p.strip()]
        for phrase in dict.fromkeys([plain, *parts]):
            out.append(("confound", name, phrase))
    return out


def expected_for(expected: dict[str, Any], kind: str, name: str) -> list[str]:
    if kind == "variable":
        return list(expected.get("variables", {}).get(name, []))
    for prefix, terms in expected.get("confounds", {}).items():
        if name.startswith(prefix):
            return list(terms)
    return []


def found_terms(neighbours: list[dict[str, Any]], wanted: list[str]) -> list[str]:
    """Which of the wanted terms some neighbour is (or contains, or is contained in)."""
    names = [n["name"].casefold() for n in neighbours]
    hits = []
    for term in wanted:
        low = term.casefold()
        if any(low in name or (len(name) >= 6 and name in low) for name in names):
            hits.append(term)
    return hits


def neighbours(
    store: VectorStore,
    embed_phrases: Callable[[list[str]], list[list[float]]],
    known: dict[str, Any],
    expected: dict[str, Any],
    k: int = 25,
    min_works: int = 0,
    distinct: bool = False,
) -> list[dict[str, Any]]:
    queries = construct_queries(known)
    vectors = embed_phrases([phrase for _, _, phrase in queries])
    flt = {"works": {"$gte": min_works}} if min_works else None
    results = []
    for (kind, name, phrase), vector in zip(queries, vectors):
        matches = store.query(vector, pool_size(k) if distinct else k, flt)
        if distinct:
            matches = without_echoes(matches, phrase, k)
        found = [
            {
                "name": m.metadata.get("name", m.id),
                "works": m.metadata.get("works"),
                "score": round(m.score, 3),
            }
            for m in matches
        ]
        wanted = expected_for(expected, kind, name)
        results.append({"kind": kind, "construct": name, "query": phrase, "neighbours": found,
                        "expected": wanted, "expected_found": found_terms(found, wanted)})  # fmt: skip
    return results


def neighbours_markdown(results: list[dict[str, Any]]) -> str:
    lines = []
    asked = hit = 0
    for r in results:
        lines.append(
            f"### {r['query']}" + (f"  _(for {r['construct']})_" if r["query"] != r["construct"] else "")
        )
        if r["expected"]:
            lines.append(f"Expected terms found: {len(r['expected_found'])}/{len(r['expected'])}: "
                         + ", ".join(r["expected_found"] or ["none"]))  # fmt: skip
            asked += len(r["expected"])
            hit += len(r["expected_found"])
        lines.append("")
        lines += [f"- {n['name']}  ({n['works']} papers, {n['score']})" for n in r["neighbours"]]
        lines.append("")
    summary = f"Expected terms found in the neighbours overall: {hit}/{asked}\n" if asked else ""
    return summary + "\n" + "\n".join(lines)


# -- query -----------------------------------------------------------------------------------------------


def nearest(
    out: Path, phrase: str, k: int, key: str, post: Post = urllib_post
) -> list[tuple[float, dict[str, Any]]]:
    """The k keywords closest in meaning to the phrase. Needs numpy."""
    np = importlib.import_module("numpy")
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    query = np.array(embed([phrase], meta["model"], meta["dims"], key, post)[0], dtype="float32")
    matrix = np.memmap(out / "vectors.f16", dtype="float16", mode="r", shape=(meta["count"], meta["dims"]))
    scores = np.empty(meta["count"], dtype="float32")
    for start in range(0, meta["count"], 100_000):
        scores[start : start + 100_000] = matrix[start : start + 100_000].astype("float32") @ query
    top = np.argsort(-scores)[:k]
    wanted = {int(i) for i in top}
    found: dict[int, dict[str, Any]] = {}
    with (out / "keywords.jsonl").open(encoding="utf-8") as handle:
        for number, line in enumerate(handle):
            if number in wanted:
                found[number] = json.loads(line)
    return [(float(scores[i]), found[int(i)]) for i in top]


def make_store(args: argparse.Namespace) -> VectorStore:
    key = os.environ.get("PINECONE_API_KEY", "")
    if not key:
        raise SystemExit("error: set PINECONE_API_KEY")
    return PineconeVectorStore(key, args.index, args.namespace, args.cloud, args.region)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("estimate", "build", "query", "push", "neighbors"):
        sub = commands.add_parser(name)
        sub.add_argument("--out", type=Path, default=Path("keyword_vectors"))
        sub.add_argument("--model", default="text-embedding-3-small")
        sub.add_argument("--dims", type=int, default=512)
        sub.add_argument(
            "--no-description",
            dest="with_description",
            action="store_false",
            help="embed only the name and alternative names (the default adds the description)",
        )
        if name in ("query", "push", "neighbors"):
            sub.add_argument("--index", help="the Pinecone index")
            sub.add_argument("--namespace", default="")
            sub.add_argument("--cloud", default="aws")
            sub.add_argument("--region", default="us-east-1")
        if name in ("query", "neighbors"):
            sub.add_argument(
                "--distinct",
                action="store_true",
                help="leave out keywords whose names merely contain the query ('firm size and growth')",
            )
        if name == "query":
            sub.add_argument(
                "--min-works", type=int, default=0, help="only keywords used by at least this many papers"
            )
        if name == "neighbors":
            sub.add_argument(
                "known", type=Path, help="a known-confounds JSON, e.g. examples/known_confounds.json"
            )
            sub.add_argument(
                "--expect", type=Path, help="terms to look for (examples/known_construct_terms.json)"
            )
            sub.add_argument("-k", type=int, default=25)
            sub.add_argument(
                "--min-works", type=int, default=0, help="only keywords used by at least this many papers"
            )
            sub.add_argument("--report", type=Path, help="also write the markdown report here")
        if name == "push":
            sub.add_argument("--workers", type=int, default=8, help="requests in flight at once (default 8)")
            sub.add_argument(
                "--min-works", type=int, default=0, help="skip keywords used by fewer papers than this"
            )
            sub.add_argument("--clobber", action="store_true", help="first delete every record in the index")
            sub.add_argument(
                "--fresh",
                action="store_true",
                help="like --clobber, but carry on instead if this very matrix is already being pushed",
            )
            sub.add_argument("--yes", action="store_true", help="do not ask before deleting")
        if name in ("estimate", "build", "push"):
            sub.add_argument("--limit", type=int)
        if name == "build":
            sub.add_argument("--max-usd", type=float, default=5.0)
        if name == "query":
            sub.add_argument("phrase")
            sub.add_argument("-k", type=int, default=20)
    args = parser.parse_args(argv)
    load_dotenv(Path.cwd() / ".env")  # keys in a .env file work without `just`; a variable already set wins
    key = os.environ.get("OPENAI_API_KEY", "")
    if args.command == "estimate":
        paths, date = download_snapshot(args.out / "snapshot")
        rows = list(itertools.islice(read_keywords(paths, args.with_description), args.limit))
        print(
            json.dumps(
                {
                    "snapshot": date,
                    "model": args.model,
                    "dims": args.dims,
                    **estimate(rows, args.model, args.dims),
                },
                indent=2,
            )
        )
        return 0
    if args.command == "push":
        if not args.index:
            raise SystemExit("error: push needs --index NAME")
        if (args.clobber or args.fresh) and not args.yes:
            if not sys.stdin.isatty():
                raise SystemExit("error: this may delete every record; with no terminal to ask, add --yes")
            if input(f"Delete EVERY record in index {args.index!r}? [y/N] ") != "y":
                raise SystemExit("cancelled: nothing was deleted")
        try:
            result = push(
                args.out,
                make_store(args),
                args.limit,
                args.index,
                args.workers,
                args.min_works,
                args.clobber,
                args.fresh,
            )
        except VectorStoreError as err:
            if not plan_limit_reached(err):
                raise
            raise SystemExit(
                "error: the plan's limit is reached, so Pinecone stopped accepting writes.\n"
                "Progress is saved: run the same command again once the limit resets or you upgrade.\n"
                "To fit a free plan, push fewer keywords with --min-works (for example 500).\n"
                f"Pinecone said: {err}"
            ) from err
        print(json.dumps(result, indent=2))
        return 0
    if args.command == "query" and args.index:
        # A name that is a keyword is looked up in the index and needs no OpenAI key; any other phrase does.
        embedder = (lambda text: embed([text], args.model, args.dims, key)[0]) if key else None
        index = KeywordIndex(make_store(args), embedder)
        try:
            found = index.query(args.phrase, args.k, args.min_works, distinct=args.distinct)
        except VectorStoreError as err:
            raise SystemExit(f"error: {err}; set OPENAI_API_KEY to search by any phrase") from err
        for match in found:
            print(f"{match.score:.3f}  {match.metadata.get('name')}  ({match.metadata.get('works')} papers)")
        return 0
    if not key:
        raise SystemExit("error: set OPENAI_API_KEY")
    if args.command == "neighbors":
        if not args.index:
            raise SystemExit("error: neighbors needs --index NAME")
        meta = json.loads((args.out / "meta.json").read_text(encoding="utf-8"))
        known = json.loads(args.known.read_text(encoding="utf-8"))
        expect = json.loads(args.expect.read_text(encoding="utf-8")) if args.expect else {}
        report = neighbours_markdown(
            neighbours(
                make_store(args),
                lambda phrases: embed(phrases, meta["model"], meta["dims"], key),
                known,
                expect,
                args.k,
                args.min_works,
                args.distinct,
            )
        )
        print(report)
        if args.report:
            args.report.write_text(report, encoding="utf-8")
        return 0
    if args.command == "build":
        print(
            json.dumps(
                build(args.out, args.model, args.dims, args.limit, args.with_description, args.max_usd, key),
                indent=2,
            )
        )
        return 0
    for score, row in nearest(args.out, args.phrase, args.k, key):
        print(f"{score:.3f}  {row['name']}  ({row['works']} papers)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
