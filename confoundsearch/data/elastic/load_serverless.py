#!/usr/bin/env python3
"""Load the paper corpus into Elasticsearch Serverless (and measure what it costs).

Standard library only; it talks to the Elasticsearch REST API directly, so there is
no client version to match. The steps, each skipped if already done:

  1. check the connection to Elasticsearch (fail early, before any download)
     (`--check` stops here, `--count` also prints the index's document count;
     `--download` skips it and only does step 2)
  2. download the Semantic Scholar `papers` and `abstracts` datasets
  3. extract ALL files at once: keep social-science papers and attach abstracts. Files are processed
     in parallel (--workers), the abstracts are scanned as few times as memory allows, and every
     finished file is remembered. (--extract-files N limits this to the first N files.)
  4. create the index (trimmed mapping, see papers_mapping.json) and bulk-load every extracted file
     (or only the first --files N, for a trial)
  5. write load_report.json with the timings to compare against Elastic's billing page

It reuses the file names and `.done` markers of the older pipeline, so anything
already in /tmp/s2, /tmp/s2_abstracts_raw or /tmp/s2_social_science is not redone.

    # put S2_KEY, ES_URL and ES_API_KEY in a .env file (or export them), then:
    python confoundsearch/data/elastic/load_serverless.py --files 3 --index papers_trial   # a trial: 3 files
    python confoundsearch/data/elastic/load_serverless.py                          # everything, into `papers`
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import multiprocessing
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
LOG_EVERY_LINES = 2_000_000  # progress while reading a dataset file
LOG_EVERY_CHUNKS = 25  # progress while loading
S2_API = "https://api.semanticscholar.org/datasets/v1"
DEFAULT_RELEASE = "2026-06-24"
TOTAL_PAPERS_FILES = 60  # the papers dataset is split into about this many files
FIELDS_OF_STUDY = {
    "Business", "Economics", "Education", "Geography", "History",
    "Law", "Political Science", "Psychology", "Sociology",
}  # fmt: skip
KEEP = ("corpusid", "title", "abstract", "url", "year", "citationcount", "s2fieldsofstudy")
RETRY_STATUS = {429, 502, 503, 504}


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def fail(message: str) -> SystemExit:
    log(f"error: {message}")
    return SystemExit(1)


# -- HTTP ----------------------------------------------------------------------


def http(
    method: str,
    url: str,
    headers: dict[str, str] | None = None,
    body: bytes | None = None,
    timeout: float = 120.0,
) -> tuple[int, bytes]:
    """One request. HTTP errors come back as (status, body); only network failures raise."""
    request = urllib.request.Request(url, data=body, headers=headers or {}, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            return response.status, response.read()
    except urllib.error.HTTPError as err:
        return err.code, err.read()
    except urllib.error.URLError as err:
        raise fail(f"could not reach {url}: {err.reason}") from err


# -- downloading ---------------------------------------------------------------


def manifest(api: str, release: str, dataset: str, key: str) -> list[tuple[str, str]]:
    status, body = http("GET", f"{api}/release/{release}/dataset/{dataset}", {"x-api-key": key}, timeout=30)
    if status != 200:
        raise fail(f"Semantic Scholar returned HTTP {status} for the {dataset} dataset: {body[:200]!r}")
    return [(Path(u.split("?", 1)[0]).name, u) for u in json.loads(body)["files"]]


def done_marker(path: Path) -> Path:
    return path.with_name(f"{path.name}.done")


def download(entries: Sequence[tuple[str, str]], dest: Path, label: str) -> list[Path]:
    """Fetch every file not already present with a .done marker. Returns all of them."""
    dest.mkdir(parents=True, exist_ok=True)
    paths, fetched = [], 0
    for i, (name, url) in enumerate(entries, 1):
        path = dest / name
        paths.append(path)
        if path.exists() and done_marker(path).exists():
            continue
        log(f"[download {label}] {i}/{len(entries)} {name}")
        partial = path.with_name(f"{name}.part")
        try:
            with urllib.request.urlopen(url, timeout=300) as response, partial.open("wb") as out:  # noqa: S310
                shutil.copyfileobj(response, out, 1024 * 1024)
        except (urllib.error.URLError, OSError) as err:
            raise fail(f"downloading {name} failed: {err}") from err
        partial.rename(path)
        done_marker(path).touch()
        fetched += 1
    log(
        f"[download {label}] {len(paths)} files ready "
        f"({fetched} downloaded, {len(paths) - fetched} already there)"
    )
    return paths


def local_files(directory: Path) -> list[Path]:
    """Files in `directory` that were fully downloaded earlier."""
    found = sorted(p for p in directory.glob("*.gz") if done_marker(p).exists())
    if not found:
        raise fail(f"no completed downloads in {directory}; run without --skip-download")
    return found


# -- extracting social-science papers with abstracts ---------------------------


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for n, line in enumerate(handle, 1):
            if n % LOG_EVERY_LINES == 0:
                log(f"  {path.name}: {n:,} lines")
            line = line.strip()
            if not line:
                continue
            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(doc, dict):
                yield doc


def is_social_science(doc: dict[str, Any]) -> bool:
    categories = {f["category"] for f in (doc.get("s2fieldsofstudy") or []) if "category" in f}
    return bool(categories & FIELDS_OF_STUDY)


def corpus_id(doc: dict[str, Any]) -> int | None:
    try:
        return int(doc["corpusid"])
    except (KeyError, TypeError, ValueError):
        return None


def extract_output(paper_file: Path, out_dir: Path) -> Path:
    return out_dir / f"{paper_file.stem}_social_science.jsonl.gz"


def ids_file(paper_file: Path, out_dir: Path) -> Path:
    return out_dir / f"{paper_file.stem}.social_ids.gz"


def pass3_marker(paper_file: Path, out_dir: Path) -> Path:
    return out_dir / f"{paper_file.stem}.pass3.done"  # the same marker the older pipeline used


# Work shared with forked worker processes. Set just before a pool is created, so children inherit it.
_WANTED: frozenset[int] = frozenset()
_ABSTRACTS: dict[int, str] = {}
_OUT_DIR: Path = Path()


def find_social_ids(paper_file: Path) -> Path:
    """Pass 1 for one file: write the corpus ids of its social-science papers next to the output."""
    log(f"[extract 1/3] finding social-science papers in {paper_file.name}")
    ids = [cid for d in read_jsonl(paper_file) if is_social_science(d) and (cid := corpus_id(d)) is not None]
    target = ids_file(paper_file, _OUT_DIR)
    tmp = target.with_name(target.name + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as handle:
        handle.write("\n".join(map(str, ids)) + ("\n" if ids else ""))
    tmp.rename(target)
    return target


def read_ids(path: Path) -> set[int]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return {int(line) for line in handle if line.strip()}


def abstracts_in(path: Path) -> dict[int, str]:
    """Pass 2 for one abstracts file: the abstracts of the papers in _WANTED."""
    log(f"[extract 2/3] abstracts in {path.name}")
    found: dict[int, str] = {}
    for doc in read_jsonl(path):
        cid = corpus_id(doc)
        if cid is not None and cid in _WANTED and doc.get("abstract"):
            found[cid] = doc["abstract"]
    return found


def write_extract(paper_file: Path) -> int:
    """Pass 3 for one file: write its social-science papers, adding abstracts they lack."""
    out, tmp = extract_output(paper_file, _OUT_DIR), _OUT_DIR / f"{paper_file.stem}.pass3.tmp.gz"
    kept = 0
    with gzip.open(tmp, "wt", encoding="utf-8") as handle:
        for doc in read_jsonl(paper_file):
            if not is_social_science(doc):
                continue
            cid = corpus_id(doc)
            if cid in _ABSTRACTS and not doc.get("abstract"):
                doc["abstract"] = _ABSTRACTS[cid]
            handle.write(json.dumps(doc) + "\n")
            kept += 1
    tmp.rename(out)
    pass3_marker(paper_file, _OUT_DIR).touch()
    log(f"[extract 3/3] {out.name}: {kept:,} papers")
    return kept


def pmap(function: Callable[[Any], Any], items: Sequence[Any], workers: int) -> list[Any]:
    """function over items, in parallel with forked processes when workers > 1. Order is kept."""
    if workers <= 1 or len(items) <= 1:
        return [function(item) for item in items]
    with multiprocessing.get_context("fork").Pool(min(workers, len(items))) as pool:
        return pool.map(function, items)


BYTES_PER_ABSTRACT = 2_500  # text plus Python's dict and string overhead, rounded up


def available_memory() -> int:
    """Bytes of memory free for new work (Linux); 8 GB if it cannot be read."""
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return 8 * 1024**3


def plan_batches(counts: Sequence[tuple[Path, int]], budget: int) -> list[list[Path]]:
    """Group files so each group's abstract lookup fits in `budget` bytes (at least one file per group)."""
    batches: list[list[Path]] = []
    used = 0
    for path, count in counts:
        cost = count * BYTES_PER_ABSTRACT
        if not batches or used + cost > budget:
            batches.append([])
            used = 0
        batches[-1].append(path)
        used += cost
    return batches


def extract(
    paper_files: Sequence[Path],
    abstract_files: Sequence[Path],
    out_dir: Path,
    workers: int = 1,
    memory_bytes: int | None = None,
) -> list[Path]:
    """Write <stem>_social_science.jsonl.gz for every papers file that has none yet.

    All files are handled together: pass 1 finds the social-science ids in every file, then each batch
    of files shares ONE scan of the abstracts. Files are processed in parallel, and a finished file or
    ids list is never redone. The batches are sized so the abstract lookup fits in memory; when
    everything fits (the usual case), the abstracts are scanned once.
    """
    global _WANTED, _ABSTRACTS, _OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    _OUT_DIR = out_dir
    todo = [p for p in paper_files if not pass3_marker(p, out_dir).exists()]
    if not todo:
        log("[extract] all files were already extracted")
        return [extract_output(p, out_dir) for p in paper_files]

    missing = [p for p in todo if not ids_file(p, out_dir).exists()]
    log(f"[extract] {len(todo)} files to extract; finding social-science papers in {len(missing)} of them")
    pmap(find_social_ids, missing, workers)
    ids = {p: read_ids(ids_file(p, out_dir)) for p in todo}

    total = sum(len(v) for v in ids.values())
    budget = memory_bytes if memory_bytes is not None else int(available_memory() * 0.6)
    batches = plan_batches([(p, len(ids[p])) for p in todo], budget)
    log(
        f"[extract] {total:,} social-science papers; the abstract lookup needs about "
        f"{total * BYTES_PER_ABSTRACT / 1024**3:.1f} GB of {budget / 1024**3:.1f} GB allowed, "
        f"so the abstracts will be scanned {len(batches)} time(s)"
    )
    for n, batch in enumerate(batches, 1):
        _WANTED = frozenset().union(*(ids[p] for p in batch))
        log(
            f"[extract] batch {n}/{len(batches)}: {len(batch)} files, {len(_WANTED):,} papers; "
            "scanning abstracts"
        )
        _ABSTRACTS = {}
        for found in pmap(abstracts_in, list(abstract_files), workers):
            _ABSTRACTS.update(found)
        log(f"[extract] batch {n}/{len(batches)}: {len(_ABSTRACTS):,} abstracts found; writing files")
        pmap(write_extract, batch, workers)
        _WANTED, _ABSTRACTS = frozenset(), {}
    return [extract_output(p, out_dir) for p in paper_files]


# -- loading into Elasticsearch ------------------------------------------------


def trim(doc: dict[str, Any], require_abstract: bool = True) -> dict[str, Any] | None:
    """Keep only the fields the agent uses. None if the paper has no abstract to show."""
    if "corpusid" not in doc or (require_abstract and not doc.get("abstract")):
        return None
    return {k: doc[k] for k in KEEP if doc.get(k) is not None}


def bulk_body(index: str, docs: Sequence[dict[str, Any]]) -> bytes:
    lines: list[str] = []
    for doc in docs:
        lines.append(json.dumps({"index": {"_index": index, "_id": str(doc["corpusid"])}}))
        lines.append(json.dumps(doc))
    return ("\n".join(lines) + "\n").encode("utf-8")


@dataclass
class LoadStats:
    indexed: int = 0
    failed: int = 0
    skipped_no_abstract: int = 0
    bytes_sent: int = 0
    errors: list[str] = field(default_factory=list)

    def add(self, other: LoadStats) -> None:
        self.indexed += other.indexed
        self.failed += other.failed
        self.skipped_no_abstract += other.skipped_no_abstract
        self.bytes_sent += other.bytes_sent
        self.errors = (self.errors + other.errors)[:10]


class Elasticsearch:
    def __init__(
        self,
        url: str,
        api_key: str,
        index: str,
        chunk_docs: int = 2000,
        retries: int = 6,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.url, self.index = url.rstrip("/"), index
        self.chunk_docs, self.retries, self.sleep = chunk_docs, retries, sleep
        self.headers = {"Authorization": f"ApiKey {api_key}", "Content-Type": "application/json"}

    def check(self) -> None:
        status, body = http("GET", f"{self.url}/", self.headers, timeout=30)
        if status != 200:
            raise fail(
                f"Elasticsearch answered HTTP {status} at {self.url}: {body[:200]!r}. "
                "Check the URL and API key."
            )

    def ensure_index(self, mapping: dict[str, Any], recreate: bool) -> None:
        target = f"{self.url}/{self.index}"
        exists = http("HEAD", target, self.headers)[0] == 200
        if exists and recreate:
            status, body = http("DELETE", target, self.headers)
            if status not in (200, 404):
                raise fail(f"could not delete index {self.index}: HTTP {status} {body[:200]!r}")
            exists = False
        if exists:
            log(f"[index] {self.index} already exists; adding to it")
            return
        status, body = http("PUT", target, self.headers, json.dumps(mapping).encode())
        if status not in (200, 201):
            raise fail(f"could not create index {self.index}: HTTP {status} {body[:300]!r}")
        log(f"[index] created {self.index}")

    def count(self) -> int:
        status, body = http("GET", f"{self.url}/{self.index}/_count", self.headers, timeout=60)
        return int(json.loads(body)["count"]) if status == 200 else -1

    def _send(self, docs: list[dict[str, Any]], stats: LoadStats) -> None:
        """Index `docs`, retrying throttled requests and throttled items with backoff."""
        pending = docs
        for attempt in range(self.retries + 1):
            body = bulk_body(self.index, pending)
            status, raw = http(
                "POST",
                f"{self.url}/_bulk",
                {**self.headers, "Content-Type": "application/x-ndjson"},
                body,
                300,
            )
            stats.bytes_sent += len(body)
            retry: list[dict[str, Any]] = []
            if status == 200:
                for doc, item in zip(pending, json.loads(raw)["items"]):
                    result = item.get("index", {})
                    if result.get("status", 500) < 300:
                        stats.indexed += 1
                    elif result.get("status") in RETRY_STATUS:
                        retry.append(doc)
                    else:
                        stats.failed += 1
                        stats.errors.append(f"{doc['corpusid']}: {json.dumps(result.get('error'))[:200]}")
            elif status in RETRY_STATUS:
                retry = pending
            else:
                stats.failed += len(pending)
                stats.errors.append(f"HTTP {status}: {raw[:200]!r}")
                return
            if not retry:
                return
            pending = retry
            if attempt < self.retries:
                self.sleep(min(2.0 * 2**attempt, 60.0))
        stats.failed += len(pending)
        stats.errors.append(f"gave up on {len(pending)} documents after {self.retries} retries")

    def load_file(self, path: Path, require_abstract: bool = True) -> LoadStats:
        stats, chunk = LoadStats(), []
        for doc in read_jsonl(path):
            trimmed = trim(doc, require_abstract)
            if trimmed is None:
                stats.skipped_no_abstract += 1
                continue
            chunk.append(trimmed)
            if len(chunk) >= self.chunk_docs:
                self._send(chunk, stats)
                chunk = []
                if stats.indexed % (self.chunk_docs * LOG_EVERY_CHUNKS) < self.chunk_docs:
                    log(f"  {path.name}: {stats.indexed:,} indexed")
        if chunk:
            self._send(chunk, stats)
        return stats


# -- the trial -----------------------------------------------------------------


def load_marker(path: Path, es: Elasticsearch) -> Path:
    """Records that `path` was loaded into this index of this project. A new project starts empty."""
    project = hashlib.sha1(es.url.encode()).hexdigest()[:8]
    return path.with_name(f"{path.name}.{es.index}.{project}.loaded")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_dotenv(path: Path) -> None:
    """Read KEY=VALUE lines from a .env file into the environment. Never overrides a variable already set."""
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        line = line[len("export ") :] if line.startswith("export ") else line
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ.setdefault(key.strip(), value)


def build_parser() -> argparse.ArgumentParser:
    env = os.environ.get
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--files", type=int, default=None, help="how many files to load (default: all of them; try 3 first)"
    )
    p.add_argument(
        "--extract-files", type=int, default=0, help="extract only the first N files (default all)"
    )
    p.add_argument(
        "--workers", type=int, default=min(os.cpu_count() or 1, 8), help="files to process in parallel"
    )
    p.add_argument("--memory-gb", type=float, default=None, help="memory the abstract lookup may use")
    p.add_argument("--index", default="papers", help="index to load into (default papers)")
    p.add_argument("--recreate", action="store_true", help="delete the index first and reload every file")
    p.add_argument("--mapping", type=Path, default=HERE / "papers_mapping.json")
    p.add_argument("--keep-no-abstract", action="store_true", help="also load papers without an abstract")
    p.add_argument("--chunk-docs", type=int, default=2000, help="documents per _bulk request")
    step = p.add_mutually_exclusive_group()
    step.add_argument("--check", action="store_true", help="only test the connection, then exit")
    step.add_argument(
        "--count", action="store_true", help="print how many documents the index holds, then exit"
    )
    step.add_argument("--download", action="store_true", help="only download the corpus, then exit")
    step.add_argument("--dry-run", action="store_true", help="download and extract, but do not load")
    p.add_argument("--skip-download", action="store_true", help="use only files already downloaded")
    p.add_argument("--es-url", default=env("ES_URL"), help="project endpoint (or $ES_URL)")
    p.add_argument("--es-api-key", default=env("ES_API_KEY"), help="API key (or $ES_API_KEY)")
    p.add_argument("--s2-key", default=env("S2_KEY"), help="Semantic Scholar API key (or $S2_KEY)")
    p.add_argument(
        "--release", default=DEFAULT_RELEASE, help=f"Semantic Scholar release (default {DEFAULT_RELEASE})"
    )
    p.add_argument("--s2-api", default=S2_API, help=argparse.SUPPRESS)
    p.add_argument("--papers-dir", type=Path, default=Path("/tmp/s2"))
    p.add_argument("--abstracts-dir", type=Path, default=Path("/tmp/s2_abstracts_raw"))
    p.add_argument("--out-dir", type=Path, default=Path("/tmp/s2_social_science"))
    return p


def fetch_corpus(args: argparse.Namespace) -> tuple[list[Path], list[Path], int]:
    """Download both S2 datasets, skipping files already there. Returns papers, abstracts, #papers files."""
    if not args.s2_key:
        raise fail("set --s2-key (or S2_KEY), or use --skip-download")
    paper_entries = manifest(args.s2_api, args.release, "papers", args.s2_key)
    papers = download(paper_entries, args.papers_dir, "papers")
    abstract_entries = manifest(args.s2_api, args.release, "abstracts", args.s2_key)
    return papers, download(abstract_entries, args.abstracts_dir, "abstracts"), len(paper_entries)


def run(args: argparse.Namespace, sleep: Callable[[float], None] = time.sleep) -> dict[str, Any]:
    if args.files is not None and args.files < 1:
        raise fail("--files must be at least 1")
    if args.download:
        if args.skip_download:
            raise fail("--download and --skip-download contradict each other")
        fetch_corpus(args)
        log("[download] done; run again without --download to extract and load")
        return {}
    es: Elasticsearch | None = None
    if not args.dry_run:
        if not args.es_url or not args.es_api_key:
            raise fail("set --es-url and --es-api-key (or ES_URL and ES_API_KEY)")
        es = Elasticsearch(args.es_url, args.es_api_key, args.index, args.chunk_docs, sleep=sleep)
        es.check()
        log(f"[check] connected to {args.es_url}")
    if args.count:
        assert es is not None  # --count excludes --dry-run, so Elasticsearch is configured
        total = es.count()
        if total < 0:
            raise fail(f"could not count documents in {args.index}; does the index exist?")
        print(f"{args.index}: {total:,} documents")
        return {}
    if args.check:
        return {}

    total_files = TOTAL_PAPERS_FILES
    if args.skip_download:
        papers, abstracts = local_files(args.papers_dir), local_files(args.abstracts_dir)
    else:
        papers, abstracts, total_files = fetch_corpus(args)
    everything = sorted(papers)
    scope = everything[: args.extract_files] if args.extract_files else everything
    memory = int(args.memory_gb * 1024**3) if args.memory_gb else None
    extracted = extract(scope, sorted(abstracts), args.out_dir, args.workers, memory)
    outputs = extracted if args.files is None else extracted[: args.files]
    if es is None:
        log("[dry run] stopping before the load")
        return {}

    mapping = json.loads(args.mapping.read_text())
    started, clock = utc_now(), time.monotonic()
    es.ensure_index(mapping, args.recreate)
    totals = LoadStats()
    for i, path in enumerate(outputs, 1):
        marker = load_marker(path, es)
        if marker.exists() and not args.recreate:
            log(f"[load] {i}/{len(outputs)} {path.name} already loaded into {args.index}; skipping")
            continue
        log(f"[load] {i}/{len(outputs)} {path.name}")
        stats = es.load_file(path, require_abstract=not args.keep_no_abstract)
        totals.add(stats)
        if stats.failed == 0:
            marker.touch()
        log(
            f"[load] {path.name}: {stats.indexed:,} indexed, {stats.failed:,} failed, "
            f"{stats.skipped_no_abstract:,} without abstract"
        )
    seconds = time.monotonic() - clock
    report = {
        "index": args.index,
        "started_utc": started,
        "finished_utc": utc_now(),
        "seconds": round(seconds, 1),
        "files_loaded": len(outputs),
        "files_in_corpus": total_files,
        "scale_to_full_corpus": round(total_files / len(outputs), 2),
        "documents_in_index": es.count(),
        "docs_per_second": round(totals.indexed / seconds, 1) if seconds > 0 else 0.0,
        **asdict(totals),
    }
    (args.out_dir / "load_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def scale_note(report: dict[str, Any]) -> str:
    factor = report["scale_to_full_corpus"]
    if factor <= 1:
        return "That was the whole corpus, so the ingest cost is what the billing page shows."
    return f"The corpus is about {factor}x this size; scale the ingest cost by that."


def summary(report: dict[str, Any], out_dir: Path) -> str:
    return (
        f"\nFinished in {report['seconds']}s: {report['indexed']:,} papers indexed "
        f"({report['failed']:,} failed, {report['skipped_no_abstract']:,} skipped for no abstract).\n"
        f"Billing window (UTC): {report['started_utc']} to {report['finished_utc']}\n"
        f"{scale_note(report)}\n"
        f"Report saved to {out_dir / 'load_report.json'}; the billing page is in the Elastic console.\n"
    )


def main(argv: Sequence[str] | None = None) -> int:
    for env_file in (Path.cwd() / ".env", HERE.parents[2] / ".env"):
        load_dotenv(env_file)
    args = build_parser().parse_args(argv)
    report = run(args)
    if report:
        print(summary(report, args.out_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
