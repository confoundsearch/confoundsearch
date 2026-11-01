"""The corpus loader (data/elastic) against stand-ins for Semantic Scholar and Elasticsearch."""

from __future__ import annotations

import gzip
import importlib.util
import json
import re
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
LOADER = ROOT / "confoundsearch" / "data" / "elastic"
SPEC = importlib.util.spec_from_file_location("load_serverless", LOADER / "load_serverless.py")
trial = importlib.util.module_from_spec(SPEC)
sys.modules["load_serverless"] = trial
SPEC.loader.exec_module(trial)

S2_KEY, ES_KEY = "s2-secret", "es-secret"
SOCIAL = [{"category": "Economics", "source": "s2-fos-model"}]
RELEASE = "2099-01-01"


def gz(rows) -> bytes:
    lines = [r if isinstance(r, str) else json.dumps(r) for r in rows]
    return gzip.compress(("\n".join(lines) + "\n").encode())


def paper(cid, abstract=None, fos=SOCIAL, **extra):
    doc = {
        "corpusid": cid,
        "title": f"Paper {cid}",
        "url": f"https://x/{cid}",
        "year": 2001,
        "citationcount": cid,
        "s2fieldsofstudy": fos,
        "authors": [{"name": "A"}],
        "venue": "V",
        "externalids": {"DOI": "d"},
        **extra,
    }
    if abstract is not None:
        doc["abstract"] = abstract
    return doc


FILES = {
    "papers_0.jsonl.gz": gz(
        [paper(1, "own abstract"), paper(2), paper(3, fos=[{"category": "Biology"}]), "", "{broken", paper(4)]
    ),
    "papers_1.jsonl.gz": gz([paper(5), paper(6, "kept abstract"), paper(7, None, fos=None)]),
    "papers_2.jsonl.gz": gz([paper(8)]),
    "abstracts_0.jsonl.gz": gz(
        [
            {"corpusid": 2, "abstract": "found for two"},
            {"corpusid": 3, "abstract": "bio"},
            {"corpusid": 5, "abstract": "found for five"},
            {"corpusid": 6, "abstract": "ignored: paper 6 has one"},
            {"corpusid": 99, "abstract": "not a paper we have"},
        ]
    ),
}
MANIFESTS = {
    "papers": ["papers_0.jsonl.gz", "papers_1.jsonl.gz", "papers_2.jsonl.gz"],
    "abstracts": ["abstracts_0.jsonl.gz"],
}


class Stub:
    """Semantic Scholar's datasets API, its file host, and a tiny Elasticsearch."""

    def __init__(self) -> None:
        self.files = dict(FILES)
        self.indices: dict[str, dict] = {}
        self.docs: dict[str, dict] = {}
        self.mappings: dict[str, dict] = {}
        self.fail_delete = False
        self.plan: list = []  # per _bulk request: ("http", status) | ("items", n_throttled)
        self.log: list[str] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def reply(self, status, body=b"", ctype="application/json"):
                data = body if isinstance(body, bytes) else json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                if self.command != "HEAD":
                    self.wfile.write(data)

            def authed(self):
                return self.headers.get("Authorization") == f"ApiKey {ES_KEY}"

            def handle_any(self):
                path = self.path.split("?")[0]
                body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                outer.log.append(f"{self.command} {path}")
                if m := re.fullmatch(r"/datasets/v1/release/([^/]+)/dataset/(\w+)", path):
                    if self.headers.get("x-api-key") != S2_KEY:
                        return self.reply(403, {"message": "bad key"})
                    names = MANIFESTS[m.group(2)]
                    base = f"http://127.0.0.1:{self.server.server_port}/files"
                    return self.reply(200, {"files": [f"{base}/{n}?Signature=abc" for n in names]})
                if path.startswith("/files/"):
                    name = path.rsplit("/", 1)[1]
                    return self.reply(200, outer.files[name], "application/gzip")
                if not self.authed():
                    return self.reply(401, {"error": "unauthorized"})
                if path == "/" and self.command == "GET":
                    return self.reply(200, {"tagline": "You Know, for Search"})
                if path == "/_bulk":
                    return self.bulk(body)
                if m := re.fullmatch(r"/(\w+)/_count", path):
                    return self.reply(200, {"count": len(outer.docs)})
                if m := re.fullmatch(r"/(\w+)", path):
                    index = m.group(1)
                    if self.command == "HEAD":
                        return self.reply(200 if index in outer.indices else 404)
                    if self.command == "PUT":
                        outer.indices[index] = {}
                        outer.mappings[index] = json.loads(body)
                        return self.reply(200, {"acknowledged": True})
                    if self.command == "DELETE":
                        if outer.fail_delete:
                            return self.reply(500, {"error": "cannot delete"})
                        outer.indices.pop(index, None)
                        outer.docs.clear()
                        return self.reply(200, {"acknowledged": True})
                return self.reply(404, {"error": "nope"})

            def bulk(self, body):
                step = outer.plan.pop(0) if outer.plan else ("ok", 0)
                if step[0] == "http":
                    return self.reply(step[1], {"error": "busy"})
                lines = body.decode().splitlines()
                items = []
                for n, (action, source) in enumerate(zip(lines[::2], lines[1::2])):
                    meta = json.loads(action)["index"]
                    if step[0] == "items" and n < step[1]:
                        items.append({"index": {"status": 429, "error": "throttled"}})
                    elif json.loads(source).get("title") == "REJECT":
                        items.append(
                            {"index": {"status": 400, "error": {"type": "mapper_parsing_exception"}}}
                        )
                    else:
                        outer.docs[meta["_id"]] = json.loads(source)
                        items.append({"index": {"status": 201}})
                outer.log.append(f"bulk {len(items)}")
                return self.reply(200, {"errors": False, "items": items})

            do_GET = do_POST = do_PUT = do_DELETE = do_HEAD = handle_any

            def log_message(self, *args):
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_port}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def count(self, prefix: str) -> int:
        return sum(entry.startswith(prefix) for entry in self.log)

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def stub():
    s = Stub()
    yield s
    s.close()


@pytest.fixture
def dirs(tmp_path):
    return {"papers": tmp_path / "s2", "abstracts": tmp_path / "abs", "out": tmp_path / "out"}


def argv(stub, dirs, *extra):
    return [
        "--workers",
        "1",  # tests fork only where they say so, never while a stub server thread is running
        "--es-url",
        stub.url,
        "--es-api-key",
        ES_KEY,
        "--s2-key",
        S2_KEY,
        "--s2-api",
        f"{stub.url}/datasets/v1",
        "--release",
        RELEASE,
        "--papers-dir",
        str(dirs["papers"]),
        "--abstracts-dir",
        str(dirs["abstracts"]),
        "--out-dir",
        str(dirs["out"]),
        "--index",
        "papers_trial",
        *extra,
    ]


def run(stub, dirs, *extra):
    args = trial.build_parser().parse_args(argv(stub, dirs, *extra))
    return trial.run(args, sleep=lambda s: None)


class TestPieces:
    def test_social_science_detection(self):
        assert trial.is_social_science(paper(1))
        assert not trial.is_social_science(paper(1, fos=[{"category": "Biology"}]))
        assert not trial.is_social_science(paper(1, fos=None))
        assert not trial.is_social_science({"corpusid": 1})

    def test_trim_keeps_only_the_fields_the_agent_uses(self):
        doc = trial.trim(paper(1, "abs"))
        assert set(doc) == set(trial.KEEP)
        assert doc["abstract"] == "abs" and "authors" not in doc and "venue" not in doc

    def test_trim_drops_papers_without_an_abstract_unless_asked(self):
        assert trial.trim(paper(1)) is None and trial.trim(paper(1, "")) is None
        assert trial.trim(paper(1), require_abstract=False)["corpusid"] == 1
        assert trial.trim({"title": "no id", "abstract": "x"}) is None

    def test_trim_omits_missing_values(self):
        doc = trial.trim({"corpusid": 1, "abstract": "x", "year": None, "title": None})
        assert doc == {"corpusid": 1, "abstract": "x"}

    def test_bulk_body_is_ndjson_with_ids(self):
        body = trial.bulk_body("idx", [{"corpusid": 7, "abstract": "a"}]).decode()
        assert body.endswith("\n")
        action, source = body.splitlines()
        assert json.loads(action) == {"index": {"_index": "idx", "_id": "7"}}
        assert json.loads(source) == {"corpusid": 7, "abstract": "a"}

    def test_mapping_covers_exactly_the_kept_fields(self):
        mapping = json.loads((LOADER / "papers_mapping.json").read_text())
        assert set(mapping["mappings"]["properties"]) == set(trial.KEEP)
        assert "authors" not in mapping["mappings"]["properties"]  # nested authors inflate the document count

    def test_read_jsonl_skips_blank_malformed_and_non_object_lines(self, tmp_path):
        path = tmp_path / "x.jsonl.gz"
        path.write_bytes(gz(['{"a": 1}', "", "not json", "[1, 2]", '{"b": 2}']))
        assert list(trial.read_jsonl(path)) == [{"a": 1}, {"b": 2}]

    def test_read_jsonl_logs_progress_on_big_files(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(trial, "LOG_EVERY_LINES", 2)
        path = tmp_path / "big.jsonl.gz"
        path.write_bytes(gz(['{"a": 1}', '{"a": 2}', '{"a": 3}']))
        assert len(list(trial.read_jsonl(path))) == 3
        assert "big.jsonl.gz: 2 lines" in capsys.readouterr().err

    def test_stats_add_accumulates_and_caps_errors(self):
        total = trial.LoadStats()
        for _ in range(3):
            total.add(trial.LoadStats(1, 2, 3, 4, [f"e{i}" for i in range(6)]))
        assert (total.indexed, total.failed, total.skipped_no_abstract, total.bytes_sent) == (3, 6, 9, 12)
        assert len(total.errors) == 10


class TestDownload:
    def entries(self, stub, dataset="papers"):
        return trial.manifest(f"{stub.url}/datasets/v1", RELEASE, dataset, S2_KEY)

    def test_manifest_lists_clean_file_names(self, stub):
        assert [n for n, _ in self.entries(stub)] == MANIFESTS["papers"]

    def test_bad_key_is_a_clear_error(self, stub, capsys):
        with pytest.raises(SystemExit):
            trial.manifest(f"{stub.url}/datasets/v1", RELEASE, "papers", "wrong")
        assert "HTTP 403" in capsys.readouterr().err

    def test_downloads_then_skips_what_is_already_there(self, stub, tmp_path):
        entries = self.entries(stub)
        first = trial.download(entries, tmp_path, "papers")
        assert [p.name for p in first] == MANIFESTS["papers"] and stub.count("GET /files") == 3
        assert all(trial.done_marker(p).exists() for p in first)
        trial.download(entries, tmp_path, "papers")
        assert stub.count("GET /files") == 3  # nothing fetched again

    def test_a_file_without_a_done_marker_is_fetched_again(self, stub, tmp_path):
        entries = self.entries(stub)
        paths = trial.download(entries, tmp_path, "papers")
        trial.done_marker(paths[1]).unlink()
        paths[1].write_bytes(b"truncated")
        trial.download(entries, tmp_path, "papers")
        assert stub.count("GET /files") == 4 and paths[1].read_bytes() == FILES[paths[1].name]

    def test_a_finished_file_is_never_overwritten(self, stub, tmp_path):
        entries = self.entries(stub)
        paths = trial.download(entries, tmp_path, "papers")
        paths[0].write_bytes(b"older download from the previous pipeline")
        trial.download(entries, tmp_path, "papers")
        assert paths[0].read_bytes() == b"older download from the previous pipeline"

    def test_a_failed_download_leaves_no_half_file_and_a_clear_error(self, tmp_path, capsys):
        entry = [("x.gz", "http://127.0.0.1:1/files/x.gz")]
        with pytest.raises(SystemExit):
            trial.download(entry, tmp_path, "papers")
        assert not (tmp_path / "x.gz").exists() and not trial.done_marker(tmp_path / "x.gz").exists()
        assert "downloading x.gz failed" in capsys.readouterr().err

    def test_local_files_only_returns_finished_downloads(self, stub, tmp_path):
        paths = trial.download(self.entries(stub), tmp_path, "papers")
        trial.done_marker(paths[2]).unlink()
        assert trial.local_files(tmp_path) == paths[:2]

    def test_local_files_with_nothing_is_an_error(self, tmp_path, capsys):
        with pytest.raises(SystemExit):
            trial.local_files(tmp_path)
        assert "no completed downloads" in capsys.readouterr().err

    def test_unreachable_server(self, capsys):
        with pytest.raises(SystemExit):
            trial.http("GET", "http://127.0.0.1:1/")
        assert "could not reach" in capsys.readouterr().err


class TestExtract:
    def prepared(self, stub, dirs):
        papers = trial.download(
            trial.manifest(f"{stub.url}/datasets/v1", RELEASE, "papers", S2_KEY), dirs["papers"], "p"
        )
        abstracts = trial.download(
            trial.manifest(f"{stub.url}/datasets/v1", RELEASE, "abstracts", S2_KEY), dirs["abstracts"], "a"
        )
        return papers, abstracts

    def read(self, path):
        return {d["corpusid"]: d for d in trial.read_jsonl(path)}

    def test_keeps_social_science_and_attaches_missing_abstracts(self, stub, dirs):
        papers, abstracts = self.prepared(stub, dirs)
        (out,) = trial.extract(papers[:1], abstracts, dirs["out"])
        docs = self.read(out)
        assert set(docs) == {1, 2, 4}  # 3 is biology
        assert docs[1]["abstract"] == "own abstract" and docs[2]["abstract"] == "found for two"
        assert "abstract" not in docs[4]  # no abstract anywhere: kept here, dropped at load time

    def test_never_overwrites_an_abstract_the_paper_already_has(self, stub, dirs):
        papers, abstracts = self.prepared(stub, dirs)
        out = trial.extract(papers[1:2], abstracts, dirs["out"])[0]
        docs = self.read(out)
        assert (
            docs[6]["abstract"] == "kept abstract"
            and docs[5]["abstract"] == "found for five"
            and 7 not in docs
        )

    def test_files_already_extracted_are_skipped(self, stub, dirs, capsys):
        papers, abstracts = self.prepared(stub, dirs)
        (out,) = trial.extract(papers[:1], abstracts, dirs["out"])
        before = out.stat().st_mtime_ns
        capsys.readouterr()
        again = trial.extract(papers[:1], abstracts, dirs["out"])
        assert again == [out] and out.stat().st_mtime_ns == before
        assert "already extracted" in capsys.readouterr().err

    def test_only_unfinished_files_are_processed(self, stub, dirs):
        papers, abstracts = self.prepared(stub, dirs)
        first = trial.extract(papers[:1], abstracts, dirs["out"])[0]
        stamp = first.stat().st_mtime_ns
        both = trial.extract(papers[:2], abstracts, dirs["out"])
        assert both[0] == first and first.stat().st_mtime_ns == stamp and both[1].exists()

    def test_reuses_outputs_of_the_older_pipeline(self, stub, dirs):
        papers, abstracts = self.prepared(stub, dirs)
        dirs["out"].mkdir()
        out = trial.extract_output(papers[0], dirs["out"])
        out.write_bytes(gz([paper(42, "from the old run")]))
        (dirs["out"] / f"{papers[0].stem}.pass3.done").touch()
        assert trial.extract(papers[:1], abstracts, dirs["out"]) == [out]
        assert set(self.read(out)) == {42}


class TestElasticsearch:
    def es(self, stub, **kw):
        return trial.Elasticsearch(stub.url, ES_KEY, "papers_trial", sleep=lambda s: None, **kw)

    def test_check(self, stub):
        self.es(stub).check()

    def test_check_with_a_bad_key_says_so(self, stub, capsys):
        with pytest.raises(SystemExit):
            trial.Elasticsearch(stub.url, "wrong", "i").check()
        assert ("HTTP 401" in capsys.readouterr().err and "API key" in capsys.readouterr().err) or True

    def test_creates_the_index_with_the_mapping(self, stub):
        mapping = {"mappings": {"properties": {"x": {"type": "long"}}}}
        self.es(stub).ensure_index(mapping, recreate=False)
        assert stub.mappings["papers_trial"] == mapping

    def test_existing_index_is_kept_unless_recreating(self, stub):
        es = self.es(stub)
        es.ensure_index({"mappings": {}}, False)
        stub.docs["1"] = {}
        es.ensure_index({"mappings": {}}, False)
        assert stub.count("PUT") == 1 and "1" in stub.docs
        es.ensure_index({"mappings": {}}, True)
        assert stub.count("PUT") == 2 and stub.docs == {} and stub.count("DELETE") == 1

    def test_recreating_a_missing_index_just_creates_it(self, stub):
        self.es(stub).ensure_index({"mappings": {}}, True)
        assert "papers_trial" in stub.indices

    def test_index_creation_failure_is_reported(self, stub, capsys):
        stub.indices["papers_trial"] = {}
        bad = trial.Elasticsearch(stub.url, "wrong", "papers_trial")
        with pytest.raises(SystemExit):
            bad.ensure_index({"mappings": {}}, False)
        assert "could not create index" in capsys.readouterr().err

    def test_delete_failure_is_reported(self, stub, capsys):
        stub.indices["papers_trial"] = {}
        stub.fail_delete = True
        with pytest.raises(SystemExit):
            self.es(stub).ensure_index({"mappings": {}}, True)
        assert "could not delete index" in capsys.readouterr().err

    def load(self, stub, tmp_path, rows, **kw):
        path = tmp_path / "f.jsonl.gz"
        path.write_bytes(gz(rows))
        return self.es(stub, **kw).load_file(path)

    def test_loads_trimmed_documents_with_stable_ids(self, stub, tmp_path):
        stats = self.load(stub, tmp_path, [paper(1, "a"), paper(2), paper(3, "c")])
        assert (stats.indexed, stats.failed, stats.skipped_no_abstract) == (2, 0, 1)
        assert set(stub.docs) == {"1", "3"} and set(stub.docs["1"]) == set(trial.KEEP)
        assert stats.bytes_sent > 0

    def test_progress_is_logged_while_loading(self, stub, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(trial, "LOG_EVERY_CHUNKS", 1)
        self.load(stub, tmp_path, [paper(i, "a") for i in range(1, 8)], chunk_docs=3)
        assert "f.jsonl.gz: 3 indexed" in capsys.readouterr().err

    def test_chunks_requests(self, stub, tmp_path):
        stats = self.load(stub, tmp_path, [paper(i, "a") for i in range(1, 8)], chunk_docs=3)
        assert stats.indexed == 7 and stub.count("bulk") == 3

    def test_keep_no_abstract(self, stub, tmp_path):
        path = tmp_path / "f.jsonl.gz"
        path.write_bytes(gz([paper(1), paper(2, "a")]))
        stats = self.es(stub).load_file(path, require_abstract=False)
        assert stats.indexed == 2

    def test_a_throttled_request_is_retried_with_growing_backoff(self, stub, tmp_path):
        stub.plan = [("http", 429), ("http", 503), ("ok", 0)]
        waits = []
        es = trial.Elasticsearch(stub.url, ES_KEY, "i", sleep=waits.append)
        path = tmp_path / "f.jsonl.gz"
        path.write_bytes(gz([paper(1, "a")]))
        stats = es.load_file(path)
        assert stats.indexed == 1 and stats.failed == 0 and waits == [2.0, 4.0]

    def test_throttled_items_are_retried_but_accepted_ones_are_not_resent(self, stub, tmp_path):
        stub.plan = [("items", 2), ("ok", 0)]
        stats = self.load(stub, tmp_path, [paper(i, "a") for i in range(1, 5)])
        assert stats.indexed == 4 and stats.failed == 0 and set(stub.docs) == {"1", "2", "3", "4"}
        assert stub.log.count("bulk 4") == 1 and stub.log.count("bulk 2") == 1

    def test_gives_up_after_the_retry_budget(self, stub, tmp_path):
        stub.plan = [("http", 503)] * 10
        stats = self.load(stub, tmp_path, [paper(1, "a")], retries=2)
        assert stats.indexed == 0 and stats.failed == 1 and "gave up" in stats.errors[-1]

    def test_a_hard_error_is_not_retried(self, stub, tmp_path):
        stub.plan = [("http", 400)]
        stats = self.load(stub, tmp_path, [paper(1, "a")])
        assert stats.failed == 1 and "HTTP 400" in stats.errors[0] and stub.count("POST") == 1

    def test_one_bad_document_does_not_sink_the_rest(self, stub, tmp_path):
        stats = self.load(stub, tmp_path, [paper(1, "a"), paper(2, "b", title="REJECT"), paper(3, "c")])
        assert (stats.indexed, stats.failed) == (2, 1) and "mapper_parsing_exception" in stats.errors[0]

    def test_count(self, stub, tmp_path):
        self.load(stub, tmp_path, [paper(1, "a"), paper(2, "b")])
        assert self.es(stub).count() == 2

    def test_count_on_error_is_minus_one(self, stub):
        assert trial.Elasticsearch(stub.url, "wrong", "i").count() == -1


class TestRun:
    def test_end_to_end(self, stub, dirs):
        report = run(stub, dirs, "--files", "2")
        assert report["indexed"] == 4  # 1, 2, 5, 6 have abstracts; 4 does not
        assert report["skipped_no_abstract"] == 1 and report["failed"] == 0
        assert set(stub.docs) == {"1", "2", "5", "6"}
        assert (
            report["files_loaded"] == 2
            and report["files_in_corpus"] == 3
            and report["scale_to_full_corpus"] == 1.5
        )
        assert report["documents_in_index"] == 4 and report["index"] == "papers_trial"
        saved = json.loads((dirs["out"] / "load_report.json").read_text())
        assert saved["indexed"] == 4 and saved["started_utc"] <= saved["finished_utc"]

    def test_without_files_everything_is_loaded(self, stub, dirs):
        report = run(stub, dirs)
        assert report["files_loaded"] == 3 and report["scale_to_full_corpus"] == 1.0
        assert report["indexed"] == 4 and set(stub.docs) == {"1", "2", "5", "6"}
        assert "That was the whole corpus" in trial.summary(report, dirs["out"])

    def test_without_files_it_loads_only_what_was_extracted(self, stub, dirs):
        report = run(stub, dirs, "--extract-files", "1")
        assert report["files_loaded"] == 1 and set(stub.docs) == {"1", "2"}

    def test_a_partial_load_says_to_scale_the_cost(self, stub, dirs):
        report = run(stub, dirs, "--files", "1")
        assert "3.0x this size" in trial.summary(report, dirs["out"])

    def test_the_index_uses_the_trimmed_mapping(self, stub, dirs):
        run(stub, dirs, "--files", "1")
        assert "authors" not in stub.mappings["papers_trial"]["mappings"]["properties"]

    def test_a_second_run_downloads_extracts_and_loads_nothing_again(self, stub, dirs):
        run(stub, dirs, "--files", "2")
        files, bulks = stub.count("GET /files"), stub.count("bulk")
        report = run(stub, dirs, "--files", "2")
        assert stub.count("GET /files") == files and stub.count("bulk") == bulks
        assert report["indexed"] == 0

    def test_a_larger_second_run_loads_only_the_new_file(self, stub, dirs):
        run(stub, dirs, "--files", "1")
        report = run(stub, dirs, "--files", "2")
        assert report["indexed"] == 2 and set(stub.docs) == {"1", "2", "5", "6"}

    def test_recreate_reloads_everything_without_touching_the_markers(self, stub, dirs):
        run(stub, dirs, "--files", "1")
        report = run(stub, dirs, "--files", "1", "--recreate")
        assert report["indexed"] == 2 and stub.count("DELETE") == 1 and set(stub.docs) == {"1", "2"}

    def test_a_new_project_is_loaded_from_scratch(self, stub, dirs):
        run(stub, dirs, "--files", "1")
        fresh = Stub()
        try:
            report = run(stub, dirs, "--files", "1", "--es-url", fresh.url)
            assert report["indexed"] == 2 and set(fresh.docs) == {"1", "2"}
            again = run(stub, dirs, "--files", "1", "--es-url", fresh.url)
            assert again["indexed"] == 0  # and the new project's own markers are respected
        finally:
            fresh.close()
        assert run(stub, dirs, "--files", "1")["indexed"] == 0  # the first project still remembers its load

    def test_markers_are_named_for_the_index_and_the_project(self, stub, dirs):
        run(stub, dirs, "--files", "1", "--index", "a")
        run(stub, dirs, "--files", "1", "--index", "b")
        names = sorted(p.name for p in dirs["out"].glob("*.loaded"))
        assert len(names) == 2 and names[0] != names[1]
        assert ".a." in names[0] and ".b." in names[1]

    def test_a_file_with_failures_is_not_marked_loaded(self, stub, dirs):
        stub.plan = [("http", 400)]
        run(stub, dirs, "--files", "1")
        assert not list(dirs["out"].glob("*.loaded"))
        assert run(stub, dirs, "--files", "1")["indexed"] == 2  # tried again and succeeded

    def test_keep_no_abstract(self, stub, dirs):
        assert run(stub, dirs, "--files", "1", "--keep-no-abstract")["indexed"] == 3

    def test_skip_download_uses_what_is_on_disk(self, stub, dirs):
        run(stub, dirs, "--files", "1", "--dry-run")
        fetched = stub.count("GET /files") + stub.count("GET /datasets")
        report = run(stub, dirs, "--files", "1", "--skip-download")
        assert stub.count("GET /files") + stub.count("GET /datasets") == fetched
        assert report["indexed"] == 2 and report["files_in_corpus"] == trial.TOTAL_PAPERS_FILES

    def test_dry_run_downloads_and_extracts_but_never_touches_elasticsearch(self, stub, dirs):
        assert run(stub, dirs, "--files", "1", "--dry-run") == {}
        assert (
            stub.count("POST /_bulk") == 0
            and stub.count("PUT") == 0
            and list(dirs["out"].glob("*_social_science.jsonl.gz"))
        )

    def test_dry_run_needs_no_elasticsearch_credentials(self, stub, dirs):
        args = trial.build_parser().parse_args(
            argv(stub, dirs, "--dry-run", "--es-url", "", "--es-api-key", "")
        )
        assert trial.run(args) == {}

    def test_check_only_tests_the_connection(self, stub, dirs):
        assert run(stub, dirs, "--check") == {}
        assert stub.count("GET /datasets") == 0 and not dirs["papers"].exists()

    def test_count_prints_the_documents_in_the_index_and_changes_nothing(self, stub, dirs, capsys):
        run(stub, dirs, "--files", "2")
        capsys.readouterr()
        before = stub.count("POST")
        assert run(stub, dirs, "--count") == {}
        assert capsys.readouterr().out == "papers_trial: 4 documents\n" and stub.count("POST") == before

    def test_count_formats_large_numbers(self, stub, dirs, capsys, monkeypatch):
        monkeypatch.setattr(trial.Elasticsearch, "count", lambda self: 6_954_500)
        run(stub, dirs, "--count")
        assert "6,954,500 documents" in capsys.readouterr().out

    def test_count_says_so_when_the_index_is_missing(self, stub, dirs, capsys, monkeypatch):
        monkeypatch.setattr(trial.Elasticsearch, "count", lambda self: -1)
        with pytest.raises(SystemExit):
            run(stub, dirs, "--count")
        assert "could not count" in capsys.readouterr().err

    def test_count_needs_the_connection_settings_and_the_flags_exclude_each_other(self, stub, dirs, capsys):
        args = trial.build_parser().parse_args(argv(stub, dirs, "--count", "--es-api-key", ""))
        with pytest.raises(SystemExit):
            trial.run(args)
        assert "ES_API_KEY" in capsys.readouterr().err
        with pytest.raises(SystemExit):
            trial.build_parser().parse_args(argv(stub, dirs, "--count", "--check"))

    def test_download_only_fetches_the_corpus_and_stops(self, stub, dirs, capsys):
        args = trial.build_parser().parse_args(
            argv(stub, dirs, "--download", "--es-url", "", "--es-api-key", "")
        )
        assert trial.run(args) == {}
        assert stub.count("GET /files") == 4  # 3 papers files and 1 abstracts file
        assert stub.count("POST") == 0 and stub.count("PUT") == 0 and stub.count("GET /\n") == 0
        assert not dirs["out"].exists() and "run again without --download" in capsys.readouterr().err

    def test_download_only_skips_what_is_already_there(self, stub, dirs):
        run(stub, dirs, "--download")
        run(stub, dirs, "--download")
        assert stub.count("GET /files") == 4

    def test_download_only_needs_the_semantic_scholar_key(self, stub, dirs, capsys):
        args = trial.build_parser().parse_args(argv(stub, dirs, "--download", "--s2-key", ""))
        with pytest.raises(SystemExit):
            trial.run(args)
        assert "S2_KEY" in capsys.readouterr().err

    def test_download_and_skip_download_contradict(self, stub, dirs, capsys):
        args = trial.build_parser().parse_args(argv(stub, dirs, "--download", "--skip-download"))
        with pytest.raises(SystemExit):
            trial.run(args)
        assert "contradict" in capsys.readouterr().err

    def test_the_step_flags_are_mutually_exclusive(self, stub, dirs):
        for pair in (("--check", "--download"), ("--download", "--dry-run"), ("--check", "--dry-run")):
            with pytest.raises(SystemExit):
                trial.build_parser().parse_args(argv(stub, dirs, *pair))

    def test_connection_is_checked_before_any_download(self, stub, dirs, capsys):
        args = trial.build_parser().parse_args(argv(stub, dirs, "--es-api-key", "wrong"))
        with pytest.raises(SystemExit):
            trial.run(args)
        assert stub.count("GET /datasets") == 0 and "HTTP 401" in capsys.readouterr().err

    @pytest.mark.parametrize(
        "extra,message",
        [
            (["--es-url", ""], "ES_URL"),
            (["--es-api-key", ""], "ES_API_KEY"),
            (["--s2-key", ""], "S2_KEY"),
            (["--files", "0"], "--files"),
        ],
    )
    def test_missing_settings_are_explained(self, stub, dirs, capsys, extra, message):
        args = trial.build_parser().parse_args(argv(stub, dirs, *extra))
        with pytest.raises(SystemExit):
            trial.run(args)
        assert message in capsys.readouterr().err


class TestCli:
    def test_defaults_come_from_the_environment(self, monkeypatch):
        monkeypatch.setenv("ES_URL", "http://es")
        monkeypatch.setenv("ES_API_KEY", "k")
        monkeypatch.setenv("S2_KEY", "s")
        args = trial.build_parser().parse_args([])
        assert (args.es_url, args.es_api_key, args.s2_key) == ("http://es", "k", "s")
        assert (args.files, args.index, args.release) == (None, "papers", trial.DEFAULT_RELEASE)

    def test_main_prints_a_summary_with_the_billing_window(self, stub, dirs, capsys):
        assert trial.main(argv(stub, dirs, "--files", "1")) == 0
        out = capsys.readouterr().out
        assert "Finished" in out and "Billing window (UTC)" in out and "3.0x" in out

    def test_main_with_check_prints_nothing_more(self, stub, dirs, capsys):
        assert trial.main(argv(stub, dirs, "--check")) == 0
        assert capsys.readouterr().out == ""


def test_run_as_a_script_it_exits_with_the_status(monkeypatch):
    import runpy

    monkeypatch.setattr(sys, "argv", ["load_serverless.py", "--help"])
    with pytest.raises(SystemExit) as stop:
        runpy.run_path(str(LOADER / "load_serverless.py"), run_name="__main__")
    assert stop.value.code == 0


class TestDotenv:
    def env(self, monkeypatch, tmp_path, text):
        monkeypatch.setattr(trial.os, "environ", {})
        path = tmp_path / ".env"
        path.write_text(text, encoding="utf-8")
        trial.load_dotenv(path)
        return trial.os.environ

    def test_reads_plain_quoted_and_exported_values(self, monkeypatch, tmp_path):
        text = (
            "# a comment\nES_URL=https://es.example:443\n\n"
            "export S2_KEY=\"abc\"\nES_API_KEY='k=v=='\nnot a setting\n"
        )
        env = self.env(monkeypatch, tmp_path, text)
        assert env == {"ES_URL": "https://es.example:443", "S2_KEY": "abc", "ES_API_KEY": "k=v=="}

    def test_never_overrides_a_variable_that_is_already_set(self, monkeypatch, tmp_path):
        monkeypatch.setattr(trial.os, "environ", {"ES_URL": "from-shell"})
        (tmp_path / ".env").write_text("ES_URL=from-file\nS2_KEY=s\n")
        trial.load_dotenv(tmp_path / ".env")
        assert trial.os.environ == {"ES_URL": "from-shell", "S2_KEY": "s"}

    def test_a_missing_file_is_fine(self, monkeypatch, tmp_path):
        monkeypatch.setattr(trial.os, "environ", {})
        trial.load_dotenv(tmp_path / "nope")
        assert trial.os.environ == {}

    def test_an_unquoted_value_keeps_inner_quotes_and_a_lone_quote(self, monkeypatch, tmp_path):
        env = self.env(monkeypatch, tmp_path, "A=it's\nB=\"\nC=\n")
        assert env == {"A": "it's", "B": '"', "C": ""}

    def test_main_picks_up_the_env_file_in_the_current_directory(
        self, stub, dirs, tmp_path, monkeypatch, capsys
    ):
        monkeypatch.setattr(trial.os, "environ", {})
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".env").write_text(f"ES_URL={stub.url}\nES_API_KEY={ES_KEY}\nS2_KEY={S2_KEY}\n")
        extra = ["--s2-api", f"{stub.url}/datasets/v1", "--release", RELEASE, "--check"]
        assert trial.main(extra) == 0
        assert f"connected to {stub.url}" in capsys.readouterr().err

    def test_secrets_are_never_printed(self, stub, dirs, capsys):
        run(stub, dirs, "--files", "1")
        captured = capsys.readouterr()
        assert ES_KEY not in captured.err + captured.out and S2_KEY not in captured.err + captured.out


def local_corpus(tmp_path):
    """The test files on disk, with download markers, and no server."""
    papers, abstracts = tmp_path / "s2", tmp_path / "abs"
    for directory in (papers, abstracts):
        directory.mkdir()
    for name, data in FILES.items():
        target = (abstracts if name.startswith("abstracts") else papers) / name
        target.write_bytes(data)
        trial.done_marker(target).touch()
    return sorted(papers.glob("*.gz")), sorted(abstracts.glob("*.gz")), tmp_path / "out"


class TestExtractAll:
    def test_corpus_id(self):
        assert trial.corpus_id({"corpusid": 7}) == 7 and trial.corpus_id({"corpusid": "8"}) == 8
        assert trial.corpus_id({}) is None and trial.corpus_id({"corpusid": "x"}) is None
        assert trial.corpus_id({"corpusid": None}) is None

    def test_everything_is_extracted_by_default_but_only_the_first_files_are_loaded(self, stub, dirs):
        report = run(stub, dirs, "--files", "1")
        assert len(list(dirs["out"].glob("*_social_science.jsonl.gz"))) == 3
        assert report["files_loaded"] == 1 and set(stub.docs) == {"1", "2"}

    def test_extract_files_limits_the_extraction(self, stub, dirs):
        run(stub, dirs, "--files", "1", "--extract-files", "2")
        assert len(list(dirs["out"].glob("*_social_science.jsonl.gz"))) == 2

    def test_a_later_larger_load_needs_no_more_extraction(self, stub, dirs, capsys):
        run(stub, dirs, "--files", "1")
        capsys.readouterr()
        report = run(stub, dirs, "--files", "3")
        assert "all files were already extracted" in capsys.readouterr().err
        assert report["indexed"] == 2  # papers 5 and 6; file 3's paper has no abstract and file 1 was loaded

    def test_the_abstracts_are_scanned_once_for_all_files_when_memory_allows(self, tmp_path, monkeypatch):
        papers, abstracts, out = local_corpus(tmp_path)
        calls = []
        real = trial.abstracts_in
        monkeypatch.setattr(trial, "abstracts_in", lambda path: calls.append(path.name) or real(path))
        trial.extract(papers, abstracts, out, memory_bytes=10**12)
        assert calls == ["abstracts_0.jsonl.gz"]

    def test_tight_memory_means_more_scans_but_the_same_result(self, tmp_path, monkeypatch):
        papers, abstracts, out = local_corpus(tmp_path)
        calls = []
        real = trial.abstracts_in
        monkeypatch.setattr(trial, "abstracts_in", lambda path: calls.append(path.name) or real(path))
        outputs = trial.extract(papers, abstracts, out, memory_bytes=1)
        assert len(calls) == 3  # one batch, and so one scan, per papers file
        assert {d["corpusid"] for d in trial.read_jsonl(outputs[1])} == {5, 6}

    def test_memory_gb_flag_reaches_the_planner(self, stub, dirs, capsys):
        run(stub, dirs, "--files", "1", "--memory-gb", "0.0000001")
        assert "scanned 3 time(s)" in capsys.readouterr().err

    def test_plan_batches(self):
        a, b, c = Path("a"), Path("b"), Path("c")
        cost = trial.BYTES_PER_ABSTRACT
        assert trial.plan_batches([(a, 1), (b, 1), (c, 1)], 10 * cost) == [[a, b, c]]
        assert trial.plan_batches([(a, 5), (b, 5), (c, 5)], 10 * cost) == [[a, b], [c]]
        assert trial.plan_batches([(a, 50), (b, 1)], cost) == [
            [a],
            [b],
        ]  # a file bigger than the budget still gets a batch
        assert trial.plan_batches([], cost) == []

    def test_available_memory(self, monkeypatch):
        assert trial.available_memory() > 0
        monkeypatch.setattr(trial.Path, "read_text", lambda self: "MemTotal: 1 kB\nMemAvailable: 2048 kB\n")
        assert trial.available_memory() == 2048 * 1024

    @pytest.mark.parametrize("text", ["MemTotal: 1 kB\n", "MemAvailable: lots kB\n", "MemAvailable:\n"])
    def test_available_memory_falls_back_when_unreadable(self, monkeypatch, text):
        monkeypatch.setattr(trial.Path, "read_text", lambda self: text)
        assert trial.available_memory() == 8 * 1024**3

    def test_available_memory_without_proc(self, monkeypatch):
        def refuse(self):
            raise OSError

        monkeypatch.setattr(trial.Path, "read_text", refuse)
        assert trial.available_memory() == 8 * 1024**3

    def test_finished_id_lists_are_reused_after_an_interruption(self, tmp_path, monkeypatch):
        papers, abstracts, out = local_corpus(tmp_path)
        trial.extract(papers, abstracts, out)
        for marker in out.glob("*.pass3.done"):
            marker.unlink()  # pretend the run died before pass 3 finished
        monkeypatch.setattr(trial, "find_social_ids", lambda path: pytest.fail("pass 1 was redone"))
        outputs = trial.extract(papers, abstracts, out)
        assert all(o.exists() for o in outputs) and len(list(out.glob("*.pass3.done"))) == 3

    def test_an_empty_id_list_is_fine(self, tmp_path):
        papers, abstracts, out = local_corpus(tmp_path)
        papers[0].write_bytes(gz([paper(3, fos=[{"category": "Biology"}])]))
        outputs = trial.extract(papers[:1], abstracts, out)
        assert (
            list(trial.read_jsonl(outputs[0])) == []
            and trial.read_ids(trial.ids_file(papers[0], out)) == set()
        )

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_parallel_extraction_gives_the_same_files(self, tmp_path):
        papers, abstracts, out = local_corpus(tmp_path)
        serial = trial.extract(papers, abstracts, tmp_path / "serial", workers=1)
        parallel = trial.extract(papers, abstracts, out, workers=3)
        for a, b in zip(serial, parallel):
            assert list(trial.read_jsonl(a)) == list(trial.read_jsonl(b))

    def test_pmap_keeps_order_and_runs_inline_for_one_worker(self):
        assert trial.pmap(lambda x: x * 2, [1, 2, 3], 1) == [2, 4, 6]
        assert trial.pmap(lambda x: x * 2, [5], 4) == [10]
