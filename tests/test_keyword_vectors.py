"""devops/tools/keyword_vectors.py: embed the OpenAlex keywords into one matrix."""

import gzip
import importlib.util
import json
import os
import struct
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
spec = importlib.util.spec_from_file_location(
    "keyword_vectors", ROOT / "devops" / "tools" / "keyword_vectors.py"
)
kv = importlib.util.module_from_spec(spec)
sys.modules["keyword_vectors"] = kv
spec.loader.exec_module(kv)

RECORDS = [
    {
        "id": "https://openalex.org/keywords/a",
        "display_name": "span of control",
        "works_count": 3,
        "display_name_alternatives": ["span of supervision", "managerial span"],
        "description": "People per manager.",
    },
    {
        "id": "https://openalex.org/keywords/b",
        "display_name": "lbos",
        "works_count": None,
        "display_name_alternatives": None,
    },
]


def snapshot():
    blob = gzip.compress("\n".join(json.dumps(r) for r in RECORDS).encode())
    manifest = {"date": "2026-10-14", "files": [{"url": "s3://openalex/data/jsonl/keywords/p/part_0.gz",
                                                    "meta": {"content_length": len(blob)}}]}  # fmt: skip

    def get(url):
        return json.dumps(manifest).encode() if url.endswith("manifest.json") else blob

    return get


def fake_post(calls, fail=()):
    def post(url, headers, body):
        payload = json.loads(body)
        calls.append(payload)
        if fail and len(calls) <= len(fail):
            raise kv.HttpError(fail[len(calls) - 1], "busy", "0")
        dims = payload.get("dimensions", 4)
        data = [
            {"index": i, "embedding": [float(i + 1)] + [0.0] * (dims - 1)}
            for i in range(len(payload["input"]))
        ]
        return json.dumps({"data": data[::-1]}).encode()  # out of order, as the API may

    return post


def test_keyword_text_has_the_name_alternatives_and_optionally_the_description():
    assert kv.keyword_text(RECORDS[0]) == "span of control | span of supervision | managerial span"
    assert kv.keyword_text(RECORDS[0], True).endswith(". People per manager.")
    assert kv.keyword_text(RECORDS[1], True) == "lbos"
    many = {"display_name": "x", "display_name_alternatives": [str(i) for i in range(20)]}
    assert kv.keyword_text(many).count("|") == kv.MAX_ALTERNATIVES
    assert len(kv.keyword_text({"display_name": "x" * 5000})) == kv.MAX_CHARS


def test_download_reads_the_manifest_and_skips_files_already_there(tmp_path):
    get = snapshot()
    paths, date = kv.download_snapshot(tmp_path, get)
    assert date == "2026-10-14" and len(paths) == 1 and paths[0].exists()
    rows = list(kv.read_keywords(paths))
    assert [r["id"] for r in rows] == ["a", "b"] and rows[1]["works"] == 0
    stamp = paths[0].stat().st_mtime_ns
    kv.download_snapshot(tmp_path, get)
    assert paths[0].stat().st_mtime_ns == stamp


def test_estimate_prices_known_models_and_sizes_the_matrix():
    rows = [{"text": "x" * 400}] * 10  # 100 tokens each
    cost = kv.estimate(rows, "text-embedding-3-small", 512)
    assert cost["tokens"] == 1000 and cost["usd"] == 0.0 and cost["matrix_gb_float16"] == 0.0
    big = kv.estimate([{"text": "x" * 4_000_000}] * 1000, "text-embedding-3-large", 512)
    assert big["usd"] == 130.0 and big["matrix_gb_float32"] == round(1000 * 512 * 4 / 1e9, 2)
    assert kv.estimate(rows, "some-new-model", 8)["usd"] is None


def test_embed_asks_for_the_dimensions_normalizes_and_restores_the_order():
    calls = []
    vectors = kv.embed(["a", "b"], "text-embedding-3-small", 4, "k", fake_post(calls))
    assert calls[0]["dimensions"] == 4 and vectors == [[1.0, 0, 0, 0], [1.0, 0, 0, 0]]
    kv.embed(["a"], "text-embedding-ada-002", 4, "k", fake_post(calls))
    assert "dimensions" not in calls[1]  # the old model cannot be shortened


def test_embed_retries_when_rate_limited_and_gives_up_on_other_errors():
    calls, waits = [], []
    assert kv.embed(["a"], "text-embedding-3-small", 4, "k", fake_post(calls, fail=(429, 503)), waits.append)
    assert len(calls) == 3 and waits == [0.0, 0.0]
    with pytest.raises(kv.HttpError, match="HTTP 401"):
        kv.embed(["a"], "text-embedding-3-small", 4, "k", fake_post([], fail=(401,)), lambda s: None)
    with pytest.raises(kv.HttpError):
        kv.embed(
            ["a"], "text-embedding-3-small", 4, "k", fake_post([], fail=(429,) * 3), lambda s: None, tries=3
        )


def test_retry_waits_grow_without_a_retry_after_header():
    waits, state = [], {"n": 0}

    def post(url, headers, body):
        state["n"] += 1
        if state["n"] < 3:
            raise kv.HttpError(429, "slow", None)
        return json.dumps({"data": [{"index": 0, "embedding": [3.0, 4.0]}]}).encode()

    assert kv.embed(["a"], "text-embedding-3-small", 2, "k", post, waits.append) == [[0.6, 0.8]]
    assert waits == [1.0, 2.0]


def test_normalize_leaves_a_zero_vector_alone():
    assert kv.normalize([0.0, 0.0]) == [0.0, 0.0]


def test_build_writes_the_matrix_names_and_metadata_and_cleans_up(tmp_path):
    calls = []
    meta = kv.build(
        tmp_path, "text-embedding-3-small", 4, None, False, 5.0, "k", fake_post(calls), snapshot()
    )
    assert meta["count"] == 2 and meta["dims"] == 4 and meta["snapshot"] == "2026-10-14"
    assert (tmp_path / "vectors.f16").stat().st_size == 2 * 4 * 2
    assert not list((tmp_path / "shards").glob("shard_*.f16"))
    names = [json.loads(line)["name"] for line in (tmp_path / "keywords.jsonl").read_text().splitlines()]
    assert names == ["span of control", "lbos"] and len(calls) == 1
    values = struct.unpack("<8e", (tmp_path / "vectors.f16").read_bytes())
    assert values[:4] == (1.0, 0.0, 0.0, 0.0)


def test_build_resumes_without_re_embedding_finished_shards(tmp_path):
    calls = []
    kv.build(tmp_path, "text-embedding-3-small", 4, None, False, 5.0, "k", fake_post(calls), snapshot())
    shard = tmp_path / "shards" / "shard_00000.f16"
    shard.write_bytes(b"\0" * (2 * 4 * 2))  # a finished shard left by an interrupted run
    kv.build(tmp_path, "text-embedding-3-small", 4, None, False, 5.0, "k", fake_post(calls), snapshot())
    assert len(calls) == 1  # the second run found the shard and made no request


def test_a_change_of_model_or_size_throws_away_old_shards(tmp_path):
    calls = []
    kv.build(tmp_path, "text-embedding-3-small", 4, None, False, 5.0, "k", fake_post(calls), snapshot())
    stale = tmp_path / "shards" / "shard_00000.f16"
    stale.write_bytes(b"\0" * (2 * 4 * 2))  # looks finished: right size for 2 keywords at 4 dimensions
    kv.build(tmp_path, "text-embedding-3-large", 4, None, False, 5.0, "k", fake_post(calls), snapshot())
    assert len(calls) == 2  # same size, different model: embedded again, not reused


def test_build_with_a_limit_and_a_budget(tmp_path):
    calls = []
    meta = kv.build(tmp_path, "text-embedding-3-small", 4, 1, True, 5.0, "k", fake_post(calls), snapshot())
    assert meta["count"] == 1 and meta["with_description"] is True
    with pytest.raises(SystemExit, match="over --max-usd"):
        kv.build(
            tmp_path / "x", "text-embedding-3-large", 4, None, False, -1.0, "k", fake_post([]), snapshot()
        )


def test_build_over_several_shards(tmp_path, monkeypatch):
    monkeypatch.setattr(kv, "SHARD", 1)
    monkeypatch.setattr(kv, "MAX_BATCH", 1)
    calls = []
    kv.build(tmp_path, "text-embedding-3-small", 4, None, False, 5.0, "k", fake_post(calls), snapshot())
    assert len(calls) == 2 and (tmp_path / "vectors.f16").stat().st_size == 16


def test_nearest_ranks_by_cosine_similarity(tmp_path):
    pytest.importorskip("numpy")
    meta = {"model": "text-embedding-3-small", "dims": 2, "count": 3}
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    (tmp_path / "vectors.f16").write_bytes(pack_rows([[1, 0], [0, 1], [0.8, 0.6]]))
    names = [{"name": n, "works": 1} for n in ("x-axis", "y-axis", "between")]
    (tmp_path / "keywords.jsonl").write_text("\n".join(json.dumps(n) for n in names))

    def post(url, headers, body):
        return json.dumps({"data": [{"index": 0, "embedding": [1.0, 0.0]}]}).encode()

    found = kv.nearest(tmp_path, "anything", 2, "k", post)
    assert [row["name"] for _, row in found] == ["x-axis", "between"] and found[0][0] == pytest.approx(1.0)


def pack_rows(rows):
    return b"".join(struct.pack("<2e", *r) for r in rows)


def test_main_estimate_build_and_query(tmp_path, monkeypatch, capsys):
    real = kv.download_snapshot
    monkeypatch.setattr(kv, "download_snapshot", lambda folder: real(folder, snapshot()))
    monkeypatch.setattr(kv, "build", lambda *args, **kwargs: {"built": True})
    monkeypatch.setattr(
        kv, "nearest", lambda *args, **kwargs: [(0.9, {"name": "span of control", "works": 3})]
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert kv.main(["estimate", "--out", str(tmp_path), "--limit", "5"]) == 0
    assert '"keywords": 2' in capsys.readouterr().out
    with pytest.raises(SystemExit, match="OPENAI_API_KEY"):
        kv.main(["build", "--out", str(tmp_path)])
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    assert kv.main(["build", "--out", str(tmp_path)]) == 0 and '"built": true' in capsys.readouterr().out
    assert kv.main(["query", "span", "--out", str(tmp_path)]) == 0
    assert "span of control" in capsys.readouterr().out


def built(tmp_path, count=3):
    """A small finished build: `count` keywords with 4-dimensional vectors."""
    rows = [{"id": f"k{i}", "name": f"name {i}", "works": i * 100, "text": f"name {i}"} for i in range(count)]
    (tmp_path / "keywords.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    vectors = [[float(i + 1), 0.0, 0.0, 1.0] for i in range(count)]
    (tmp_path / "vectors.f16").write_bytes(b"".join(struct.pack("<4e", *v) for v in vectors))
    meta = {"model": "text-embedding-3-small", "dims": 4, "count": count, "dtype": "float16"}
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    return rows


def test_push_writes_every_keyword_with_its_name_and_works(tmp_path):
    from confoundsearch import InMemoryVectorStore

    built(tmp_path)
    store = InMemoryVectorStore("kw")
    assert kv.push(tmp_path, store, label="kw") == {
        "index": "kw",
        "pushed": 3,
        "passed_over": 3,
        "dims": 4,
        "min_works": 0,
    }
    assert sorted(store.records) == ["k0", "k1", "k2"]
    assert store.records["k2"][1] == {"name": "name 2", "works": 200}
    assert store.records["k2"][0][:2] == (3.0, 0.0)


def test_push_carries_on_after_an_interruption(tmp_path, monkeypatch):
    from confoundsearch import InMemoryVectorStore

    monkeypatch.setattr(kv, "PUSH_CHUNK", 1)
    built(tmp_path)
    store = InMemoryVectorStore("kw")
    kv.push(tmp_path, store, limit=2, label="kw")
    assert sorted(store.records) == ["k0", "k1"]
    seen = []
    original = store.upsert
    store.upsert = lambda records: seen.append([r.id for r in records]) or original(records)
    assert kv.push(tmp_path, store, label="kw")["pushed"] == 3
    assert seen == [["k2"]]  # only the keyword the first push had not reached


def test_a_push_to_another_index_or_size_starts_again(tmp_path):
    from confoundsearch import InMemoryVectorStore

    built(tmp_path)
    kv.push(tmp_path, InMemoryVectorStore("a"), label="a")
    other = InMemoryVectorStore("b")
    kv.push(tmp_path, other, label="b")
    assert sorted(other.records) == ["k0", "k1", "k2"]


def test_main_push_and_query_through_a_store(tmp_path, monkeypatch, capsys):
    from confoundsearch import InMemoryVectorStore

    built(tmp_path)
    store = InMemoryVectorStore("kw")
    monkeypatch.setattr(kv, "make_store", lambda args: store)
    monkeypatch.setattr(kv, "embed", lambda texts, model, dims, key: [[3.0, 0.0, 0.0, 1.0]])
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    with pytest.raises(SystemExit, match="push needs --index"):
        kv.main(["push", "--out", str(tmp_path)])
    assert kv.main(["push", "--out", str(tmp_path), "--index", "kw"]) == 0
    assert '"pushed": 3' in capsys.readouterr().out
    assert kv.main(["query", "x", "--out", str(tmp_path), "--index", "kw", "-k", "1"]) == 0
    assert "name 2  (200 papers)" in capsys.readouterr().out


def test_make_store_needs_a_key_and_builds_a_pinecone_store(monkeypatch):
    import argparse

    args = argparse.Namespace(index="kw", namespace="ns", cloud="gcp", region="us-central1")
    monkeypatch.delenv("PINECONE_API_KEY", raising=False)
    with pytest.raises(SystemExit, match="PINECONE_API_KEY"):
        kv.make_store(args)
    monkeypatch.setenv("PINECONE_API_KEY", "pk")
    store = kv.make_store(args)
    assert (store.index, store.namespace, store.cloud, store.region) == ("kw", "ns", "gcp", "us-central1")


def test_a_finished_build_is_not_embedded_again(tmp_path):
    calls = []
    kv.build(tmp_path, "text-embedding-3-small", 4, None, False, 5.0, "k", fake_post(calls), snapshot())
    matrix = (tmp_path / "vectors.f16").read_bytes()
    assert len(calls) == 1
    meta = kv.build(
        tmp_path, "text-embedding-3-small", 4, None, False, 5.0, "k", fake_post(calls), snapshot()
    )
    assert len(calls) == 1 and meta["count"] == 2  # no second request: it was already built
    assert (tmp_path / "vectors.f16").read_bytes() == matrix


@pytest.mark.parametrize(
    "change",
    [
        {"model": "text-embedding-3-large"},
        {"dims": 8},
        {"limit": 1},
        {"with_description": True},
    ],
)
def test_a_build_with_different_settings_is_made_again(tmp_path, change):
    calls = []
    settings = {"model": "text-embedding-3-small", "dims": 4, "limit": None, "with_description": False}
    run = lambda **s: kv.build(  # noqa: E731
        tmp_path,
        s["model"],
        s["dims"],
        s["limit"],
        s["with_description"],
        5.0,
        "k",
        fake_post(calls),
        snapshot(),
    )
    run(**settings)
    run(**{**settings, **change})
    assert len(calls) == 2


def test_a_truncated_matrix_is_not_taken_for_finished(tmp_path):
    calls = []
    kv.build(tmp_path, "text-embedding-3-small", 4, None, False, 5.0, "k", fake_post(calls), snapshot())
    (tmp_path / "vectors.f16").write_bytes(b"\0" * 5)  # damaged: the wrong size
    kv.build(tmp_path, "text-embedding-3-small", 4, None, False, 5.0, "k", fake_post(calls), snapshot())
    assert len(calls) == 2 and (tmp_path / "vectors.f16").stat().st_size == 2 * 4 * 2


def test_a_parallel_push_writes_the_same_thing(tmp_path, monkeypatch):
    from confoundsearch import InMemoryVectorStore

    monkeypatch.setattr(kv, "PUSH_REQUEST", 1)
    built(tmp_path, 7)
    store = InMemoryVectorStore("kw")
    assert kv.push(tmp_path, store, label="kw", workers=4)["pushed"] == 7
    assert sorted(store.records) == [f"k{i}" for i in range(7)]
    assert json.loads((tmp_path / "pushed.json").read_text())["count"] == 7


def test_a_failed_request_in_a_parallel_push_stops_it_and_keeps_the_progress(tmp_path, monkeypatch):
    from confoundsearch import InMemoryVectorStore, VectorStoreError

    monkeypatch.setattr(kv, "PUSH_CHUNK", 2)
    monkeypatch.setattr(kv, "PUSH_REQUEST", 1)
    built(tmp_path, 6)
    store = InMemoryVectorStore("kw")
    original = store.upsert

    def flaky(records):
        if records[0].id == "k4":
            raise VectorStoreError("boom")
        return original(records)

    store.upsert = flaky
    with pytest.raises(VectorStoreError, match="boom"):
        kv.push(tmp_path, store, label="kw", workers=2)
    assert json.loads((tmp_path / "pushed.json").read_text())["count"] == 4  # the two finished chunks


KNOWN = {
    "iv": "Organizational structure",
    "dv": "Firm performance",
    "controls": [
        {"name": "Product-market competition / trade liberalization exposure"},
        {"name": "Firm age"},
        {"name": "Diversification / product mix (lagged) \u2014 note"},
    ],
}
EXPECTED = {
    "variables": {"Firm performance": ["productivity", "Tobin's q"]},
    "confounds": {"Product-market competition": ["trade liberalization", "tariffs"]},
}


def test_each_construct_is_asked_whole_and_in_parts():
    queries = kv.construct_queries(KNOWN)
    assert [q[2] for q in queries if q[0] == "variable"] == ["Organizational structure", "Firm performance"]
    competition = [q[2] for q in queries if q[1].startswith("Product-market")]
    assert competition == [
        "Product-market competition trade liberalization exposure",
        "Product-market competition",
        "trade liberalization exposure",
    ]
    assert [q[2] for q in queries if q[1] == "Firm age"] == ["Firm age"]  # one phrase: no repeat
    assert "Diversification product mix" in [q[2] for q in queries]  # parentheses and the note are dropped


def test_expected_terms_are_matched_loosely_by_name():
    names = [{"name": "Trade liberalization"}, {"name": "labour productivity"}, {"name": "tax"}]
    assert kv.found_terms(names, ["trade liberalization", "productivity", "Tobin's q"]) == [
        "trade liberalization",
        "productivity",
    ]
    assert kv.found_terms(names, ["taxes"]) == []  # a short name does not count as inside a longer term
    assert kv.found_terms([{"name": "import tariffs"}], ["tariffs"]) == ["tariffs"]


def test_expected_terms_are_looked_up_by_variable_name_or_confound_prefix():
    assert kv.expected_for(EXPECTED, "variable", "Firm performance") == ["productivity", "Tobin's q"]
    assert kv.expected_for(EXPECTED, "variable", "Other") == []
    assert kv.expected_for(EXPECTED, "confound", "Product-market competition / x") == [
        "trade liberalization",
        "tariffs",
    ]
    assert kv.expected_for(EXPECTED, "confound", "Firm age") == []


def test_neighbours_embed_every_phrase_once_filter_and_report(tmp_path):
    from confoundsearch import InMemoryVectorStore, VectorRecord

    store = InMemoryVectorStore("kw")
    store.create_index(2)
    store.upsert(
        [
            VectorRecord("a", [1.0, 0.0], {"name": "labour productivity", "works": 900}),
            VectorRecord("b", [0.9, 0.1], {"name": "rare thing", "works": 5}),
        ]
    )
    seen = []

    def embed_phrases(phrases):
        seen.append(list(phrases))
        return [[1.0, 0.0] for _ in phrases]

    results = kv.neighbours(store, embed_phrases, KNOWN, EXPECTED, k=5, min_works=100)
    assert len(seen) == 1 and len(seen[0]) == len(results)  # one embedding request for all of them
    firm = next(r for r in results if r["query"] == "Firm performance")
    assert [n["name"] for n in firm["neighbours"]] == ["labour productivity"]  # the rare keyword is filtered
    assert firm["expected_found"] == ["productivity"]
    text = kv.neighbours_markdown(results)
    assert "Expected terms found: 1/2: productivity" in text and "overall: 1/8" in text
    assert "(for Product-market competition / trade liberalization exposure)" in text
    assert "overall" not in kv.neighbours_markdown([{**firm, "expected": [], "expected_found": []}])


def test_main_neighbors(tmp_path, monkeypatch, capsys):
    from confoundsearch import InMemoryVectorStore, VectorRecord

    built(tmp_path)
    store = InMemoryVectorStore("kw")
    store.create_index(4)
    store.upsert([VectorRecord("k0", [1.0, 0.0, 0.0, 1.0], {"name": "labour productivity", "works": 10})])
    monkeypatch.setattr(kv, "make_store", lambda args: store)
    monkeypatch.setattr(kv, "embed", lambda texts, model, dims, key: [[1.0, 0.0, 0.0, 1.0] for _ in texts])
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    known, expect, report = tmp_path / "known.json", tmp_path / "expect.json", tmp_path / "report.md"
    known.write_text(json.dumps(KNOWN))
    expect.write_text(json.dumps(EXPECTED))
    with pytest.raises(SystemExit, match="neighbors needs --index"):
        kv.main(["neighbors", str(known), "--out", str(tmp_path)])
    argv = ["neighbors", str(known), "--out", str(tmp_path), "--index", "kw", "--expect", str(expect)]
    assert kv.main([*argv, "--report", str(report)]) == 0
    assert "labour productivity" in capsys.readouterr().out and "labour productivity" in report.read_text()
    assert kv.main(["neighbors", str(known), "--out", str(tmp_path), "--index", "kw"]) == 0


def test_push_can_skip_keywords_with_few_papers_and_says_how_many_it_passed_over(tmp_path):
    from confoundsearch import InMemoryVectorStore

    built(tmp_path, 5)  # works are 0, 100, 200, 300, 400
    store = InMemoryVectorStore("kw")
    result = kv.push(tmp_path, store, label="kw", min_works=250)
    assert sorted(store.records) == ["k3", "k4"]
    assert result["pushed"] == 2 and result["passed_over"] == 5 and result["min_works"] == 250
    again = kv.push(tmp_path, store, label="kw", min_works=250)  # finished: nothing more to do
    assert again["pushed"] == 2 and again["passed_over"] == 5


def test_a_different_min_works_starts_the_push_again(tmp_path):
    from confoundsearch import InMemoryVectorStore

    built(tmp_path, 5)
    store = InMemoryVectorStore("kw")
    kv.push(tmp_path, store, label="kw", min_works=250)
    kv.push(tmp_path, store, label="kw", min_works=0)  # lower threshold: the earlier, longer list is redone
    assert len(store.records) == 5


def test_main_push_explains_a_spent_plan_limit_and_keeps_the_progress(tmp_path, monkeypatch):
    from confoundsearch import VectorStoreError

    built(tmp_path)

    class Spent:
        def create_index(self, dimension, metric="cosine"):
            pass

        def upsert(self, records):
            raise VectorStoreError(
                "HTTP 429: You've reached your write unit limit for the current month. upgrade your plan"
            )

    monkeypatch.setattr(kv, "make_store", lambda args: Spent())
    with pytest.raises(SystemExit, match="plan's limit is reached"):
        kv.main(["push", "--out", str(tmp_path), "--index", "kw"])
    monkeypatch.setattr(
        kv,
        "make_store",
        lambda args: type(
            "Broken",
            (),
            {
                "create_index": lambda self, d, m="cosine": None,
                "upsert": lambda self, r: (_ for _ in ()).throw(VectorStoreError("HTTP 500: other")),
            },
        )(),
    )
    with pytest.raises(VectorStoreError, match="other"):  # any other failure is not hidden
        kv.main(["push", "--out", str(tmp_path), "--index", "kw"])


def keyword_store():
    """A store with four keywords on a line: ids are OpenAlex slugs, names and paper counts are metadata."""
    from confoundsearch import InMemoryVectorStore, VectorRecord

    rows = {
        "firm-size": ("firm size", 900, [1.0, 0.0]),
        "large-firms": ("large firms", 600, [0.9, 0.1]),
        "rare-firms": ("rare firms", 5, [0.95, 0.05]),
        "pt3sn": ("Pt3Sn", 700, [0.0, 1.0]),
    }
    store = InMemoryVectorStore("kw")
    store.create_index(2)
    store.upsert([VectorRecord(i, v, {"name": n, "works": w}) for i, (n, w, v) in rows.items()])
    return store


def test_query_of_a_keyword_name_needs_no_openai_key(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(kv, "make_store", lambda args: keyword_store())
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert kv.main(["query", "Firm Size", "--out", str(tmp_path), "--index", "kw", "-k", "1"]) == 0
    assert "rare firms  (5 papers)" in capsys.readouterr().out  # the nearest, and not firm size itself
    assert kv.main(["query", "firm-size", "--out", str(tmp_path), "--index", "kw", "--min-works", "100"]) == 0
    out = capsys.readouterr().out
    assert "large firms" in out and "rare firms" not in out


def test_query_of_another_phrase_needs_the_openai_key_and_says_so(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(kv, "make_store", lambda args: keyword_store())
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(SystemExit, match="set OPENAI_API_KEY to search by any phrase"):
        kv.main(["query", "how big a firm is", "--out", str(tmp_path), "--index", "kw"])
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setattr(kv, "embed", lambda texts, model, dims, key: [[0.0, 1.0]])
    assert (
        kv.main(["query", "something with a dash", "--out", str(tmp_path), "--index", "kw", "-k", "1"]) == 0
    )
    assert "Pt3Sn" in capsys.readouterr().out


def test_keys_in_a_dotenv_file_are_used_without_just(tmp_path, monkeypatch):
    built(tmp_path)
    (tmp_path / ".env").write_text(
        "# keys\nPINECONE_API_KEY=from-the-file\nOPENAI_API_KEY=also-from-the-file\n"
    )
    for name in ("PINECONE_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.setenv(name, "x")
        monkeypatch.delenv(name)  # registers the clean-up: whatever the tool sets is removed after the test
    seen = {}

    class Recorder:
        def __init__(self, key, index, namespace, cloud, region):
            seen["key"] = key

        def create_index(self, dimension, metric="cosine"):
            pass

        def upsert(self, records):
            pass

    monkeypatch.setattr(kv, "PineconeVectorStore", Recorder)
    monkeypatch.chdir(tmp_path)
    assert kv.main(["push", "--out", str(tmp_path), "--index", "kw"]) == 0
    assert seen["key"] == "from-the-file" and os.environ["OPENAI_API_KEY"] == "also-from-the-file"


def test_a_variable_already_set_beats_the_dotenv_file(tmp_path, monkeypatch):
    built(tmp_path)
    (tmp_path / ".env").write_text("PINECONE_API_KEY=from-the-file\n")
    monkeypatch.setenv("PINECONE_API_KEY", "from-the-shell")
    seen = {}

    class Recorder:
        def __init__(self, key, index, namespace, cloud, region):
            seen["key"] = key

        def create_index(self, dimension, metric="cosine"):
            pass

        def upsert(self, records):
            pass

    monkeypatch.setattr(kv, "PineconeVectorStore", Recorder)
    monkeypatch.chdir(tmp_path)
    kv.main(["push", "--out", str(tmp_path), "--index", "kw"])
    assert seen["key"] == "from-the-shell"


def test_push_with_clobber_deletes_what_was_there_and_starts_again(tmp_path):
    from confoundsearch import InMemoryVectorStore, VectorRecord

    built(tmp_path, 3)
    store = InMemoryVectorStore("kw")
    kv.push(tmp_path, store, label="kw")
    store.upsert([VectorRecord("stale", [1.0, 0.0, 0.0, 1.0], {"name": "old", "works": 1})])
    result = kv.push(tmp_path, store, label="kw", clobber=True)
    assert (
        sorted(store.records) == ["k0", "k1", "k2"] and result["pushed"] == 3
    )  # 'stale' is gone, all re-pushed


def clobber_setup(tmp_path, monkeypatch, answer=None):
    monkeypatch.setattr(kv.sys.stdin, "isatty", lambda: True)  # pretend a person is at the keyboard
    from confoundsearch import InMemoryVectorStore

    built(tmp_path, 2)
    store = InMemoryVectorStore("kw")
    kv.push(tmp_path, store, label="kw")
    monkeypatch.setattr(kv, "make_store", lambda args: store)
    if answer is None:
        monkeypatch.setattr("builtins.input", lambda prompt="": pytest.fail("should not have asked"))
    else:
        monkeypatch.setattr("builtins.input", lambda prompt="": answer)
    return store


def test_clobber_asks_first_and_a_no_deletes_nothing(tmp_path, monkeypatch):
    store = clobber_setup(tmp_path, monkeypatch, answer="n")
    with pytest.raises(SystemExit, match="cancelled: nothing was deleted"):
        kv.main(["push", "--out", str(tmp_path), "--index", "kw", "--clobber"])
    assert sorted(store.records) == ["k0", "k1"]


def test_clobber_after_a_yes_and_with_yes_flag(tmp_path, monkeypatch, capsys):
    clobber_setup(tmp_path, monkeypatch, answer="y")
    assert kv.main(["push", "--out", str(tmp_path), "--index", "kw", "--clobber"]) == 0
    assert '"pushed": 2' in capsys.readouterr().out
    clobber_setup(tmp_path, monkeypatch)  # no answer wanted: --yes must not ask
    assert kv.main(["push", "--out", str(tmp_path), "--index", "kw", "--clobber", "--yes"]) == 0


def test_descriptions_are_embedded_unless_switched_off(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(
        kv, "build", lambda out, model, dims, limit, with_description, *a: seen.append(with_description) or {}
    )
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    kv.main(["build", "--out", str(tmp_path)])
    kv.main(["build", "--out", str(tmp_path), "--no-description"])
    assert seen == [True, False]


def test_clobber_without_a_terminal_and_without_yes_refuses_instead_of_crashing(tmp_path, monkeypatch):
    store = clobber_setup(tmp_path, monkeypatch)
    monkeypatch.setattr(kv.sys.stdin, "isatty", lambda: False)  # as in a script or a pipeline
    with pytest.raises(SystemExit, match="with no terminal to ask, add --yes"):
        kv.main(["push", "--out", str(tmp_path), "--index", "kw", "--clobber"])
    assert sorted(store.records) == ["k0", "k1"]
    assert kv.main(["push", "--out", str(tmp_path), "--index", "kw", "--clobber", "--yes"]) == 0


def test_fresh_wipes_for_a_new_matrix_but_carries_on_with_the_one_being_pushed(tmp_path):
    from confoundsearch import InMemoryVectorStore, VectorRecord

    built(tmp_path, 4)
    store = InMemoryVectorStore("kw")
    kv.push(tmp_path, store, limit=2, label="kw", fresh=True)  # a new load: nothing there yet, starts at 0
    assert sorted(store.records) == ["k0", "k1"]
    store.upsert([VectorRecord("marker", [1.0, 0.0, 0.0, 1.0], {"name": "marker", "works": 1})])
    kv.push(tmp_path, store, label="kw", fresh=True)  # the same matrix, half done: carry on, delete nothing
    assert sorted(store.records) == ["k0", "k1", "k2", "k3", "marker"]
    again = kv.push(tmp_path, store, label="kw", fresh=True)  # finished: still nothing deleted
    assert "marker" in store.records and again["pushed"] == 4
    matrix = tmp_path / "vectors.f16"
    matrix.write_bytes(matrix.read_bytes())  # rebuilt: a new matrix at the same path
    os.utime(matrix, ns=(1, 1))
    kv.push(tmp_path, store, label="kw", fresh=True)
    assert sorted(store.records) == ["k0", "k1", "k2", "k3"]  # the marker is gone: replaced wholesale


def test_main_fresh_asks_like_clobber_and_the_prompt_can_be_skipped(tmp_path, monkeypatch, capsys):
    store = clobber_setup(tmp_path, monkeypatch, answer="n")
    with pytest.raises(SystemExit, match="cancelled"):
        kv.main(["push", "--out", str(tmp_path), "--index", "kw", "--fresh"])
    assert sorted(store.records) == ["k0", "k1"]
    assert kv.main(["push", "--out", str(tmp_path), "--index", "kw", "--fresh", "--yes"]) == 0
    assert "carrying on" in capsys.readouterr().err  # the same matrix, already pushed: nothing deleted
    assert sorted(store.records) == ["k0", "k1"]


def test_neighbours_distinct_drops_the_echoes_of_each_phrase():
    from confoundsearch import InMemoryVectorStore, VectorRecord

    store = InMemoryVectorStore("kw")
    store.create_index(2)
    store.upsert(
        [
            VectorRecord("a", [1.0, 0.0], {"name": "firm performance and size", "works": 9}),
            VectorRecord("b", [0.9, 0.1], {"name": "profitability", "works": 9}),
        ]
    )
    known = {"iv": "x", "dv": "firm performance", "controls": []}
    results = kv.neighbours(
        store, lambda phrases: [[1.0, 0.0] for _ in phrases], known, {}, k=5, distinct=True
    )
    dv = next(r for r in results if r["query"] == "firm performance")
    assert [n["name"] for n in dv["neighbours"]] == ["profitability"]
    plain = kv.neighbours(store, lambda phrases: [[1.0, 0.0] for _ in phrases], known, {}, k=5)
    assert len(next(r for r in plain if r["query"] == "firm performance")["neighbours"]) == 2


def test_main_query_distinct(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(kv, "make_store", lambda args: keyword_store())
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert (
        kv.main(["query", "firm size", "--out", str(tmp_path), "--index", "kw", "-k", "5", "--distinct"]) == 0
    )
    out = capsys.readouterr().out
    assert "large firms" in out and "firm size" not in out


def test_query_through_the_service_needs_no_keys_for_a_keyword(monkeypatch, capsys, tmp_path):
    seen = []

    class Service:
        def __init__(self, url, embedder=None):
            seen.append((url, embedder))

        def query(self, name, k, min_works, distinct):
            from confoundsearch import Match

            seen.append((name, k, min_works, distinct))
            return [Match("large-firms", 0.9, {"name": "large firms", "works": 600})]

    monkeypatch.setattr(kv, "KeywordService", Service)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    argv = [
        "query",
        "firm size",
        "--api",
        "https://svc.test",
        "--out",
        str(tmp_path),
        "-k",
        "3",
        "--distinct",
    ]
    assert kv.main(argv) == 0
    assert "0.900  large firms  (600 papers)" in capsys.readouterr().out
    assert seen[0] == ("https://svc.test", None) and seen[1] == ("firm size", 3, 0, True)


def test_query_through_the_service_embeds_a_phrase_with_your_own_key(monkeypatch, tmp_path):
    seen = []

    class Service:
        def __init__(self, url, embedder=None):
            seen.append(embedder)

        def query(self, name, k, min_works, distinct):
            return []

    monkeypatch.setattr(kv, "KeywordService", Service)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-mine")
    kv.main(["query", "x", "--api", "https://svc.test", "--out", str(tmp_path)])
    assert seen[0].api_key == "sk-mine" and seen[0].dims == 512


def test_query_through_the_service_reports_a_missing_keyword_plainly(monkeypatch, tmp_path):
    from confoundsearch import KeywordNotFound

    class Service:
        def __init__(self, url, embedder=None):
            pass

        def query(self, name, k, min_works, distinct):
            raise KeywordNotFound(
                "no keyword named 'x'; to search any phrase, give an embedder (an OpenAI key)"
            )

    monkeypatch.setattr(kv, "KeywordService", Service)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(SystemExit, match="an OpenAI key"):
        kv.main(["query", "x", "--api", "https://svc.test", "--out", str(tmp_path)])
