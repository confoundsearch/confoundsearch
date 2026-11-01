"""The vector store interface, its in-memory store, and the Pinecone store (against a fake Pinecone)."""

import json
import math
from urllib.parse import parse_qs, urlsplit

import pytest

from confoundsearch import (
    InMemoryVectorStore,
    Match,
    PineconeVectorStore,
    VectorRecord,
    VectorStoreError,
)
from confoundsearch.data.vectors import BATCH, matches, plan_limit_reached
from confoundsearch.errors import HttpError

A = VectorRecord("a", [1.0, 0.0], {"works": 10, "field": "econ"})
B = VectorRecord("b", [0.0, 1.0], {"works": 500, "field": "econ"})
C = VectorRecord("c", [0.6, 0.8], {"works": 1000, "field": "bio"})


class FakePinecone:
    """Enough of Pinecone's REST API to run the store against: control plane, upsert, update, query."""

    def __init__(self, ready_after=1):
        self.calls, self.indexes, self.vectors, self.fail = [], {}, {}, []
        self.describes, self.ready_after = 0, ready_after
        self.missing_namespace_is_404 = False

    def __call__(self, url, headers, body, timeout):
        payload = json.loads(body) if body else None
        self.calls.append((url, headers, payload))
        if self.fail:
            status = self.fail.pop(0)
            if status:
                raise HttpError(f"HTTP {status} from {url}: nope", status)
        path = url.split("pinecone.io", 1)[-1] if "api.pinecone.io" in url else "/" + url.split("/", 3)[3]
        if url.endswith("/indexes") and payload:
            if payload["name"] in self.indexes:
                raise HttpError(f"HTTP 409 from {url}: ALREADY_EXISTS", 409)
            self.indexes[payload["name"]] = {"dimension": payload["dimension"], "metric": payload["metric"]}
            return json.dumps({"name": payload["name"]}).encode()
        if "/indexes/" in url:
            self.describes += 1
            name = url.rsplit("/", 1)[-1]
            info = self.indexes[name]
            ready = self.describes > self.ready_after
            return json.dumps(
                {"host": f"{name}-x.svc.pinecone.io", "status": {"ready": ready}, **info}
            ).encode()
        if path.startswith("/vectors/fetch"):
            query = parse_qs(urlsplit(url).query)
            ns = query.get("namespace", [""])[0]
            wanted = query["ids"]
            assert all(w.isascii() for w in wanted)  # the id travels in the URL, still ASCII
            return json.dumps(
                {"vectors": {w: self.vectors[(ns, w)] for w in wanted if (ns, w) in self.vectors}}
            ).encode()
        if path == "/vectors/delete":
            ns = payload.get("namespace", "")
            assert payload.get("deleteAll") is True
            mine = [key for key in self.vectors if key[0] == ns]
            if not mine and self.missing_namespace_is_404:
                raise HttpError(f"HTTP 404 from {url}: Namespace not found", 404)
            for key in mine:
                del self.vectors[key]
            return b"{}"
        if path in ("/vectors/upsert", "/vectors/update"):
            for vid in [v["id"] for v in payload.get("vectors", [])] + [payload.get("id", "")]:
                if not vid.isascii():  # the real service refuses these
                    message = f"Vector ID must be ASCII, but got {vid!r}"
                    raise HttpError(f"HTTP 400 from {url}: {message}", 400)
        if path == "/vectors/upsert":
            for v in payload["vectors"]:
                self.vectors[(payload.get("namespace", ""), v["id"])] = v
            return json.dumps({"upsertedCount": len(payload["vectors"])}).encode()
        if path == "/vectors/update":
            v = self.vectors[(payload.get("namespace", ""), payload["id"])]
            v["values"] = payload.get("values", v["values"])
            v["metadata"] = {**v["metadata"], **payload.get("setMetadata", {})}
            return b""
        if path == "/query":
            ns = payload.get("namespace", "")
            found = []
            for (space, id), v in self.vectors.items():
                if space != ns or ("filter" in payload and not matches(v["metadata"], payload["filter"])):
                    continue
                dot = sum(x * y for x, y in zip(payload["vector"], v["values"]))
                norms = math.hypot(*payload["vector"]) * math.hypot(*v["values"])
                found.append({"id": id, "score": dot / norms, "metadata": v["metadata"]})
            found.sort(key=lambda m: -m["score"])
            return json.dumps({"matches": found[: payload["topK"]]}).encode()
        raise AssertionError(f"unexpected call {url}")


def pinecone(fake=None, **kwargs):
    fake = fake or FakePinecone()
    kwargs.setdefault("sleep", lambda s: None)
    return PineconeVectorStore("key", "kw", transport=fake, **kwargs), fake


@pytest.fixture(params=["memory", "pinecone"])
def store(request):
    """The same contract tests run against every store."""
    if request.param == "memory":
        made = InMemoryVectorStore("kw")
    else:
        made, _ = pinecone()
    made.create_index(2)
    return made


class TestEveryStore:
    def test_upsert_then_query_returns_the_closest_first(self, store):
        assert store.upsert([A, B, C]) == 3
        found = store.query([1.0, 0.1], 2)
        assert [m.id for m in found] == ["a", "c"] and isinstance(found[0], Match)
        assert found[0].score == pytest.approx(0.995, abs=1e-3) and found[0].metadata["field"] == "econ"

    def test_upsert_replaces_a_record_with_the_same_id(self, store):
        store.upsert([A, B])
        store.upsert([VectorRecord("a", [0.0, 1.0], {"works": 1})])
        assert store.query([0.0, 1.0], 5)[0].score == pytest.approx(1.0)
        assert {m.id: m.metadata["works"] for m in store.query([0.0, 1.0], 5)}["a"] == 1

    def test_update_changes_the_vector_and_merges_metadata(self, store):
        store.upsert([A])
        store.update("a", metadata={"works": 99})
        store.update("a", values=[0.0, 1.0])
        (match,) = store.query([0.0, 1.0], 1)
        assert match.score == pytest.approx(1.0) and match.metadata == {"works": 99, "field": "econ"}

    def test_filters_use_pinecones_syntax(self, store):
        store.upsert([A, B, C])
        assert [m.id for m in store.query([1.0, 0.0], 5, {"works": {"$gte": 500}})] == ["c", "b"]
        assert [m.id for m in store.query([1.0, 0.0], 5, {"field": "econ", "works": {"$lt": 100}})] == ["a"]
        assert store.query([1.0, 0.0], 5, {"field": {"$in": ["none"]}}) == []

    def test_a_query_for_nothing_returns_nothing(self, store):
        store.upsert([A])
        assert store.query([1.0, 0.0], 0) == []

    def test_update_needs_something_to_change(self, store):
        with pytest.raises(VectorStoreError, match="needs new values or new metadata"):
            store.update("a")

    def test_create_index_twice_is_harmless(self, store):
        store.create_index(2)
        store.upsert([A])
        assert len(store.query([1.0, 0.0], 5)) == 1


NAMES = [
    "qur\u02beānic-concordance",
    "100%-sure",
    "a/b c",
    "plain-id",
]  # \u02be: a letter keyword ids really use


class TestClear:
    def test_clear_deletes_every_record_and_the_index_stays_usable(self, store):
        store.upsert([A, B, C])
        store.clear()
        assert store.query([1.0, 0.0], 5) == [] and store.fetch(["a"]) == []
        store.clear()  # clearing an empty store is fine
        store.upsert([A])
        assert [m.id for m in store.query([1.0, 0.0], 5)] == ["a"]

    def test_pinecone_clear_only_touches_its_namespace(self):
        first, fake = pinecone(namespace="one")
        first.create_index(2)
        first.upsert([A])
        second, _ = pinecone(fake, namespace="two")
        second.create_index(2)
        second.upsert([B])
        first.clear()
        assert first.query([1.0, 0.0], 5) == [] and [m.id for m in second.query([0.0, 1.0], 5)] == ["b"]

    def test_a_namespace_that_does_not_exist_yet_is_not_an_error(self):
        store, fake = pinecone()
        store.create_index(2)
        fake.missing_namespace_is_404 = True
        store.clear()  # 404 means there was nothing to delete

    def test_other_errors_are_raised(self):
        store, fake = pinecone()
        store.create_index(2)
        fake.fail = [403]
        with pytest.raises(VectorStoreError, match="HTTP 403"):
            store.clear()


class TestFetch:
    def test_fetch_returns_the_stored_vectors_and_leaves_out_unknown_ids(self, store):
        store.upsert([A, B])
        got = store.fetch(["b", "nope", "a"])
        assert sorted(r.id for r in got) == ["a", "b"]
        by_id = {r.id: r for r in got}
        assert list(by_id["b"].values) == [0.0, 1.0] and by_id["a"].metadata["field"] == "econ"
        assert store.fetch([]) == []

    def test_a_fetched_vector_can_be_used_to_find_its_neighbours(self, store):
        store.upsert([A, B, C])
        (record,) = store.fetch(["c"])
        assert store.query(record.values, 3)[0].id == "c"

    def test_fetch_handles_non_ascii_ids(self, store):
        name = "qur\u02beānic-concordance"
        store.upsert([VectorRecord(name, [1.0, 0.0], {"n": 1})])
        assert [r.id for r in store.fetch([name])] == [name]

    def test_pinecone_fetch_goes_in_batches_and_keeps_the_namespace(self):
        from confoundsearch.data import vectors as module

        store, fake = pinecone(namespace="ns")
        store.create_index(2)
        records = [VectorRecord(str(i), [1.0, 0.0]) for i in range(module.FETCH_BATCH + 5)]
        store.upsert(records)
        got = store.fetch([r.id for r in records] + ["missing"])
        calls = [c[0] for c in fake.calls if "/vectors/fetch" in c[0]]
        assert len(got) == len(records) and len(calls) == 2 and all("namespace=ns" in c for c in calls)

    @pytest.mark.parametrize("body", [b'{"vectors": {"a": {"values": [1]}}}', b'{"vectors": {"a": 5}}'])
    def test_pinecone_fetch_unusable_replies(self, body):
        class Bad:
            def __call__(self, url, headers, body_, timeout):
                return body

        store = PineconeVectorStore("key", "kw", host="h.example", transport=Bad())
        with pytest.raises(VectorStoreError, match="unexpected response"):
            store.fetch(["a"])


class TestIds:
    """OpenAlex keyword ids are slugs that are not always ASCII; Pinecone only takes ASCII."""

    def test_any_id_survives_a_round_trip_in_every_store(self, store):
        store.upsert([VectorRecord(name, [1.0, 0.0], {"n": i}) for i, name in enumerate(NAMES)])
        assert sorted(m.id for m in store.query([1.0, 0.0], 10)) == sorted(NAMES)
        store.update("qur\u02beānic-concordance", metadata={"n": 99})
        by_id = {m.id: m.metadata["n"] for m in store.query([1.0, 0.0], 10)}
        assert by_id["qur\u02beānic-concordance"] == 99

    def test_pinecone_receives_only_ascii_ids(self):
        store, fake = pinecone()
        store.create_index(2)
        store.upsert([VectorRecord("qur\u02beānic-concordance", [1.0, 0.0])])
        sent = [c[2]["vectors"][0]["id"] for c in fake.calls if c[0].endswith("/vectors/upsert")]
        assert sent == ["qur%CA%BE%C4%81nic-concordance"] and sent[0].isascii()

    def test_an_id_that_is_too_long_once_encoded_is_refused(self):
        store, _ = pinecone()
        store.create_index(2)
        with pytest.raises(VectorStoreError, match="longer than 512"):
            store.upsert([VectorRecord("ā" * 200, [1.0, 0.0])])


class TestFilterSyntax:
    @pytest.mark.parametrize(
        "condition,expected",
        [
            ({"n": 5}, True),
            ({"n": {"$eq": 4}}, False),
            ({"n": {"$ne": 4}}, True),
            ({"n": {"$gt": 5}}, False),
            ({"n": {"$gte": 5}}, True),
            ({"n": {"$lt": 6}}, True),
            ({"n": {"$lte": 4}}, False),
            ({"n": {"$in": [1, 5]}}, True),
            ({"n": {"$nin": [5]}}, False),
            ({"n": {"$gte": 1, "$lte": 9}}, True),
            ({"missing": {"$gt": 1}}, False),
            ({"missing": {"$lte": 1}}, False),
            ({}, True),
        ],
    )
    def test_matches(self, condition, expected):
        assert matches({"n": 5}, condition) is expected

    def test_an_unknown_operator_is_an_error(self):
        with pytest.raises(VectorStoreError, match=r"unsupported filter operator '\$regex'"):
            matches({"n": 5}, {"n": {"$regex": "x"}})


class TestInMemory:
    def test_it_must_be_created_first(self):
        with pytest.raises(VectorStoreError, match="call create_index first"):
            InMemoryVectorStore().upsert([A])

    def test_dimensions_are_checked(self):
        store = InMemoryVectorStore()
        store.create_index(2)
        with pytest.raises(VectorStoreError, match="expected 2 dimensions, got 3"):
            store.upsert([VectorRecord("x", [1, 2, 3])])
        with pytest.raises(VectorStoreError, match="expected 2 dimensions"):
            store.query([1.0], 1)
        assert store.records == {}  # a bad batch writes nothing

    def test_bad_settings(self):
        store = InMemoryVectorStore()
        with pytest.raises(VectorStoreError, match="cosine and dotproduct"):
            store.create_index(2, "euclidean")
        with pytest.raises(VectorStoreError, match="must be positive"):
            store.create_index(0)
        store.create_index(2)
        with pytest.raises(VectorStoreError, match="exists with 2 dimensions, cosine"):
            store.create_index(3)

    def test_update_of_a_missing_record(self):
        store = InMemoryVectorStore()
        store.create_index(2)
        with pytest.raises(VectorStoreError, match="no record with id 'zz'"):
            store.update("zz", metadata={"a": 1})

    def test_dotproduct_uses_length_and_cosine_does_not(self):
        for metric, expected in (("cosine", 1.0), ("dotproduct", 3.0)):
            store = InMemoryVectorStore()
            store.create_index(2, metric)
            store.upsert([VectorRecord("a", [3.0, 0.0])])
            assert store.query([1.0, 0.0], 1)[0].score == pytest.approx(expected)

    def test_a_zero_vector_does_not_divide_by_zero(self):
        store = InMemoryVectorStore()
        store.create_index(2)
        store.upsert([VectorRecord("z", [0.0, 0.0])])
        assert store.query([0.0, 0.0], 1)[0].score == 0.0


class TestPinecone:
    def test_create_index_sends_the_spec_and_waits_until_ready(self):
        waits = []
        store, fake = pinecone(
            FakePinecone(ready_after=2), cloud="gcp", region="us-central1", sleep=waits.append
        )
        store.create_index(512, "dotproduct")
        url, headers, payload = fake.calls[0]
        assert url == "https://api.pinecone.io/indexes" and headers["Api-Key"] == "key"
        assert "X-Pinecone-API-Version" in headers
        assert payload == {
            "name": "kw",
            "dimension": 512,
            "metric": "dotproduct",
            "spec": {"serverless": {"cloud": "gcp", "region": "us-central1"}},
        }
        assert waits == [2.0, 2.0] and store.host == "kw-x.svc.pinecone.io"

    def test_it_gives_up_if_the_index_never_gets_ready(self):
        store, _ = pinecone(FakePinecone(ready_after=10**6), ready_timeout=4.0)
        with pytest.raises(VectorStoreError, match="not ready after 4s"):
            store.create_index(2)

    def test_an_existing_index_is_used_but_its_size_must_match(self):
        fake = FakePinecone()
        fake.indexes["kw"] = {"dimension": 8, "metric": "cosine"}
        store, _ = pinecone(fake)
        with pytest.raises(VectorStoreError, match="exists with 8 dimensions"):
            store.create_index(2)
        fake.indexes["kw"] = {"dimension": 2, "metric": "cosine"}
        pinecone(fake)[0].create_index(2)

    def test_other_creation_errors_are_raised(self):
        store, _ = pinecone(
            FakePinecone(),
        )
        store.transport.fail = [401]
        with pytest.raises(VectorStoreError, match="HTTP 401"):
            store.create_index(2)

    def test_the_metric_is_checked(self):
        with pytest.raises(VectorStoreError, match="metric must be one of"):
            pinecone()[0].create_index(2, "manhattan")

    def test_upsert_goes_in_batches_and_counts(self):
        store, fake = pinecone()
        store.create_index(2)
        records = [VectorRecord(str(i), [1.0, 0.0]) for i in range(BATCH * 2 + 50)]
        assert store.upsert(records) == len(records)
        sent = [c for c in fake.calls if c[0].endswith("/vectors/upsert")]
        assert [len(c[2]["vectors"]) for c in sent] == [BATCH, BATCH, 50]
        assert "namespace" not in sent[0][2]

    def test_a_namespace_is_sent_and_keeps_vectors_apart(self):
        first, fake = pinecone(namespace="one")
        first.create_index(2)
        first.upsert([A])
        second, _ = pinecone(fake, namespace="two")
        second.create_index(2)
        assert second.query([1.0, 0.0], 5) == [] and len(first.query([1.0, 0.0], 5)) == 1

    def test_rate_limits_are_retried_then_succeed(self):
        waits = []
        store, fake = pinecone(sleep=waits.append)
        store.create_index(2)
        waits.clear()
        fake.fail = [429, 503]
        assert store.upsert([A]) == 1 and waits == [1.0, 2.0]

    def test_it_stops_retrying_and_reports_the_failure(self):
        store, fake = pinecone()
        store.create_index(2)
        fake.fail = [500] * 9
        with pytest.raises(VectorStoreError, match="HTTP 500"):
            store.upsert([A])

    def test_a_plan_limit_is_not_retried(self):
        waits = []
        store, _ = pinecone(sleep=waits.append)
        store.create_index(2)
        waits.clear()
        message = "You've reached your write unit limit for the current month (2000000). upgrade your plan."
        calls = []

        def spent(url, headers, body, timeout):
            calls.append(url)
            raise HttpError(f"HTTP 429 from {url}: {message}", 429)

        store.transport = spent
        with pytest.raises(VectorStoreError, match="write unit limit") as info:
            store.upsert([A])
        assert len(calls) == 1 and waits == [] and plan_limit_reached(info.value)
        assert not plan_limit_reached(HttpError("HTTP 429 from x: slow down", 429))

    def test_a_bad_request_is_not_retried(self):
        store, fake = pinecone()
        store.create_index(2)
        fake.fail = [400]
        with pytest.raises(VectorStoreError, match="HTTP 400"):
            store.upsert([A])
        assert fake.fail == []

    def test_the_host_is_looked_up_when_not_given(self):
        fake = FakePinecone(ready_after=0)
        fake.indexes["kw"] = {"dimension": 2, "metric": "cosine"}
        fake.describes = 1
        store, _ = pinecone(fake)
        assert store.query([1.0, 0.0], 1) == [] and store.host == "kw-x.svc.pinecone.io"

    def test_a_given_host_skips_the_lookup(self):
        store, fake = pinecone(host="https://my-host.svc.pinecone.io")
        assert store.host == "my-host.svc.pinecone.io"
        store.upsert([A])
        assert fake.calls[0][0] == "https://my-host.svc.pinecone.io/vectors/upsert"

    def test_a_missing_index_has_no_host(self):
        class NoHost:
            def __call__(self, url, headers, body, timeout):
                return b'{"status": {"ready": false}}'

        store = PineconeVectorStore("key", "kw", transport=NoHost())
        with pytest.raises(VectorStoreError, match="has no host yet"):
            store.query([1.0, 0.0], 1)

    @pytest.mark.parametrize("body", [b"not json", b'{"matches": 5}', b'{"matches": [{"id": "a"}]}', b"{}"])
    def test_unusable_replies(self, body):
        class Bad:
            def __call__(self, url, headers, body_, timeout):
                return body

        store = PineconeVectorStore("key", "kw", host="h.example", transport=Bad())
        with pytest.raises(VectorStoreError, match="unexpected response"):
            store.query([1.0, 0.0], 1)

    def test_an_empty_reply_is_fine_for_an_update(self):
        store, fake = pinecone()
        store.create_index(2)
        store.upsert([A])
        store.update("a", metadata={"x": 1})
        assert fake.calls[-1][2] == {"id": "a", "setMetadata": {"x": 1}}
