"""devops/proxy/handler.py: the Lambda that lets anyone read the keyword index and nothing else."""

import base64
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from confoundsearch import InMemoryVectorStore, VectorRecord, VectorStoreError

ROOT = Path(__file__).parent.parent
spec = importlib.util.spec_from_file_location("proxy_handler", ROOT / "devops" / "proxy" / "handler.py")
h = importlib.util.module_from_spec(spec)
sys.modules["proxy_handler"] = h
spec.loader.exec_module(h)

CONFIG = h.Config(dims=2, max_k=5, model="m", count=3, file_url="https://files.example.test/kw.f16")


def make_store():
    store = InMemoryVectorStore("kw")
    store.create_index(2)
    store.upsert(
        [
            VectorRecord("firm-size", [1.0, 0.0], {"name": "firm size", "works": 900, "secret": "x"}),
            VectorRecord("large-firms", [0.9, 0.1], {"name": "large firms", "works": 600}),
            VectorRecord("pt3sn", [0.0, 1.0], {"name": "Pt3Sn", "works": 5}),
        ]
    )
    return store


def event(method, path, body=None, **extra):
    raw = body if isinstance(body, str) else (json.dumps(body) if body is not None else None)
    return {
        "rawPath": path,
        "requestContext": {"http": {"method": method, "path": path}},
        "body": raw,
        **extra,
    }


def call(method, path, body=None, store=None, config=CONFIG, corpus=None, **extra):
    logs = []
    store = store or make_store()
    response = h.handle(
        event(method, path, body, **extra), lambda: store, config, logs.append, lambda: corpus
    )
    return response, json.loads(response["body"]), logs


class TestReading:
    def test_health_and_manifest(self):
        response, body, _ = call("GET", "/v1/health")
        assert response["statusCode"] == 200 and body == {"ok": True}
        assert response["headers"]["content-type"] == "application/json"
        assert response["headers"]["cache-control"] == "no-store"
        _, manifest, _ = call("GET", "/v1/manifest")
        assert manifest == {
            "version": 1,
            "model": "m",
            "dims": 2,
            "count": 3,
            "file": "https://files.example.test/kw.f16",
            "papers": False,
            "max_k": 5,
        }

    def test_query_returns_the_nearest_and_only_the_public_metadata(self):
        response, body, _ = call("POST", "/v1/query", {"vector": [1, 0], "k": 2})
        assert response["statusCode"] == 200
        assert [m["id"] for m in body["matches"]] == ["firm-size", "large-firms"]
        assert body["matches"][0]["metadata"] == {"name": "firm size", "works": 900}  # 'secret' never leaves

    def test_query_filter_and_default_k(self):
        _, body, _ = call("POST", "/v1/query", {"vector": [1, 0], "filter": {"works": {"$gte": 500}}})
        assert [m["id"] for m in body["matches"]] == ["firm-size", "large-firms"]

    def test_fetch_returns_vectors_for_known_ids(self):
        _, body, _ = call("POST", "/v1/fetch", {"ids": ["large-firms", "nothing"]})
        assert body["records"] == [
            {"id": "large-firms", "values": [0.9, 0.1], "metadata": {"name": "large firms", "works": 600}}
        ]

    def test_a_base64_body_is_decoded(self):
        raw = base64.b64encode(json.dumps({"ids": ["pt3sn"]}).encode()).decode()
        _, body, _ = call("POST", "/v1/fetch", raw, isBase64Encoded=True)
        assert body["records"][0]["id"] == "pt3sn"

    def test_the_path_can_come_from_the_request_context(self):
        e = event("GET", "/v1/health")
        del e["rawPath"]
        assert h.handle(e, make_store, CONFIG, lambda s: None)["statusCode"] == 200


class TestRefusals:
    @pytest.mark.parametrize(
        "path,method,status",
        [
            ("/v1/other", "GET", 404),
            ("/", "GET", 404),
            ("/v1/query", "GET", 405),
            ("/v1/health", "POST", 405),
            ("/v1/fetch", "DELETE", 405),
        ],
    )
    def test_unknown_routes_and_wrong_methods(self, path, method, status):
        response, body, _ = call(method, path)
        assert response["statusCode"] == status and "error" in body

    @pytest.mark.parametrize(
        "body",
        [
            {},
            {"vector": [1]},
            {"vector": [1, 2, 3]},
            {"vector": "no"},
            {"vector": [1, "x"]},
            {"vector": [1, True]},
            {"vector": [1, float("nan")]},
            {"vector": [1, float("inf")]},
            {"vector": [1, 0], "k": 0},
            {"vector": [1, 0], "k": 6},
            {"vector": [1, 0], "k": 2.5},
            {"vector": [1, 0], "k": True},
            {"vector": [1, 0], "k": "3"},
            {"vector": [1, 0], "filter": {"name": "x"}},
            {"vector": [1, 0], "filter": {"works": 5}},
            {"vector": [1, 0], "filter": {"works": {}}},
            {"vector": [1, 0], "filter": {"works": {"$regex": "x"}}},
            {"vector": [1, 0], "filter": {"works": {"$gte": "5"}}},
            {"vector": [1, 0], "filter": []},
        ],
    )
    def test_bad_queries_are_refused_with_400_before_the_database_is_asked(self, body):
        class Never:
            def query(self, *args):
                raise AssertionError("the database should not have been asked")

        response, out, _ = call("POST", "/v1/query", body, store=Never())
        assert response["statusCode"] == 400 and "error" in out

    @pytest.mark.parametrize(
        "body",
        [
            {},
            {"ids": []},
            {"ids": "a"},
            {"ids": ["a"] * 11},
            {"ids": [1]},
            {"ids": [""]},
            {"ids": ["x" * 513]},
        ],
    )
    def test_bad_fetches_are_refused(self, body):
        assert call("POST", "/v1/fetch", body)[0]["statusCode"] == 400

    @pytest.mark.parametrize("raw", ["not json", "[1, 2]", "", None])
    def test_a_body_that_is_not_a_json_object(self, raw):
        assert call("POST", "/v1/fetch", raw)[0]["statusCode"] == 400

    def test_bad_base64_and_oversize_bodies(self):
        assert call("POST", "/v1/fetch", "***", isBase64Encoded=True)[0]["statusCode"] == 400
        assert call("POST", "/v1/fetch", "x" * (h.MAX_BODY + 1))[0]["statusCode"] == 413

    def test_k_is_capped_by_the_configuration(self):
        assert call("POST", "/v1/query", {"vector": [1, 0], "k": 5})[0]["statusCode"] == 200
        assert call("POST", "/v1/query", {"vector": [1, 0], "k": 6})[0]["statusCode"] == 400


class TestFailures:
    def test_a_database_failure_is_a_502_that_does_not_repeat_its_message(self):
        class Broken:
            def query(self, *args):
                raise VectorStoreError("HTTP 500 from https://myindex-123.svc.example.io: secret detail")

        response, _, logs = call("POST", "/v1/query", {"vector": [1, 0], "k": 1}, store=Broken())
        assert response["statusCode"] == 502 and "secret detail" not in response["body"]
        assert any("secret detail" in line for line in logs)  # but it is logged, for you

    def test_anything_unexpected_is_a_500_with_no_details(self):
        class Odd:
            def fetch(self, ids):
                raise RuntimeError("internal path /var/task/x.py")

        response, _, logs = call("POST", "/v1/fetch", {"ids": ["a"]}, store=Odd())
        assert response["statusCode"] == 500 and "var/task" not in response["body"]
        assert any("RuntimeError" in line for line in logs)

    def test_every_request_is_logged_once_with_its_status(self):
        _, _, logs = call("GET", "/v1/health")
        entry = json.loads(logs[-1])
        assert entry["event"] == "request" and entry["status"] == 200 and entry["path"] == "/v1/health"


class TestNeighbors:
    def test_a_keyword_name_is_looked_up(self):
        response, body, _ = call("POST", "/v1/neighbors", {"name": "Firm Size", "k": 2})
        assert response["statusCode"] == 200
        assert [m["id"] for m in body["matches"]] == ["large-firms", "pt3sn"]  # not firm-size itself
        assert body["matches"][0]["metadata"] == {"name": "large firms", "works": 600}

    def test_a_name_that_is_not_a_keyword_is_a_404_that_says_how_to_search_a_phrase(self):
        response, body, _ = call("POST", "/v1/neighbors", {"name": "tiny plasma thing"})
        assert response["statusCode"] == 404
        assert "embed the phrase yourself" in body["error"] and "/v1/query" in body["error"]

    def test_min_works_distinct_and_the_default_k(self):
        _, body, _ = call("POST", "/v1/neighbors", {"name": "firm size", "min_works": 100})
        assert [m["id"] for m in body["matches"]] == ["large-firms"]
        _, body, _ = call("POST", "/v1/neighbors", {"name": "firm size", "distinct": True, "k": 5})
        assert "firm-size" not in [m["id"] for m in body["matches"]]
        small = h.Config(dims=2, max_k=2)
        assert (
            call("POST", "/v1/neighbors", {"name": "firm size"}, config=small)[0]["statusCode"] == 200
        )  # 25 -> 2

    @pytest.mark.parametrize(
        "body",
        [
            {},
            {"name": ""},
            {"name": "   "},
            {"name": 5},
            {"name": "x" * 201},
            {"name": "a", "k": 0},
            {"name": "a", "k": 6},
            {"name": "a", "min_works": -1},
            {"name": "a", "min_works": 1.5},
            {"name": "a", "min_works": True},
            {"name": "a", "distinct": "yes"},
        ],
    )
    def test_bad_requests_are_refused_before_the_database_is_asked(self, body):
        class Never:
            def fetch(self, ids):
                raise AssertionError("the database should not have been asked")

        response, out, _ = call("POST", "/v1/neighbors", body, store=Never())
        assert response["statusCode"] == 400 and "error" in out


class Papers:
    """A corpus of two papers that remembers what it was asked."""

    def __init__(self):
        self.asked = []

    def search(self, query, k):
        from confoundsearch import Passage

        self.asked.append((query, k))
        found = [
            Passage("p1", "Larger firms delegate.", "Firm size", "https://s2/p1"),
            Passage("p2", "Text.", "", ""),
        ]
        return found[:k]


class TestPapers:
    def test_it_returns_passages(self):
        corpus = Papers()
        response, body, _ = call("POST", "/v1/papers", {"query": " family firms ", "k": 1}, corpus=corpus)
        assert response["statusCode"] == 200 and corpus.asked == [("family firms", 1)]
        assert body["passages"] == [
            {"id": "p1", "title": "Firm size", "text": "Larger firms delegate.", "url": "https://s2/p1"}
        ]

    def test_the_default_is_ten_papers_and_never_more_than_the_limits(self):
        corpus = Papers()
        call("POST", "/v1/papers", {"query": "x"}, corpus=corpus, config=h.Config(dims=2, max_k=5))
        assert corpus.asked[-1] == ("x", 5)  # the configured maximum is below ten
        call("POST", "/v1/papers", {"query": "x"}, corpus=corpus, config=h.Config(dims=2))
        assert corpus.asked[-1] == ("x", 10)
        assert (
            call("POST", "/v1/papers", {"query": "x", "k": 51}, corpus=corpus, config=h.Config(dims=2))[0][
                "statusCode"
            ]
            == 400
        )

    @pytest.mark.parametrize(
        "body",
        [
            {},
            {"query": ""},
            {"query": "  "},
            {"query": 5},
            {"query": "x" * 301},
            {"query": "x", "k": 0},
            {"query": "x", "k": "3"},
        ],
    )
    def test_bad_requests_are_refused_before_elasticsearch_is_asked(self, body):
        class Never:
            def search(self, query, k):
                raise AssertionError("elasticsearch should not have been asked")

        response, out, _ = call("POST", "/v1/papers", body, corpus=Never())
        assert response["statusCode"] == 400 and "error" in out

    def test_without_elasticsearch_it_is_a_404_not_a_crash(self):
        response, body, _ = call("POST", "/v1/papers", {"query": "firm size"})
        assert response["statusCode"] == 404 and "not switched on" in body["error"]

    def test_an_elasticsearch_failure_is_a_502_that_does_not_repeat_its_message(self):
        from confoundsearch import CorpusError

        class Broken:
            def search(self, query, k):
                raise CorpusError("HTTP 401 from https://my-project.es.example.io: key abc123 refused")

        response, _, logs = call("POST", "/v1/papers", {"query": "x"}, corpus=Broken())
        assert (
            response["statusCode"] == 502
            and "abc123" not in response["body"]
            and "es.example" not in response["body"]
        )
        assert any("abc123" in line for line in logs)

    def test_the_manifest_says_whether_paper_search_is_on(self):
        assert call("GET", "/v1/manifest")[1]["papers"] is False
        assert call("GET", "/v1/manifest", config=h.Config(dims=2, papers=True))[1]["papers"] is True

    def test_wrong_method_on_papers(self):
        assert call("GET", "/v1/papers")[0]["statusCode"] == 405


class TestSetup:
    def test_the_corpus_is_built_from_the_environment_or_absent(self):
        assert h.make_corpus({}) is None
        corpus = h.make_corpus(
            {"ELASTIC_URL": "https://es.example", "ELASTIC_API_KEY": "k", "ELASTIC_INDEX": "docs"}
        )
        assert (corpus.url, corpus.api_key, corpus.index) == ("https://es.example", "k", "docs")
        keyless = h.make_corpus({"ELASTIC_URL": "https://es.example"})
        assert keyless.api_key is None and keyless.index == "papers"

    def test_config_from_the_environment(self):
        config = h.config_from_env(
            {
                "EMBED_DIMS": "256",
                "MAX_K": "50",
                "EMBED_MODEL": "x",
                "KEYWORD_COUNT": "7",
                "LOCAL_COPY_URL": "https://f",
            }
        )
        assert config == h.Config(dims=256, max_k=50, model="x", count=7, file_url="https://f")
        assert h.config_from_env({}) == h.Config()
        assert h.config_from_env({"ELASTIC_URL": "https://es"}).papers is True

    def test_secrets_come_from_the_environment_or_the_secure_store(self):
        assert h.read_secret("K", "P", {"K": "k"}) == "k"
        assert h.read_secret("K", "P", {}) is None

        class Ssm:
            def get_parameter(self, Name, WithDecryption):
                assert Name == "/kw/key" and WithDecryption is True
                return {"Parameter": {"Value": "from-ssm"}}

        assert h.read_secret("K", "P", {"P": "/kw/key"}, Ssm()) == "from-ssm"

    def test_boto3_is_used_for_the_secure_store_when_no_client_is_given(self, monkeypatch):
        class Client:
            def get_parameter(self, Name, WithDecryption):
                return {"Parameter": {"Value": "via-boto3"}}

        class Boto3:
            @staticmethod
            def client(service):
                assert service == "ssm"
                return Client()

        monkeypatch.setattr(h.importlib, "import_module", lambda name: Boto3)
        assert h.read_secret("K", "P", {"P": "/kw/key"}) == "via-boto3"

    def test_the_store_is_built_from_the_environment(self):
        store = h.make_store(
            {
                "PINECONE_API_KEY": "k",
                "PINECONE_INDEX": "kw",
                "PINECONE_NAMESPACE": "ns",
                "PINECONE_HOST": "https://h.example",
            }
        )
        assert (store.api_key, store.index, store.namespace, store.host) == ("k", "kw", "ns", "h.example")
        with pytest.raises(VectorStoreError, match="no Pinecone key"):
            h.make_store({})


class TestEntryPoint:
    def test_the_lambda_entry_point_builds_the_store_and_the_corpus_once(self, monkeypatch):
        built = []
        monkeypatch.setattr(h, "_store", None)
        monkeypatch.setattr(h, "_corpus", None)
        monkeypatch.setattr(h, "_corpus_ready", False)
        monkeypatch.setattr(h, "make_store", lambda: built.append("store") or make_store())
        monkeypatch.setattr(h, "make_corpus", lambda: built.append("corpus") or Papers())
        monkeypatch.setenv("EMBED_DIMS", "2")
        for _ in range(3):
            assert h.lambda_handler(event("GET", "/v1/health"), None)["statusCode"] == 200
            assert (
                h.lambda_handler(event("POST", "/v1/neighbors", {"name": "firm size", "k": 1}), None)[
                    "statusCode"
                ]
                == 200
            )
            assert (
                h.lambda_handler(event("POST", "/v1/papers", {"query": "x", "k": 1}), None)["statusCode"]
                == 200
            )
        assert sorted(built) == ["corpus", "store"]

    def test_with_no_elasticsearch_configured_the_entry_point_answers_404(self, monkeypatch):
        monkeypatch.setattr(h, "_corpus", None)
        monkeypatch.setattr(h, "_corpus_ready", False)
        monkeypatch.delenv("ELASTIC_URL", raising=False)
        r = h.lambda_handler(event("POST", "/v1/papers", {"query": "x"}), None)
        assert r["statusCode"] == 404
