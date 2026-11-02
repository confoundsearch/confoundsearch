"""The clients of the keyword proxy: HttpVectorStore, KeywordService and HttpCorpus."""

import json

import pytest

from confoundsearch import (
    HttpCorpus,
    HttpVectorStore,
    KeywordIndex,
    KeywordNotFound,
    KeywordService,
    VectorRecord,
    VectorStoreError,
)
from confoundsearch.data.keyword_service import FETCH_LIMIT
from confoundsearch.errors import HttpError

VECTORS = {"firm-size": [1.0, 0.0], "large-firms": [0.9, 0.1], "pt3sn": [0.0, 1.0]}
NAMES = {"firm-size": "firm size", "large-firms": "large firms", "pt3sn": "Pt3Sn"}


class FakeService:
    """The proxy's answers, enough to run the client against."""

    def __init__(self, fail=()):
        self.calls, self.fail = [], list(fail)

    def __call__(self, url, headers, body, timeout):
        payload = json.loads(body) if body else None
        self.calls.append((url, headers, payload))
        if self.fail:
            status = self.fail.pop(0)
            if status:
                raise HttpError(f"HTTP {status} from {url}: busy", status, 0.0 if status == 429 else None)
        path = url.split("//", 1)[1].split("/", 1)[1]
        if path == "v1/manifest":
            return json.dumps({"version": 1, "dims": 2, "max_k": 400}).encode()
        if path == "v1/fetch":
            records = [
                {"id": i, "values": VECTORS[i], "metadata": {"name": NAMES[i], "works": 10}}
                for i in payload["ids"]
                if i in VECTORS
            ]
            return json.dumps({"records": records}).encode()
        if path == "v1/neighbors":
            if payload["name"] not in NAMES.values():
                raise HttpError(f"HTTP 404 from {url}: no keyword has that name", 404)
            found = [i for i in VECTORS if NAMES[i].casefold() != payload["name"].casefold()]
            matches = [{"id": i, "score": 0.5, "metadata": {"name": NAMES[i], "works": 10}} for i in found]
            return json.dumps({"via": "keyword", "matches": matches[: payload["k"]]}).encode()
        if path == "v1/papers":
            papers = [
                {"id": "p1", "title": "Firm size", "text": "Larger firms delegate.", "url": "https://s2/p1"},
                {"id": "p2", "title": None, "text": "No title here.", "url": None},
            ]
            return json.dumps({"passages": papers[: payload["k"]]}).encode()
        if path == "v1/query":
            found = sorted(VECTORS, key=lambda i: -sum(a * b for a, b in zip(payload["vector"], VECTORS[i])))
            matches = [{"id": i, "score": 0.5, "metadata": {"name": NAMES[i], "works": 10}} for i in found]
            return json.dumps({"matches": matches[: payload["k"]]}).encode()
        raise AssertionError(path)


def service(**kwargs):
    fake = FakeService(**kwargs)
    return HttpVectorStore("https://api.example.test/", transport=fake, sleep=lambda s: None), fake


def test_the_keyword_index_works_on_top_of_it():
    store, fake = service()
    found = KeywordIndex(store).query("Firm Size", 2)
    assert [m.metadata["name"] for m in found] == ["large firms", "Pt3Sn"]
    urls = [c[0] for c in fake.calls]
    assert urls == [
        "https://api.example.test/v1/fetch",
        "https://api.example.test/v1/query",
    ]  # no double slash


def test_query_sends_the_vector_k_and_filter_and_k_zero_asks_nothing():
    store, fake = service()
    store.query([1, 0], 3, {"works": {"$gte": 500}})
    assert fake.calls[0][2] == {"vector": [1.0, 0.0], "k": 3, "filter": {"works": {"$gte": 500}}}
    store.query([1, 0], 3)
    assert "filter" not in fake.calls[1][2]
    assert store.query([1, 0], 0) == [] and len(fake.calls) == 2


def test_fetch_goes_in_batches_and_leaves_out_unknown_ids():
    store, fake = service()
    ids = ["firm-size", "nothing"] + [f"x{i}" for i in range(FETCH_LIMIT + 3)]
    got = store.fetch(ids)
    assert [r.id for r in got] == ["firm-size"] and len(fake.calls) == 2
    assert len(fake.calls[0][2]["ids"]) == FETCH_LIMIT
    assert store.fetch([]) == []


def test_the_manifest_is_a_plain_dictionary():
    assert service()[0].manifest() == {"version": 1, "dims": 2, "max_k": 400}


def test_busy_replies_are_retried_then_succeed():
    store, fake = service(fail=[429, 503])
    assert store.manifest()["version"] == 1 and len(fake.calls) == 3


def test_other_errors_are_not_hidden():
    store, _ = service(fail=[400])
    with pytest.raises(VectorStoreError, match="HTTP 400"):
        store.manifest()


@pytest.mark.parametrize("body", [b'{"matches": 5}', b'{"matches": [{"id": "a"}]}', b"[]"])
def test_unusable_query_replies(body):
    store = HttpVectorStore("https://x.test", transport=lambda *a: body)
    with pytest.raises(VectorStoreError, match="unexpected response"):
        store.query([1.0], 1)


@pytest.mark.parametrize("body", [b'{"records": [{"values": [1]}]}', b'{"records": 5}', b"{}"])
def test_unusable_fetch_replies(body):
    store = HttpVectorStore("https://x.test", transport=lambda *a: body)
    with pytest.raises(VectorStoreError, match="unexpected response"):
        store.fetch(["a"])


def test_an_unusable_manifest():
    store = HttpVectorStore("https://x.test", transport=lambda *a: b"[1, 2]")
    with pytest.raises(VectorStoreError, match="unexpected response"):
        store.manifest()


def test_it_cannot_write_anything():
    store, fake = service()
    for call in (
        lambda: store.create_index(2),
        lambda: store.upsert([VectorRecord("a", [1.0, 0.0])]),
        lambda: store.update("a", metadata={"x": 1}),
        lambda: store.clear(),
    ):
        with pytest.raises(VectorStoreError, match="read-only"):
            call()
    assert fake.calls == []  # refused before any request was made


class TestKeywordService:
    @staticmethod
    def db(**kwargs):
        fake = FakeService(**kwargs)
        return KeywordService("https://api.example.test", transport=fake, sleep=lambda s: None), fake

    def test_one_call_returns_matches_and_sends_the_options(self):
        db, fake = self.db()
        found = db.query("firm size", k=1, min_works=500, distinct=True)
        assert [m.metadata["name"] for m in found] == ["large firms"]
        assert fake.calls[0][2] == {"name": "firm size", "k": 1, "min_works": 500, "distinct": True}
        assert fake.calls[0][0] == "https://api.example.test/v1/neighbors"

    def test_k_zero_asks_nothing(self):
        db, fake = self.db()
        assert db.query("firm size", k=0) == [] and fake.calls == []

    def test_a_phrase_that_is_not_a_keyword_needs_an_embedder_and_says_so(self):
        db, _ = self.db()
        with pytest.raises(KeywordNotFound, match=r"no keyword named 'unknown'.*an OpenAI key"):
            db.query("unknown")

    def test_a_phrase_is_embedded_on_our_side_and_only_the_vector_is_sent(self):
        seen = []

        def embed(text):
            seen.append(text)
            return [0.0, 1.0]

        db, fake = self.db()
        found = db.query("a tiny plasma device", k=1, min_works=5, embedder=embed)
        assert seen == ["a tiny plasma device"] and [m.metadata["name"] for m in found] == ["Pt3Sn"]
        url, _, payload = fake.calls[-1]
        assert url.endswith("/v1/query") and payload == {
            "vector": [0.0, 1.0],
            "k": 1,
            "filter": {"works": {"$gte": 5}},
        }
        assert "plasma" not in json.dumps(payload)  # the phrase itself never goes to the service

    def test_the_embedder_can_be_given_once_and_distinct_drops_echoes_of_the_phrase(self):
        db = KeywordService(
            "https://api.example.test",
            embedder=lambda t: [1.0, 0.0],
            transport=FakeService(),
            sleep=lambda s: None,
        )
        names = [m.metadata["name"] for m in db.query("large", k=5, distinct=True)]
        assert names == ["firm size", "Pt3Sn"]  # 'large firms' contains the phrase 'large', so it is an echo
        calls = []
        db.store.transport = lambda *a: calls.append(a) or FakeService()(*a)
        db.query("zzz", k=1)  # not distinct: asks for exactly k
        assert json.loads(calls[-1][2])["k"] == 1

    def test_distinct_asks_for_a_wider_pool_but_never_more_than_the_service_allows(self):
        asked = []

        def transport(url, headers, body, timeout):
            if url.endswith("/v1/neighbors"):
                raise HttpError(f"HTTP 404 from {url}: not a keyword", 404)
            asked.append(json.loads(body)["k"])
            return b'{"matches": []}'

        db = KeywordService(
            "https://x.test", embedder=lambda t: [1.0, 0.0], transport=transport, sleep=lambda s: None
        )
        db.query("zzz", k=60, distinct=True)
        assert asked == [100]  # the pool would be 240, but the service's limit is 100

    def test_other_errors_pass_through(self):
        db, _ = self.db(fail=[400])
        with pytest.raises(VectorStoreError, match="HTTP 400") as info:
            db.query("firm size")
        assert not isinstance(info.value, KeywordNotFound)

    @pytest.mark.parametrize("body", [b'{"matches": 5}', b'{"matches": [{"id": "a"}]}', b"{}"])
    def test_unusable_replies(self, body):
        db = KeywordService("https://x.test", transport=lambda *a: body)
        with pytest.raises(VectorStoreError, match="unexpected response"):
            db.query("firm size")

    def test_manifest(self):
        assert self.db()[0].manifest()["version"] == 1


class TestHttpCorpus:
    @staticmethod
    def corpus(**kwargs):
        fake = FakeService(**kwargs)
        return HttpCorpus("https://api.example.test", transport=fake, sleep=lambda s: None), fake

    def test_search_returns_passages_and_sends_the_query(self):
        from confoundsearch import Passage

        corpus, fake = self.corpus()
        found = corpus.search("firm size", 5)
        assert found == [
            Passage("p1", "Larger firms delegate.", "Firm size", "https://s2/p1"),
            Passage("p2", "No title here.", "", ""),
        ]
        assert fake.calls[0][0] == "https://api.example.test/v1/papers"
        assert fake.calls[0][2] == {"query": "firm size", "k": 5}

    def test_it_never_asks_for_more_papers_than_the_service_gives(self):
        corpus, fake = self.corpus()
        corpus.search("firm size", 500)
        assert fake.calls[0][2]["k"] == 50

    @pytest.mark.parametrize("query,k", [("", 5), ("   ", 5), ("x", 0), ("x", -1)])
    def test_nothing_to_search_asks_nothing(self, query, k):
        corpus, fake = self.corpus()
        assert corpus.search(query, k) == [] and fake.calls == []

    def test_failures_are_corpus_errors(self):
        from confoundsearch import CorpusError

        corpus, _ = self.corpus(fail=[403])
        with pytest.raises(CorpusError, match="HTTP 403"):
            corpus.search("x", 1)

    @pytest.mark.parametrize("body", [b'{"passages": 5}', b'{"passages": [{"id": "a"}]}', b"{}"])
    def test_unusable_replies(self, body):
        from confoundsearch import CorpusError

        corpus = HttpCorpus("https://x.test", transport=lambda *a: body)
        with pytest.raises(CorpusError, match="unexpected response"):
            corpus.search("x", 1)


class TestWhenTheServiceIsDown:
    """The default searches every source; one that is down is skipped and the others still answer."""

    @staticmethod
    def down():
        def transport(url, headers, body, timeout):
            raise HttpError(f"HTTP 503 from {url}: unavailable", 503)

        return HttpCorpus("https://down.example.test", transport=transport, sleep=lambda s: None, tries=2)

    def test_the_other_sources_still_answer(self):
        from confoundsearch import InMemoryCorpus, Passage
        from confoundsearch.data.multi import MultiCorpus

        working = InMemoryCorpus([Passage("a", "firm size and growth", "A paper", "")])
        logged = []
        multi = MultiCorpus(
            {"service": self.down(), "file": working}, lambda kind, **info: logged.append((kind, info))
        )
        assert [p.title for p in multi.search("firm size", 5)] == ["A paper"]
        assert logged[0][0] == "source_failed" and logged[0][1]["source"] == "service"

    def test_only_when_every_source_is_down_is_it_an_error(self):
        from confoundsearch import CorpusError
        from confoundsearch.data.multi import MultiCorpus

        with pytest.raises(CorpusError, match="every source failed"):
            MultiCorpus({"service": self.down()}).search("firm size", 5)


class TestServiceUrl:
    def test_the_order_is_setting_then_environment_then_the_built_in_address(self, monkeypatch):
        from confoundsearch.data import keyword_service as module

        monkeypatch.setattr(module, "DEFAULT_SERVICE_URL", "https://built-in.test")
        monkeypatch.delenv("CONFOUNDSEARCH_API", raising=False)
        assert module.service_url() == "https://built-in.test"
        monkeypatch.setenv("CONFOUNDSEARCH_API", "https://env.test")
        assert module.service_url() == "https://env.test"
        assert module.service_url("https://flag.test") == "https://flag.test"

    def test_the_built_in_address_is_the_projects_domain_over_https(self):
        import re
        from pathlib import Path

        source = (Path(__file__).parent.parent / "confoundsearch" / "data" / "keyword_service.py").read_text()
        shipped = re.search(r'^DEFAULT_SERVICE_URL = "([^"]*)"', source, re.M).group(1)
        assert (
            shipped == "https://confoundsearch.org"
        )  # not the generated execute-api address, which can change

    def test_an_empty_built_in_address_means_no_service(self, monkeypatch):
        from confoundsearch.data import keyword_service as module

        monkeypatch.delenv("CONFOUNDSEARCH_API", raising=False)
        monkeypatch.setattr(module, "DEFAULT_SERVICE_URL", "")
        assert module.service_url() is None

    @pytest.mark.parametrize("off", ["off", "Off", " none ", "no", "false", "0", "", "   "])
    def test_off_words_switch_it_off(self, off, monkeypatch):
        from confoundsearch.data import keyword_service as module

        monkeypatch.setattr(module, "DEFAULT_SERVICE_URL", "https://built-in.test")
        assert module.service_url(off) is None

    def test_the_clients_use_the_built_in_address_and_say_what_to_do_without_one(self, monkeypatch):
        from confoundsearch.data import keyword_service as module

        monkeypatch.delenv("CONFOUNDSEARCH_API", raising=False)
        monkeypatch.setattr(module, "DEFAULT_SERVICE_URL", "https://built-in.test/")
        for client in (HttpVectorStore(), HttpCorpus().store, KeywordService().store):
            assert client.base_url == "https://built-in.test"  # and no double slash
        monkeypatch.setattr(module, "DEFAULT_SERVICE_URL", "")
        for make in (HttpVectorStore, HttpCorpus, KeywordService):
            with pytest.raises(VectorStoreError, match="no service address"):
                make()


class TestLimits:
    def test_the_service_says_how_many_neighbours_one_answer_may_hold(self):
        db = KeywordService("https://api.example.test", transport=FakeService(), sleep=lambda s: None)
        assert db.max_k == 100 and db.learn_limits() == 400 and db.max_k == 400

    def test_without_an_answer_the_old_limit_stays(self):
        for body in (b'{"version": 1}', b'{"max_k": "lots"}', b'{"max_k": 0}', b'{"max_k": true}'):
            db = KeywordService("https://x.test", transport=lambda *a, body=body: body)
            assert db.learn_limits() == 100
        down = KeywordService("https://x.test", transport=FakeService(fail=[400]), sleep=lambda s: None)
        assert down.learn_limits() == 100

    def test_a_distinct_search_asks_for_a_wider_pool_the_higher_the_limit(self):
        asked = []

        def transport(url, headers, body, timeout):
            if url.endswith("/v1/neighbors"):
                raise HttpError(f"HTTP 404 from {url}: not a keyword", 404)
            asked.append(json.loads(body)["k"])
            return b'{"matches": []}'

        db = KeywordService("https://x.test", embedder=lambda t: [1.0, 0.0], transport=transport, max_k=1000)
        db.query("zzz", k=60, distinct=True)
        assert asked == [240]  # the pool, now that the service allows more than 100
