from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from confoundsearch import Passage
from confoundsearch.data.es import ElasticsearchCorpus
from confoundsearch.errors import CorpusError

HITS = {
    "hits": {
        "hits": [
            {
                "_id": "abc",
                "_source": {
                    "corpusid": 42,
                    "title": "Firm size",
                    "abstract": "Big firms win.",
                    "url": "https://x/42",
                },
            },
            {"_id": "def", "_source": {"abstract": "No title here."}},
        ]
    }
}


def stub(response: dict | bytes = HITS, status: int = 200):
    seen: list[dict] = []
    body = response if isinstance(response, bytes) else json.dumps(response).encode()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append({"path": self.path, "headers": dict(self.headers), "body": payload})
            self.send_response(status)
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    httpd = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, f"http://127.0.0.1:{httpd.server_port}", seen


@pytest.fixture
def served():
    servers = []

    def make(*args, **kwargs):
        httpd, url, seen = stub(*args, **kwargs)
        servers.append(httpd)
        return url, seen

    yield make
    for httpd in servers:
        httpd.shutdown()
        httpd.server_close()


def test_searches_the_index_and_parses_hits(served):
    url, seen = served()
    passages = ElasticsearchCorpus(url, index="papers").search("firm size", 2)
    assert passages == [
        Passage("42", "Big firms win.", "Firm size", "https://x/42"),
        Passage("def", "No title here.", "", ""),
    ]
    (request,) = seen
    assert request["path"] == "/papers/_search" and request["body"]["size"] == 2
    query = request["body"]["query"]["bool"]
    assert query["must"]["multi_match"] == {"query": "firm size", "fields": ["title^2", "abstract"]}
    assert query["filter"] == [{"exists": {"field": "abstract"}}]


def test_sends_the_api_key_as_elastic_cloud_expects(served):
    url, seen = served()
    ElasticsearchCorpus(url, api_key="secret").search("q", 1)
    assert seen[0]["headers"]["Authorization"] == "ApiKey secret"


def test_no_credentials_means_no_authorization_header(served):
    url, seen = served()
    ElasticsearchCorpus(url).search("q", 1)
    assert "Authorization" not in seen[0]["headers"]


def test_minimum_citations_filter(served):
    url, seen = served()
    ElasticsearchCorpus(url, min_citations=10).search("q", 1)
    assert {"range": {"citationcount": {"gte": 10}}} in seen[0]["body"]["query"]["bool"]["filter"]


def test_custom_fields_and_trailing_slash(served):
    url, seen = served({"hits": {"hits": [{"_id": "1", "_source": {"body": "text", "name": "N"}}]}})
    corpus = ElasticsearchCorpus(url + "/", index="docs", text_field="body", title_field="name")
    assert corpus.search("q", 3) == [Passage("1", "text", "N", "")]
    assert seen[0]["path"] == "/docs/_search"
    assert seen[0]["body"]["query"]["bool"]["must"]["multi_match"]["fields"] == ["name^2", "body"]


def test_non_positive_k_skips_the_request(served):
    url, seen = served()
    assert ElasticsearchCorpus(url).search("q", 0) == [] and seen == []


def test_http_errors(served):
    url, _ = served(b'{"error": "nope"}', status=403)
    with pytest.raises(CorpusError, match="HTTP 403"):
        ElasticsearchCorpus(url).search("q", 1)


@pytest.mark.parametrize(
    "response", [b"not json", {"nope": 1}, {"hits": {"hits": [{"_id": "1", "_source": {}}]}}]
)
def test_malformed_responses(served, response):
    url, _ = served(response)
    with pytest.raises(CorpusError, match="unexpected response"):
        ElasticsearchCorpus(url).search("q", 1)


def test_transport_is_injectable():
    corpus = ElasticsearchCorpus("http://x", transport=lambda *a: json.dumps({"hits": {"hits": []}}).encode())
    assert corpus.search("q", 1) == []
