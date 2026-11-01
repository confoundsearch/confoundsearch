"""Semantic Scholar and OpenAlex corpora, against local stand-ins for their APIs."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlsplit

import pytest

from confoundsearch import OpenAlexCorpus, Passage, SemanticScholarCorpus
from confoundsearch.data.sources import abstract_from_inverted_index
from confoundsearch.errors import CorpusError


class Api:
    """Answers every GET with a fixed status and body, and remembers what was asked."""

    def __init__(self, body, status=200):
        self.requests = []
        outer = self
        payload = body if isinstance(body, bytes) else json.dumps(body).encode()

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                parts = urlsplit(self.path)
                outer.requests.append(
                    {
                        "path": parts.path,
                        "query": parse_qs(parts.query),
                        "headers": {k.lower(): v for k, v in self.headers.items()},
                    }
                )
                self.send_response(status)
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args):
                pass

        self.httpd = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_port}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def api():
    made = []

    def make(body, status=200):
        made.append(Api(body, status))
        return made[-1]

    yield make
    for item in made:
        item.close()


S2_BODY = {
    "total": 6,
    "data": [
        {
            "paperId": "abc",
            "corpusId": 42,
            "title": "Firm size",
            "abstract": "Big firms win.",
            "url": "https://s2/42",
        },
        {
            "paperId": "def",
            "corpusId": None,
            "title": "No corpus id",
            "abstract": "Uses paper id.",
            "url": None,
        },
        {"paperId": "ghi", "corpusId": 7, "title": "No abstract", "abstract": None, "url": "https://s2/7"},
        {"paperId": "jkl", "corpusId": 8, "title": None, "abstract": "Untitled.", "url": ""},
        {
            "paperId": "mno",
            "corpusId": 9,
            "title": "Has a tldr",
            "abstract": None,
            "tldr": {"text": "Short."},
            "url": "",
        },
        {"paperId": "pqr", "corpusId": 10, "title": None, "abstract": None, "tldr": None, "url": ""},
    ],
}


class TestSemanticScholar:
    def test_searches_and_falls_back_to_tldr_then_title_without_an_abstract(self, api):
        server = api(S2_BODY)
        passages = SemanticScholarCorpus(base_url=server.url).search("firm size", 5)
        assert passages == [
            Passage("42", "Big firms win.", "Firm size", "https://s2/42"),
            Passage("def", "Uses paper id.", "No corpus id", ""),
            Passage("7", "No abstract", "No abstract", "https://s2/7"),
            Passage("8", "Untitled.", "", ""),
            Passage("9", "Short.", "Has a tldr", ""),
        ]
        (request,) = server.requests
        assert request["path"] == "/paper/search" and request["query"]["query"] == ["firm size"]
        assert {"abstract", "tldr"} <= set(request["query"]["fields"][0].split(",")) and request["query"][
            "limit"
        ] == ["10"]

    def test_never_returns_more_than_asked(self, api):
        assert len(SemanticScholarCorpus(base_url=api(S2_BODY).url).search("q", 2)) == 2

    def test_the_api_key_goes_in_the_header_when_given(self, api):
        server = api(S2_BODY)
        SemanticScholarCorpus("secret", base_url=server.url).search("q", 1)
        SemanticScholarCorpus(base_url=server.url).search("q", 1)
        assert server.requests[0]["headers"]["x-api-key"] == "secret"
        assert "x-api-key" not in server.requests[1]["headers"]

    def test_the_page_size_is_capped(self, api):
        server = api(S2_BODY)
        SemanticScholarCorpus(base_url=server.url).search("q", 500)
        assert server.requests[0]["query"]["limit"] == ["100"]

    @pytest.mark.parametrize("query,k", [("", 5), ("   ", 5), ("q", 0), ("q", -1)])
    def test_nothing_to_search_makes_no_request(self, api, query, k):
        server = api(S2_BODY)
        assert SemanticScholarCorpus(base_url=server.url).search(query, k) == [] and server.requests == []

    def test_an_empty_result_has_no_data_key(self, api):
        assert SemanticScholarCorpus(base_url=api({"total": 0, "offset": 0}).url).search("q", 3) == []

    def test_rate_limiting_says_how_to_fix_it(self, api):
        with pytest.raises(CorpusError, match=r"HTTP 429.*semanticscholar\.org/product/api.*S2_KEY"):
            SemanticScholarCorpus(base_url=api({"message": "slow down"}, 429).url).search("q", 3)

    def test_other_http_errors_are_plain(self, api):
        with pytest.raises(CorpusError, match="HTTP 500") as info:
            SemanticScholarCorpus(base_url=api({"error": "x"}, 500).url).search("q", 3)
        assert "S2_KEY" not in str(info.value)

    @pytest.mark.parametrize(
        "body", [b"not json", {"data": "oops"}, {"data": [1]}, {"data": [{"abstract": "a"}]}, []]
    )
    def test_malformed_responses(self, api, body):
        with pytest.raises(CorpusError, match="unexpected response"):
            SemanticScholarCorpus(base_url=api(body).url).search("q", 3)

    def test_transport_is_injectable(self):
        seen = []

        def transport(url, headers, body, timeout):
            seen.append((url, body, timeout))
            return b'{"data": []}'

        assert (
            SemanticScholarCorpus(transport=transport, timeout=3.0, base_url="http://x/").search("a b", 1)
            == []
        )
        url, body, timeout = seen[0]
        assert url.startswith("http://x/paper/search?query=a+b") and body is None and timeout == 3.0


def inverted(text):
    index = {}
    for position, word in enumerate(text.split()):
        index.setdefault(word, []).append(position)
    return index


OA_BODY = {
    "results": [
        {
            "id": "https://openalex.org/W1",
            "title": "Firm size",
            "doi": "https://doi.org/10.1/x",
            "abstract_inverted_index": inverted("Big firms invest in IT and big firms win"),
        },
        {
            "id": "https://openalex.org/W2",
            "title": None,
            "doi": None,
            "abstract_inverted_index": inverted("No doi here"),
        },
        {
            "id": "https://openalex.org/W3",
            "title": "No abstract",
            "doi": None,
            "abstract_inverted_index": None,
        },
    ]
}


class TestOpenAlex:
    def test_rebuilds_abstracts_and_skips_works_without_one(self, api):
        server = api(OA_BODY)
        passages = OpenAlexCorpus(base_url=server.url).search("firm size", 5)
        assert passages == [
            Passage("W1", "Big firms invest in IT and big firms win", "Firm size", "https://doi.org/10.1/x"),
            Passage("W2", "No doi here", "", "https://openalex.org/W2"),
        ]
        (request,) = server.requests
        assert request["path"] == "/works" and request["query"]["search"] == ["firm size"]
        assert (
            request["query"]["per_page"] == ["10"]
            and "abstract_inverted_index" in request["query"]["select"][0]
        )

    def test_a_repeated_word_lands_in_every_position(self):
        assert abstract_from_inverted_index({"a": [0, 2], "b": [1]}) == "a b a"
        assert abstract_from_inverted_index({}) == ""

    def test_never_returns_more_than_asked(self, api):
        assert len(OpenAlexCorpus(base_url=api(OA_BODY).url).search("q", 1)) == 1

    def test_the_key_and_email_are_sent_when_given(self, api):
        server = api(OA_BODY)
        OpenAlexCorpus("k", "me@example.org", base_url=server.url).search("q", 1)
        OpenAlexCorpus(base_url=server.url).search("q", 1)
        assert server.requests[0]["query"]["api_key"] == ["k"] and server.requests[0]["query"]["mailto"] == [
            "me@example.org"
        ]
        assert "api_key" not in server.requests[1]["query"] and "mailto" not in server.requests[1]["query"]

    @pytest.mark.parametrize("query,k", [("", 5), ("q", 0)])
    def test_nothing_to_search_makes_no_request(self, api, query, k):
        server = api(OA_BODY)
        assert OpenAlexCorpus(base_url=server.url).search(query, k) == [] and server.requests == []

    def test_rate_limiting_says_how_to_fix_it(self, api):
        with pytest.raises(CorpusError, match=r"HTTP 429.*OPENALEX_EMAIL"):
            OpenAlexCorpus(base_url=api({}, 429).url).search("q", 3)

    @pytest.mark.parametrize(
        "body", [b"nope", {}, {"results": None}, {"results": [1]}, {"results": [{"id": "x"}]}, []]
    )
    def test_malformed_responses(self, api, body):
        if body == {"results": [{"id": "x"}]}:
            assert (
                OpenAlexCorpus(base_url=api(body).url).search("q", 3) == []
            )  # no abstract: dropped, not an error
            return
        with pytest.raises(CorpusError, match="unexpected response"):
            OpenAlexCorpus(base_url=api(body).url).search("q", 3)

    def test_a_work_missing_its_id_is_an_error_not_a_crash(self, api):
        with pytest.raises(CorpusError):
            OpenAlexCorpus(base_url=api({"results": [{"abstract_inverted_index": {"a": [0]}}]}).url).search(
                "q", 3
            )


class TestDotenv:
    def load(self, monkeypatch, tmp_path, text):
        from confoundsearch.dotenv import load_dotenv

        monkeypatch.setattr("os.environ", {})
        path = tmp_path / ".env"
        path.write_text(text)
        load_dotenv(path)
        import os

        return os.environ

    def test_reads_plain_quoted_and_exported_values(self, monkeypatch, tmp_path):
        env = self.load(
            monkeypatch, tmp_path, '# c\nA=1\n\nexport B="two"\nC=\'k=v==\'\nnot a setting\nD="\n'
        )
        assert env == {"A": "1", "B": "two", "C": "k=v==", "D": '"'}

    def test_never_overrides_and_tolerates_a_missing_file(self, monkeypatch, tmp_path):
        from confoundsearch.dotenv import load_dotenv

        monkeypatch.setattr("os.environ", {"A": "shell"})
        (tmp_path / ".env").write_text("A=file\nB=2\n")
        load_dotenv(tmp_path / ".env")
        load_dotenv(tmp_path / "missing")
        import os

        assert os.environ == {"A": "shell", "B": "2"}
