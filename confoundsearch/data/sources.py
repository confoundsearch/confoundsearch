"""Paper sources that need no corpus of your own: Semantic Scholar's API and OpenAlex's.

Both implement the same one-method `Corpus` protocol as the file and Elasticsearch corpora, so the agent
does not know or care which it is searching. The passage the agent reads and cites is the abstract. Semantic
Scholar withholds many abstracts, so there a paper without one is kept, with its tldr or title as the text.

    SemanticScholarCorpus(api_key="...")      # a free key raises the rate limit; request one at S2_KEY_URL
    OpenAlexCorpus(mailto="you@example.org")  # an email (or an api_key) puts you in OpenAlex's polite pool
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import urlencode

from confoundsearch.errors import CorpusError, RemoteError
from confoundsearch.http import Transport, urllib_transport
from confoundsearch.logic.graph import Passage

MAX_PAGE = 100
S2_KEY_URL = "https://www.semanticscholar.org/product/api"  # where to request a free Semantic Scholar API key


def _get(transport: Transport, url: str, headers: Mapping[str, str], timeout: float, hint: str) -> Any:
    try:
        raw = transport(url, headers, None, timeout)
    except RemoteError as err:
        extra = f" {hint}" if "HTTP 429" in str(err) else ""
        raise CorpusError(f"{err}{extra}") from err
    try:
        return json.loads(raw)
    except ValueError as err:
        raise CorpusError(f"unexpected response: {raw[:200]!r}") from err


def _page_size(k: int) -> int:
    """Ask for extra results, since some papers have no abstract and are dropped."""
    return min(max(2 * k, k), MAX_PAGE)


class SemanticScholarCorpus:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://api.semanticscholar.org/graph/v1",
        timeout: float = 30.0,
        transport: Transport = urllib_transport,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.transport = transport

    def search(self, query: str, k: int) -> list[Passage]:
        if k <= 0 or not query.strip():
            return []
        params = urlencode(
            {
                "query": query,
                "limit": _page_size(k),
                "fields": "title,abstract,tldr,url,year,citationCount,corpusId,authors,venue,journal,"
                "publicationVenue",
            }
        )
        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        hint = (
            f"Semantic Scholar is rate limiting you. Request a free API key at {S2_KEY_URL} "
            "and set it as S2_KEY (or --s2-key); it raises the limit."
        )
        data = _get(self.transport, f"{self.base_url}/paper/search?{params}", headers, self.timeout, hint)
        try:
            papers = data.get("data") or []  # an empty result has no "data" at all
            passages = [self._passage(p) for p in papers if self._text(p)]
        except (AttributeError, KeyError, TypeError) as err:
            raise CorpusError(f"unexpected response: {str(data)[:200]!r}") from err
        return passages[:k]

    @staticmethod
    def _text(paper: dict[str, Any]) -> str:
        """The abstract, else the one-sentence summary, else the title: many papers have no abstract here."""
        return paper.get("abstract") or (paper.get("tldr") or {}).get("text") or paper.get("title") or ""

    @classmethod
    def _passage(cls, paper: dict[str, Any]) -> Passage:
        return Passage(
            id=str(paper.get("corpusId") or paper["paperId"]),
            text=cls._text(paper),
            title=paper.get("title") or "",
            url=paper.get("url") or "",
            authors=authors_text([(a or {}).get("name") or "" for a in paper.get("authors") or []]),
            venue=venue_text(paper),
            year=paper.get("year"),
            cited_by=paper.get("citationCount"),
        )


def authors_text(names: Sequence[str], shown: int = 3) -> str:
    """The first few authors, then "et al.": "Maria Guadalupe, Hongyi Li, Julie Wulf"."""
    named = [" ".join(n.split()) for n in names if isinstance(n, str) and n.strip()]
    return ", ".join(named[:shown]) + (" et al." if len(named) > shown else "")


def venue_text(paper: Mapping[str, Any]) -> str:
    """Semantic Scholar's venue, else the journal it names, else its publication venue."""
    named = (
        paper.get("venue"),
        (paper.get("journal") or {}).get("name"),
        (paper.get("publicationVenue") or {}).get("name"),
    )
    return next((str(name) for name in named if name), "")


def abstract_from_inverted_index(inverted: Mapping[str, list[int]]) -> str:
    """OpenAlex stores an abstract as {word: [positions]}. Put the words back in order."""
    placed = sorted((position, word) for word, positions in inverted.items() for position in positions)
    return " ".join(word for _, word in placed)


class OpenAlexCorpus:
    def __init__(
        self,
        api_key: str | None = None,
        mailto: str | None = None,
        base_url: str = "https://api.openalex.org",
        timeout: float = 30.0,
        transport: Transport = urllib_transport,
    ) -> None:
        self.api_key = api_key
        self.mailto = mailto
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.transport = transport

    def search(self, query: str, k: int) -> list[Passage]:
        if k <= 0 or not query.strip():
            return []
        params = {
            "search": query,
            "per_page": _page_size(k),
            "select": "id,title,abstract_inverted_index,doi,publication_year,cited_by_count,"
            "authorships,primary_location",
        }
        if self.api_key:
            params["api_key"] = self.api_key
        if self.mailto:
            params["mailto"] = self.mailto
        hint = (
            "OpenAlex is rate limiting you; add an email (OPENALEX_EMAIL) or an API key (OPENALEX_API_KEY)."
        )
        data = _get(
            self.transport,
            f"{self.base_url}/works?{urlencode(params)}",
            {"Accept": "application/json"},
            self.timeout,
            hint,
        )
        try:
            works = data["results"]
            passages = [self._passage(w) for w in works if w.get("abstract_inverted_index")]
        except (KeyError, TypeError, AttributeError) as err:
            raise CorpusError(f"unexpected response: {str(data)[:200]!r}") from err
        return passages[:k]

    @staticmethod
    def _passage(work: dict[str, Any]) -> Passage:
        return Passage(
            id=work["id"].rsplit("/", 1)[-1],
            text=abstract_from_inverted_index(work["abstract_inverted_index"]),
            title=work.get("title") or "",
            url=work.get("doi") or work["id"],
            authors=authors_text(
                [
                    ((a or {}).get("author") or {}).get("display_name") or ""
                    for a in work.get("authorships") or []
                ]
            ),
            venue=(((work.get("primary_location") or {}).get("source")) or {}).get("display_name") or "",
            year=work.get("publication_year"),
            cited_by=work.get("cited_by_count"),
        )
