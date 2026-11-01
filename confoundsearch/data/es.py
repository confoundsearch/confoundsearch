"""An Elasticsearch-backed corpus, over the plain REST API.

Works with a local node or Elastic Cloud. Documents need a text field (default
`abstract`) and may have `title`, `url` and `corpusid`, as in the Semantic
Scholar index the paper's experiments use.

    ElasticsearchCorpus("https://my-deployment.es.us-east-1.aws.elastic.cloud", api_key="...")
"""

from __future__ import annotations

import json
from typing import Any

from confoundsearch.errors import CorpusError, RemoteError
from confoundsearch.http import Transport, urllib_transport
from confoundsearch.logic.graph import Passage


class ElasticsearchCorpus:
    def __init__(
        self,
        url: str,
        index: str = "papers",
        api_key: str | None = None,
        text_field: str = "abstract",
        title_field: str = "title",
        min_citations: int = 0,
        timeout: float = 30.0,
        transport: Transport = urllib_transport,
    ) -> None:
        self.url = url.rstrip("/")
        self.index = index
        self.api_key = api_key
        self.text_field = text_field
        self.title_field = title_field
        self.min_citations = min_citations
        self.timeout = timeout
        self.transport = transport

    def _query(self, query: str, k: int) -> dict[str, Any]:
        match = {"multi_match": {"query": query, "fields": [f"{self.title_field}^2", self.text_field]}}
        bool_query: dict[str, Any] = {"must": match, "filter": [{"exists": {"field": self.text_field}}]}
        if self.min_citations:
            bool_query["filter"].append({"range": {"citationcount": {"gte": self.min_citations}}})
        return {"query": {"bool": bool_query}, "size": k}

    def search(self, query: str, k: int) -> list[Passage]:
        if k <= 0:
            return []
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"ApiKey {self.api_key}"
        body = json.dumps(self._query(query, k)).encode()
        try:
            raw = self.transport(f"{self.url}/{self.index}/_search", headers, body, self.timeout)
        except RemoteError as err:
            raise CorpusError(str(err)) from err
        try:
            hits = json.loads(raw)["hits"]["hits"]
            return [self._passage(hit) for hit in hits]
        except (ValueError, KeyError, TypeError) as err:
            raise CorpusError(f"unexpected response: {raw[:200]!r}") from err

    def _passage(self, hit: dict[str, Any]) -> Passage:
        source = hit["_source"]
        return Passage(
            id=str(source.get("corpusid", hit["_id"])),
            text=source[self.text_field],
            title=source.get(self.title_field) or "",
            url=source.get("url") or "",
        )
