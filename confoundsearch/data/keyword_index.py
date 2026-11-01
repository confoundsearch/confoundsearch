"""Ask for the keywords near a name: `db.query("firm size")`.

A keyword index is a vector store holding OpenAlex's research keywords, one record each. The id is the
keyword's OpenAlex slug (`firm-size`); its name and paper count are the metadata. `KeywordIndex` takes words:

    db = KeywordIndex(PineconeVectorStore(key, "keywords"))
    db.query("firm size")       # a keyword: its stored vector is looked up, then its nearest keywords found
    db.query("Firm-Size")       # the same: spaces, hyphens and capitals do not matter
    db.query("how big a firm is", embedder=...)    # not a keyword: the phrase is embedded, then searched

A name that is a keyword needs no embedding model, so no model key. Anything else needs one (`embedder`).
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable, Sequence

from confoundsearch.data.vectors import Match, VectorRecord, VectorStore
from confoundsearch.errors import VectorStoreError

Embedder = Callable[[str], Sequence[float]]
DISTINCT_POOL = (
    100  # how many neighbours to ask for, at least, when the echoes of the query are to be dropped
)
MAX_POOL = 1000


def without_echoes(matches: Sequence[Match], phrase: str, k: int) -> list[Match]:
    """The first k matches whose names lack the phrase: 'firm size and growth' is an echo of 'firm size'."""
    wanted = phrase.casefold().strip()
    kept = [m for m in matches if wanted not in str(m.metadata.get("name", m.id)).casefold()]
    return kept[:k]


def pool_size(k: int) -> int:
    """How many neighbours to fetch so that k remain after the echoes are dropped."""
    return min(max(DISTINCT_POOL, 4 * k), MAX_POOL)


def slugify(name: str) -> str:
    """OpenAlex's keyword slug for a name: accents dropped, lower case, other characters become one hyphen."""
    folded = "".join(c for c in unicodedata.normalize("NFKD", name) if not unicodedata.combining(c))
    return re.sub(r"[\W_]+", "-", folded.casefold()).strip("-")


class KeywordIndex:
    def __init__(self, store: VectorStore, embedder: Embedder | None = None) -> None:
        self.store = store
        self.embedder = embedder

    def keyword(self, name: str) -> VectorRecord | None:
        """The keyword's stored record, or None: found by its slug, then checked against the name."""
        slug = slugify(name)
        if not slug:
            return None
        for record in self.store.fetch([slug]):
            # A slug can outlive a rename (OpenAlex kept 'female-dogs' when 'bitches' became the name), so
            # a record whose name is not the one asked for is some other keyword.
            if slugify(str(record.metadata.get("name", record.id))) == slug:
                return record
        return None

    def query(
        self,
        name: str,
        k: int = 25,
        min_works: int = 0,
        embedder: Embedder | None = None,
        distinct: bool = False,
    ) -> list[Match]:
        """The k keywords nearest the name, nearest first, not counting the keyword itself.

        With distinct, keywords whose names merely contain the name (the echoes of the query) are left out,
        so the k that remain are other ideas.
        """
        flt = {"works": {"$gte": min_works}} if min_works else None
        wanted = pool_size(k) if distinct else k
        record = self.keyword(name)
        if record is not None:
            found = [m for m in self.store.query(record.values, wanted + 1, flt) if m.id != record.id]
            phrase = str(record.metadata.get("name", name))
        else:
            embed = embedder or self.embedder
            if embed is None:
                raise VectorStoreError(
                    f"no keyword named {name!r}; to search by any phrase, give the index an embedder"
                )
            found, phrase = self.store.query(embed(name), wanted, flt), name
        return without_echoes(found, phrase, k) if distinct else found[:k]
