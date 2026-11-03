"""Wire a KeywordExpander to a language model and the keyword service.

    service = KeywordService(address, embedder=OpenAIEmbedder())
    expander = build_expander(client, service, rounds=5, min_works=0)
    expander.expand("firm size").texts()

Wiring is this layer's job: the logic knows neither which model nor which database is behind its two steps.
"""

from __future__ import annotations

from confoundsearch.data.keyword_service import KeywordService
from confoundsearch.logic.terms import KeywordExpander
from confoundsearch.model.expansion import FIELDS, LLMTermSynonyms, LLMWideNet
from confoundsearch.model.llm import ChatClient


def build_expander(
    client: ChatClient,
    service: KeywordService,
    rounds: int = 5,
    per_round: int = 12,
    context: str = "",
    min_works: int = 0,
    per_term: int = 100,
    limit: int = 1000,
    workers: int = 8,
    fields: tuple[str, ...] = FIELDS,
    per_field: int = 8,
    per_variant: int = 4,
    max_seeds: int = 150,
    per_hypernym: int = 3,
    per_mechanism: int = 6,
    per_attribute: int = 8,
    neighbor_top: int = 100,
    per_neighbor: int = 2,
) -> KeywordExpander:
    """The grand query expansion: a KeywordExpander that casts a wide net with `client` and
    fetches neighbours from `service`.

    The model is asked the five ways, then once per field in `fields` (what that field calls the idea or uses
    as a stand-in for it), then for synonyms of those field terms. Empty `fields` or a zero `per_field` skips
    the field steps, and a zero `per_variant` skips their synonyms. `per_hypernym` asks for broader terms (0:
    none) and `per_mechanism` for the channels it works through and `per_attribute` for the
    characteristics papers measure (0: none).
    Of the `neighbor_top` best keywords found, the model is asked for `per_neighbor` other names each
    (either zero: none). It first asks the service how many
    neighbours one answer may hold, and never asks for more.
    """
    service.learn_limits()
    net = LLMWideNet(
        client,
        rounds,
        per_round,
        fields,
        per_field,
        per_variant,
        context,
        per_hypernym=per_hypernym,
        per_mechanism=per_mechanism,
        per_attribute=per_attribute,
    )
    hop = LLMTermSynonyms(client, per_neighbor) if per_neighbor > 0 else None
    return KeywordExpander(
        net,
        lambda term, k: service.query(term, k, min_works),
        min(per_term, service.max_k),
        limit,
        workers,
        max_seeds,
        hop,
        neighbor_top,
    )
