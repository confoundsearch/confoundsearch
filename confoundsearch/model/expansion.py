"""Many angles on one construct: ask the model for the terms papers use, several different ways.

    variants = LLMTermVariants(client, rounds=5)
    variants("firm size")        # ['company size', 'number of employees', 'economies of scale', ...]

One question gets one kind of answer, so each round asks for something else: other names for it, how it is
measured, the ideas studied beside it, the words other fields use for it, and phrases that appear in titles.
The aim is a long list, not a clean one: filtering comes later. A round that fails is skipped; only if
every round fails is that an error.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from confoundsearch.errors import LLMError
from confoundsearch.logic.graph import normalize
from confoundsearch.model.llm import _FENCE, ChatClient, _strings

ANGLES = (
    ("synonyms", "synonyms and other names for the same idea, including abbreviations and British spellings"),
    ("measures", "narrower terms: the specific ways it is measured or operationalized in empirical studies"),
    ("related", "closely related concepts that studies discuss alongside it"),
    (
        "fields",
        "the words that other fields use for the same or a similar idea (economics, finance, "
        "management, sociology, psychology)",
    ),
    ("phrases", "phrases that appear in the titles and abstracts of empirical papers about it"),
)
MAX_TERM_LENGTH = 80  # characters: a "term" longer than this is a sentence


class LLMTermVariants:
    def __init__(self, client: ChatClient, rounds: int = 5, per_round: int = 12, context: str = "") -> None:
        self.client, self.rounds, self.per_round, self.context = client, rounds, per_round, context.strip()

    def _prompt(self, construct: str, what: str) -> str:
        setting = f" {self.context}" if self.context else ""
        return (
            f'In social science research, "{construct}" is an idea that papers name in many ways.{setting} '
            f"Give up to {self.per_round} {what}. Each is 1 to 4 words, as a paper would write it. "
            'Reply with JSON only: {"terms": ["..."]}'
        )

    def __call__(self, construct: str) -> list[str]:
        found: list[str] = []
        last_error: LLMError | None = None
        for number in range(max(self.rounds, 0)):
            _, what = ANGLES[number % len(ANGLES)]  # past five rounds the angles repeat; answers still differ
            try:
                found += _strings(
                    self.client.chat([{"role": "user", "content": self._prompt(construct, what)}]), "terms"
                )
            except LLMError as err:
                last_error = err
        if not found and last_error is not None:
            raise last_error
        seen = {normalize(construct)}
        terms = []
        for item in found:
            term = " ".join(item.split())
            key = normalize(term)
            if term and len(term) <= MAX_TERM_LENGTH and key and key not in seen:
                seen.add(key)
                terms.append(term)
        return terms


class LLMMeasures:
    """The specific measures, forms or outcomes that papers name when they study a variable.

    Not synonyms of its label: for "firm performance", terms like "return on equity" or "labor productivity".
    """

    def __init__(self, client: ChatClient, count: int = 8) -> None:
        self.client, self.count = client, count

    def __call__(self, term: str) -> list[str]:
        prompt = (
            f'In empirical social science papers, "{term}" is studied through specific measures, forms or '
            "outcomes that papers name directly (not synonyms of the label). "
            f"List {self.count}, 1 to 4 words each. "
            'Reply with JSON only: {"terms": ["..."]}'
        )
        return clean(
            _strings(self.client.chat([{"role": "user", "content": prompt}]), "terms"), {normalize(term)}
        )


class LLMMechanisms:
    """The specific channels through which a construct could act (on a variable, when one is given).

    For "CEO style" acting on "organizational structure": "delegation of authority", "incentive design".
    Papers that study an effect usually name the channel, not the construct's label.
    """

    def __init__(self, client: ChatClient, count: int = 6) -> None:
        self.client, self.count = client, count

    def __call__(self, construct: str, target: str | None = None) -> list[str]:
        on = f' affect "{target}"' if target else " work"
        prompt = (
            f'Through what specific channels or mechanisms could "{construct}"{on}? '
            f"Name {self.count} concrete mechanisms, 1 to 4 words each, as empirical papers would name them. "
            'Reply with JSON only: {"terms": ["..."]}'
        )
        return clean(
            _strings(self.client.chat([{"role": "user", "content": prompt}]), "terms"), {normalize(construct)}
        )


FIELDS = (
    "economics",
    "finance",
    "accounting",
    "management and organization studies",
    "sociology",
    "political science",
    "psychology",
    "marketing",
    "public administration and policy",
)
EXAMPLE = (
    'For example, in finance "free cash flow" can stand in for "firm size", and in accounting "total assets" '
    'can; in labour economics it is "number of employees". The same idea goes by very different names.'
)


def clean(items: list[str], skip: set[str]) -> list[str]:
    """Tidy spacing, drop the empty, the too long and the repeated (`skip` holds what is already taken)."""
    kept = []
    for item in items:
        term = " ".join(item.split())
        key = normalize(term)
        if term and len(term) <= MAX_TERM_LENGTH and key and key not in skip:
            skip.add(key)
            kept.append(term)
    return kept


def in_parallel(calls: list[Callable[[], Any]], workers: int) -> list[Any]:
    """Run the functions side by side, in order. One that raises an LLMError gives that error instead."""

    def attempt(call: Callable[[], Any]) -> Any:
        try:
            return call()
        except LLMError as err:
            return err

    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(calls) or 1))) as pool:
        return list(pool.map(attempt, calls))


class LLMFieldVariants:
    """What each social science field calls the same idea, or uses as a stand-in (a proxy) for it.

    One question per field, each starting from the same example (free cash flow in finance for firm size) to
    show how far apart the names can be, and asking for that kind of reach. The answers are not synonyms in
    the dictionary sense, which is the point: they are what a paper in that field might say instead.
    """

    def __init__(
        self,
        client: ChatClient,
        fields: tuple[str, ...] = FIELDS,
        per_field: int = 8,
        context: str = "",
        workers: int = 4,
    ) -> None:
        self.client, self.fields, self.per_field = client, tuple(fields), per_field
        self.context, self.workers = context.strip(), workers

    def _prompt(self, construct: str, field: str) -> str:
        setting = f" {self.context}" if self.context else ""
        return (
            f'{EXAMPLE}{setting} Now take the idea "{construct}". Give up to {self.per_field} terms that '
            f"papers in {field} use for it, or use as a measure, proxy or close stand-in for it, even when "
            f"the word is not an obvious synonym. Be creative: think of how a {field} researcher who never "
            f'writes "{construct}" would refer to it. Each is 1 to 4 words, as a paper would write it. '
            'Reply with JSON only: {"terms": ["..."]}'
        )

    def __call__(self, construct: str) -> list[str]:
        def ask(field: str):  # type: ignore[no-untyped-def]
            return lambda: _strings(
                self.client.chat([{"role": "user", "content": self._prompt(construct, field)}]), "terms"
            )

        answers = in_parallel([ask(field) for field in self.fields], self.workers)
        errors = [a for a in answers if isinstance(a, LLMError)]
        if errors and len(errors) == len(answers):
            raise errors[-1]
        found = [term for a in answers if not isinstance(a, LLMError) for term in a]
        return clean(found, {normalize(construct)})


class _PerTerm:
    """Ask about a list of terms in batches: for each term, a few related terms (the shape of the answer)."""

    key = ""  # the JSON key of the reply
    what = ""  # what to give for each term

    def __init__(self, client: ChatClient, per_term: int = 4, batch: int = 10, workers: int = 4) -> None:
        self.client, self.per_term, self.batch, self.workers = client, per_term, max(batch, 1), workers

    def _prompt(self, terms: list[str]) -> str:
        listing = "\n".join(f"- {t}" for t in terms)
        return (
            f"For each term below, as used in social science research, give up to {self.per_term} "
            f"{self.what} (1 to 4 words each).\n"
            f"{listing}\n"
            f'Reply with JSON only: {{"{self.key}": {{"<term>": ["..."]}}}}, '
            "using each term exactly as written."
        )

    def _parse(self, reply: str) -> list[str]:
        try:
            mapping = json.loads(_FENCE.sub("", reply.strip())).get(self.key, {})
        except (ValueError, AttributeError, TypeError):
            return []
        if not isinstance(mapping, dict):
            return []
        return [
            x for value in mapping.values() if isinstance(value, list) for x in value if isinstance(x, str)
        ]

    def __call__(self, terms: list[str]) -> list[str]:
        chunks = [terms[i : i + self.batch] for i in range(0, len(terms), self.batch)]

        def ask(chunk: list[str]) -> Callable[[], list[str]]:
            prompt = self._prompt(chunk)
            return lambda: self._parse(self.client.chat([{"role": "user", "content": prompt}]))

        answers = in_parallel([ask(chunk) for chunk in chunks], self.workers)
        errors = [a for a in answers if isinstance(a, LLMError)]
        if errors and len(errors) == len(answers):
            raise errors[-1]
        found = [term for a in answers if not isinstance(a, LLMError) for term in a]
        return clean(found, {normalize(t) for t in terms})


class LLMTermSynonyms(_PerTerm):
    """Other names for each of a list of terms (for the field variants, and for the nearest keywords)."""

    key = "synonyms"
    what = "synonyms or other names that papers use for it"


class LLMHypernyms(_PerTerm):
    """Broader terms for each of a list of terms: the category it belongs to, the wider idea around it."""

    key = "hypernyms"
    what = "broader terms: the more general idea or category that papers file it under"


class LLMWideNet:
    """The whole net as one step of a `KeywordExpander`: angles, fields, broader terms, then synonyms.

    1. `LLMTermVariants`: other names, measures, related ideas, other fields' words, title phrases.
    2. `LLMFieldVariants`: what each social science field calls it or uses as a stand-in.
    3. `LLMHypernyms`: broader terms for the construct and for each of those field variants.
    4. `LLMTermSynonyms`: other names for each of those field variants.
    5. `LLMMechanisms`: the specific channels the construct works through (papers name these, not the label).

    It returns (term, kind) pairs, with kinds "synonym", "variant", "hypernym", "variant synonym" and
    "mechanism", so that
    where every term came from survives into the result. A stage that fails is skipped and noted in
    `problems`; if all fail, that is an error.
    """

    def __init__(
        self,
        client: ChatClient,
        rounds: int = 5,
        per_round: int = 12,
        fields: tuple[str, ...] = FIELDS,
        per_field: int = 8,
        per_variant: int = 4,
        context: str = "",
        workers: int = 4,
        per_hypernym: int = 3,
        per_mechanism: int = 6,
    ) -> None:
        self.angles = LLMTermVariants(client, rounds, per_round, context)
        self.variants = LLMFieldVariants(client, fields, per_field, context, workers)
        self.synonyms = LLMTermSynonyms(client, per_variant, workers=workers)
        self.hypernyms = LLMHypernyms(client, per_hypernym, workers=workers)
        self.use_fields, self.use_synonyms = bool(fields) and per_field > 0, per_variant > 0
        self.use_hypernyms = per_hypernym > 0
        self.mechanisms = LLMMechanisms(client, per_mechanism)
        self.use_mechanisms = per_mechanism > 0
        self.problems: list[str] = []

    def __call__(self, construct: str) -> list[tuple[str, str]]:
        self.problems = []
        taken = {normalize(construct)}
        pairs: list[tuple[str, str]] = []

        def stage(name: str, kind: str, call: Callable[[], list[str]]) -> list[str]:
            try:
                found = call()
            except LLMError as err:
                self.problems.append(f"{name}: {err}")
                return []
            fresh = clean(found, taken)
            pairs.extend((term, kind) for term in fresh)
            return fresh

        stage("angles", "synonym", lambda: self.angles(construct))
        variants = (
            stage("field variants", "variant", lambda: self.variants(construct)) if self.use_fields else []
        )
        if self.use_hypernyms:
            stage("hypernyms", "hypernym", lambda: self.hypernyms([construct, *variants]))
        if variants and self.use_synonyms:
            stage("synonyms of the variants", "variant synonym", lambda: self.synonyms(variants))
        if self.use_mechanisms:
            stage("mechanisms", "mechanism", lambda: self.mechanisms(construct))
        if not pairs and self.problems:
            raise LLMError("; ".join(self.problems))
        return pairs
