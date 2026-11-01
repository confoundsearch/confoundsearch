"""What the language model has cost so far, estimated from the tokens it reported.

The number is an estimate: prices change and a gateway may bill differently. It is meant for a spending cap
(`--max-budget`), not for accounts. When the provider reports its own `cost` in the usage it sends back
(OpenRouter does), that is used instead.
"""

from __future__ import annotations

from typing import Any

# dollars per million tokens (input, output); the longest matching name prefix wins
PRICES: dict[str, tuple[float, float]] = {
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
}
UNKNOWN_MODEL = (
    10.0,
    30.0,
)  # a model not listed is assumed to be an expensive one, so a cap errs on the safe side


def price_of(model: str) -> tuple[float, float]:
    name = model.rsplit("/", 1)[-1].lower()  # "openai/gpt-4.1" on OpenRouter is "gpt-4.1" here
    known = [prefix for prefix in PRICES if name.startswith(prefix)]
    return PRICES[max(known, key=len)] if known else UNKNOWN_MODEL


class Spend:
    def __init__(self, model: str) -> None:
        self.model, self.tokens_in, self.tokens_out, self.reported = model, 0, 0, 0.0

    def add(self, usage: Any) -> None:
        """Count one reply's `usage` block; anything that is not the expected shape counts as nothing."""
        if not isinstance(usage, dict):
            return
        for field, name in (("tokens_in", "prompt_tokens"), ("tokens_out", "completion_tokens")):
            value = usage.get(name)
            if isinstance(value, (int, float)):
                setattr(self, field, getattr(self, field) + int(value))
        cost = usage.get("cost")
        if isinstance(cost, (int, float)):
            self.reported += float(cost)

    @property
    def dollars(self) -> float:
        if self.reported:
            return self.reported
        price_in, price_out = price_of(self.model)
        return (self.tokens_in * price_in + self.tokens_out * price_out) / 1_000_000
