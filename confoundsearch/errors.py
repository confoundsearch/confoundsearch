"""Exceptions shared across the package."""

from __future__ import annotations


class InvalidAction(ValueError):
    """An edit or action violates a graph invariant or the MDP's rules."""


class BudgetReached(RuntimeError):
    """The money the user allowed for the language model has been spent; the run stops here."""


class ReplayMiss(RuntimeError):
    """A replay-only run needed an answer from the model that has not been saved.

    Not a RemoteError on purpose: the code that carries on when the model is unreachable would otherwise
    swallow it, and a replay that found nothing would look like a finished run.
    """


class RemoteError(RuntimeError):
    """A remote HTTP service failed or returned something unusable."""


class HttpError(RemoteError):
    """An HTTP request got an error status. `retry_after` is the server's "wait N seconds", if it said so."""

    def __init__(self, message: str, status: int, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.status, self.retry_after = status, retry_after


class LLMError(RemoteError):
    """An OpenAI-compatible endpoint failed or returned something unusable."""


class CorpusError(RemoteError):
    """The search backend failed or returned something unusable."""


class VectorStoreError(RemoteError):
    """A vector database failed, or was asked for something it cannot do."""
