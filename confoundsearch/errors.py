"""Exceptions shared across the package."""

from __future__ import annotations


class InvalidAction(ValueError):
    """An edit or action violates a graph invariant or the MDP's rules."""


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
