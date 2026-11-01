"""One HTTP request (a POST with a body, or a GET without), using only the standard library."""

from __future__ import annotations

import http.client
import socket
import time
import urllib.error
import urllib.request
from typing import Callable, Mapping, Optional

from confoundsearch.errors import HttpError, RemoteError

Transport = Callable[[str, Mapping[str, str], Optional[bytes], float], bytes]
Log = Callable[..., None]  # log("http", url=..., status=...): where the wrapper below reports each request
RETRY_STATUSES = frozenset(
    {429, 500, 502, 503, 504}
)  # rate limited, or the server had a problem: worth another go


def _seconds(header: str | None) -> float | None:
    try:
        return float(header) if header is not None else None
    except ValueError:  # an HTTP date rather than a number of seconds
        return None


def urllib_transport(url: str, headers: Mapping[str, str], body: bytes | None, timeout: float) -> bytes:
    request = urllib.request.Request(
        url, data=body, headers=dict(headers), method="GET" if body is None else "POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            data: bytes = response.read()
            return data
    except urllib.error.HTTPError as err:
        detail = err.read().decode("utf-8", "replace")
        retry_after = _seconds(err.headers.get("Retry-After"))
        raise HttpError(f"HTTP {err.code} from {url}: {detail}", err.code, retry_after) from err
    except urllib.error.URLError as err:
        raise RemoteError(f"could not reach {url}: {err.reason}") from err
    except (
        socket.timeout,
        TimeoutError,
    ) as err:  # a server that accepts the request and never finishes answering
        raise RemoteError(f"timed out waiting for {url}") from err
    except (OSError, http.client.HTTPException) as err:  # a connection that drops half way
        raise RemoteError(f"the connection to {url} failed: {type(err).__name__}: {err}") from err


def resilient(
    transport: Transport,
    log: Log | None = None,
    min_interval: float = 0.0,
    attempts: int = 5,
    backoff: float = 2.0,
    max_wait: float = 60.0,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> Transport:
    """`transport`, paced to one request per `min_interval` seconds, and trying again when told to wait.

    A 429 or a 5xx is retried, up to `attempts` requests in all, after the server's own Retry-After or else
    `backoff` seconds doubling each time (never more than `max_wait`). Any other failure is raised at once.
    Every request is reported to `log`: its status, how long it took, and how long it waits before the next.
    """
    started: list[float] = []  # when the last request began

    def send(url: str, headers: Mapping[str, str], body: bytes | None, timeout: float) -> bytes:
        attempt = 0
        while True:
            attempt += 1
            if started and min_interval - (clock() - started[-1]) > 0:
                sleep(min_interval - (clock() - started[-1]))
            started.append(clock())
            if log:
                log("request", url=url, attempt=attempt)  # so a hang shows as a request with no answer
            try:
                data = transport(url, headers, body, timeout)
            except HttpError as err:
                retry = err.status in RETRY_STATUSES and attempt < attempts
                wait = min(
                    max_wait, err.retry_after if err.retry_after is not None else backoff * 2 ** (attempt - 1)
                )
                if log:
                    log(
                        "http",
                        url=url,
                        attempt=attempt,
                        status=err.status,
                        seconds=round(clock() - started[-1], 3),
                        retry_after=err.retry_after,
                        retrying_in=wait if retry else None,
                        error=str(err)[:300],
                    )
                if not retry:
                    raise
                sleep(wait)
                continue
            if log:
                log("http", url=url, attempt=attempt, status=200, seconds=round(clock() - started[-1], 3))
            return data

    return send
