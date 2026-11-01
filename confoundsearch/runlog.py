"""A log of what a run did as it did it: one JSON line per event, flushed at once so a crash loses nothing.

It is for finding out why a run was slow or stopped: each search request and its status, each step, the
error. It is not evidence (the audit trail, `confoundsearch.logic.audit`, is). Keys are never written to it.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class RunLog:
    def __init__(
        self,
        path: Path | None = None,
        clock: Callable[[], float] = time.monotonic,
        keep: bool = False,
        echo: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.path, self.clock, self.began, self.keep, self.echo = path, clock, clock(), keep, echo
        self.events: list[dict[str, Any]] = []
        self._file = None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._file = path.open("a", encoding="utf-8")

    def event(self, kind: str, **data: Any) -> None:
        """Record `kind` with `data`, the time, and the seconds since the log began."""
        record = {
            "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "t": round(self.clock() - self.began, 3),
            "event": kind,
            **data,
        }
        if self.echo is not None:  # e.g. the status line on the terminal
            self.echo(record)
        if self.keep:
            self.events.append(record)
        if self._file is not None:
            self._file.write(json.dumps(record, default=str) + "\n")
            self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
