"""An audit trail: what the agent searched for, found, and decided about each piece of evidence.

Every run keeps a log. It records, in order,

    run_started    the question, the step budget and the policy
    model_call     each prompt sent to the AI model and its reply (when the client is an AuditedClient)
    action         each move the agent chose, its reward, and the score before and after
    retrieved      each search, with the full text of every passage it brought in
    judged         each citation: the claim, the passage, and whether the judge accepted or rejected it
    run_finished   why the run ended, the final score, and which passages back each confound

The log is append-only and tamper-evident: every event carries a SHA-256 hash of its content chained to the
previous event's hash, so editing, reordering or deleting an event in the middle is detectable (`verify`).
A log with no `run_finished` event is incomplete, which is how a truncated or crashed run shows up.

`AuditLog` is the record. `AuditTrail` reads one back: `trajectory.audit_trail()` for a run you just made,
`AuditTrail.load(path)` for a saved file, and `.supporting(confound, target)` answers "which passages back
this link, and where did they come from?".
"""

from __future__ import annotations

import contextvars
import hashlib
import json
import threading
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any

from confoundsearch.logic.actions import Action, AttachPassage, Retrieve
from confoundsearch.logic.graph import TARGETS, Graph, Passage
from confoundsearch.logic.reward import phi

if TYPE_CHECKING:
    from confoundsearch.logic.episode import Step
    from confoundsearch.model.llm import ChatClient

GENESIS = "0" * 64
FINAL_KINDS = ("run_finished", "run_failed")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


@dataclass(frozen=True)
class Event:
    seq: int
    step: int
    time: str
    kind: str
    data: dict[str, Any]
    hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "step": self.step,
            "time": self.time,
            "kind": self.kind,
            "data": self.data,
            "hash": self.hash,
        }


def _digest(previous: str, payload: Mapping[str, Any]) -> str:
    body = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256((previous + body).encode("utf-8")).hexdigest()


def verify(events: Iterable[Event | Mapping[str, Any]]) -> bool:
    """True if the events form an unbroken chain from the first one: nothing edited, reordered or removed."""
    previous = GENESIS
    for expected, event in enumerate(events):
        record = event.to_dict() if isinstance(event, Event) else dict(event)
        claimed = record.pop("hash", None)
        if record.get("seq") != expected or claimed != _digest(previous, record):
            return False
        previous = claimed
    return True


class AuditLog:
    """The append-only record of a run. Pass `path` to also stream every event to a file as it happens."""

    def __init__(self, path: str | Path | None = None, clock: Callable[[], str] = utc_now) -> None:
        self.path = Path(path) if path is not None else None
        self.step = 0
        self._clock = clock
        self._events: list[Event] = []
        self._lock = threading.Lock()
        self._sink: IO[str] | None = open(self.path, "w", encoding="utf-8") if self.path else None  # noqa: SIM115

    @property
    def events(self) -> tuple[Event, ...]:
        with self._lock:
            return tuple(self._events)

    @property
    def head(self) -> str:
        """The hash of the latest event. Keep it somewhere safe to be able to detect a log cut short."""
        with self._lock:
            return self._events[-1].hash if self._events else GENESIS

    def record(self, kind: str, **data: Any) -> Event:
        with self._lock:
            previous = self._events[-1].hash if self._events else GENESIS
            seq, stamp = len(self._events), self._clock()
            payload = {"seq": seq, "step": self.step, "time": stamp, "kind": kind, "data": data}
            event = Event(seq, self.step, stamp, kind, data, _digest(previous, payload))
            self._events.append(event)
            if self._sink is not None:
                self._sink.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")
                self._sink.flush()
            return event

    def to_jsonl(self) -> str:
        return "".join(json.dumps(e.to_dict(), ensure_ascii=False) + "\n" for e in self.events)

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.write_text(self.to_jsonl(), encoding="utf-8")
        return target

    def close(self) -> None:
        if self._sink is not None:
            self._sink.close()
            self._sink = None

    def __enter__(self) -> AuditLog:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


# -- recording during a run ----------------------------------------------------

_CURRENT: contextvars.ContextVar[AuditLog | None] = contextvars.ContextVar(
    "confoundsearch_audit", default=None
)


def current() -> AuditLog | None:
    """The log of the episode running in this thread, if any."""
    return _CURRENT.get()


_REPLAYED: contextvars.ContextVar[bool] = contextvars.ContextVar("confoundsearch_replayed", default=False)


def mark_replayed() -> None:
    """Say that the model call now being answered was answered from a saved reply, not by the model."""
    _REPLAYED.set(True)


@contextmanager
def recording(log: AuditLog) -> Iterator[AuditLog]:
    """Make `log` the current log, so any AuditedClient used inside records to it."""
    token = _CURRENT.set(log)
    try:
        yield log
    finally:
        _CURRENT.reset(token)


class AuditedClient:
    """Wraps a chat client and records every prompt and reply in the current run's log.

    Use it in place of the client: `AuditedClient(OpenAICompatibleClient("gpt-4.1"))`. Inside `run_episode`
    the calls go to that run's log. Outside a run, or with no `log`, it simply passes calls through.
    """

    def __init__(self, client: ChatClient, log: AuditLog | None = None) -> None:
        self.client = client
        self.log = log

    def chat(self, messages: Sequence[Mapping[str, str]]) -> str:
        log = self.log or current()
        model = getattr(self.client, "model", None)
        sent = [dict(m) for m in messages]
        _REPLAYED.set(False)  # a wrapped client sets it again when it answers from a saved reply
        try:
            reply = self.client.chat(messages)
        except Exception as err:
            if log is not None:
                log.record("model_error", model=model, messages=sent, error=f"{type(err).__name__}: {err}")
            raise
        if log is not None:
            # a replayed call says so: it is an earlier answer, not a fresh judgement by the model
            extra = {"replayed": True} if _REPLAYED.get() else {}
            log.record("model_call", model=model, messages=sent, reply=reply, **extra)
        return reply


def action_dict(action: Action) -> dict[str, Any]:
    values = {f.name: getattr(action, f.name) for f in fields(action)}
    return {
        "type": type(action).__name__,
        **{k: sorted(v) if isinstance(v, frozenset) else v for k, v in values.items()},
    }


def passage_dict(passage: Passage) -> dict[str, str]:
    return {"id": passage.id, "title": passage.title, "url": passage.url, "text": passage.text}


def graph_summary(graph: Graph) -> dict[str, Any]:
    """Each confound, its phrasings, and the ids of the passages that back each of its links."""
    return {
        "confounds": [
            {
                "label": v.label,
                "strings": sorted(v.strings),
                "connected": graph.is_connected(v.label),
                "supporting_passages": {t: sorted(p.id for p in graph.passages(v.label, t)) for t in TARGETS},
            }
            for v in graph.confounds
        ]
    }


def record_start(log: AuditLog, x: str, y: str, max_steps: int, policy: object) -> None:
    from confoundsearch import __version__

    settings = {
        k: getattr(policy, k)
        for k in (
            "epsilon",
            "temperature",
            "explore_stop",
            "max_confounds",
            "synonyms_per_term",
            "queries_per_side",
            "results_per_query",
            "attach_per_side",
            "mine_count",
            "mine_queries",
            "mine_depth",
        )
        if hasattr(policy, k)
    }
    log.record(
        "run_started",
        independent_variable=x,
        dependent_variable=y,
        max_steps=max_steps,
        policy=type(policy).__name__,
        policy_settings=settings,
        package_version=__version__,
    )


def record_step(log: AuditLog, step: Step) -> None:
    """Log one step: the move and its effect on the score, then the evidence it collected or judged."""
    before, after, action = step.state, step.next_state, step.action
    log.record(
        "action",
        action=action_dict(action),
        reward=step.reward,
        score_before=phi(before.graph),
        score_after=phi(after.graph),
    )
    if isinstance(action, Retrieve):
        new = sorted(after.observed - before.observed, key=lambda p: p.id)
        log.record(
            "retrieved",
            query=action.query,
            requested=action.k,
            new_passages=[passage_dict(p) for p in new],
            passages_seen_so_far=len(after.observed),
        )
    elif isinstance(action, AttachPassage):
        passage = {p.id: p for p in before.observed}[action.passage_id]
        accepted = passage in after.graph.passages(action.confound, action.target)
        log.record(
            "judged",
            claim=f"{action.confound} affects {before.graph.target_name(action.target)}",
            confound=action.confound,
            target=action.target,
            verdict="supported" if accepted else "rejected",
            passage=passage_dict(passage),
        )


def record_finish(log: AuditLog, reason: str, steps: int, final: Any) -> None:
    log.record(
        "run_finished",
        reason=reason,
        steps=steps,
        final_score=phi(final.graph),
        passages_observed=len(final.observed),
        passages_rejected=len(final.rejected),
        graph=graph_summary(final.graph),
    )


# -- reading a log back --------------------------------------------------------


@dataclass(frozen=True)
class EvidenceRow:
    """One passage that was judged for one claim, with where it came from."""

    confound: str
    target: str
    passage_id: str
    title: str
    url: str
    text: str
    verdict: str
    retrieved_by: str
    retrieved_at_step: int
    judged_at_step: int


@dataclass(frozen=True)
class AuditTrail:
    """A run's log, read back.

    Get one from `trajectory.audit_trail()`, `AuditTrail.from_log(log)`, or `AuditTrail.load(path)`.
    """

    events: tuple[Event, ...]

    @classmethod
    def from_log(cls, log: AuditLog) -> AuditTrail:
        return cls(log.events)

    @classmethod
    def load(cls, path: str | Path) -> AuditTrail:
        lines = [line for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
        return cls(tuple(Event(**json.loads(line)) for line in lines))

    @property
    def intact(self) -> bool:
        """True if the hash chain is unbroken, so no event was edited, reordered or removed."""
        return verify(self.events)

    @property
    def complete(self) -> bool:
        """True if the run ended and said so. A log that stops early was cut short or crashed."""
        return bool(self.events) and self.events[-1].kind in FINAL_KINDS

    @property
    def head(self) -> str:
        """The hash of the last event. Keep it elsewhere to be able to show later that nothing was cut off."""
        return self.events[-1].hash if self.events else GENESIS

    def to_dict(self) -> dict[str, Any]:
        """A machine-readable summary: the question, how it ended, whether the log is sound, the evidence."""
        started, finished = (
            (self.of_kind("run_started") or [None])[0],
            (self.of_kind("run_finished") or [None])[-1],
        )
        return {
            "question": {
                "independent_variable": started.data["independent_variable"] if started else None,
                "dependent_variable": started.data["dependent_variable"] if started else None,
            },
            "ended": finished.data["reason"] if finished else None,
            "steps": finished.data["steps"] if finished else None,
            "final_score": finished.data["final_score"] if finished else None,
            "events": len(self.events),
            "intact": self.intact,
            "complete": self.complete,
            "head": self.head,
            "model_calls": len(self.of_kind("model_call")),
            "models": sorted(
                {str(e.data["model"]) for e in self.of_kind("model_call") if e.data.get("model")}
            ),
            "evidence": [asdict(row) for row in self.evidence],
        }

    def of_kind(self, kind: str) -> tuple[Event, ...]:
        return tuple(e for e in self.events if e.kind == kind)

    @property
    def evidence(self) -> tuple[EvidenceRow, ...]:
        found: dict[str, tuple[str, int]] = {}
        for event in self.of_kind("retrieved"):
            for passage in event.data["new_passages"]:
                found.setdefault(passage["id"], (event.data["query"], event.step))
        rows = []
        for event in self.of_kind("judged"):
            passage = event.data["passage"]
            query, at = found.get(passage["id"], ("", 0))
            rows.append(
                EvidenceRow(
                    confound=event.data["confound"],
                    target=event.data["target"],
                    passage_id=passage["id"],
                    title=passage["title"],
                    url=passage["url"],
                    text=passage["text"],
                    verdict=event.data["verdict"],
                    retrieved_by=query,
                    retrieved_at_step=at,
                    judged_at_step=event.step,
                )
            )
        return tuple(rows)

    def supporting(self, confound: str, target: str | None = None) -> tuple[EvidenceRow, ...]:
        """The passages the judge accepted for `confound` (for one target, "X" or "Y", or both)."""
        rows = (r for r in self.evidence if r.confound == confound and r.verdict == "supported")
        return tuple(r for r in rows if target in (None, r.target))

    def to_markdown(self) -> str:
        started, finished = (
            (self.of_kind("run_started") or [None])[0],
            (self.of_kind("run_finished") or [None])[-1],
        )
        lines = ["# Audit trail", ""]
        if started is not None:
            d = started.data
            lines.append(
                f"Question: does **{d['independent_variable']}** affect **{d['dependent_variable']}**?  "
            )
        if finished is not None:
            lines.append(
                f"Ended after {finished.data['steps']} steps ({finished.data['reason'].replace('_', ' ')}); "
                f"final score {finished.data['final_score']:.3f}.  "
            )
        lines += [
            f"{len(self.events)} events; hash chain {'intact' if self.intact else 'BROKEN'}; "
            f"{'complete' if self.complete else 'INCOMPLETE (no end-of-run event)'}.",
        ]
        calls = self.of_kind("model_call")
        replayed = sum(1 for e in calls if e.data.get("replayed"))
        if replayed:
            lines.append(
                f"{replayed} of {len(calls)} model calls were answered from saved replies, not asked afresh."
            )
        lines.append("")
        rows = self.evidence
        for label in dict.fromkeys(r.confound for r in rows):
            lines += [f"## {label}", ""]
            for target in TARGETS:
                mine = [r for r in rows if r.confound == label and r.target == target]
                if not mine:
                    continue
                name = (
                    (
                        started.data["independent_variable"]
                        if target == "X"
                        else started.data["dependent_variable"]
                    )
                    if started
                    else target
                )
                ok = [r for r in mine if r.verdict == "supported"]
                lines.append(f"**{label} → {name}**: {len(ok)} supported, {len(mine) - len(ok)} rejected")
                for r in ok:
                    link = f" <{r.url}>" if r.url else ""
                    lines.append(
                        f"- {r.title or r.passage_id} (`{r.passage_id}`){link}: "
                        f'found by "{r.retrieved_by}" at step {r.retrieved_at_step}, '
                        f"accepted at step {r.judged_at_step}"
                    )
                lines.append("")
        return "\n".join(lines).rstrip() + "\n"
