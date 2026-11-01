"""HTTP backend for the ConfoundSearch interface. Standard library only.

    GET  /                              the interface (ui/app.html)
    POST /api/runs                      start a run: {"iv", "dv", "max_steps", "epsilon", "seed"}
    GET  /api/runs                      list runs
    GET  /api/runs/<id>                 status, reward so far, recent steps
    POST /api/runs/<id>/stop            stop a run early
    GET  /api/runs/<id>/share_data.json the graph in the interface's data format
    GET  /api/runs/<id>/export.zip      the same, zipped like the Gradio app's exports (plus audit.jsonl)
    GET  /api/runs/<id>/audit.jsonl     the run's audit trail: every search, passage and verdict, hash-chained
    GET  /api/runs/<id>/audit.md        the same, as a readable report
    GET  /runs/<id>/standalone.html     the standalone page (add ?download=1 to save it)
    GET  /api/health

Each run is one episode of the MDP in a background thread. After every step the
run's graph is re-exported, so the page can show it growing.
"""

from __future__ import annotations

import dataclasses
import json
import random
import re
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, urlsplit

from confoundsearch.interface.export import export_zip, to_share_data
from confoundsearch.interface.ui import standalone_html
from confoundsearch.logic.actions import Action
from confoundsearch.logic.audit import AuditLog, AuditTrail
from confoundsearch.logic.episode import ActionSource, Step, run_episode
from confoundsearch.logic.policy import EpsilonGreedy, LookaheadQ, Policy
from confoundsearch.logic.transition import Environment

MAX_BODY = 8192
MAX_STEPS = 1000
SANDBOX = "sandbox allow-scripts allow-popups"  # the exported page runs in an opaque origin


class Busy(RuntimeError):
    """Too many runs are already in flight."""


def describe(action: Action) -> str:
    fields = []
    for f in dataclasses.fields(action):
        value = getattr(action, f.name)
        fields.append(f"{f.name}={sorted(value) if isinstance(value, frozenset) else value}")
    return f"{type(action).__name__}({', '.join(fields)})"


class Run:
    def __init__(self, iv: str, dv: str, max_steps: int, epsilon: float, seed: int | None) -> None:
        self.id = uuid.uuid4().hex[:12]
        self.iv, self.dv = iv, dv
        self.max_steps, self.epsilon, self.seed = max_steps, epsilon, seed
        self.status = "running"
        self.steps: list[dict[str, Any]] = []
        self.phi = 0.0
        self.share: dict[str, Any] = {}
        self.error: str | None = None
        self.stop = threading.Event()
        self.audit = AuditLog()
        self.started = time.time()

    def summary(self, recent: int = 100) -> dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status,
            "iv": self.iv,
            "dv": self.dv,
            "step": len(self.steps),
            "max_steps": self.max_steps,
            "phi": self.phi,
            "audit_events": len(self.audit.events),
            "steps": self.steps[-recent:] if recent else [],
            "error": self.error,
        }


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not 1 <= len(value.strip()) <= 200 or re.search(r"[<>\x00-\x1f]", value):
        raise ValueError(f"{name} must be 1-200 characters of plain text")
    return value.strip()


class RunManager:
    def __init__(
        self,
        env: Environment,
        proposer: ActionSource,
        max_concurrent: int = 2,
        keep: int = 50,
        audit_dir: Path | None = None,
        policy_factory: Callable[[Run], Policy] | None = None,
        default_steps: int = 40,
    ) -> None:
        """`policy_factory(run)` makes each run's policy; without it every run is epsilon-greedy."""
        self.env, self.proposer = env, proposer
        self.policy_factory, self.default_steps = policy_factory, default_steps
        self.max_concurrent, self.keep = max_concurrent, keep
        self.audit_dir = audit_dir  # when set, every run's audit trail is also written here as it happens
        self.runs: dict[str, Run] = {}
        self._lock = threading.Lock()

    def start(self, params: dict[str, Any]) -> Run:
        iv, dv = _text(params.get("iv"), "iv"), _text(params.get("dv"), "dv")
        max_steps = params.get("max_steps")
        if max_steps is None:
            max_steps = self.default_steps
        epsilon, seed = params.get("epsilon", 0.1), params.get("seed")
        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or not 1 <= max_steps <= MAX_STEPS:
            raise ValueError(f"max_steps must be an integer from 1 to {MAX_STEPS}")
        if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)) or not 0 <= epsilon <= 1:
            raise ValueError("epsilon must be a number from 0 to 1")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
            raise ValueError("seed must be an integer")
        run = Run(iv, dv, max_steps, float(epsilon), seed)
        run.share = to_share_data(self.env.initial_state(iv, dv))
        if self.audit_dir is not None:
            self.audit_dir.mkdir(parents=True, exist_ok=True)
            run.audit = AuditLog(self.audit_dir / f"{run.id}.audit.jsonl")
        with self._lock:
            if sum(r.status == "running" for r in self.runs.values()) >= self.max_concurrent:
                raise Busy(f"already running {self.max_concurrent} runs; try again shortly")
            self.runs[run.id] = run
            finished = [r for r in self.runs.values() if r.status != "running"]
            for old in finished[: max(0, len(self.runs) - self.keep)]:
                del self.runs[old.id]
        threading.Thread(target=self._work, args=(run,), daemon=True).start()
        return run

    def _work(self, run: Run) -> None:
        if self.policy_factory is not None:
            policy = self.policy_factory(run)
        else:
            policy = EpsilonGreedy(
                LookaheadQ(self.env.preview), run.epsilon, random.Random(run.seed), explore_stop=False
            )

        def on_step(step: Step) -> None:
            run.share = to_share_data(step.next_state)
            run.phi = run.share["phi"]
            run.steps.append(
                {"n": len(run.steps) + 1, "action": describe(step.action), "reward": step.reward}
            )

        try:
            run_episode(
                self.env,
                policy,
                self.proposer,
                run.iv,
                run.dv,
                run.max_steps,
                on_step,
                run.stop.is_set,
                run.audit,
            )
            run.status = "stopped" if run.stop.is_set() else "done"
        except Exception as err:  # noqa: BLE001 - surface any backend failure to the page
            run.status, run.error = "error", f"{type(err).__name__}: {err}"[:300]
        finally:
            run.audit.close()

    def get(self, run_id: str) -> Run | None:
        return self.runs.get(run_id)


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:40] or "run"


def make_handler(manager: RunManager) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "ConfoundSearch"

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            pass

        def _send(
            self, status: int, body: bytes, content_type: str, extra: dict[str, str] | None = None
        ) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            for key, value in (extra or {}).items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, payload: Any) -> None:
            self._send(status, json.dumps(payload).encode(), "application/json")

        def _body(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BODY:
                raise ValueError("request body too large")
            data = json.loads(self.rfile.read(length) or b"{}")
            if not isinstance(data, dict):
                raise ValueError("body must be a JSON object")
            return data

        def do_GET(self) -> None:  # noqa: N802
            url = urlsplit(self.path)
            path = url.path
            if path == "/":
                page = (Path(__file__).resolve().parent / "ui" / "app.html").read_bytes()
                return self._send(
                    200, page, "text/html; charset=utf-8", {"Content-Security-Policy": "frame-src 'self'"}
                )
            if path == "/api/health":
                return self._json(200, {"ok": True})
            if path == "/api/runs":
                return self._json(200, [r.summary(0) for r in manager.runs.values()])
            match = re.fullmatch(r"/(?:api/runs|runs)/([0-9a-f]+)(?:/(.+))?", path)
            run = manager.get(match.group(1)) if match else None
            if match is None or run is None:
                return self._json(404, {"error": "not found"})
            leaf = match.group(2)
            if leaf is None and path.startswith("/api/"):
                return self._json(200, run.summary())
            if leaf == "share_data.json":
                return self._json(200, run.share)
            if leaf == "audit.jsonl":
                return self._send(200, run.audit.to_jsonl().encode(), "application/x-ndjson")
            if leaf == "audit.md":
                text = AuditTrail.from_log(run.audit).to_markdown()
                return self._send(200, text.encode(), "text/markdown; charset=utf-8")
            if leaf == "export.zip":
                disposition = f'attachment; filename="confoundsearch_{run.id}.zip"'
                return self._send(
                    200,
                    export_zip(run.share, run.audit),
                    "application/zip",
                    {"Content-Disposition": disposition},
                )
            if leaf == "standalone.html" and path.startswith("/runs/"):
                headers = {"Content-Security-Policy": SANDBOX}
                if parse_qs(url.query).get("download"):
                    name = f"confoundsearch_{_slug(run.iv)}_{_slug(run.dv)}.html"
                    headers["Content-Disposition"] = f'attachment; filename="{name}"'
                return self._send(
                    200, standalone_html(run.share).encode(), "text/html; charset=utf-8", headers
                )
            return self._json(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            path = urlsplit(self.path).path
            try:
                if path == "/api/runs":
                    return self._json(201, manager.start(self._body()).summary())
                stop = re.fullmatch(r"/api/runs/([0-9a-f]+)/stop", path)
                target = manager.get(stop.group(1)) if stop else None
                if target is not None:
                    target.stop.set()
                    return self._json(202, target.summary())
                return self._json(404, {"error": "not found"})
            except Busy as err:
                return self._json(429, {"error": str(err)})
            except ValueError as err:
                return self._json(400, {"error": str(err)})

    return Handler


def make_server(manager: RunManager, host: str = "127.0.0.1", port: int = 7860) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), make_handler(manager))
    server.daemon_threads = True
    return server
