from __future__ import annotations

import io
import json
import threading
import time
import urllib.error
import urllib.request
import zipfile

import pytest

from confoundsearch import AddConfound, Environment, InMemoryCorpus, Proposer, Retrieve
from confoundsearch.errors import RemoteError
from confoundsearch.interface.server import SANDBOX, RunManager, describe, make_server
from confoundsearch.logic.state import State
from tests.conftest import P1, P2, P3, P4, StubJudge


class Open:
    """Closed-form actions plus fixed open-ended ones, like an LLM proposer would add."""

    def __init__(self, delay: float = 0.0) -> None:
        self.inner, self.delay = Proposer(), delay

    def propose(self, state: State):
        time.sleep(self.delay)
        if state.done:
            return []
        extra = [Retrieve("larger firms"), Retrieve("profits")]
        if not state.graph.confounds:
            extra.append(AddConfound("firm size", frozenset({"firm size"})))
        return [*self.inner.propose(state), *extra]


class Boom:
    def propose(self, state: State):
        raise RuntimeError("backend exploded")


def start(proposer=None, max_concurrent=2, keep=50, **options):
    env = Environment(InMemoryCorpus([P1, P2, P3, P4]), StubJudge())
    manager = RunManager(env, proposer or Open(), max_concurrent, keep, **options)
    server = make_server(manager, "127.0.0.1", 0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return manager, server, f"http://127.0.0.1:{server.server_port}"


@pytest.fixture
def backend():
    made = []

    def make(*args, **kwargs):
        manager, server, url = start(*args, **kwargs)
        made.append(server)
        return manager, url

    yield make
    for server in made:
        server.shutdown()
        server.server_close()


def call(url, path, payload=None, method=None, raw=None):
    data = raw if raw is not None else (None if payload is None else json.dumps(payload).encode())
    request = urllib.request.Request(
        url + path, data=data, method=method or ("POST" if data is not None else "GET")
    )
    try:
        with urllib.request.urlopen(request) as response:
            return response.status, dict(response.headers), response.read()
    except urllib.error.HTTPError as err:
        return err.code, dict(err.headers), err.read()


def jcall(*args, **kwargs):
    status, _, body = call(*args, **kwargs)
    return status, json.loads(body)


def wait_for(url, run_id, states=("done", "stopped", "error"), timeout=20.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        _, run = jcall(url, f"/api/runs/{run_id}")
        if run["status"] in states:
            return run
        time.sleep(0.02)
    raise AssertionError(f"run did not finish: {run}")


def launch(url, **params):
    params = {"iv": "IT", "dv": "perf", "max_steps": 30, "seed": 1, **params}
    status, run = jcall(url, "/api/runs", params)
    assert status == 201, run
    return run["id"]


class TestBasics:
    def test_health(self, backend):
        _, url = backend()
        assert jcall(url, "/api/health") == (200, {"ok": True})

    def test_serves_the_interface(self, backend):
        _, url = backend()
        status, headers, body = call(url, "/")
        assert status == 200 and headers["Content-Type"].startswith("text/html")
        assert b"ConfoundSearch" in body and b"/api/runs" in body and b"audit.jsonl" in body
        assert headers["X-Content-Type-Options"] == "nosniff" and headers["Cache-Control"] == "no-store"

    @pytest.mark.parametrize(
        "path", ["/nope", "/api/runs/abc123", "/runs/abc123/standalone.html", "/api/runs/zz"]
    )
    def test_unknown_routes_are_404(self, backend, path):
        _, url = backend()
        assert jcall(url, path)[0] == 404

    def test_unknown_post_routes_are_404(self, backend):
        _, url = backend()
        assert jcall(url, "/api/nothing", {})[0] == 404
        assert jcall(url, "/api/runs/abc123/stop", {})[0] == 404

    def test_unknown_leaf_is_404(self, backend):
        _, url = backend()
        run_id = launch(url)
        assert jcall(url, f"/api/runs/{run_id}/other")[0] == 404
        wait_for(url, run_id)


class TestRuns:
    def test_a_run_goes_from_running_to_done(self, backend):
        _, url = backend()
        run_id = launch(url, iv="IT investment", dv="firm performance")
        run = wait_for(url, run_id)
        assert run["status"] == "done" and run["step"] == 30 and run["max_steps"] == 30
        assert run["iv"] == "IT investment" and run["error"] is None and run["phi"] >= 0
        assert run["steps"][0]["n"] == 1 and {"action", "reward"} <= set(run["steps"][0])

    def test_run_is_reproducible_given_a_seed(self, backend):
        _, url = backend()
        first = wait_for(url, launch(url, seed=5))
        second = wait_for(url, launch(url, seed=5))
        assert [s["action"] for s in first["steps"]] == [s["action"] for s in second["steps"]]

    def test_the_agent_actually_finds_a_confound(self, backend):
        _, url = backend()
        run_id = launch(url, max_steps=60, epsilon=0.2)
        assert wait_for(url, run_id)["phi"] > 1.0
        _, data = jcall(url, f"/api/runs/{run_id}/share_data.json")
        assert data["all_md_names"] == ["firm size"]

    def test_recent_steps_are_capped_in_the_summary(self, backend):
        _, url = backend()
        run = wait_for(url, launch(url, max_steps=150))
        assert len(run["steps"]) == 100 and run["steps"][-1]["n"] == 150

    def test_listing_runs(self, backend):
        _, url = backend()
        a = launch(url)
        b = launch(url)
        wait_for(url, a), wait_for(url, b)
        status, runs = jcall(url, "/api/runs")
        assert status == 200 and {r["id"] for r in runs} == {a, b} and all(r["steps"] == [] for r in runs)

    def test_stopping_a_run(self, backend):
        _, url = backend(Open(delay=0.05))
        run_id = launch(url, max_steps=200)
        time.sleep(0.3)
        status, _ = jcall(url, f"/api/runs/{run_id}/stop", {})
        assert status == 202
        run = wait_for(url, run_id)
        assert run["status"] == "stopped" and 0 < run["step"] < 200

    def test_a_backend_failure_is_reported_not_raised(self, backend):
        _, url = backend(Boom())
        run = wait_for(url, launch(url))
        assert run["status"] == "error" and "backend exploded" in run["error"]

    def test_too_many_concurrent_runs(self, backend):
        _, url = backend(Open(delay=0.05), max_concurrent=1)
        first = launch(url, max_steps=200)
        status, body = jcall(url, "/api/runs", {"iv": "a", "dv": "b"})
        assert status == 429 and "try again" in body["error"]
        jcall(url, f"/api/runs/{first}/stop", {})
        wait_for(url, first)
        launch(url)  # capacity is free again

    def test_old_finished_runs_are_evicted(self, backend):
        manager, url = backend(keep=2)
        ids = []
        for _ in range(4):
            ids.append(launch(url, max_steps=2))
            wait_for(url, ids[-1])
        assert len(manager.runs) <= 3 and ids[-1] in manager.runs and ids[0] not in manager.runs


class TestValidation:
    @pytest.mark.parametrize(
        "params,message",
        [
            ({}, "iv"),
            ({"iv": "a"}, "dv"),
            ({"iv": "", "dv": "b"}, "iv"),
            ({"iv": "   ", "dv": "b"}, "iv"),
            ({"iv": 5, "dv": "b"}, "iv"),
            ({"iv": "a" * 201, "dv": "b"}, "iv"),
            ({"iv": "<b>a</b>", "dv": "b"}, "iv"),
            ({"iv": "a", "dv": "b\x00"}, "dv"),
            ({"iv": "a", "dv": "b", "max_steps": 0}, "max_steps"),
            ({"iv": "a", "dv": "b", "max_steps": 1001}, "max_steps"),
            ({"iv": "a", "dv": "b", "max_steps": "9"}, "max_steps"),
            ({"iv": "a", "dv": "b", "max_steps": True}, "max_steps"),
            ({"iv": "a", "dv": "b", "epsilon": 1.5}, "epsilon"),
            ({"iv": "a", "dv": "b", "epsilon": "x"}, "epsilon"),
            ({"iv": "a", "dv": "b", "epsilon": True}, "epsilon"),
            ({"iv": "a", "dv": "b", "seed": 1.5}, "seed"),
            ({"iv": "a", "dv": "b", "seed": True}, "seed"),
        ],
    )
    def test_bad_parameters_are_400(self, backend, params, message):
        manager, url = backend()
        status, body = jcall(url, "/api/runs", params)
        assert status == 400 and message in body["error"] and manager.runs == {}

    @pytest.mark.parametrize("raw", [b"not json", b"[1]", b'"x"'])
    def test_bad_bodies_are_400(self, backend, raw):
        _, url = backend()
        assert jcall(url, "/api/runs", raw=raw)[0] == 400

    def test_oversized_bodies_are_400(self, backend):
        _, url = backend()
        status, body = jcall(url, "/api/runs", raw=b'{"iv": "' + b"a" * 9000 + b'"}')
        assert status == 400 and "too large" in body["error"]

    def test_empty_body_is_a_missing_parameter(self, backend):
        _, url = backend()
        assert jcall(url, "/api/runs", raw=b"")[0] == 400

    def test_input_is_stripped(self, backend):
        _, url = backend()
        run = wait_for(url, launch(url, iv="  IT  ", dv=" perf ", max_steps=1))
        assert (run["iv"], run["dv"]) == ("IT", "perf")


class TestDownloads:
    def finished(self, backend):
        _, url = backend()
        run_id = launch(url, iv="IT investment", dv="firm performance", max_steps=40, epsilon=0.2)
        wait_for(url, run_id)
        return url, run_id

    def test_share_data(self, backend):
        url, run_id = self.finished(backend)
        status, data = jcall(url, f"/api/runs/{run_id}/share_data.json")
        assert status == 200 and data["iv"] == "IT investment" and data["mode"] == "explore"

    def test_export_zip_matches_the_gradio_apps_format(self, backend):
        url, run_id = self.finished(backend)
        status, headers, body = call(url, f"/api/runs/{run_id}/export.zip")
        assert status == 200 and headers["Content-Type"] == "application/zip"
        assert f"confoundsearch_{run_id}.zip" in headers["Content-Disposition"]
        with zipfile.ZipFile(io.BytesIO(body)) as zf:
            assert sorted(zf.namelist()) == ["audit.jsonl", "share_data.json"]  # share_data.json at the root
            assert json.loads(zf.read("share_data.json"))["dv"] == "firm performance"

    def test_standalone_page_is_sandboxed_and_viewable(self, backend):
        url, run_id = self.finished(backend)
        status, headers, body = call(url, f"/runs/{run_id}/standalone.html")
        assert status == 200 and headers["Content-Security-Policy"] == SANDBOX
        assert "attachment" not in headers.get("Content-Disposition", "")
        assert b"firm performance" in body

    def test_standalone_page_download(self, backend):
        url, run_id = self.finished(backend)
        _, headers, _ = call(url, f"/runs/{run_id}/standalone.html?download=1")
        assert (
            headers["Content-Disposition"]
            == 'attachment; filename="confoundsearch_it_investment_firm_performance.html"'
        )

    def test_standalone_page_is_not_served_under_api(self, backend):
        url, run_id = self.finished(backend)
        assert call(url, f"/api/runs/{run_id}/standalone.html")[0] == 404

    def test_a_run_in_flight_already_has_a_page(self, backend):
        _, url = backend(Open(delay=0.05))
        run_id = launch(url, max_steps=200)
        status, _, body = call(url, f"/runs/{run_id}/standalone.html")
        assert status == 200 and b"Showing possible confounds" in body
        jcall(url, f"/api/runs/{run_id}/stop", {})
        wait_for(url, run_id)


class TestAuditEndpoints:
    def finished(self, backend):
        _, url = backend()
        run_id = launch(url, iv="IT investment", dv="firm performance", max_steps=40, epsilon=0.2)
        wait_for(url, run_id)
        return url, run_id

    def test_the_run_summary_counts_its_audit_events(self, backend):
        url, run_id = self.finished(backend)
        _, run = jcall(url, f"/api/runs/{run_id}")
        assert run["audit_events"] > 40

    def test_audit_jsonl_is_the_hash_chained_log(self, backend):
        url, run_id = self.finished(backend)
        status, headers, body = call(url, f"/api/runs/{run_id}/audit.jsonl")
        assert status == 200 and headers["Content-Type"] == "application/x-ndjson"
        rows = [json.loads(line) for line in body.decode().splitlines()]
        assert rows[0]["kind"] == "run_started" and rows[-1]["kind"] == "run_finished"
        assert rows[0]["data"]["independent_variable"] == "IT investment"
        from confoundsearch import verify

        assert verify(rows)

    def test_audit_md_is_a_readable_report(self, backend):
        url, run_id = self.finished(backend)
        status, headers, body = call(url, f"/api/runs/{run_id}/audit.md")
        text = body.decode()
        assert status == 200 and headers["Content-Type"].startswith("text/markdown")
        assert text.startswith("# Audit trail") and "hash chain intact" in text

    def test_the_export_zip_carries_the_audit_trail(self, backend):
        url, run_id = self.finished(backend)
        _, _, body = call(url, f"/api/runs/{run_id}/export.zip")
        with zipfile.ZipFile(io.BytesIO(body)) as archive:
            assert sorted(archive.namelist()) == ["audit.jsonl", "share_data.json"]
            assert archive.read("audit.jsonl").decode().startswith('{"seq": 0')

    def test_a_run_in_flight_already_has_a_partial_log(self, backend):
        _, url = backend(Open(delay=0.05))
        run_id = launch(url, max_steps=200)
        time.sleep(0.3)
        status, _, body = call(url, f"/api/runs/{run_id}/audit.jsonl")
        kinds = [json.loads(line)["kind"] for line in body.decode().splitlines()]
        assert status == 200 and kinds[0] == "run_started" and "run_finished" not in kinds
        jcall(url, f"/api/runs/{run_id}/stop", {})
        wait_for(url, run_id)

    def test_a_stopped_run_says_so_in_its_log(self, backend):
        _, url = backend(Open(delay=0.05))
        run_id = launch(url, max_steps=200)
        time.sleep(0.3)
        jcall(url, f"/api/runs/{run_id}/stop", {})
        wait_for(url, run_id)
        _, _, body = call(url, f"/api/runs/{run_id}/audit.jsonl")
        assert json.loads(body.decode().splitlines()[-1])["data"]["reason"] == "stopped_by_user"

    def test_a_backend_failure_ends_the_log_with_run_failed(self, backend):
        _, url = backend(Boom())
        run_id = launch(url)
        wait_for(url, run_id)
        _, _, body = call(url, f"/api/runs/{run_id}/audit.jsonl")
        last = json.loads(body.decode().splitlines()[-1])
        assert last["kind"] == "run_failed" and "backend exploded" in last["data"]["error"]

    def test_with_an_audit_folder_every_run_is_also_written_to_disk(self, tmp_path):
        env = Environment(InMemoryCorpus([P1, P2, P3, P4]), StubJudge())
        manager = RunManager(env, Open(), audit_dir=tmp_path / "logs")
        server = make_server(manager, "127.0.0.1", 0)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        url = f"http://127.0.0.1:{server.server_port}"
        try:
            run_id = launch(url, max_steps=10)
            wait_for(url, run_id)
            time.sleep(0.1)
            path = tmp_path / "logs" / f"{run_id}.audit.jsonl"
            assert path.read_text() == manager.get(run_id).audit.to_jsonl()
            assert manager.get(run_id).audit._sink is None  # closed when the run ended
        finally:
            server.shutdown()
            server.server_close()


def test_describe_renders_actions_readably():
    from confoundsearch import AddEdge, Stop

    assert describe(Stop()) == "Stop()"
    assert describe(AddEdge("a", "X")) == "AddEdge(confound=a, target=X)"
    assert describe(AddConfound("a", frozenset({"z", "b"}))) == "AddConfound(label=a, strings=['b', 'z'])"


def test_remote_errors_in_the_backend_surface_to_the_page(backend):
    class Failing:
        def propose(self, state):
            raise RemoteError("HTTP 401 from llm")

    _, url = backend(Failing())
    run = wait_for(url, launch(url))
    assert run["status"] == "error" and "HTTP 401" in run["error"]


class TestPolicies:
    def test_a_policy_factory_makes_each_runs_policy(self, backend):
        from confoundsearch import DeterministicPolicy

        made = []

        def factory(run):
            made.append(run)
            return DeterministicPolicy(lambda x, y: ["firm size"], lambda term: [], max_confounds=1)

        _, url = backend(policy_factory=factory)
        run = wait_for(url, launch(url, max_steps=60))
        assert run["status"] == "done" and [m.id for m in made] == [run["id"]]
        assert run["step"] > 3  # it asked for a confound and went on to search for it

    def test_without_a_max_steps_the_managers_default_is_used(self, backend):
        manager, url = backend(default_steps=7)
        run_id = launch(url, max_steps=None)
        assert wait_for(url, run_id)["max_steps"] == 7
        assert manager.default_steps == 7

    def test_an_explicit_max_steps_beats_the_default(self, backend):
        _, url = backend(default_steps=7)
        assert wait_for(url, launch(url, max_steps=9))["max_steps"] == 9
