"""Rate limits, retries, the run log, several sources at once, and a run that stops part way."""

from __future__ import annotations

import io
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from confoundsearch.data.corpus import InMemoryCorpus
from confoundsearch.data.multi import MultiCorpus, paper_id, title_key
from confoundsearch.errors import CorpusError, HttpError, RemoteError
from confoundsearch.http import resilient, urllib_transport
from confoundsearch.interface import cli
from confoundsearch.interface.review import where_it_went
from confoundsearch.logic.graph import Passage
from confoundsearch.runlog import RunLog
from tests.test_cli import COMMON, ScriptedClient, fake_backend, parse  # noqa: F401


class Clock:
    def __init__(self):
        self.now, self.slept = 0.0, []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.slept.append(seconds)
        self.now += seconds


def flaky(*outcomes):
    """A transport that gives each outcome in turn: bytes are returned, exceptions raised."""
    left = list(outcomes)

    def transport(url, headers, body, timeout):
        outcome = left.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    return transport


def make(transport, **kw):
    clock, events = Clock(), []
    send = resilient(
        transport, lambda kind, **d: events.append((kind, d)), sleep=clock.sleep, clock=clock, **kw
    )
    return send, clock, events


class TestResilient:
    def test_a_rate_limit_is_waited_out_using_the_servers_own_figure(self):
        send, clock, events = make(flaky(HttpError("slow down", 429, 7.0), b"ok"))
        assert send("u", {}, None, 1) == b"ok"
        assert clock.slept == [7.0]
        assert [d["status"] for k, d in events if k == "http"] == [429, 200]
        assert events[1][1]["retrying_in"] == 7.0 and events[1][1]["retry_after"] == 7.0

    def test_without_a_figure_it_backs_off_exponentially_up_to_a_cap(self):
        errors = [HttpError("busy", 503)] * 4
        send, clock, _ = make(flaky(*errors, b"ok"), backoff=2.0, max_wait=5.0)
        send("u", {}, None, 1)
        assert clock.slept == [2.0, 4.0, 5.0, 5.0]

    def test_it_gives_up_after_the_attempts_and_says_so(self):
        send, clock, events = make(flaky(*[HttpError("no", 429)] * 3), attempts=3, backoff=1.0)
        with pytest.raises(HttpError) as raised:
            send("u", {}, None, 1)
        assert raised.value.status == 429 and len(clock.slept) == 2
        assert events[-1][1]["retrying_in"] is None

    def test_other_errors_are_raised_at_once(self):
        send, clock, events = make(flaky(HttpError("nope", 404), b"x"))
        with pytest.raises(HttpError):
            send("u", {}, None, 1)
        assert clock.slept == [] and [k for k, _ in events] == ["request", "http"]

    def test_without_a_log_it_still_works(self):
        send = resilient(flaky(b"a"))
        assert send("u", {}, None, 1) == b"a"

    def test_without_a_log_a_failure_still_raises(self):
        with pytest.raises(HttpError):
            resilient(flaky(HttpError("no", 400)))("u", {}, None, 1)

    def test_requests_are_spaced_by_the_minimum_interval(self):
        send, clock, _ = make(flaky(b"1", b"2", b"3"), min_interval=1.0)
        for _ in range(3):
            send("u", {}, None, 1)
        assert clock.slept == [1.0, 1.0]
        clock.now += 5
        send2, clock2, _ = make(flaky(b"1", b"2"), min_interval=1.0)
        send2("u", {}, None, 1)
        clock2.now += 5
        send2("u", {}, None, 1)
        assert clock2.slept == []  # already waited long enough


class Server:
    """A local server that answers every request with a fixed status and headers."""

    def __init__(self, status, headers=None, body=b"{}"):
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802
                self.send_response(outer.status)
                for name, value in outer.headers.items():
                    self.send_header(name, value)
                self.end_headers()
                self.wfile.write(outer.body)

            def log_message(self, *args):
                pass

        self.status, self.headers, self.body = status, headers or {}, body
        self.httpd = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_port}/"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


class TestTransport:
    @pytest.mark.parametrize(
        ("header", "expected"),
        [({"Retry-After": "12"}, 12.0), ({"Retry-After": "Wed, 21 Oct"}, None), ({}, None)],
    )
    def test_a_refusal_carries_its_status_and_retry_after(self, header, expected):
        server = Server(429, header, b"too many")
        try:
            with pytest.raises(HttpError) as raised:
                urllib_transport(server.url, {}, None, 5)
        finally:
            server.close()
        assert raised.value.status == 429 and raised.value.retry_after == expected
        assert "too many" in str(raised.value) and isinstance(raised.value, RemoteError)

    def test_an_unreachable_host_is_a_remote_error(self):
        with pytest.raises(RemoteError, match="could not reach"):
            urllib_transport("http://127.0.0.1:1/", {}, None, 1)

    def test_a_server_that_never_answers_is_a_timeout_not_a_crash(self):
        import socket

        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        threading.Thread(target=lambda: (listener.accept(), time.sleep(2)), daemon=True).start()
        try:
            with pytest.raises(RemoteError, match="timed out waiting for"):
                urllib_transport(f"http://127.0.0.1:{listener.getsockname()[1]}/", {}, None, 0.3)
        finally:
            listener.close()

    def test_a_connection_dropped_half_way_is_a_remote_error(self):
        import socket

        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)

        def hang_up():
            connection, _ = listener.accept()
            connection.recv(1024)
            connection.sendall(
                b"HTTP/1.1 200 OK\r\nContent-Length: 100\r\n\r\nshort"
            )  # promises 100, sends 5
            connection.close()

        threading.Thread(target=hang_up, daemon=True).start()
        try:
            with pytest.raises(RemoteError, match=r"connection to .* failed"):
                urllib_transport(f"http://127.0.0.1:{listener.getsockname()[1]}/", {}, None, 5)
        finally:
            listener.close()

    def test_a_good_answer_is_returned(self):
        server = Server(200, body=b"fine")
        try:
            assert urllib_transport(server.url, {}, None, 5) == b"fine"
        finally:
            server.close()


class TestRunLog:
    def test_events_are_written_as_json_lines_as_they_happen(self, tmp_path):
        path = tmp_path / "deep" / "run.log.jsonl"
        log = RunLog(path)
        log.event("one", a=1)
        before_close = path.read_text()  # already on disk
        log.event("two", obj=Path)  # anything unserialisable is stringified, not fatal
        log.close()
        log.close()
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert before_close.count("\n") == 1
        assert [r["event"] for r in rows] == ["one", "two"] and rows[0]["a"] == 1
        assert {"time", "t"} <= set(rows[0])

    def test_it_can_keep_events_in_memory_with_no_file(self):
        log = RunLog(keep=True)
        log.event("x")
        assert [e["event"] for e in log.events] == ["x"] and log.path is None
        quiet = RunLog()
        quiet.event("y")
        assert quiet.events == []


def paper(i, title, text="text"):
    return Passage(i, text, title, None)


class Store:
    def __init__(self, *papers, error=None):
        self.papers, self.error = papers, error

    def search(self, query, k):
        if self.error:
            raise self.error
        return list(self.papers)[:k]


class TestMultiCorpus:
    def test_titles_match_regardless_of_case_and_punctuation(self):
        assert title_key("Firm  Size: A Study!") == title_key("firm size a study")

    def test_the_same_paper_from_two_sources_appears_once_with_the_fuller_text(self):
        a = Store(paper("1", "Firm Size", "short"), paper("2", "Other"))
        b = Store(paper("9", "firm size.", "a much longer abstract"))
        found = MultiCorpus({"a": a, "b": b}).search("q", 10)
        assert [p.title for p in found] == ["firm size.", "Other"]
        assert found[0].text == "a much longer abstract" and found[0].id == paper_id(
            "firm size", "b", found[0]
        )

    def test_a_paper_keeps_one_id_whichever_source_or_search_found_it(self):
        from_a = MultiCorpus({"a": Store(paper("1", "Firm Size"))}).search("q", 1)[0]
        from_b = MultiCorpus({"b": Store(paper("zz", "firm size!"))}).search("other", 1)[0]
        assert from_a.id == from_b.id and from_a.id.startswith("p")

    def test_agreement_ranks_higher_and_ties_are_stable(self):
        a = Store(paper("1", "Only A"), paper("2", "Both"))
        b = Store(paper("3", "Both"), paper("4", "Only B"))
        assert [p.title for p in MultiCorpus({"a": a, "b": b}).search("q", 10)] == [
            "Both",
            "Only A",
            "Only B",
        ]

    def test_k_limits_and_empty_asks_return_nothing(self):
        multi = MultiCorpus({"a": Store(paper("1", "A"), paper("2", "B"))})
        assert len(multi.search("q", 1)) == 1
        assert multi.search("q", 0) == [] and multi.search("  ", 5) == []

    def test_papers_without_titles_are_kept_apart_by_source_and_id(self):
        found = MultiCorpus({"a": Store(paper("1", "")), "b": Store(paper("1", ""))}).search("q", 5)
        assert sorted(p.id for p in found) == ["a:1", "b:1"]

    def test_a_failing_source_is_skipped_and_logged(self):
        events = []
        multi = MultiCorpus(
            {"a": Store(error=HttpError("429", 429)), "b": Store(paper("1", "Fine"))},
            lambda kind, **d: events.append((kind, d)),
        )
        assert [p.title for p in multi.search("q", 5)] == ["Fine"]
        kinds = [k for k, _ in events]
        assert kinds == ["source_failed", "search"]
        assert events[1][1]["failed"] == ["a"] and events[1][1]["answered"] == {"b": 1}

    def test_a_failing_source_without_a_log_is_still_skipped(self):
        multi = MultiCorpus({"a": Store(error=RemoteError("x")), "b": Store(paper("1", "Fine"))})
        assert len(multi.search("q", 5)) == 1

    def test_all_failing_is_an_error_naming_each(self):
        multi = MultiCorpus({"a": Store(error=RemoteError("x")), "b": Store(error=RemoteError("y"))})
        with pytest.raises(CorpusError, match="every source failed: a: x; b: y"):
            multi.search("q", 5)

    def test_it_needs_a_store(self):
        with pytest.raises(ValueError):
            MultiCorpus({})


class TestRunLogging:
    def test_a_run_writes_a_log_next_to_the_page_without_any_key(self, tmp_path, monkeypatch, capsys, env):
        monkeypatch.setattr(cli, "build_backend", fake_backend(env))
        monkeypatch.setenv("OPENAI_API_KEY", "sk-secret-value")
        out = tmp_path / "page.html"
        code = cli.main(
            [
                "run",
                *COMMON,
                "--corpus-file",
                "x",
                "--iv",
                "firm size",
                "--dv",
                "profit",
                "--out",
                str(out),
                "--max-steps",
                "3",
                "--api-key",
                "sk-secret-value",
            ]
        )
        assert code == 0
        text = (tmp_path / "page.log.jsonl").read_text()
        rows = [json.loads(line) for line in text.splitlines()]
        assert rows[0]["event"] == "run_started" and rows[0]["keys_present"]["model"] is True
        assert rows[-1]["event"] == "run_finished" and any(r["event"] == "step" for r in rows)
        assert "sk-secret-value" not in text
        assert "page.log.jsonl" in capsys.readouterr().out

    def test_log_path_can_be_chosen(self, tmp_path, monkeypatch, env):
        monkeypatch.setattr(cli, "build_backend", fake_backend(env))
        where = tmp_path / "logs" / "mine.jsonl"
        cli.main(
            [
                "run",
                *COMMON,
                "--corpus-file",
                "x",
                "--iv",
                "a",
                "--dv",
                "b",
                "--out",
                str(tmp_path / "p.html"),
                "--max-steps",
                "2",
                "--log",
                str(where),
            ]
        )
        assert where.exists()

    def test_a_search_that_fails_keeps_the_work_and_says_why(self, tmp_path, monkeypatch, capsys, env):
        monkeypatch.setattr(cli, "build_backend", fake_backend(env))
        calls = []

        def boom(*a, **k):
            calls.append(1)
            raise HttpError("HTTP 429 from s2", 429)

        monkeypatch.setattr(cli, "run_episode", boom)
        out = tmp_path / "page.html"
        code = cli.main(
            ["run", *COMMON, "--corpus-file", "x", "--iv", "a", "--dv", "b", "--out", str(out), "--json"]
        )
        captured = capsys.readouterr()
        assert code == 1 and out.exists() and (tmp_path / "page.audit.jsonl").exists()
        assert "HTTP 429" in json.loads(captured.out)["error"]
        assert "stopped early" in captured.err and "page.log.jsonl" in captured.err
        events = [
            json.loads(line)["event"] for line in (tmp_path / "page.log.jsonl").read_text().splitlines()
        ]
        assert events[-1] == "run_failed"

    def test_a_failure_without_json_still_exits_one(self, tmp_path, monkeypatch, capsys, env):
        monkeypatch.setattr(cli, "build_backend", fake_backend(env))
        monkeypatch.setattr(cli, "run_episode", lambda *a, **k: (_ for _ in ()).throw(RemoteError("down")))
        code = cli.main(
            [
                "run",
                *COMMON,
                "--corpus-file",
                "x",
                "--iv",
                "a",
                "--dv",
                "b",
                "--out",
                str(tmp_path / "p.html"),
            ]
        )
        assert code == 1 and "steps" in capsys.readouterr().out

    def test_a_bug_is_logged_with_its_traceback_and_raised(self, tmp_path, monkeypatch, env):
        monkeypatch.setattr(cli, "build_backend", fake_backend(env))
        monkeypatch.setattr(cli, "run_episode", lambda *a, **k: 1 / 0)
        with pytest.raises(ZeroDivisionError):
            cli.main(
                [
                    "run",
                    *COMMON,
                    "--corpus-file",
                    "x",
                    "--iv",
                    "a",
                    "--dv",
                    "b",
                    "--out",
                    str(tmp_path / "p.html"),
                ]
            )
        last = json.loads((tmp_path / "p.log.jsonl").read_text().splitlines()[-1])
        assert last["event"] == "crashed" and "ZeroDivisionError" in last["traceback"]

    def test_the_summary_points_at_the_log(self, tmp_path):
        text = where_it_went(tmp_path / "p.html", tmp_path / "a.jsonl", True, tmp_path / "p.log.jsonl")
        assert "p.log.jsonl" in text and "read this first" in text


class TestDescribeRun:
    def test_it_reports_which_keys_exist_and_never_their_values(self, monkeypatch):
        for name in ("OPENAI_API_KEY", "OPENROUTER_API_KEY", "S2_KEY", "ES_URL", "ES_API_KEY"):
            monkeypatch.delenv(name, raising=False)
        args = parse(
            "run",
            *COMMON,
            "--iv",
            "a",
            "--dv",
            "b",
            "--out",
            "o.html",
            "--s2-key",
            "sekret",
            "--base-url",
            "https://h.example/v1",
        )
        info = cli.describe_run(args)
        assert info["keys_present"]["semantic_scholar"] is True
        assert info["model_host"] == "h.example" and info["sources"] == ["semanticscholar", "openalex"]
        assert "sekret" not in json.dumps(info)

    def test_a_papers_file_is_the_source_when_given(self):
        args = parse("run", *COMMON, "--corpus-file", "c", "--iv", "a", "--dv", "b", "--out", "o.html")
        assert cli.describe_run(args)["sources"] == ["file"]


def test_the_corpus_builder_logs_requests_through_the_run_log(monkeypatch, tmp_path):
    args = parse("serve", *COMMON, "--corpus-file", str(tmp_path / "c.jsonl"))
    (tmp_path / "c.jsonl").write_text(json.dumps({"id": "1", "text": "t"}) + "\n")
    assert isinstance(cli.build_corpus(args), InMemoryCorpus)


class Tty(io.StringIO):
    def isatty(self):
        return True


class TestStatusLine:
    def asked(self, kind, **data):
        return {"event": kind, **data}

    def test_a_search_says_where_and_for_what(self):
        url = "https://api.semanticscholar.org/graph/v1/paper/search?query=firm+size&limit=5"
        assert (
            cli.status_text(self.asked("request", url=url, attempt=1))
            == "searching semanticscholar.org for 'firm size'"
        )
        assert "(try 3)" in cli.status_text(self.asked("request", url=url, attempt=3))

    def test_a_request_without_a_query_still_says_where(self):
        assert (
            cli.status_text(self.asked("request", url="https://es.example/papers/_search", attempt=1))
            == "searching es.example"
        )
        assert "the search service" in cli.status_text(self.asked("request", url="", attempt=1))

    def test_a_wait_says_why_and_for_how_long(self):
        url = "https://api.openalex.org/works?search=x"
        text = cli.status_text(self.asked("http", url=url, status=429, retrying_in=7.0))
        assert text == "openalex.org said HTTP 429; waiting 7s, then trying again"
        assert "giving up" in cli.status_text(self.asked("http", url=url, status=429, retrying_in=None))

    def test_success_and_other_events_say_nothing(self):
        assert cli.status_text(self.asked("http", url="u", status=200, retrying_in=None)) is None
        assert cli.status_text(self.asked("step")) is None

    def test_a_failed_source_is_mentioned(self):
        assert "openalex failed" in cli.status_text(self.asked("source_failed", source="openalex"))
        resting = cli.status_text(self.asked("source_skipped", source="openalex", retry_in=20.0))
        assert "openalex is resting" in resting and "again in 20s" in resting
        assert (
            cli.status_text(self.asked("source_recovered", source="openalex"))
            == "openalex is answering again"
        )

    def test_on_a_terminal_one_line_is_rewritten_with_the_activity_step_and_score(self):
        out, clock = Tty(), Clock()
        status = cli.StatusLine(out, clock=clock)
        status.total = 20
        status.stop()  # nothing shown yet: nothing written
        assert out.getvalue() == ""
        clock.now = 75
        status(self.asked("request", url="https://api.openalex.org/works?search=x", attempt=1))
        status(self.asked("step", n=3, phi=1.25))
        status(self.asked("asking_model"))
        drawn = out.getvalue().split("\r\x1b[K")[1:]
        assert "searching openalex.org for 'x'" in drawn[0] and "1:15" in drawn[0] and "step 0/20" in drawn[0]
        assert (
            "choosing the next action" in drawn[1] and "step 3/20" in drawn[1] and "\u03a6 1.25" in drawn[1]
        )
        assert "asking the model" in drawn[2]
        assert "\n" not in out.getvalue()
        status.stop()
        assert out.getvalue().endswith("\r\x1b[K")
        status.stop()  # clearing twice is harmless

    def test_a_long_line_is_cut_to_the_terminal_width(self, monkeypatch):
        monkeypatch.setattr(cli.shutil, "get_terminal_size", lambda fallback: os.terminal_size((30, 5)))
        out = Tty()
        cli.StatusLine(out)(self.asked("source_failed", source="x" * 80))
        assert len(out.getvalue()[len("\r\x1b[K") :]) == 29

    def test_without_a_total_it_just_counts_steps(self):
        out = Tty()
        cli.StatusLine(out)(self.asked("step", n=2))
        assert "step 2 " in out.getvalue()

    def test_the_spinner_keeps_moving_while_nothing_happens(self):
        class Ticking(Tty):
            def __init__(self):
                super().__init__()
                self.enough = threading.Event()

            def write(self, text):
                super().write(text)
                if self.getvalue().count("\r\x1b[K") >= 3:
                    self.enough.set()

        out = Ticking()
        status = cli.StatusLine(out, interval=0.01)
        status.start()
        status.start()  # starting twice does not start two
        assert out.enough.wait(5)
        status.stop()
        assert status.thread is None

    def test_when_piped_only_trouble_is_printed(self):
        out = io.StringIO()
        status = cli.StatusLine(out)
        status.start()  # no thread without a terminal
        assert status.thread is None
        status(self.asked("request", url="https://x.org/?q=a", attempt=1))
        status(self.asked("asking_model"))
        status(self.asked("step", n=1))
        status(self.asked("http", url="https://api.x.org/", status=200, retrying_in=None))
        status(self.asked("http", url="https://api.x.org/", status=429, retrying_in=3.0))
        status(self.asked("source_failed", source="a"))
        status.stop()
        assert out.getvalue() == (
            "  x.org said HTTP 429; waiting 3s, then trying again\n"
            "  a failed, carrying on with the other sources\n"
        )

    def test_the_log_echoes_events_to_the_status(self):
        seen = []
        RunLog(echo=seen.append).event("x", a=1)
        assert seen[0]["event"] == "x" and seen[0]["a"] == 1

    def test_the_model_is_announced_before_each_call(self):
        seen = []
        client = cli.Announced(ScriptedClient(), lambda kind, **d: seen.append((kind, d)))
        assert client.chat([{"role": "user", "content": "hi"}]) == "yes" and client.model == "scripted"
        assert seen == [("asking_model", {"model": "scripted"})]
        assert cli.Announced(ScriptedClient(), None).chat([{"role": "user", "content": "hi"}]) == "yes"

    def test_a_run_on_a_terminal_prints_no_line_per_step(self, tmp_path, monkeypatch, capsys, env):
        monkeypatch.setattr(cli, "build_backend", fake_backend(env))
        real = cli.StatusLine
        monkeypatch.setattr(cli, "StatusLine", lambda: real(Tty()))
        cli.main(
            [
                "run",
                *COMMON,
                "--corpus-file",
                "x",
                "--iv",
                "a",
                "--dv",
                "b",
                "--out",
                str(tmp_path / "p.html"),
                "--max-steps",
                "3",
            ]
        )
        assert "\u25cf a1" not in capsys.readouterr().err
