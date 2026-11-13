"""Saved model answers: the key, the database, the wrapper, the audit flag, and the command line."""

from __future__ import annotations

import json
import sqlite3

import pytest

from confoundsearch.data.cache import PaperDatabase, call_key, database_path, flush, wait_for_writes
from confoundsearch.data.demo import RECORDED_MODEL
from confoundsearch.errors import RemoteError, ReplayMiss
from confoundsearch.interface import cli
from confoundsearch.interface.modelcache import CachedClient, find_cached
from confoundsearch.logic.audit import AuditedClient, AuditLog, recording
from confoundsearch.model.cost import Spend
from tests.test_cli import COMMON, ScriptedClient, write_corpus

QUESTION = ("a", "b")
HI = [{"role": "user", "content": "hi"}]


class Model:
    """A stand-in model that counts how often it is asked."""

    def __init__(self, reply="yes", temperature=0.0, fail=False):
        self.model, self.temperature, self.reply, self.fail = "m", temperature, reply, fail
        self.spend = Spend("m")
        self.asked = []

    def chat(self, messages):
        self.asked.append(list(messages))
        if self.fail:
            raise RemoteError("down")
        self.spend.add({"prompt_tokens": 1_000_000})
        return self.reply


def cached(inner, folder, scope=lambda: QUESTION, log=None, model=None):
    return CachedClient(inner, folder, scope, log, model=model)


def ask(client, messages=HI):
    reply = client.chat(messages)
    wait_for_writes()
    return reply


class TestTheKey:
    def test_the_same_call_gives_the_same_key_whatever_order_the_fields_come_in(self):
        a = call_key("m", 0.0, [{"role": "user", "content": "hi"}])
        assert a == call_key("m", 0.0, [{"content": "hi", "role": "user"}]) and len(a) == 64

    def test_anything_that_changes_the_answer_changes_the_key(self):
        base = call_key("m", 0.0, HI)
        assert len({base, call_key("n", 0.0, HI), call_key("m", 0.7, HI), call_key("m", None, HI)}) == 4
        assert call_key("m", 0.0, [{"role": "user", "content": "ho"}]) != base
        two = [*HI, {"role": "assistant", "content": "x"}]
        assert call_key("m", 0.0, two) != base and call_key("m", 0.0, two[::-1]) != call_key("m", 0.0, two)

    def test_text_in_any_language_is_fine(self):
        assert call_key("m", 0.0, [{"role": "user", "content": "éé 中文"}])


class TestTheDatabase:
    def test_a_saved_reply_comes_back_and_a_new_one_replaces_it(self, tmp_path):
        db = PaperDatabase(tmp_path / "d.sqlite3")
        assert db.load_call("k") is None and db.call_count() == 0
        db.save_call("k", "m", HI, "one")
        assert db.load_call("k") == "one" and db.call_count() == 1
        db.save_call("k", "m", HI, "two")
        assert db.load_call("k") == "two" and db.call_count() == 1
        db.save_call("j", None, HI, "x")  # a call to an unnamed model is fine
        assert db.call_count() == 2

    def test_what_was_sent_is_kept_beside_the_reply(self, tmp_path):
        db = PaperDatabase(tmp_path / "d.sqlite3")
        db.save_call("k", "m", HI, "yes")
        con = sqlite3.connect(db.path)
        try:
            model, messages = con.execute("SELECT model, messages FROM model_calls").fetchone()
        finally:
            con.close()
        assert model == "m" and json.loads(messages) == HI

    def test_a_damaged_or_unwritable_file_is_never_fatal(self, tmp_path):
        bad = tmp_path / "bad.sqlite3"
        bad.write_bytes(b"not a database")
        db = PaperDatabase(bad)
        assert db.load_call("k") is None and db.call_count() == 0
        db.save_call("k", "m", HI, "x")  # cannot save: carries on
        blocked = tmp_path / "file-not-folder"
        blocked.write_text("x")
        PaperDatabase(blocked / "d.sqlite3").save_call("k", "m", HI, "x")

    def test_an_older_database_without_the_table_gains_it(self, tmp_path):
        path = tmp_path / "old.sqlite3"
        con = sqlite3.connect(path)
        con.execute("CREATE TABLE papers (key TEXT PRIMARY KEY, id TEXT, title TEXT, url TEXT, text TEXT)")
        con.commit()
        con.close()
        db = PaperDatabase(path)
        db.save_call("k", "m", HI, "x")
        assert db.load_call("k") == "x"


class TestTheWrapper:
    def test_the_model_is_asked_once_and_the_answer_is_saved(self, tmp_path):
        inner, events = Model(), []
        client = cached(inner, tmp_path, log=lambda kind, **d: events.append(kind))
        assert ask(client) == "yes" and ask(client) == "yes"
        assert len(inner.asked) == 1 and (client.hits, client.misses) == (1, 1)
        assert events == ["model_cache_miss", "model_cache_hit"]

    def test_a_replay_costs_nothing_and_a_fresh_call_is_counted(self, tmp_path):
        inner = Model()
        client = cached(inner, tmp_path)
        ask(client)
        assert client.spend is inner.spend and client.spend.dollars == pytest.approx(
            10.0
        )  # $10 per million, once
        ask(client)
        assert client.spend.dollars == pytest.approx(10.0)  # the replay added nothing

    def test_it_looks_like_the_client_it_wraps(self, tmp_path):
        client = cached(Model(temperature=0.3), tmp_path)
        assert (client.model, client.temperature) == ("m", 0.3)
        assert (
            cached(Model(temperature=None), tmp_path).temperature is None
        )  # a model left at its own default
        assert cached(None, tmp_path, model="m").temperature == 0.0  # a replay assumes the usual default
        assert (
            cached(None, tmp_path, model="gpt-6-astra").temperature is None
        )  # or none, for models that need it

    def test_a_different_model_temperature_or_prompt_is_asked_afresh(self, tmp_path):
        inner = Model()
        ask(cached(inner, tmp_path), HI)
        ask(cached(inner, tmp_path), [{"role": "user", "content": "other"}])
        warm = Model(temperature=0.9)
        ask(cached(warm, tmp_path), HI)
        assert len(inner.asked) == 2 and len(warm.asked) == 1

    def test_each_question_has_its_own_answers(self, tmp_path):
        inner = Model()
        ask(cached(inner, tmp_path, scope=lambda: ("a", "b")))
        ask(cached(inner, tmp_path, scope=lambda: ("c", "d")))
        ask(cached(inner, tmp_path, scope=lambda: None))  # outside any run
        assert len(inner.asked) == 3
        assert database_path(tmp_path, ("a", "b")).is_file() and database_path(tmp_path, None).is_file()

    def test_it_survives_a_new_process(self, tmp_path):
        ask(cached(Model(), tmp_path))
        later = Model()
        assert ask(cached(later, tmp_path)) == "yes" and later.asked == []

    def test_a_failed_call_is_not_saved_and_the_error_is_raised(self, tmp_path):
        client = cached(Model(fail=True), tmp_path)
        with pytest.raises(RemoteError):
            client.chat(HI)
        wait_for_writes()
        assert PaperDatabase(database_path(tmp_path, QUESTION)).call_count() == 0 and client.misses == 0

    def test_without_a_log_it_still_works(self, tmp_path):
        client = cached(Model(), tmp_path)
        assert ask(client) == ask(client)

    def test_a_replay_only_client_serves_what_is_saved_and_says_so_when_it_is_not(self, tmp_path):
        ask(cached(Model(), tmp_path))
        replay = cached(None, tmp_path, model="m")
        assert ask(replay) == "yes" and replay.spend.dollars == 0
        with pytest.raises(ReplayMiss, match=r"no saved reply.*set a model key"):
            replay.chat([{"role": "user", "content": "never asked"}])

    def test_a_missing_reply_is_not_one_of_the_errors_that_code_carries_on_after(self):
        assert not issubclass(ReplayMiss, RemoteError)  # else the policy would swallow it and look finished

    def test_the_wrapper_is_found_inside_other_wrappers(self, tmp_path):
        inner = cached(Model(), tmp_path)
        stack = AuditedClient(cli.Announced(inner, None))
        assert find_cached(stack) is inner and find_cached(Model()) is None and find_cached(None) is None


class TestTheAuditTrail:
    def run(self, client):
        log = AuditLog()
        with recording(log):
            log.record("run_started", independent_variable="a", dependent_variable="b")
            client.chat(HI)
            wait_for_writes()
            client.chat(HI)
        return log

    def test_a_replayed_call_says_so_and_a_fresh_one_does_not(self, tmp_path):
        log = self.run(AuditedClient(cached(Model(), tmp_path)))
        first, second = [e.data for e in log.events if e.kind == "model_call"]
        assert "replayed" not in first and second["replayed"] is True

    def test_after_a_replay_the_next_fresh_call_is_not_marked(self, tmp_path):
        ask(cached(Model(), tmp_path))  # saved
        audited = AuditedClient(cached(Model(), tmp_path))
        log = AuditLog()
        with recording(log):
            log.record("run_started", independent_variable="a", dependent_variable="b")
            audited.chat(HI)  # a replay
            audited.chat([{"role": "user", "content": "new"}])  # asked afresh
        flags = [e.data.get("replayed", False) for e in log.events if e.kind == "model_call"]
        assert flags == [True, False]

    def test_the_report_says_how_many_were_replays(self, tmp_path):
        log = self.run(AuditedClient(cached(Model(), tmp_path)))
        log.record("run_finished", steps=1, reason="stop_action", final_score=0.0)
        from confoundsearch.logic.audit import AuditTrail

        text = AuditTrail(log.events).to_markdown()
        assert "1 of 2 model calls were answered from saved replies, not asked afresh." in text

    def test_a_trail_with_no_replays_reads_as_before(self):
        from confoundsearch.logic.audit import AuditTrail

        log = AuditLog()
        log.record("run_started", independent_variable="a", dependent_variable="b")
        assert "saved replies" not in AuditTrail(log.events).to_markdown()


class TestTheCommandLine:
    def run_args(self, tmp_path, *more):
        corpus = str(write_corpus(tmp_path))
        return [
            *["run", *COMMON, "--corpus-file", corpus, "--iv", "firm size", "--dv", "profit"],
            *[
                "--cache-dir",
                str(tmp_path / "data"),
                "--out",
                str(tmp_path / "p.html"),
                "--max-steps",
                "40",
                *more,
            ],
        ]

    def scripted(self, monkeypatch):
        scripted = ScriptedClient()
        scripted.model = "m"  # what --model says, as a real client's would
        calls = []
        real = scripted.chat
        scripted.chat = lambda messages: calls.append(1) or real(messages)
        monkeypatch.setattr(cli, "build_client", lambda args: scripted)
        return calls

    def test_the_second_identical_run_asks_the_model_nothing(self, tmp_path, monkeypatch, capsys):
        calls = self.scripted(monkeypatch)
        assert cli.main(self.run_args(tmp_path)) == 0
        first = len(calls)
        assert first > 0 and "answered from saved replies" not in capsys.readouterr().out
        assert cli.main(self.run_args(tmp_path)) == 0
        assert len(calls) == first  # not one more call
        assert (
            f"{first} of {first} model calls were answered from saved replies, at no cost"
            in capsys.readouterr().out
        )

    def test_the_audit_trail_of_the_second_run_marks_every_call_as_a_replay(self, tmp_path, monkeypatch):
        self.scripted(monkeypatch)
        cli.main(self.run_args(tmp_path))
        cli.main(self.run_args(tmp_path))
        events = [json.loads(line) for line in (tmp_path / "p.audit.jsonl").read_text().splitlines()]
        calls = [e["data"] for e in events if e["kind"] == "model_call"]
        assert calls and all(c.get("replayed") for c in calls)

    def test_no_cache_asks_the_model_every_time_and_saves_nothing(self, tmp_path, monkeypatch):
        calls = self.scripted(monkeypatch)
        cli.main(self.run_args(tmp_path, "--no-cache"))
        first = len(calls)
        cli.main(self.run_args(tmp_path, "--no-cache"))
        assert len(calls) == 2 * first and not (tmp_path / "data").exists()

    def test_replay_answers_from_what_is_saved_with_no_model_at_all(self, tmp_path, monkeypatch, capsys):
        self.scripted(monkeypatch)
        cli.main(self.run_args(tmp_path))  # record
        monkeypatch.setattr(cli, "build_client", lambda args: pytest.fail("a replay must not build a client"))
        assert cli.main(self.run_args(tmp_path, "--replay")) == 0
        assert "answered from saved replies" in capsys.readouterr().out

    def test_a_replay_that_misses_stops_loudly_and_keeps_what_it_had(self, tmp_path, capsys):
        code = cli.main(self.run_args(tmp_path, "--replay"))  # nothing was ever saved
        err = capsys.readouterr().err
        assert (
            code == 1
            and "no saved reply" in err
            and "stopped early" in err
            and (tmp_path / "p.html").exists()
        )

    def test_replay_and_no_cache_contradict_each_other(self, tmp_path, capsys):
        assert cli.main(self.run_args(tmp_path, "--replay", "--no-cache")) == 1
        assert "--no-cache" in capsys.readouterr().err

    def test_the_log_records_that_it_was_a_replay(self, tmp_path):
        cli.main(self.run_args(tmp_path, "--replay"))
        started = json.loads((tmp_path / "p.log.jsonl").read_text().splitlines()[0])
        assert started["replay"] is True

    def test_flush_counts_the_model_replies_it_removes(self, tmp_path, capsys):
        folder = tmp_path / "c"
        PaperDatabase(database_path(folder, ("a", "b"))).save_call("k", "m", HI, "x")
        assert cli.main(["flush", "--cache-dir", str(folder)]) == 0
        assert "1 model replies" in capsys.readouterr().out and flush(folder) == (0, 0, 0)


class TestTheDemoWithoutAKey:
    def capture(self, monkeypatch):
        seen = []
        real = cli.main
        monkeypatch.setattr(cli, "main", lambda argv=None: seen.append(list(argv)) or 0)
        return seen, real

    def test_with_no_key_the_curated_run_is_replayed_and_the_user_is_told(
        self, monkeypatch, capsys, tmp_path
    ):
        seen, real = self.capture(monkeypatch)
        assert real(["demo", "--out", str(tmp_path / "d.html")]) == 0
        (argv,) = seen
        assert "--replay" in argv and argv[argv.index("--model") + 1] == RECORDED_MODEL
        assert argv[argv.index("--corpus") + 1] == "demo"  # the shipped example papers
        said = " ".join(capsys.readouterr().err.split())
        assert "You have not run setup yet" in said and "confoundsearch setup" in said

    def test_with_no_key_and_live_it_replays_on_the_sources_whose_searches_ship_saved(
        self, monkeypatch, tmp_path
    ):
        seen, real = self.capture(monkeypatch)
        assert real(["demo", "--live", "--out", str(tmp_path / "d.html")]) == 0
        (argv,) = seen
        assert "--replay" in argv and argv[argv.index("--corpus") + 1] == "semanticscholar,openalex"
        assert argv[argv.index("--confounds") + 1] == "4"

    def test_with_a_key_it_runs_live_and_does_not_replay_only(self, monkeypatch, tmp_path):
        monkeypatch.setenv("OPENAI_API_KEY", "k")
        seen, real = self.capture(monkeypatch)
        real(["demo", "--out", str(tmp_path / "d.html")])
        assert "--replay" not in seen[0]
