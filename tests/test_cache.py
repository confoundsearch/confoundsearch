"""Saved searches: where they live, one database per question, a write queue, one copy of each paper."""

from __future__ import annotations

import contextlib
import contextvars
import getpass
import json
import sqlite3
import tempfile
import threading
from pathlib import Path

import pytest

from confoundsearch.data.cache import (
    ENV_VAR,
    PREFIX,
    SUFFIX,
    WRITES,
    CachedCorpus,
    PaperDatabase,
    WriteQueue,
    cache_dir,
    database_path,
    databases,
    default_dir,
    flush,
    wait_for_writes,
)
from confoundsearch.data.multi import MultiCorpus
from confoundsearch.errors import RemoteError
from confoundsearch.interface import cli
from confoundsearch.logic.audit import AuditLog, recording
from confoundsearch.logic.graph import Passage
from tests.test_cli import COMMON, write_corpus


class Counting:
    def __init__(self, *papers, error=None):
        self.papers, self.error, self.asked = papers, error, []

    def search(self, query, k):
        self.asked.append((query, k))
        if self.error:
            raise self.error
        return list(self.papers)[:k]


PAPER = Passage("p1", "firms and profit", "A Title", "https://example.org/p1")
QUESTION = ("Organizational structure", "Firm performance")


def go(cached, query="q", k=5):
    """Search, then wait for the write queue, so what was saved can be looked at."""
    found = cached.search(query, k)
    wait_for_writes()
    return found


def cached(inner, folder, name="s2", log=None, question=QUESTION):
    return CachedCorpus(inner, folder, name, log, (lambda: question) if question else None)


class TestWhere:
    def test_the_variable_wins_and_may_use_a_tilde(self):
        assert cache_dir({ENV_VAR: "/data/papers"}) == Path("/data/papers")
        assert cache_dir({ENV_VAR: "~/papers"}) == Path.home() / "papers"

    def test_by_default_it_is_a_folder_of_its_own_in_the_temporary_directory(self, monkeypatch):
        monkeypatch.setattr(getpass, "getuser", lambda: "ada")
        assert cache_dir({}) == Path(tempfile.gettempdir()) / "confoundsearch-ada"
        assert cache_dir({"XDG_CACHE_HOME": "/x/cache"}) == cache_dir({})  # not the home cache any more

    def test_the_user_name_is_made_safe_and_may_be_unknown(self, monkeypatch):
        monkeypatch.setattr(getpass, "getuser", lambda: "a b/c")
        assert default_dir().name == "confoundsearch-a_b_c"

        def nobody():
            raise KeyError("no such user")

        monkeypatch.setattr(getpass, "getuser", nobody)
        assert default_dir().name == "confoundsearch-user"

    def test_the_folder_is_made_private(self, tmp_path):
        go(cached(Counting(PAPER), tmp_path / "shared" / "papers"))
        assert (tmp_path / "shared" / "papers").stat().st_mode & 0o777 == 0o700

    def test_with_no_argument_it_reads_the_real_environment(self, monkeypatch, tmp_path):
        monkeypatch.setenv(ENV_VAR, str(tmp_path / "mine"))
        assert cache_dir() == tmp_path / "mine"


class TestOneDatabasePerQuestion:
    def test_each_pair_of_variables_has_its_own_readable_file(self, tmp_path):
        a = database_path(tmp_path, ("Organizational structure", "Firm performance"))
        b = database_path(tmp_path, ("IT investment", "Firm performance"))
        assert a != b and a.parent == tmp_path
        assert a.name.startswith(PREFIX + "organizational-structure--firm-performance-")
        assert a.name.endswith(SUFFIX)

    def test_case_and_spacing_do_not_make_a_new_database(self, tmp_path):
        assert database_path(tmp_path, ("Firm  Size", "profit")) == database_path(
            tmp_path, ("firm size", "PROFIT")
        )

    def test_the_order_matters_and_odd_text_stays_a_safe_name(self, tmp_path):
        assert database_path(tmp_path, ("a", "b")) != database_path(tmp_path, ("b", "a"))
        odd = database_path(tmp_path, ("../../etc/passwd", "x" * 500))
        assert odd.parent == tmp_path and len(odd.name) < 100 and "/" not in odd.name
        assert database_path(tmp_path, ("???", "!!!")).parent == tmp_path  # nothing left to name it by

    def test_a_search_outside_any_run_has_a_general_database(self, tmp_path):
        assert database_path(tmp_path, None).name == f"{PREFIX}unscoped{SUFFIX}"

    def test_a_new_pair_starts_a_new_database_so_nothing_is_shared(self, tmp_path):
        inner = Counting(PAPER)
        go(cached(inner, tmp_path, question=("a", "b")))
        go(cached(inner, tmp_path, question=("a", "b")))  # same pair: answered from its database
        go(cached(inner, tmp_path, question=("c", "d")))  # a new pair: searched again, saved separately
        assert len(inner.asked) == 2 and len(databases(tmp_path)) == 2
        assert PaperDatabase(database_path(tmp_path, ("c", "d"))).counts() == (1, 1)

    def test_with_no_scope_the_general_database_is_used(self, tmp_path):
        go(cached(Counting(PAPER), tmp_path, question=None))
        assert [p.name for p in databases(tmp_path)] == [f"{PREFIX}unscoped{SUFFIX}"]


class TestCachedCorpus:
    def test_a_search_is_made_once_and_the_answer_comes_back_the_same(self, tmp_path):
        inner, events = Counting(PAPER), []
        corpus = cached(inner, tmp_path, log=lambda kind, **d: events.append((kind, d["papers"])))
        first = go(corpus, "Firm  Size")
        second = go(corpus, "firm size")  # case and spacing are not a different question
        assert first == second == [PAPER] and len(inner.asked) == 1
        assert events == [("cache_miss", 1), ("cache_hit", 1)]

    def test_the_file_holds_the_search_and_the_paper(self, tmp_path):
        go(cached(Counting(PAPER), tmp_path))
        (path,) = databases(tmp_path)
        with contextlib.closing(sqlite3.connect(path)) as db:
            assert db.execute("SELECT source, query, k FROM searches").fetchall() == [("s2", "q", 5)]
            assert db.execute("SELECT id, title, url, text, source FROM papers").fetchall() == [
                ("p1", "A Title", "https://example.org/p1", "firms and profit", "s2")
            ]

    def test_a_different_query_size_or_source_is_a_different_search(self, tmp_path):
        inner = Counting(PAPER)
        go(cached(inner, tmp_path), "a")
        go(cached(inner, tmp_path), "b")
        go(cached(inner, tmp_path), "a", 3)
        go(cached(inner, tmp_path, "openalex"), "a")
        assert len(inner.asked) == 4
        assert PaperDatabase(database_path(tmp_path, QUESTION)).counts() == (4, 1)  # one paper, kept once

    def test_it_survives_a_new_process(self, tmp_path):
        go(cached(Counting(PAPER), tmp_path))
        later = Counting()
        assert go(cached(later, tmp_path)) == [PAPER] and later.asked == []

    def test_the_order_of_the_papers_is_kept(self, tmp_path):
        second = Passage("p2", "more text", "Another Title")
        go(cached(Counting(second, PAPER), tmp_path))
        assert go(cached(Counting(), tmp_path)) == [second, PAPER]

    def test_a_failed_search_is_not_saved(self, tmp_path):
        with pytest.raises(RemoteError):
            cached(Counting(error=RemoteError("429")), tmp_path).search("q", 5)
        wait_for_writes()
        assert databases(tmp_path) == []

    def test_a_damaged_database_is_searched_again(self, tmp_path):
        go(cached(Counting(PAPER), tmp_path))
        (path,) = databases(tmp_path)
        path.write_bytes(b"this is not a database")
        inner = Counting(PAPER)
        assert go(cached(inner, tmp_path)) == [PAPER] and len(inner.asked) == 1

    def test_a_search_whose_papers_went_missing_is_searched_again(self, tmp_path):
        go(cached(Counting(PAPER), tmp_path))
        (path,) = databases(tmp_path)
        with contextlib.closing(sqlite3.connect(path)) as db:
            db.execute("DELETE FROM papers")
            db.commit()
        inner = Counting(PAPER)
        assert go(cached(inner, tmp_path)) == [PAPER] and len(inner.asked) == 1

    def test_an_unwritable_place_does_not_stop_the_search(self, tmp_path):
        blocked = tmp_path / "file-not-folder"
        blocked.write_text("x")
        inner = Counting(PAPER)
        assert go(cached(inner, blocked)) == [PAPER]  # cannot save; still answers
        assert PaperDatabase(database_path(blocked, QUESTION)).counts() == (0, 0)

    def test_without_a_log_it_still_works(self, tmp_path):
        corpus = cached(Counting(PAPER), tmp_path)
        assert go(corpus) == go(corpus)


class TestTheWriteQueue:
    def test_writes_happen_in_the_order_they_were_asked_on_one_thread(self):
        queue_, seen, threads = WriteQueue(), [], set()
        for number in range(50):
            queue_.submit(lambda n=number: (seen.append(n), threads.add(threading.get_ident())))
        queue_.wait()
        assert seen == list(range(50)) and len(threads) == 1 and threading.get_ident() not in threads

    def test_a_failing_job_does_not_stop_the_ones_after_it(self):
        queue_, seen = WriteQueue(), []
        queue_.submit(lambda: 1 / 0)
        queue_.submit(lambda: seen.append("after"))
        queue_.wait()
        assert seen == ["after"]

    def test_waiting_with_nothing_queued_returns_at_once(self):
        WriteQueue().wait()
        WRITES.wait()

    def test_a_search_returns_before_its_write_is_made(self, tmp_path):
        gate, inner = threading.Event(), Counting(PAPER)
        WRITES.submit(gate.wait)  # the writer is busy with something slow
        corpus = cached(inner, tmp_path)
        assert corpus.search("q", 5) == [PAPER]  # did not wait for the disk
        assert databases(tmp_path) == []
        gate.set()
        wait_for_writes()
        assert len(databases(tmp_path)) == 1

    def test_many_threads_searching_at_once_all_get_saved(self, tmp_path):
        stores = {f"s{i}": Counting(Passage(f"id{i}", "text", f"Title {i}")) for i in range(6)}
        threads = [
            threading.Thread(
                target=lambda name=name, store=store: [
                    cached(store, tmp_path, name).search(f"query {n}", 5) for n in range(8)
                ]
            )
            for name, store in stores.items()
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        wait_for_writes()
        assert PaperDatabase(database_path(tmp_path, QUESTION)).counts() == (48, 6)


class TestOneCopyOfEachPaper:
    def test_the_same_title_from_two_sources_is_one_paper_with_the_fuller_abstract(self, tmp_path):
        bare = Passage("s2-1", "", "Firm Size and Profit", "")
        full = Passage("oa-9", "A long abstract.", "firm size and profit.", "https://example.org/oa")
        go(cached(Counting(bare), tmp_path, "semanticscholar"), "firm size")
        go(cached(Counting(full), tmp_path, "openalex"), "firm size")
        assert PaperDatabase(database_path(tmp_path, QUESTION)).counts() == (2, 1)
        again = go(cached(Counting(), tmp_path, "semanticscholar"), "firm size")
        # the source that had no abstract now gets the other's text and link; the first id stays
        assert again == [
            Passage("s2-1", "A long abstract.", "Firm Size and Profit", "https://example.org/oa")
        ]

    def test_a_shorter_copy_never_replaces_a_fuller_one(self, tmp_path):
        go(cached(Counting(Passage("a", "the whole abstract", "Same Title", "u1")), tmp_path))
        go(cached(Counting(Passage("b", "short", "same title", "u2")), tmp_path, "openalex"))
        assert go(cached(Counting(), tmp_path, "openalex"))[0].text == "the whole abstract"

    def test_a_missing_title_is_filled_in_from_another_copy(self, tmp_path):
        go(cached(Counting(Passage("a", "text one", "A  Title!", "")), tmp_path))
        go(
            cached(
                Counting(Passage("b", "text one", "a title", "https://example.org/b")), tmp_path, "openalex"
            )
        )
        got = go(cached(Counting(), tmp_path))[0]
        assert got.url == "https://example.org/b" and got.title == "A  Title!"

    def test_papers_with_no_title_are_kept_apart_by_source_and_id(self, tmp_path):
        go(cached(Counting(Passage("1", "x", "")), tmp_path))
        go(cached(Counting(Passage("1", "y", "")), tmp_path, "openalex"))
        assert PaperDatabase(database_path(tmp_path, QUESTION)).counts() == (2, 2)


class TestTheQuestionComesFromTheRun:
    def test_the_variables_are_read_from_the_run_that_is_recording(self):
        log = AuditLog()
        assert cli.current_question() is None  # no run
        with recording(log):
            assert cli.current_question() is None  # a run, but it has not said what it asks yet
            log.record("something_else")  # other events come first or between
            log.record("run_started", independent_variable="a", dependent_variable="b")
            assert cli.current_question() == ("a", "b")
        assert cli.current_question() is None

    def test_searches_in_worker_threads_belong_to_the_same_question(self, tmp_path):
        log = AuditLog()
        seen = {}

        class Probe:
            def __init__(self, name):
                self.name = name

            def search(self, query, k):
                seen[self.name] = cli.current_question()
                return []

        multi = MultiCorpus({n: Probe(n) for n in ("a", "b", "c")})
        with recording(log):
            log.record("run_started", independent_variable="x", dependent_variable="y")
            multi.search("q", 3)
        assert seen == {"a": ("x", "y"), "b": ("x", "y"), "c": ("x", "y")}

    def test_a_context_value_reaches_the_searches(self):
        var = contextvars.ContextVar("v", default="unset")
        got = []

        class Probe:
            def search(self, query, k):
                got.append(var.get())
                return []

        var.set("set")
        MultiCorpus({"a": Probe(), "b": Probe()}).search("q", 1)
        assert got == ["set", "set"]


class TestFlush:
    def fill(self, folder):
        for question in (("a", "b"), ("c", "d")):
            for source in ("s2", "openalex"):
                go(cached(Counting(PAPER), folder, source, question=question))

    def test_it_removes_every_database_and_reports_what_they_held(self, tmp_path):
        self.fill(tmp_path / "papers")
        searches, papers, freed = flush(tmp_path / "papers")
        assert (searches, papers) == (4, 2) and freed > 0
        assert not (tmp_path / "papers").exists()  # nothing else was in it, so the folder goes too

    def test_it_can_empty_just_one_question(self, tmp_path):
        self.fill(tmp_path)
        assert flush(tmp_path, ("a", "b"))[:2] == (2, 1)
        assert databases(tmp_path) == [database_path(tmp_path, ("c", "d"))]
        assert flush(tmp_path, ("a", "b")) == (0, 0, 0)  # already gone

    def test_it_leaves_everything_else_in_the_folder_alone(self, tmp_path):
        self.fill(tmp_path)
        (tmp_path / "thesis.docx").write_text("mine")
        (tmp_path / "other.sqlite3").write_text("someone else's database")
        (
            database_path(tmp_path, ("a", "b")).with_name(
                database_path(tmp_path, ("a", "b")).name + "-journal"
            )
        ).write_text("")
        flush(tmp_path)
        assert sorted(p.name for p in tmp_path.iterdir()) == ["other.sqlite3", "thesis.docx"]

    def test_a_folder_that_is_not_there_is_nothing_to_do(self, tmp_path):
        assert flush(tmp_path / "never-made") == (0, 0, 0)

    def test_it_waits_for_writes_still_queued(self, tmp_path):
        gate = threading.Event()
        WRITES.submit(gate.wait)
        cached(Counting(PAPER), tmp_path).search("q", 5)  # its write is stuck behind the gate
        threading.Timer(0.2, gate.set).start()
        assert flush(tmp_path)[:2] == (1, 1) and databases(tmp_path) == []

    def test_flush_then_search_starts_again(self, tmp_path):
        self.fill(tmp_path)
        flush(tmp_path)
        inner = Counting(PAPER)
        go(cached(inner, tmp_path, question=("a", "b")))
        assert len(inner.asked) == 1


class TestCommandLine:
    def run_args(self, tmp_path, *more):
        return ["run", *COMMON, "--iv", "a", "--dv", "b", "--out", str(tmp_path / "p.html"), *more]

    def test_a_source_is_wrapped_so_its_searches_are_saved_under_the_runs_question(self, tmp_path):
        args = cli.build_parser().parse_args(
            ["serve", *COMMON, "--corpus", "semanticscholar", "--cache-dir", str(tmp_path)]
        )
        corpus = cli.build_corpus(args)
        assert isinstance(corpus, CachedCorpus) and corpus.name == "semanticscholar"
        assert corpus.folder == tmp_path and corpus.scope is cli.current_question

    def test_the_variable_sets_the_default_place(self, tmp_path, monkeypatch):
        monkeypatch.setenv(ENV_VAR, str(tmp_path / "from-env"))
        args = cli.build_parser().parse_args(["serve", *COMMON, "--corpus", "openalex"])
        assert cli.build_corpus(args).folder == tmp_path / "from-env"
        parsed = cli.build_parser().parse_args(self.run_args(tmp_path))
        assert cli.describe_run(parsed)["cache"] == str(tmp_path / "from-env")

    def test_no_cache_searches_every_time_and_a_papers_file_is_never_wrapped(self, tmp_path):
        args = cli.build_parser().parse_args(["serve", *COMMON, "--corpus", "openalex", "--no-cache"])
        assert not isinstance(cli.build_corpus(args), CachedCorpus)
        parsed = cli.build_parser().parse_args(self.run_args(tmp_path, "--no-cache"))
        assert cli.describe_run(parsed)["cache"] is None
        args = cli.build_parser().parse_args(["serve", *COMMON, "--corpus-file", str(write_corpus(tmp_path))])
        assert not isinstance(cli.build_corpus(args), CachedCorpus)

    def test_flush_empties_everything_and_says_what_it_did(self, tmp_path, capsys):
        go(cached(Counting(PAPER), tmp_path / "c"))
        assert cli.main(["flush", "--cache-dir", str(tmp_path / "c")]) == 0
        assert "removed 1 saved searches, 1 papers and 0 model replies" in capsys.readouterr().out
        assert not (tmp_path / "c").exists()

    def test_flush_can_take_one_question(self, tmp_path, capsys):
        go(cached(Counting(PAPER), tmp_path, question=("a", "b")))
        go(cached(Counting(PAPER), tmp_path, question=("c", "d")))
        cli.main(["flush", "--cache-dir", str(tmp_path), "--iv", "a", "--dv", "b"])
        assert "removed 1 saved searches" in capsys.readouterr().out
        assert len(databases(tmp_path)) == 1

    def test_flush_needs_both_variables_or_neither(self, tmp_path, capsys):
        go(cached(Counting(PAPER), tmp_path))
        assert cli.main(["flush", "--cache-dir", str(tmp_path), "--iv", "a"]) == 1
        assert "both --iv and --dv" in capsys.readouterr().err and len(databases(tmp_path)) == 1

    def test_flush_uses_the_variable_by_default(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setenv(ENV_VAR, str(tmp_path))
        go(cached(Counting(PAPER), tmp_path))
        cli.main(["flush"])
        assert f"from {tmp_path}" in capsys.readouterr().out

    def test_the_json_in_a_saved_search_is_plain(self, tmp_path):
        go(cached(Counting(PAPER), tmp_path))
        (path,) = databases(tmp_path)
        with contextlib.closing(sqlite3.connect(path)) as db:
            (keys,) = db.execute("SELECT papers FROM searches").fetchone()
        assert json.loads(keys) == ["a title"]
