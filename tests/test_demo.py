"""The demo papers that ship with the package: present, sound, searchable offline, and rebuildable."""

from __future__ import annotations

import importlib.util
import json
import sqlite3
import sys
from pathlib import Path

import pytest

from confoundsearch.data.cache import CachedCorpus, PaperDatabase, database_path, wait_for_writes
from confoundsearch.data.demo import (
    DEMO_FILE,
    DEMO_QUESTION,
    PREWARMED_NOTE,
    RECORDED_MODEL,
    DemoCorpus,
    seed_demo_cache,
)
from confoundsearch.errors import CorpusError, RemoteError
from confoundsearch.interface import cli
from confoundsearch.logic.graph import Passage
from confoundsearch.model.llm import temperature_for
from tests.test_cli import COMMON

ROOT = Path(__file__).parent.parent


class TestTheShippedFile:
    def test_it_is_there_and_is_a_plain_saved_searches_database(self):
        assert DEMO_FILE.is_file() and DEMO_FILE.stat().st_size < 12_000_000
        db = sqlite3.connect(
            f"{DEMO_FILE.resolve().as_uri()}?mode=ro", uri=True
        )  # never write to the shipped file
        try:
            searches = db.execute("SELECT COUNT(*) FROM searches").fetchone()[0]
            papers = db.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
        finally:
            db.close()
        assert searches >= 40 and papers >= 300

    def test_every_paper_has_an_abstract_a_title_and_one_copy(self):
        db = sqlite3.connect(f"{DEMO_FILE.resolve().as_uri()}?mode=ro", uri=True)
        try:
            rows = db.execute("SELECT key, id, title, text FROM papers").fetchall()
            mode = db.execute("PRAGMA journal_mode").fetchone()[0]
        finally:
            db.close()
        assert mode == "delete"  # no journal files needed, so it opens from a read-only install
        assert all(text.strip() and title.strip() and id_ for _, id_, title, text in rows)
        assert len({key for key, *_ in rows}) == len(rows)

    def test_the_example_set_holds_every_known_study_and_some_distractors(self):
        db = sqlite3.connect(f"{DEMO_FILE.resolve().as_uri()}?mode=ro", uri=True)
        try:
            keys = {k for (k,) in db.execute("SELECT key FROM example_papers")}
            titles = dict(db.execute("SELECT key, title FROM papers").fetchall())
        finally:
            db.close()
        builder = load_builder()
        known = {builder.title_key(t) for t in builder.KNOWN_STUDIES}
        assert len(keys) == 40 and known <= keys  # the ten known studies, padded out with thirty others
        assert all(k in titles for k in keys)


class TestDemoCorpus:
    def test_it_finds_papers_for_a_question_best_first(self):
        found = DemoCorpus().search("organizational structure and firm performance", 5)
        assert 0 < len(found) <= 5 and all(isinstance(p, Passage) and p.text for p in found)
        assert all("structure" in p.title.lower() or "structure" in p.text.lower() for p in found)

    def test_rare_words_count_for_more_than_common_ones_and_the_title_counts_extra(self):
        corpus = DemoCorpus()
        top = corpus.search("environmental uncertainty organizational structure", 3)
        assert any("uncertainty" in p.title.lower() for p in top)  # not drowned out by "of", "and", "firm"
        assert corpus.search("of the and", 5) == []  # only stop words: nothing to go on

    def test_k_limits_and_no_matching_words_find_nothing(self):
        corpus = DemoCorpus()
        assert len(corpus.search("firm performance", 3)) == 3
        assert corpus.search("zzzqqq", 5) == [] and corpus.search("firm", 0) == []

    def test_the_same_question_gives_the_same_answer(self):
        assert DemoCorpus().search("firm size", 5) == DemoCorpus().search("firm size", 5)

    def test_a_missing_file_is_a_clear_error(self, tmp_path):
        with pytest.raises(CorpusError, match="not installed"):
            DemoCorpus(tmp_path / "nope.sqlite3").search("q", 5)

    def test_a_damaged_file_is_a_clear_error(self, tmp_path):
        bad = tmp_path / "bad.sqlite3"
        bad.write_bytes(b"not a database at all")
        with pytest.raises(CorpusError, match="could not be read"):
            DemoCorpus(bad).search("q", 5)

    def test_it_reads_the_file_once(self, tmp_path):
        small = tmp_path / "small.sqlite3"
        PaperDatabase(small).save("s", "q", 5, [Passage("1", "firm size matters", "T", "u")])
        corpus = DemoCorpus(small)
        assert corpus.search("firm", 5)[0].id == "1"
        small.unlink()
        assert corpus.search("firm", 5)[0].id == "1"  # already in memory


class TestCommandLine:
    def test_demo_is_a_source_that_needs_no_keys_and_is_never_saved_or_merged(self):
        args = cli.build_parser().parse_args(["serve", *COMMON, "--corpus", "demo"])
        assert isinstance(cli.build_corpus(args), DemoCorpus)
        assert cli.corpus_names(args) == ["demo"]

    def test_the_default_does_not_include_it(self):
        args = cli.build_parser().parse_args(["serve", *COMMON])
        assert "demo" not in cli.corpus_names(args)

    def test_it_can_be_searched_alongside_another_source(self):
        args = cli.build_parser().parse_args(["serve", *COMMON, "--corpus", "demo,openalex"])
        assert list(cli.build_corpus(args).stores) == ["demo", "openalex"]


def load_builder():
    spec = importlib.util.spec_from_file_location("build_demo", ROOT / "devops" / "tools" / "build_demo.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["build_demo"] = module
    spec.loader.exec_module(module)
    return module


class Stub:
    def __init__(self, fail=False):
        self.fail = fail

    def search(self, query, k):
        if self.fail:
            raise RemoteError("429")
        return [Passage("x1", "an abstract", f"A paper about {query}", "u")]


class TestTheBuilder:
    def test_it_lists_the_searches_a_default_run_makes_first_for_each_confound(self):
        builder = load_builder()
        iv, dv = "Organizational structure", "Firm performance"
        exact = builder.exact_queries(iv, dv)
        assert exact[:2] == ["firm size Organizational structure", "firm size Firm performance"]
        assert len(exact) == 2 * len(builder.CONFOUNDS) and len(set(exact)) == len(exact)

    def test_and_a_broader_set_that_gives_the_offline_demo_its_papers(self):
        builder = load_builder()
        broad = builder.broad_queries("Organizational structure", "Firm performance")
        assert broad[0] == "Organizational structure Firm performance"
        assert "firm size Firm performance" in broad and len(set(broad)) == len(broad)

    def test_it_writes_a_database_of_what_it_found(self, tmp_path, monkeypatch, capsys):
        builder = load_builder()
        monkeypatch.setattr(builder, "SemanticScholarCorpus", lambda *a, **k: Stub())
        monkeypatch.setattr(builder, "OpenAlexCorpus", lambda *a, **k: Stub())
        out = tmp_path / "demo.sqlite3"
        out.write_bytes(b"an older build")  # replaced, not added to
        assert builder.main(["--out", str(out)]) == 0
        iv, dv = builder.DEMO_QUESTION
        wanted = len(builder.exact_queries(iv, dv)) + len(builder.broad_queries(iv, dv))
        searches, papers = PaperDatabase(out).counts()
        assert searches == 2 * wanted and papers >= 1
        assert "(0 of" in capsys.readouterr().out
        db = sqlite3.connect(f"{out.resolve().as_uri()}?mode=ro", uri=True)
        try:
            sizes = {k for (k,) in db.execute("SELECT DISTINCT k FROM searches")}
        finally:
            db.close()
        assert sizes == {builder.RESULTS, builder.BROAD}  # three per search, as a default run asks for

    def test_a_build_with_many_failures_is_not_kept_quietly(self, tmp_path, monkeypatch, capsys):
        builder = load_builder()
        monkeypatch.setattr(builder, "SemanticScholarCorpus", lambda *a, **k: Stub(fail=True))
        monkeypatch.setattr(builder, "OpenAlexCorpus", lambda *a, **k: Stub())
        out = tmp_path / "demo.sqlite3"
        assert builder.main(["--out", str(out)]) == 1  # half the searches failed
        assert "nothing written" in capsys.readouterr().err and not out.exists()

    def test_a_few_failures_are_tolerated(self, tmp_path, monkeypatch):
        builder = load_builder()
        calls = []

        class Flaky(Stub):
            def search(self, query, k):
                calls.append(query)
                if len(calls) == 3:
                    raise RemoteError("429")
                return super().search(query, k)

        monkeypatch.setattr(builder, "SemanticScholarCorpus", lambda *a, **k: Flaky())
        monkeypatch.setattr(builder, "OpenAlexCorpus", lambda *a, **k: Stub())
        assert builder.main(["--out", str(tmp_path / "demo.sqlite3")]) == 0

    def test_with_models_it_also_records_real_runs_at_the_default_and_the_demo_setting(
        self, tmp_path, monkeypatch
    ):
        builder = load_builder()
        monkeypatch.setattr(builder, "SemanticScholarCorpus", lambda *a, **k: Stub())
        monkeypatch.setattr(builder, "OpenAlexCorpus", lambda *a, **k: Stub())
        ran = []
        monkeypatch.setattr(cli, "main", lambda argv=None: ran.append(list(argv)) or 0)
        assert builder.main(["--out", str(tmp_path / "demo.sqlite3"), "--models", "m1", "m2"]) == 0
        pairs = [(a[a.index("--model") + 1], a[a.index("--confounds") + 1]) for a in ran]
        assert pairs == [("m1", "8"), ("m1", "4"), ("m2", "8"), ("m2", "4")]  # `run`'s default, then `demo`'s
        assert all(a[0] == "run" and "--cache-dir" in a for a in ran)
        assert len({a[a.index("--cache-dir") + 1] for a in ran}) == 1  # one data directory for all
        assert len({a[a.index("--out") + 1] for a in ran}) == 4  # a page each, so none overwrites another

    def test_the_confound_counts_can_be_chosen(self, tmp_path, monkeypatch):
        builder = load_builder()
        monkeypatch.setattr(builder, "SemanticScholarCorpus", lambda *a, **k: Stub())
        monkeypatch.setattr(builder, "OpenAlexCorpus", lambda *a, **k: Stub())
        ran = []
        monkeypatch.setattr(cli, "main", lambda argv=None: ran.append(list(argv)) or 0)
        builder.main(["--out", str(tmp_path / "d.sqlite3"), "--models", "m", "--confounds", "5"])
        assert [a[a.index("--confounds") + 1] for a in ran] == ["5"]

    def test_clean_removes_emails_and_leaves_one_plain_file(self, tmp_path):
        builder = load_builder()
        at = "@"
        path = tmp_path / "d.sqlite3"
        PaperDatabase(path).save(
            "s", "q", 3, [Passage("1", f"mail x{at}example.org now", f"T y{at}example.org", "u")]
        )
        assert builder.clean(path) == (1, 1)
        db = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
        try:
            text, title = db.execute("SELECT text, title FROM papers").fetchone()
            mode = db.execute("PRAGMA journal_mode").fetchone()[0]
        finally:
            db.close()
        assert (text, title, mode) == ("mail x now", "T y", "delete") or (
            "@" not in text + title and mode == "delete"
        )

    def test_scrubbing_papers_counts_them_and_clean_drops_answers_recorded_before_it(self, tmp_path):
        builder = load_builder()
        at = "@"
        path = tmp_path / "d.sqlite3"
        database = PaperDatabase(path)
        database.save("s", "q", 3, [Passage("1", f"mail x{at}example.org", "T", "u")])
        database.save_call("old", "m", [{"role": "user", "content": f"x{at}example.org"}], "yes")
        database.save_call("new", "m", [{"role": "user", "content": "no address"}], "yes")
        wait_for_writes()
        assert builder.scrub_papers(path) == 1 and builder.scrub_papers(path) == 0
        builder.clean(path)
        db = sqlite3.connect(path)
        try:
            kept = [row[0] for row in db.execute("SELECT key FROM model_calls")]
        finally:
            db.close()
        assert kept == ["new"]


class TestThePrewarmedCache:
    def source(self, tmp_path):
        path = tmp_path / "shipped.sqlite3"
        PaperDatabase(path).save(
            "semanticscholar", "firm size Organizational structure", 3, [Passage("1", "t", "T", "u")]
        )
        return path

    def test_the_example_question_gets_the_saved_searches_copied_in_once(self, tmp_path):
        folder, source = tmp_path / "data", self.source(tmp_path)
        assert seed_demo_cache(folder, ("organizational  STRUCTURE", "firm performance"), source) == "seeded"
        target = database_path(folder, DEMO_QUESTION)
        assert target.read_bytes() == source.read_bytes()
        assert target.parent.stat().st_mode & 0o777 == 0o700
        assert seed_demo_cache(folder, DEMO_QUESTION, source) == "present"

    def test_a_data_directory_that_already_has_it_is_never_overwritten(self, tmp_path):
        folder, source = tmp_path / "data", self.source(tmp_path)
        seed_demo_cache(folder, DEMO_QUESTION, source)
        PaperDatabase(database_path(folder, DEMO_QUESTION)).save(
            "s", "mine", 3, [Passage("2", "x", "Mine", "u")]
        )
        assert seed_demo_cache(folder, DEMO_QUESTION, source) == "present"
        assert PaperDatabase(database_path(folder, DEMO_QUESTION)).counts() == (2, 2)

    def test_any_other_question_gets_nothing(self, tmp_path):
        assert (
            seed_demo_cache(tmp_path / "data", ("IT investment", "Firm performance"), self.source(tmp_path))
            is None
        )
        assert not (tmp_path / "data").exists()

    def test_without_the_shipped_file_or_a_place_to_write_it_the_run_just_goes_on(self, tmp_path):
        assert seed_demo_cache(tmp_path / "data", DEMO_QUESTION, tmp_path / "missing.sqlite3") is None
        blocked = tmp_path / "file-not-folder"
        blocked.write_text("x")
        assert seed_demo_cache(blocked, DEMO_QUESTION, self.source(tmp_path)) is None

    def test_the_note_is_polite_and_says_it_is_a_stub(self):
        flat = " ".join(PREWARMED_NOTE.split())
        assert "pre-warmed cache" in flat and "stub" in flat and "instantly" in flat
        assert "any other question" in PREWARMED_NOTE

    def run_args(self, tmp_path, *more, iv="Organizational structure", dv="Firm performance"):
        return [
            *["run", *COMMON, "--iv", iv, "--dv", dv, "--out", str(tmp_path / "p.html")],
            *["--cache-dir", str(tmp_path / "data"), "--max-steps", "2", *more],
        ]

    def test_running_the_example_question_says_so_and_pre_warms_the_data_directory(
        self, tmp_path, monkeypatch, capsys, env
    ):
        from tests.test_cli import fake_backend

        monkeypatch.setattr(cli, "build_backend", fake_backend(env))
        assert cli.main(self.run_args(tmp_path)) == 0
        assert "pre-warmed cache" in " ".join(capsys.readouterr().err.split())  # the note wraps over lines
        assert database_path(tmp_path / "data", DEMO_QUESTION).is_file()
        events = [json.loads(line) for line in (tmp_path / "p.log.jsonl").read_text().splitlines()]
        assert [e["state"] for e in events if e["event"] == "prewarmed_cache"] == ["seeded"]
        assert cli.main(self.run_args(tmp_path)) == 0  # the second time it is already there; still said
        assert "pre-warmed cache" in " ".join(capsys.readouterr().err.split())

    def test_no_note_for_other_questions_or_without_the_saved_searches(
        self, tmp_path, monkeypatch, capsys, env
    ):
        from tests.test_cli import fake_backend

        monkeypatch.setattr(cli, "build_backend", fake_backend(env))
        for more, kwargs in [((), {"iv": "IT investment"}), (("--no-cache",), {})]:
            assert cli.main(self.run_args(tmp_path, *more, **kwargs)) == 0
            assert "pre-warmed" not in capsys.readouterr().err
        assert not (tmp_path / "data").exists()  # nothing was copied in

    def test_the_shipped_example_is_copied_in_quietly_when_the_papers_are_the_shipped_ones(
        self, tmp_path, monkeypatch, capsys, env
    ):
        from tests.test_cli import fake_backend

        monkeypatch.setattr(cli, "build_backend", fake_backend(env))
        assert cli.main(self.run_args(tmp_path, "--corpus", "demo")) == 0
        assert "pre-warmed" not in capsys.readouterr().err  # the curated note is the demo command's to print
        assert database_path(tmp_path / "data", DEMO_QUESTION).is_file()  # but the saved answers are there

    def test_the_note_never_goes_to_standard_output(self, tmp_path, monkeypatch, capsys, env):
        from tests.test_cli import fake_backend

        monkeypatch.setattr(cli, "build_backend", fake_backend(env))
        cli.main([*self.run_args(tmp_path), "--json"])
        captured = capsys.readouterr()
        assert json.loads(captured.out)["question"] and "pre-warmed" in " ".join(captured.err.split())


class TestTheShippedSavedSearches:
    """What the package ships must really answer a default run's searches from the data directory."""

    def test_the_searches_a_default_run_makes_first_are_all_there_for_both_sources(self, tmp_path):
        builder = load_builder()
        assert seed_demo_cache(tmp_path, DEMO_QUESTION) == "seeded"
        iv, dv = DEMO_QUESTION

        class Nothing:
            def search(self, query, k):
                raise AssertionError(f"went to the source for {query!r}")

        for name in ("semanticscholar", "openalex"):
            warmed = CachedCorpus(Nothing(), tmp_path, name, scope=lambda: DEMO_QUESTION)
            for query in builder.exact_queries(iv, dv):
                assert warmed.search(query, builder.RESULTS), (name, query)

    def test_the_size_is_modest_and_the_searches_are_many(self):
        db = sqlite3.connect(f"{DEMO_FILE.resolve().as_uri()}?mode=ro", uri=True)
        try:
            searches = db.execute("SELECT COUNT(*) FROM searches").fetchone()[0]
        finally:
            db.close()
        assert DEMO_FILE.stat().st_size < 12_000_000 and searches >= 200


class TestScrubbing:
    def test_emails_are_removed_from_what_is_shipped(self):
        builder = load_builder()
        at = "@"  # built here so this file holds no address of its own
        first, second, third = f"a.b+c{at}example.com", f"d{at}example.org", f"x{at}example.net"
        assert builder.scrub(f"Contact {first} or {second} for data.") == "Contact  or  for data."
        assert builder.scrub("no address here") == "no address here"
        kept = builder.scrubbed([Passage("1", f"write to {third}", f"T {third}", "u")])
        assert kept == [Passage("1", "write to ", "T ", "u")]

    def test_the_shipped_papers_hold_no_email_address(self):
        builder = load_builder()
        db = sqlite3.connect(f"{DEMO_FILE.resolve().as_uri()}?mode=ro", uri=True)
        try:
            texts = [t for row in db.execute("SELECT title, text, url FROM papers") for t in row]
        finally:
            db.close()
        assert not [t for t in texts if builder.EMAIL.search(t)]


class TestTheDemoCommand:
    KEYS = ("OPENAI_API_KEY", "OPENROUTER_API_KEY", "OPENAI_BASE_URL")

    @pytest.fixture(autouse=True)
    def no_keys(self, monkeypatch):
        for name in self.KEYS:
            monkeypatch.delenv(name, raising=False)

    def parse(self, *argv):
        return cli.build_parser().parse_args(["demo", *argv])

    def test_the_defaults_are_a_short_example_run(self):
        args = self.parse()
        assert (args.out, args.confounds, args.max_budget, args.no_open, args.live) == (
            Path("confoundsearch-demo.html"),
            None,  # 3 for the curated example, 4 with --live
            1.0,
            False,
            False,
        )

    def test_the_model_follows_the_key_that_is_set(self, monkeypatch):
        assert cli.demo_model(self.parse()) is None
        assert cli.demo_model(self.parse("--model", "mine")) == "mine"
        assert cli.demo_model(self.parse("--api-key", "k")) == RECORDED_MODEL
        monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:1/v1")
        assert cli.demo_model(self.parse()) == RECORDED_MODEL
        monkeypatch.delenv("OPENAI_BASE_URL")
        monkeypatch.setenv("OPENAI_API_KEY", "k")
        assert cli.demo_model(self.parse()) == RECORDED_MODEL
        monkeypatch.setenv("OPENROUTER_API_KEY", "k")  # both set: OpenAI's own key is used
        assert cli.demo_model(self.parse()) == RECORDED_MODEL
        monkeypatch.delenv("OPENAI_API_KEY")
        assert cli.demo_model(self.parse()) == f"openai/{RECORDED_MODEL}"  # only OpenRouter
        monkeypatch.setenv("OPENAI_API_KEY", "k")
        assert cli.demo_model(self.parse("--openrouter")) == f"openai/{RECORDED_MODEL}"

    def test_by_default_it_is_the_curated_example_on_the_shipped_papers(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setenv("OPENAI_API_KEY", "k")
        seen = []
        real = cli.main
        monkeypatch.setattr(cli, "main", lambda argv=None: seen.append(list(argv)) or 0)
        assert real(["demo", "--out", str(tmp_path / "d.html")]) == 0
        (argv,) = seen
        assert argv[:5] == ["run", "--model", RECORDED_MODEL, "--iv", "Organizational structure"]
        assert argv[argv.index("--corpus") + 1] == "demo"  # the shipped example papers: no internet
        assert (
            argv[argv.index("--confounds") + 1] == "3" and "--replay" not in argv
        )  # a key: it can ask live too
        assert (
            argv[argv.index("--dv") + 1] == "Firm performance"
            and argv[argv.index("--max-budget") + 1] == "1.0"
        )
        assert argv[-1] == "--review"  # opens the result
        said = " ".join(capsys.readouterr().err.split())
        assert "curated example" not in said and "stub" not in said

    def test_live_searches_the_usual_sources_and_asks_for_four_confounds(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setenv("OPENAI_API_KEY", "k")
        seen = []
        real = cli.main
        monkeypatch.setattr(cli, "main", lambda argv=None: seen.append(list(argv)) or 0)
        assert real(["demo", "--live", "--out", str(tmp_path / "d.html"), "--no-open"]) == 0
        (argv,) = seen
        assert (
            "--corpus" not in argv and argv[argv.index("--confounds") + 1] == "4"
        )  # its cache is pre-warmed
        assert "curated" not in capsys.readouterr().err  # no curated note: these are real searches

    def test_the_number_of_confounds_can_be_chosen(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "k")
        seen = []
        real = cli.main
        monkeypatch.setattr(cli, "main", lambda argv=None: seen.append(list(argv)) or 0)
        real(["demo", "--confounds", "5", "--no-open"])
        assert seen[0][seen[0].index("--confounds") + 1] == "5"

    def test_options_for_the_model_and_not_opening_are_passed_on(self, monkeypatch):
        seen = []
        real = cli.main
        monkeypatch.setattr(cli, "main", lambda argv=None: seen.append(list(argv)) or 0)
        real(
            [
                "demo",
                "--no-open",
                "--openrouter",
                "--api-key",
                "k",
                "--base-url",
                "http://h/v1",
                "--model",
                "m",
            ]
        )
        (argv,) = seen
        assert "--review" not in argv and "--openrouter" in argv
        assert (
            argv[argv.index("--api-key") + 1] == "k" and argv[argv.index("--base-url") + 1] == "http://h/v1"
        )

    def test_it_really_runs_and_opens_the_page(self, monkeypatch, tmp_path, capsys, env):
        from tests.test_cli import fake_backend

        monkeypatch.setenv("OPENAI_API_KEY", "k")
        monkeypatch.setattr(cli, "build_backend", fake_backend(env))
        opened = []
        monkeypatch.setattr(cli, "open_page", lambda page: opened.append(page))
        out = tmp_path / "d.html"
        assert cli.main(["demo", "--out", str(out)]) == 0
        assert out.exists() and opened == [out]
        assert cli.main(["demo", "--out", str(out), "--no-open"]) == 0 and len(opened) == 1


class TestResumingABuild:
    def test_reuse_continues_from_an_earlier_folder_without_searching_again(self, tmp_path, monkeypatch):
        builder = load_builder()
        asked = []

        class Recording(Stub):
            def search(self, query, k):
                asked.append(query)
                return super().search(query, k)

        monkeypatch.setattr(builder, "SemanticScholarCorpus", lambda *a, **k: Recording())
        monkeypatch.setattr(builder, "OpenAlexCorpus", lambda *a, **k: Recording())
        work = tmp_path / "earlier"
        PaperDatabase(database_path(work, DEMO_QUESTION)).save(
            "semanticscholar", "q", 3, [Passage("1", "x", "T", "u")]
        )
        out = tmp_path / "demo.sqlite3"
        assert builder.main(["--out", str(out), "--reuse", str(work)]) == 0
        assert asked == [] and PaperDatabase(out).counts() == (1, 1)


class TestRecordingTheModelsAnswers:
    def fake_main(self, tmp_path, fresh_by_pass, replay_code=0):
        """A stand-in for `run` that writes a log saying how much it had to do afresh."""
        calls = []

        def main(argv=None):
            argv = list(argv)
            calls.append(argv)
            log = Path(argv[argv.index("--log") + 1])
            if "--replay" in argv:
                log.write_text(json.dumps({"event": "model_cache_hit"}) + "\n")
                return replay_code
            fresh = fresh_by_pass[
                min(len([a for a in calls if "--replay" not in a]) - 1, len(fresh_by_pass) - 1)
            ]
            events = [{"event": "cache_miss"}] * fresh + [{"event": "model_cache_hit"}, {"event": "step"}]
            log.write_text("".join(json.dumps(e) + "\n" for e in events))
            return 0

        return main, calls

    def test_it_repeats_until_a_pass_finds_everything_saved_then_proves_it_replays(
        self, tmp_path, monkeypatch
    ):
        builder = load_builder()
        main, calls = self.fake_main(tmp_path, [5, 2, 0])
        monkeypatch.setattr(cli, "main", main)
        assert builder.record_demo(tmp_path) is True
        assert len(calls) == 4 and "--replay" in calls[-1] and all("--replay" not in c for c in calls[:-1])
        first = calls[0]
        assert (
            first[first.index("--model") + 1] == RECORDED_MODEL
            and first[first.index("--confounds") + 1] == "4"
        )
        assert (
            first[first.index("--corpus") + 1] == "semanticscholar,openalex"
        )  # the sources `demo` replays on
        assert first[first.index("--cache-dir") + 1] == str(tmp_path)

    def test_it_gives_up_if_the_run_never_settles(self, tmp_path, monkeypatch):
        builder = load_builder()
        main, calls = self.fake_main(tmp_path, [3])
        monkeypatch.setattr(cli, "main", main)
        assert builder.record_demo(tmp_path, passes=3) is False and len(calls) == 3  # never got to the replay

    def test_a_replay_that_fails_is_a_failed_recording(self, tmp_path, monkeypatch):
        builder = load_builder()
        main, _ = self.fake_main(tmp_path, [0], replay_code=1)
        monkeypatch.setattr(cli, "main", main)
        assert builder.record_demo(tmp_path) is False

    def test_a_run_that_errors_is_not_counted_as_settled(self, tmp_path, monkeypatch):
        builder = load_builder()
        monkeypatch.setattr(
            cli, "main", lambda argv=None: Path(argv[argv.index("--log") + 1]).write_text("") or 1
        )
        assert builder.record_demo(tmp_path, passes=2) is False

    def test_fresh_work_counts_searches_and_model_calls_made_afresh(self, tmp_path):
        builder = load_builder()
        log = tmp_path / "l.jsonl"
        events = [
            "cache_miss",
            "cache_hit",
            "model_cache_miss",
            "model_cache_hit",
            "step",
            "model_cache_miss",
        ]
        log.write_text("".join(json.dumps({"event": e}) + "\n" for e in events))
        assert builder.fresh_work(log) == 3

    def test_with_record_the_build_keeps_the_file_only_if_the_replay_proves_out(self, tmp_path, monkeypatch):
        builder = load_builder()
        monkeypatch.setattr(builder, "SemanticScholarCorpus", lambda *a, **k: Stub())
        monkeypatch.setattr(builder, "OpenAlexCorpus", lambda *a, **k: Stub())
        monkeypatch.setattr(builder, "record_demo", lambda work: True)
        good = tmp_path / "good.sqlite3"
        assert builder.main(["--out", str(good), "--record"]) == 0 and good.exists()
        monkeypatch.setattr(builder, "record_demo", lambda work: False)
        bad = tmp_path / "bad.sqlite3"
        assert builder.main(["--out", str(bad), "--record"]) == 1 and not bad.exists()


class TestTheShippedRecording:
    """What ships must replay the whole example run by itself: if a prompt or the code changes, this fails."""

    def argv(self, tmp_path, *more):
        iv, dv = DEMO_QUESTION
        return [
            *[
                "run",
                "--corpus",
                "semanticscholar,openalex",
                "--model",
                RECORDED_MODEL,
                "--iv",
                iv,
                "--dv",
                dv,
            ],
            *[
                "--confounds",
                "4",
                "--cache-dir",
                str(tmp_path / "data"),
                "--out",
                str(tmp_path / "p.html"),
                *more,
            ],
        ]

    def test_the_example_run_replays_completely_with_no_model_and_no_network(self, tmp_path, monkeypatch):
        def no_network(*args, **kwargs):
            raise AssertionError("went to the network")

        monkeypatch.setattr(cli, "urllib_transport", no_network)
        monkeypatch.setattr(
            cli, "build_client", lambda args: pytest.fail("a replay must not build a model client")
        )
        assert cli.main(self.argv(tmp_path, "--replay")) == 0
        events = [json.loads(line)["event"] for line in (tmp_path / "p.log.jsonl").read_text().splitlines()]
        assert events.count("model_cache_hit") >= 5 and "model_cache_miss" not in events
        assert "cache_miss" not in events and events.count("cache_hit") >= 20  # every search was saved too

    def test_demo_with_no_keys_at_all_works_and_says_so(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(cli, "urllib_transport", lambda *a, **k: pytest.fail("went to the network"))
        assert cli.main(["demo", "--no-open", "--out", str(tmp_path / "d.html")]) == 0
        captured = capsys.readouterr()
        said = " ".join(captured.err.split())
        assert "No language model key found" in said and "curated example" not in said
        assert "model calls were answered from saved replies, at no cost" in captured.out
        assert (tmp_path / "d.html").exists() and (tmp_path / "d.audit.jsonl").exists()

    def test_the_curated_demo_finds_and_supports_a_known_study(self, tmp_path):
        """The point of the curated example: well-known studies show up as accepted evidence."""
        from confoundsearch.logic.audit import AuditTrail

        assert cli.main(["demo", "--no-open", "--out", str(tmp_path / "d.html")]) == 0
        text = AuditTrail.load(tmp_path / "d.audit.jsonl").to_markdown()
        assert "market competition" in text and "The Flattening Firm and Product Market Competition" in text
        assert "accepted at step" in text
        assert "Managing with Style: The Effect of Managers on Firm Policies" in text  # CEO style, too

    def test_its_audit_trail_is_intact_and_says_the_answers_were_replays(self, tmp_path):
        cli.main(self.argv(tmp_path, "--replay"))
        from confoundsearch.logic.audit import AuditTrail

        trail = AuditTrail.load(tmp_path / "p.audit.jsonl")
        assert trail.intact and trail.complete
        assert "model calls were answered from saved replies, not asked afresh." in trail.to_markdown()


class TestTheExampleSet:
    def database(self, tmp_path, example):
        path = tmp_path / "d.sqlite3"
        db = PaperDatabase(path)
        db.save(
            "s",
            "q",
            3,
            [
                Passage("1", "managers and style at the firm", "On Style"),
                Passage("2", "firms and competition in markets", "On Competition"),
                Passage("3", "something else about weather", "Weather"),
            ],
        )
        if example:
            con = sqlite3.connect(path)
            con.executemany("INSERT INTO example_papers VALUES (?)", [(k,) for k in example])
            con.commit()
            con.close()
        return path

    def test_with_an_example_set_only_those_papers_are_searched(self, tmp_path):
        path = self.database(tmp_path, ["on style"])
        assert [p.id for p in DemoCorpus(path).search("style competition", 5)] == ["1"]

    def test_without_one_every_paper_is_searched(self, tmp_path):
        path = self.database(tmp_path, [])
        assert sorted(p.id for p in DemoCorpus(path).search("style competition", 5)) == ["1", "2"]

    def test_a_file_from_before_example_sets_still_works(self, tmp_path):
        path = tmp_path / "old.sqlite3"
        con = sqlite3.connect(path)
        con.execute("CREATE TABLE papers (key TEXT PRIMARY KEY, id TEXT, title TEXT, url TEXT, text TEXT)")
        con.execute("INSERT INTO papers VALUES ('k', '9', 'T', 'u', 'firm text')")
        con.commit()
        con.close()
        assert [p.id for p in DemoCorpus(path).search("firm", 5)] == ["9"]  # no example_papers table at all


class TestCurating:
    def paper(self, title, text="an abstract about firms and structure"):
        return Passage(title[:6].replace(" ", ""), text, title, "u")

    def test_it_adds_the_known_studies_and_pads_the_set_with_other_papers(self, tmp_path, monkeypatch):
        builder = load_builder()
        path = tmp_path / "d.sqlite3"
        others = [
            Passage(f"o{i}", f"organizational structure study number {i}", f"Other study {i}")
            for i in range(40)
        ]
        PaperDatabase(path).save("s", "q", 40, others)

        class Source:
            def search(self, title, k):
                return [self_.paper(title, "abstract mentioning x@example.org here")]

        self_ = self
        size = builder.curate(path, Source(), distractors=5)
        assert size == len(builder.KNOWN_STUDIES) + 5
        con = sqlite3.connect(path)
        try:
            keys = {k for (k,) in con.execute("SELECT key FROM example_papers")}
            text = con.execute(
                "SELECT text FROM papers WHERE title = ?", (builder.KNOWN_STUDIES[0],)
            ).fetchone()[0]
        finally:
            con.close()
        assert {builder.title_key(t) for t in builder.KNOWN_STUDIES} <= keys
        assert "@" not in text  # emails are scrubbed from these abstracts too
        assert curate_twice(builder, path, Source(), size) == size  # running it again changes nothing

    def test_a_study_the_source_cannot_find_stops_the_build(self, tmp_path):
        builder = load_builder()
        PaperDatabase(tmp_path / "d.sqlite3").save("s", "q", 1, [Passage("1", "x", "T")])

        class Nothing:
            def search(self, title, k):
                return []

        with pytest.raises(RemoteError, match="could not find the known study"):
            builder.curate(tmp_path / "d.sqlite3", Nothing())

    def test_the_curated_confounds_are_saved_as_the_models_answer_to_the_very_prompt_a_run_sends(
        self, tmp_path
    ):
        builder = load_builder()
        from confoundsearch.data.cache import call_key
        from confoundsearch.data.demo import CURATED_CONFOUNDS
        from confoundsearch.model.llm import LLMConfounds

        builder.save_curated_confounds(tmp_path)
        sent = []

        class Spy:
            model, temperature = RECORDED_MODEL, temperature_for(RECORDED_MODEL)

            def chat(self, messages):
                sent.append(list(messages))
                return "unused"

        LLMConfounds(Spy(), len(CURATED_CONFOUNDS))(*DEMO_QUESTION)
        saved = PaperDatabase(database_path(tmp_path, DEMO_QUESTION)).load_call(
            call_key(RECORDED_MODEL, temperature_for(RECORDED_MODEL), sent[0])
        )
        assert json.loads(saved) == {"confounds": list(CURATED_CONFOUNDS)}

    def test_recording_a_curated_run_reads_the_file_being_built_and_asks_for_three(
        self, tmp_path, monkeypatch
    ):
        builder = load_builder()
        seen = []

        def main(argv=None):
            argv = list(argv)
            seen.append((argv, cli.DemoCorpus().path))  # which file would the run read?
            Path(argv[argv.index("--log") + 1]).write_text(json.dumps({"event": "step"}) + "\n")
            return 0

        monkeypatch.setattr(cli, "main", main)
        assert builder.record_demo(tmp_path, curated=True) is True
        argv, path = seen[0]
        assert argv[argv.index("--corpus") + 1] == "demo" and argv[argv.index("--confounds") + 1] == "3"
        assert path == database_path(tmp_path, DEMO_QUESTION)  # the work file, not the shipped one
        assert cli.DemoCorpus is DemoCorpus  # and the real corpus is put back

    def test_with_curate_the_build_stops_if_the_curated_run_does_not_replay(self, tmp_path, monkeypatch):
        builder = load_builder()
        monkeypatch.setattr(builder, "SemanticScholarCorpus", lambda *a, **k: Stub())
        monkeypatch.setattr(builder, "OpenAlexCorpus", lambda *a, **k: Stub())
        monkeypatch.setattr(builder, "curate", lambda db, source: 40)
        monkeypatch.setattr(builder, "record_demo", lambda work, curated=False: curated)
        good = tmp_path / "good.sqlite3"
        assert builder.main(["--out", str(good), "--curate"]) == 0 and good.exists()
        monkeypatch.setattr(builder, "record_demo", lambda work, curated=False: False)
        bad = tmp_path / "bad.sqlite3"
        assert builder.main(["--out", str(bad), "--curate"]) == 1 and not bad.exists()


def curate_twice(builder, path, source, size):
    return builder.curate(path, source, distractors=5)
