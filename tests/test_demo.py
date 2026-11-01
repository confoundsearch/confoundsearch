"""The demo papers that ship with the package: present, sound, searchable offline, and rebuildable."""

from __future__ import annotations

import importlib.util
import sqlite3
import sys
from pathlib import Path

import pytest

from confoundsearch.data.cache import PaperDatabase
from confoundsearch.data.demo import DEMO_FILE, DEMO_QUESTION, DemoCorpus
from confoundsearch.errors import CorpusError, RemoteError
from confoundsearch.interface import cli
from confoundsearch.logic.graph import Passage
from tests.test_cli import COMMON

ROOT = Path(__file__).parent.parent


class TestTheShippedFile:
    def test_it_is_there_and_is_a_plain_saved_searches_database(self):
        assert DEMO_FILE.is_file() and DEMO_FILE.stat().st_size < 3_000_000
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

    def test_it_holds_the_question_and_its_usual_confounds(self):
        assert DEMO_QUESTION == ("Organizational structure", "Firm performance")
        corpus = DemoCorpus()
        for confound in ("firm size", "IT investment", "environmental uncertainty"):
            assert corpus.search(f"{confound} firm performance", 5), confound


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
    def test_it_asks_about_the_question_and_each_confound_in_several_wordings(self):
        builder = load_builder()
        asked = builder.queries("Organizational structure", "Firm performance")
        assert asked[0] == "Organizational structure Firm performance"
        assert "firm size Firm performance" in asked and len(asked) == 4 + 3 * len(builder.CONFOUNDS)
        assert len(set(asked)) == len(asked)

    def test_it_writes_a_database_of_what_it_found(self, tmp_path, monkeypatch, capsys):
        builder = load_builder()
        monkeypatch.setattr(builder, "SemanticScholarCorpus", lambda *a, **k: Stub())
        monkeypatch.setattr(builder, "OpenAlexCorpus", lambda *a, **k: Stub())
        out = tmp_path / "demo.sqlite3"
        out.write_bytes(b"an older build")  # replaced, not added to
        assert builder.main(["--out", str(out)]) == 0
        searches, papers = PaperDatabase(out).counts()
        assert searches == 2 * len(builder.queries(*builder.DEMO_QUESTION)) and papers == searches // 2
        assert "(0 of" in capsys.readouterr().out

    def test_a_build_with_many_failures_is_not_kept_quietly(self, tmp_path, monkeypatch, capsys):
        builder = load_builder()
        monkeypatch.setattr(builder, "SemanticScholarCorpus", lambda *a, **k: Stub(fail=True))
        monkeypatch.setattr(builder, "OpenAlexCorpus", lambda *a, **k: Stub())
        assert builder.main(["--out", str(tmp_path / "demo.sqlite3")]) == 1  # half the searches failed
        assert "failed" in capsys.readouterr().err

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

    def test_the_defaults_are_the_happy_path(self):
        args = self.parse()
        assert (args.out, args.confounds, args.max_budget, args.no_open) == (
            Path("confoundsearch-demo.html"),
            4,
            1.0,
            False,
        )

    def test_the_model_follows_the_key_that_is_set(self, monkeypatch):
        assert cli.demo_model(self.parse()) is None
        assert cli.demo_model(self.parse("--model", "mine")) == "mine"
        assert cli.demo_model(self.parse("--api-key", "k")) == "gpt-4.1-mini"
        monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:1/v1")
        assert cli.demo_model(self.parse()) == "gpt-4.1-mini"
        monkeypatch.delenv("OPENAI_BASE_URL")
        monkeypatch.setenv("OPENAI_API_KEY", "k")
        assert cli.demo_model(self.parse()) == "gpt-4.1-mini"
        monkeypatch.setenv("OPENROUTER_API_KEY", "k")  # both set: OpenAI's own key is used
        assert cli.demo_model(self.parse()) == "gpt-4.1-mini"
        monkeypatch.delenv("OPENAI_API_KEY")
        assert cli.demo_model(self.parse()) == "openai/gpt-4.1-mini"  # only OpenRouter
        monkeypatch.setenv("OPENAI_API_KEY", "k")
        assert cli.demo_model(self.parse("--openrouter")) == "openai/gpt-4.1-mini"

    def test_without_any_key_it_says_what_to_do(self, capsys):
        assert cli.main(["demo"]) == 1
        assert "confoundsearch setup" in capsys.readouterr().err

    def test_it_is_a_run_on_the_demo_papers_for_the_demo_question(self, monkeypatch, tmp_path):
        monkeypatch.setenv("OPENAI_API_KEY", "k")
        seen = []
        real = cli.main
        monkeypatch.setattr(cli, "main", lambda argv=None: seen.append(list(argv)) or 0)
        assert real(["demo", "--out", str(tmp_path / "d.html"), "--confounds", "3"]) == 0
        (argv,) = seen
        assert argv[:5] == ["run", "--corpus", "demo", "--model", "gpt-4.1-mini"]
        assert argv[argv.index("--iv") + 1] == "Organizational structure"
        assert argv[argv.index("--dv") + 1] == "Firm performance"
        assert argv[argv.index("--confounds") + 1] == "3" and argv[argv.index("--max-budget") + 1] == "1.0"
        assert argv[-1] == "--review"  # opens the result

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
