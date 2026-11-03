from __future__ import annotations

import argparse
import json
import re
import runpy
import sys
from pathlib import Path

import pytest

from confoundsearch.data.cache import CachedCorpus
from confoundsearch.data.corpus import InMemoryCorpus
from confoundsearch.data.es import ElasticsearchCorpus
from confoundsearch.data.multi import MultiCorpus
from confoundsearch.data.sources import OpenAlexCorpus, SemanticScholarCorpus
from confoundsearch.interface import cli
from confoundsearch.interface.server import RunManager
from confoundsearch.logic.audit import AuditedClient, AuditLog, AuditTrail
from confoundsearch.logic.transition import TrustSynthesis
from confoundsearch.model.llm import LLMJudge, OpenAICompatibleClient, Proposer
from tests.conftest import P1, P2, P3, StubJudge

COMMON = ["--model", "m", "--base-url", "http://localhost:1/v1"]


def parse(*argv):
    return cli.build_parser().parse_args(list(argv))


def test_serve_defaults():
    args = parse("serve", *COMMON, "--corpus-file", "c.jsonl")
    assert (args.host, args.port, args.max_concurrent) == ("127.0.0.1", 7860, 2)
    assert args.es_index == "papers"


def test_run_arguments():
    args = parse(
        "run", *COMMON, "--es-url", "http://es", "--iv", "a", "--dv", "b", "--seed", "3", "--out", "r.html"
    )
    assert (args.iv, args.dv, args.seed, args.max_steps, args.epsilon) == ("a", "b", 3, None, 0.1)
    assert (args.policy, args.confounds, args.synonyms) == ("deterministic", 8, None)
    assert args.queries == 3
    assert str(args.out) == "r.html" and args.review is None


def test_run_needs_an_explicit_output_file(capsys):
    with pytest.raises(SystemExit):
        parse("run", *COMMON, "--corpus-file", "c", "--iv", "a", "--dv", "b")
    assert "--out" in capsys.readouterr().err


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["serve", "--corpus-file", "c", *COMMON[:0]],  # no model
        ["run", *COMMON, "--corpus-file", "c"],  # no iv/dv
    ],
)
def test_invalid_command_lines_exit(argv):
    with pytest.raises(SystemExit) as exit_info:
        parse(*argv)
    assert exit_info.value.code == 2


def test_an_unknown_source_is_refused_before_anything_starts():
    with pytest.raises(SystemExit, match="unknown source 'pubmed'"):
        cli.main(["serve", *COMMON, "--corpus", "pubmed"])


def test_version(capsys):
    with pytest.raises(SystemExit) as exit_info:
        parse("--version")
    assert exit_info.value.code == 0 and "confoundsearch" in capsys.readouterr().out


def write_corpus(tmp_path):
    path = tmp_path / "corpus.jsonl"
    rows = [{"id": p.id, "text": p.text} for p in (P1, P2, P3)]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n\n")
    return path


def test_build_backend_with_a_jsonl_corpus(tmp_path):
    args = parse("serve", *COMMON, "--corpus-file", str(write_corpus(tmp_path)), "--api-key", "k")
    backend = cli.build_backend(args)
    env, proposer = backend.env, backend.proposer
    assert [p.id for p in env.corpus.search("larger firms", 5)] == ["p1", "p2"]
    assert isinstance(env.judge, TrustSynthesis) and isinstance(proposer, Proposer)  # the synthesis judges
    kept = cli.build_backend(
        parse(
            "serve", *COMMON, "--corpus-file", str(write_corpus(tmp_path)), "--api-key", "k", "--no-synthesis"
        )
    )
    assert isinstance(kept.env.judge, LLMJudge)
    audited = backend.client
    assert isinstance(audited, AuditedClient)  # every model call goes into the run's audit trail
    client = audited.client.client.client  # announced, then cached, then the real client
    assert isinstance(client, OpenAICompatibleClient)
    assert (client.model, client.base_url, client.api_key) == ("m", "http://localhost:1/v1", "k")


def test_build_backend_with_elasticsearch():
    args = parse(
        "serve",
        *COMMON,
        "--corpus",
        "elasticsearch",
        "--es-url",
        "https://es.example",
        "--es-index",
        "docs",
        "--es-api-key",
        "key",
    )
    env = cli.build_backend(args).env
    store = env.corpus.corpus  # the source, inside its saved-searches wrapper
    assert isinstance(store, ElasticsearchCorpus)
    assert (store.url, store.index, store.api_key) == ("https://es.example", "docs", "key")


class ScriptedClient:
    """A stand-in for the audited model: the confounds to look for, their synonyms, and a judge."""

    model = "scripted"

    def chat(self, messages):
        prompt = messages[-1]["content"]
        if '{"confounds"' in prompt:
            return json.dumps({"confounds": ["firm size", "firm age"]})
        if '{"synonyms"' in prompt:
            return json.dumps({"synonyms": ["head count"] if "size" in prompt else []})
        if "JSON lines only" in prompt:  # the synthesis: every passage supports "firm size" affecting X
            ids = re.findall(r"^id: (.+)$", prompt, flags=re.M)
            return "\n".join(
                json.dumps({"confound": "firm size", "side": "X", "id": i, "quote": ""}) for i in ids
            )
        return "yes"


def fake_backend(env):
    class Open:
        def propose(self, state):
            from confoundsearch import AddConfound, Retrieve, enumerate_actions

            actions = enumerate_actions(state)
            if not state.done:
                actions += [Retrieve("larger firms"), Retrieve("profits")]
                if not state.graph.confounds:
                    actions.append(AddConfound("firm size", frozenset({"firm size"})))
            return actions

    return lambda args: cli.Backend(env, Open(), AuditedClient(ScriptedClient()))


def test_run_writes_the_standalone_page(tmp_path, monkeypatch, capsys, env):
    monkeypatch.setattr(cli, "build_backend", fake_backend(env))
    out = tmp_path / "page.html"
    argv = [
        "run",
        *COMMON,
        "--policy",
        "epsilon-greedy",
        "--corpus-file",
        "x",
        "--iv",
        "IT",
        "--dv",
        "perf",
        "--seed",
        "1",
    ]
    argv += ["--max-steps", "40", "--epsilon", "0.2", "--out", str(out)]
    assert cli.main(argv) == 0
    html = out.read_text(encoding="utf-8")
    assert html.startswith("<!DOCTYPE html>") and "IT" in html and "perf" in html
    stdout = capsys.readouterr().out
    assert "40 steps" in stdout and f"page   {out.resolve()}" in stdout


def test_serve_starts_and_closes_the_server(monkeypatch, capsys, env):
    events = []

    class FakeServer:
        server_port = 4321

        def serve_forever(self):
            events.append("serve")
            raise KeyboardInterrupt

        def server_close(self):
            events.append("close")

    def make_server(manager, host, port):
        events.append((type(manager), host, port, manager.max_concurrent))
        return FakeServer()

    monkeypatch.setattr(cli, "build_backend", fake_backend(env))
    monkeypatch.setattr(cli, "make_server", make_server)
    argv = [
        "serve",
        *COMMON,
        "--corpus-file",
        "x",
        "--host",
        "0.0.0.0",
        "--port",
        "9",
        "--max-concurrent",
        "5",
    ]
    assert cli.main(argv) == 0
    assert events == [(RunManager, "0.0.0.0", 9, 5), "serve", "close"]
    assert "http://0.0.0.0:4321" in capsys.readouterr().err


def test_python_dash_m_entry_point(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["confoundsearch", "--version"])
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_module("confoundsearch", run_name="__main__")
    assert exit_info.value.code == 0


def test_namespace_type():
    assert isinstance(
        parse("run", *COMMON, "--corpus-file", "c", "--iv", "a", "--dv", "b", "--out", "r.html"),
        argparse.Namespace,
    )
    assert StubJudge  # keep the shared stub imported for readers of this module


RUN_ARGV = [
    "run",
    *COMMON,
    "--policy",
    "epsilon-greedy",
    "--corpus-file",
    "x",
    "--iv",
    "IT",
    "--dv",
    "perf",
    "--seed",
    "1",
    "--max-steps",
    "40",
]


class TestAudit:
    def test_run_writes_an_audit_trail_next_to_the_page_by_default(self, tmp_path, monkeypatch, capsys, env):
        monkeypatch.setattr(cli, "build_backend", fake_backend(env))
        out = tmp_path / "page.html"
        assert cli.main([*RUN_ARGV, "--out", str(out)]) == 0
        trail = AuditTrail.load(tmp_path / "page.audit.jsonl")
        assert trail.intact and trail.complete and trail.of_kind("run_started")
        assert f"audit  {(tmp_path / 'page.audit.jsonl').resolve()}" in capsys.readouterr().out

    def test_the_audit_path_can_be_chosen(self, tmp_path, monkeypatch, env):
        monkeypatch.setattr(cli, "build_backend", fake_backend(env))
        target = tmp_path / "elsewhere" / "log.jsonl"
        target.parent.mkdir()
        cli.main([*RUN_ARGV, "--out", str(tmp_path / "p.html"), "--audit", str(target)])
        assert target.exists() and not (tmp_path / "p.audit.jsonl").exists()

    def test_the_audit_command_prints_the_report_and_succeeds_on_a_good_log(
        self, tmp_path, monkeypatch, capsys, env
    ):
        monkeypatch.setattr(cli, "build_backend", fake_backend(env))
        cli.main([*RUN_ARGV, "--out", str(tmp_path / "p.html")])
        capsys.readouterr()
        assert cli.main(["audit", str(tmp_path / "p.audit.jsonl")]) == 0
        out = capsys.readouterr().out
        assert out.startswith("# Audit trail") and "hash chain intact" in out and "complete" in out

    def test_the_audit_command_fails_on_a_tampered_or_unfinished_log(self, tmp_path, capsys):
        log = AuditLog(tmp_path / "unfinished.jsonl")
        log.record("run_started", independent_variable="a", dependent_variable="b", max_steps=1)
        log.close()
        assert cli.main(["audit", str(tmp_path / "unfinished.jsonl")]) == 1
        assert "INCOMPLETE" in capsys.readouterr().out
        text = (tmp_path / "unfinished.jsonl").read_text().replace('"max_steps": 1', '"max_steps": 9')
        (tmp_path / "edited.jsonl").write_text(text)
        assert cli.main(["audit", str(tmp_path / "edited.jsonl")]) == 1
        assert "BROKEN" in capsys.readouterr().out

    def test_the_audit_command_accepts_the_committed_example(self, capsys):
        example = Path(__file__).parent.parent / "examples" / "example_audit.jsonl"
        assert cli.main(["audit", str(example)]) == 0
        assert "hash chain intact" in capsys.readouterr().out

    def test_the_audit_command_needs_no_model_or_corpus(self):
        assert cli.build_parser().parse_args(["audit", "x.jsonl"]).file == Path("x.jsonl")

    def test_serve_passes_the_audit_folder_to_the_manager(self, monkeypatch, env, tmp_path):
        seen = []

        class FakeServer:
            server_port = 1

            def serve_forever(self):
                raise KeyboardInterrupt

            def server_close(self):
                pass

        monkeypatch.setattr(cli, "build_backend", fake_backend(env))
        monkeypatch.setattr(
            cli, "make_server", lambda manager, host, port: seen.append(manager) or FakeServer()
        )
        cli.main(["serve", *COMMON, "--corpus-file", "x", "--audit-dir", str(tmp_path / "logs")])
        assert seen[0].audit_dir == tmp_path / "logs"


class TestSources:
    def backend(self, monkeypatch, *argv, **env):
        for name in (
            "ES_URL",
            "ES_API_KEY",
            "S2_KEY",
            "OPENALEX_API_KEY",
            "OPENALEX_EMAIL",
            "CONFOUNDSEARCH_API",
        ):
            monkeypatch.delenv(name, raising=False)
        for name, value in env.items():
            monkeypatch.setenv(name, value)
        corpus = cli.build_backend(parse("serve", *COMMON, *argv)).env.corpus
        return corpus.corpus if isinstance(corpus, CachedCorpus) else corpus  # the source inside its wrapper

    def test_the_default_searches_every_source_that_is_set_up(self, monkeypatch):
        corpus = self.backend(monkeypatch, ES_URL="https://hosted.example:443", ES_API_KEY="k")
        assert isinstance(corpus, MultiCorpus)
        assert list(corpus.stores) == ["elasticsearch", "semanticscholar", "openalex"]
        assert corpus.stores["elasticsearch"].corpus.index == "papers"

    def test_the_default_includes_the_keyword_service_when_its_address_is_set(self, monkeypatch):
        from confoundsearch import HttpCorpus

        corpus = self.backend(monkeypatch, CONFOUNDSEARCH_API="https://api.example.test")
        assert list(corpus.stores) == ["service", "semanticscholar", "openalex"]
        assert isinstance(corpus.stores["service"].corpus, HttpCorpus)  # inside the search cache
        assert corpus.stores["service"].corpus.store.base_url == "https://api.example.test"
        by_flag = self.backend(monkeypatch, "--api", "https://flag.example.test", ES_URL="https://es.example")
        assert list(by_flag.stores) == ["service", "elasticsearch", "semanticscholar", "openalex"]

    def test_a_built_in_address_is_used_when_nothing_is_set(self, monkeypatch):
        from confoundsearch import HttpCorpus
        from confoundsearch.data import keyword_service

        monkeypatch.setattr(keyword_service, "DEFAULT_SERVICE_URL", "https://built-in.example.test")
        corpus = self.backend(monkeypatch)
        assert list(corpus.stores) == ["service", "semanticscholar", "openalex"]
        assert corpus.stores["service"].corpus.store.base_url == "https://built-in.example.test"
        assert isinstance(corpus.stores["service"].corpus, HttpCorpus)
        # a setting wins over the built-in address, and the environment wins too
        by_flag = self.backend(monkeypatch, "--api", "https://flag.example.test")
        assert by_flag.stores["service"].corpus.store.base_url == "https://flag.example.test"
        by_env = self.backend(monkeypatch, CONFOUNDSEARCH_API="https://env.example.test")
        assert by_env.stores["service"].corpus.store.base_url == "https://env.example.test"

    @pytest.mark.parametrize("off", ["off", "OFF", "none", "no", "false", "0", ""])
    def test_the_service_can_be_switched_off(self, monkeypatch, off):
        from confoundsearch.data import keyword_service

        monkeypatch.setattr(keyword_service, "DEFAULT_SERVICE_URL", "https://built-in.example.test")
        assert list(self.backend(monkeypatch, "--api", off).stores) == ["semanticscholar", "openalex"]
        assert list(self.backend(monkeypatch, CONFOUNDSEARCH_API=off).stores) == [
            "semanticscholar",
            "openalex",
        ]
        with pytest.raises(SystemExit, match="no address"):
            self.backend(monkeypatch, "--corpus", "service", "--api", off)

    def test_the_service_alone_is_a_corpus_by_itself(self, monkeypatch):
        from confoundsearch import HttpCorpus

        corpus = self.backend(monkeypatch, "--corpus", "service", "--api", "https://api.example.test")
        assert isinstance(corpus, HttpCorpus)

    def test_the_service_needs_its_address(self, monkeypatch):
        with pytest.raises(SystemExit) as exit_info:
            self.backend(monkeypatch, "--corpus", "service")
        assert "--api URL" in str(exit_info.value) and "CONFOUNDSEARCH_API" in str(exit_info.value)

    def test_without_a_hosted_index_the_default_is_the_two_open_sources(self, monkeypatch):
        corpus = self.backend(monkeypatch)
        assert list(corpus.stores) == ["semanticscholar", "openalex"]

    def test_sources_can_be_repeated_or_comma_separated(self, monkeypatch):
        for argv in (
            ("--corpus", "openalex", "--corpus", "semanticscholar"),
            ("--corpus", "openalex,semanticscholar"),
        ):
            assert list(self.backend(monkeypatch, *argv).stores) == ["openalex", "semanticscholar"]

    def test_the_hosted_index_needs_its_address(self, monkeypatch):
        with pytest.raises(SystemExit) as exit_info:
            self.backend(monkeypatch, "--corpus", "elasticsearch")
        message = str(exit_info.value)
        assert "ES_URL" in message and "--corpus semanticscholar" in message and "--corpus-file" in message

    def test_semantic_scholar_with_and_without_a_key(self, monkeypatch):
        corpus = self.backend(monkeypatch, "--corpus", "semanticscholar", S2_KEY="s2")
        assert isinstance(corpus, SemanticScholarCorpus) and corpus.api_key == "s2"
        assert self.backend(monkeypatch, "--corpus", "semanticscholar").api_key is None

    def test_openalex_takes_a_key_and_an_email(self, monkeypatch):
        corpus = self.backend(
            monkeypatch, "--corpus", "openalex", OPENALEX_API_KEY="k", OPENALEX_EMAIL="me@example.org"
        )
        assert isinstance(corpus, OpenAlexCorpus) and (corpus.api_key, corpus.mailto) == (
            "k",
            "me@example.org",
        )
        bare = self.backend(monkeypatch, "--corpus", "openalex")
        assert (bare.api_key, bare.mailto) == (None, None)

    def test_flags_override_the_environment(self, monkeypatch):
        corpus = self.backend(
            monkeypatch,
            "--corpus",
            "openalex",
            "--openalex-email",
            "flag@example.org",
            OPENALEX_EMAIL="env@example.org",
        )
        assert corpus.mailto == "flag@example.org"

    def test_a_papers_file_is_chosen_by_its_flag_alone(self, monkeypatch, tmp_path):
        corpus = self.backend(
            monkeypatch, "--corpus-file", str(write_corpus(tmp_path)), ES_URL="https://ignored.example"
        )
        assert isinstance(corpus, InMemoryCorpus)

    def test_corpus_file_needs_a_path_and_does_not_mix_with_another_source(self, monkeypatch):
        with pytest.raises(SystemExit, match="needs --corpus-file"):
            self.backend(monkeypatch, "--corpus", "file")
        with pytest.raises(SystemExit, match="not with another source"):
            self.backend(monkeypatch, "--corpus", "openalex", "--corpus-file", "x.jsonl")

    def test_main_reads_keys_from_a_dotenv_file(self, monkeypatch, tmp_path, env):
        for name in ("ES_URL", "ES_API_KEY"):
            monkeypatch.delenv(name, raising=False)
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".env").write_text("ES_URL=https://from-dotenv.example\nES_API_KEY=abc\n")
        seen = []

        class Server:
            server_port = 1

            def serve_forever(self):
                raise KeyboardInterrupt

            def server_close(self):
                pass

        real = cli.build_backend

        def spy(args):
            seen.append(real(args).env.corpus)
            return cli.Backend(env, object(), ScriptedClient())

        monkeypatch.setattr(cli, "build_backend", spy)
        monkeypatch.setattr(cli, "make_server", lambda *a: Server())
        cli.main(["serve", *COMMON])
        assert seen[0].stores["elasticsearch"].corpus.url == "https://from-dotenv.example"


class TestJsonAndSkill:
    def test_run_json_prints_only_json_to_stdout(self, tmp_path, monkeypatch, capsys, env):
        monkeypatch.setattr(cli, "build_backend", fake_backend(env))
        out = tmp_path / "page.html"
        assert cli.main([*RUN_ARGV, "--out", str(out), "--json"]) == 0
        captured = capsys.readouterr()
        result = json.loads(captured.out)  # the whole of stdout is one JSON document
        assert result["steps"] == 40 and result["files"]["page"] == str(out)
        assert result["files"]["audit"] == str(tmp_path / "page.audit.jsonl") and result["audit"]["intact"]
        assert out.exists() and (tmp_path / "page.audit.jsonl").exists()

    def test_without_json_the_human_summary_is_unchanged(self, tmp_path, monkeypatch, capsys, env):
        monkeypatch.setattr(cli, "build_backend", fake_backend(env))
        cli.main([*RUN_ARGV, "--out", str(tmp_path / "p.html")])
        out = capsys.readouterr().out
        assert "steps," in out and "Results are in" in out and not out.lstrip().startswith("{")

    def test_audit_json_summarises_the_log(self, tmp_path, monkeypatch, capsys, env):
        monkeypatch.setattr(cli, "build_backend", fake_backend(env))
        cli.main([*RUN_ARGV, "--out", str(tmp_path / "p.html")])
        capsys.readouterr()
        assert cli.main(["audit", str(tmp_path / "p.audit.jsonl"), "--json"]) == 0
        summary = json.loads(capsys.readouterr().out)
        assert (
            summary["intact"] and summary["complete"] and summary["file"] == str(tmp_path / "p.audit.jsonl")
        )
        assert summary["question"]["independent_variable"] == "IT" and summary["head"] and summary["evidence"]

    def test_audit_json_still_fails_on_a_broken_log(self, tmp_path, capsys):
        log = AuditLog(tmp_path / "x.jsonl")
        log.record("run_started", independent_variable="a", dependent_variable="b", max_steps=1)
        log.close()
        assert cli.main(["audit", str(tmp_path / "x.jsonl"), "--json"]) == 1
        summary = json.loads(capsys.readouterr().out)
        assert summary["complete"] is False and summary["ended"] is None and summary["evidence"] == []

    def test_install_skill_writes_both_packaged_skills(self, tmp_path, capsys):
        assert cli.main(["install-skill", "--dir", str(tmp_path)]) == 0
        out = capsys.readouterr().out
        for name in ("confoundsearch", "confoundsearchui"):
            target = tmp_path / name / "SKILL.md"
            assert target.read_text(encoding="utf-8") == cli.skill_text(name)
            assert f"installed {target}" in out

    def test_install_skill_is_idempotent_and_will_not_clobber_edits(self, tmp_path, capsys):
        cli.main(["install-skill", "--dir", str(tmp_path)])
        capsys.readouterr()
        assert cli.main(["install-skill", "--dir", str(tmp_path)]) == 0
        assert capsys.readouterr().out.count("already installed") == 2
        target = tmp_path / "confoundsearch" / "SKILL.md"
        target.write_text("my own edits")
        assert cli.main(["install-skill", "--dir", str(tmp_path)]) == 1
        shown = capsys.readouterr()
        assert "--force" in shown.err and target.read_text() == "my own edits"
        assert "already installed" in shown.out  # the other skill is still fine
        assert cli.main(["install-skill", "--dir", str(tmp_path), "--force"]) == 0
        assert target.read_text(encoding="utf-8") == cli.skill_text()

    def test_install_skill_defaults_to_the_users_claude_folder(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cli.Path, "home", classmethod(lambda cls: tmp_path / "home"))
        cli.main(["install-skill"])
        skills = tmp_path / "home" / ".claude" / "skills"
        assert (skills / "confoundsearch" / "SKILL.md").exists()
        assert (skills / "confoundsearchui" / "SKILL.md").exists()

    def test_install_skill_can_target_the_current_project(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        cli.main(["install-skill", "--project"])
        assert (tmp_path / ".claude" / "skills" / "confoundsearch" / "SKILL.md").exists()
        assert (tmp_path / ".claude" / "skills" / "confoundsearchui" / "SKILL.md").exists()

    def test_install_skill_can_print_instead(self, tmp_path, capsys, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert cli.main(["install-skill", "--print"]) == 0
        assert capsys.readouterr().out == cli.skill_text() and not (tmp_path / ".claude").exists()

    def test_install_skill_needs_no_model_or_corpus(self):
        assert cli.build_parser().parse_args(["install-skill"]).command == "install-skill"


class TestProgress:
    """`run` reports each step as it happens, on stderr, in the style of Claude Code's own updates."""

    def test_each_step_is_one_bulleted_action_and_what_it_earned(self, tmp_path, monkeypatch, capsys, env):
        monkeypatch.setattr(cli, "build_backend", fake_backend(env))
        out = tmp_path / "page.html"
        argv = [
            "run",
            *COMMON,
            "--policy",
            "epsilon-greedy",
            "--corpus-file",
            "x",
            "--iv",
            "IT",
            "--dv",
            "perf",
            "--seed",
            "1",
        ]
        assert cli.main([*argv, "--max-steps", "6", "--out", str(out)]) == 0
        captured = capsys.readouterr()
        bullets = [line for line in captured.err.splitlines() if line.startswith("● ")]
        assert len(bullets) == 6
        assert "a1 = " in captured.err and "r6 = " in captured.err
        assert "●" not in captured.out  # standard output keeps only the summary

    def test_json_output_stays_clean_while_progress_goes_to_stderr(self, tmp_path, monkeypatch, capsys, env):
        monkeypatch.setattr(cli, "build_backend", fake_backend(env))
        argv = [
            "run",
            *COMMON,
            "--corpus-file",
            "x",
            "--iv",
            "IT",
            "--dv",
            "perf",
            "--json",
            "--max-steps",
            "3",
        ]
        assert cli.main([*argv, "--out", str(tmp_path / "p.html")]) == 0
        captured = capsys.readouterr()
        json.loads(captured.out)
        assert captured.err.count("● ") == 3

    def test_each_action_is_described_in_the_papers_words(self, env):
        from confoundsearch.logic.actions import AddConfound, AddEdge, AttachPassage, Merge, Retrieve, Stop
        from confoundsearch.logic.episode import Step

        state = env.initial_state("IT", "perf")

        def line(number, action):
            return cli.progress(number, Step(state, action, 0.0, state))

        assert line(1, Retrieve("firm age")).startswith(
            "\u25cf a1 = Retrieve('firm age')\n  \u23bf r1 = +0.00, \u03a6 = 0.00"
        )
        assert "AddConfound('age')  a candidate C" in line(2, AddConfound("age", frozenset({"age"})))
        assert "AddEdge('age')  C \u2192 X" in line(3, AddEdge("age", "X"))
        assert "AttachPassage('age', p1)  evidence for C \u2192 Y" in line(4, AttachPassage("age", "Y", "p1"))
        assert "Merge('a', 'b')  one construct, two names" in line(5, Merge("a", "b"))
        assert line(6, Stop()).startswith("\u25cf a6 = Stop()")
        assert "0 confounds with both X \u2190 C \u2192 Y" in line(6, Stop())

    def test_the_run_opens_with_the_epigraph(self, tmp_path, monkeypatch, capsys, env):
        monkeypatch.setattr(cli, "build_backend", fake_backend(env))
        argv = [
            "run",
            *COMMON,
            "--policy",
            "epsilon-greedy",
            "--corpus-file",
            "x",
            "--iv",
            "IT",
            "--dv",
            "perf",
            "--max-steps",
            "1",
        ]
        assert cli.main([*argv, "--out", str(tmp_path / "p.html")]) == 0
        err = capsys.readouterr().err
        assert err.startswith("\u201c[T]he theorist is overloaded by demands") and "Karl Weick" in err


class TestSetup:
    def answers(self, monkeypatch, choice, *keys, folder=""):
        replies = iter([choice, folder])  # the model, then the data directory
        monkeypatch.setattr("builtins.input", lambda prompt="": next(replies))
        given = iter(keys)
        monkeypatch.setattr(cli.getpass, "getpass", lambda prompt="": next(given))

    def test_it_writes_the_keys_privately_and_never_prints_them(self, tmp_path, monkeypatch, capsys):
        self.answers(monkeypatch, "2", "sk-openai-secret", "s2-secret")
        env_file = tmp_path / ".env"
        assert cli.main(["setup", "--env-file", str(env_file)]) == 0
        assert env_file.read_text() == "OPENAI_API_KEY=sk-openai-secret\nS2_KEY=s2-secret\n"
        assert env_file.stat().st_mode & 0o777 == 0o600
        shown = capsys.readouterr()
        assert "secret" not in shown.out + shown.err
        assert "OPENAI_API_KEY, S2_KEY" in shown.out

    def test_it_tells_you_where_downloads_go_and_how_to_empty_them(self, tmp_path, monkeypatch, capsys):
        self.answers(monkeypatch, "2", "k", "")
        assert cli.main(["setup", "--env-file", str(tmp_path / ".env")]) == 0
        said = " ".join(capsys.readouterr().out.split())  # the message wraps over lines
        assert "downloads papers" in said and "Hugging Face's model folder" in said
        assert (
            "confoundsearch flush" in said and str(cli.cache_dir()) in said and "temporary directory" in said
        )

    def test_a_data_directory_you_choose_is_saved_in_full(self, tmp_path, monkeypatch):
        env_file = tmp_path / ".env"
        self.answers(monkeypatch, "2", "k", "", folder="~/confoundsearch-data")
        assert cli.main(["setup", "--env-file", str(env_file)]) == 0
        saved = Path.home() / "confoundsearch-data"
        assert env_file.read_text() == f"OPENAI_API_KEY=k\nCONFOUNDSEARCH_CACHE={saved}\n"

    def test_enter_leaves_the_data_directory_alone(self, tmp_path, monkeypatch):
        env_file = tmp_path / ".env"
        self.answers(monkeypatch, "2", "k", "")
        assert cli.main(["setup", "--env-file", str(env_file)]) == 0
        assert "CONFOUNDSEARCH_CACHE" not in env_file.read_text()

    def test_the_default_is_openrouter_and_the_semantic_scholar_key_is_optional(self, tmp_path, monkeypatch):
        self.answers(monkeypatch, "", "sk-or-secret", "")
        env_file = tmp_path / ".env"
        assert cli.main(["setup", "--env-file", str(env_file)]) == 0
        assert env_file.read_text() == "OPENROUTER_API_KEY=sk-or-secret\n"

    def test_it_changes_only_what_it_was_given(self, tmp_path, monkeypatch):
        env_file = tmp_path / ".env"
        env_file.write_text("# mine\nES_URL=https://x\nOPENAI_API_KEY=old\n")
        self.answers(monkeypatch, "2", "new", "")
        assert cli.main(["setup", "--env-file", str(env_file)]) == 0
        assert env_file.read_text() == "# mine\nES_URL=https://x\nOPENAI_API_KEY=new\n"

    def test_a_bad_choice_or_no_key_writes_nothing(self, tmp_path, monkeypatch, capsys):
        env_file = tmp_path / ".env"
        self.answers(monkeypatch, "9")
        assert cli.main(["setup", "--env-file", str(env_file)]) == 1
        self.answers(monkeypatch, "1", "", "")
        assert cli.main(["setup", "--env-file", str(env_file)]) == 1
        assert not env_file.exists()
        assert "choose 1 or 2" in capsys.readouterr().err

    def test_in_a_git_folder_it_keeps_the_file_out_of_git(self, tmp_path, monkeypatch, capsys):
        (tmp_path / ".git").mkdir()
        self.answers(monkeypatch, "1", "k", "")
        assert cli.main(["setup", "--env-file", str(tmp_path / ".env")]) == 0
        assert (tmp_path / ".gitignore").read_text() == ".env\n"
        assert "added .env to .gitignore" in capsys.readouterr().out
        self.answers(monkeypatch, "1", "k", "")
        assert cli.main(["setup", "--env-file", str(tmp_path / ".env")]) == 0  # already listed: no change
        assert (tmp_path / ".gitignore").read_text() == ".env\n"

    def test_outside_git_it_leaves_gitignore_alone(self, tmp_path, monkeypatch):
        self.answers(monkeypatch, "1", "k", "")
        assert cli.main(["setup", "--env-file", str(tmp_path / ".env")]) == 0
        assert not (tmp_path / ".gitignore").exists()
        (tmp_path / ".gitignore").write_text("build")
        self.answers(monkeypatch, "1", "k", "")
        assert cli.main(["setup", "--env-file", str(tmp_path / ".env")]) == 0
        assert (tmp_path / ".gitignore").read_text() == "build\n.env\n"


def test_build_backend_with_synonyms_wraps_the_corpus(tmp_path):
    from confoundsearch import ExpandedCorpus

    path = write_corpus(tmp_path)
    flags = ["--corpus-file", str(path), "--api-key", "k", "--policy", "epsilon-greedy"]
    plain = cli.build_backend(parse("serve", *COMMON, *flags)).env
    wrapped = cli.build_backend(parse("serve", *COMMON, *flags, "--synonyms", "3")).env
    assert isinstance(wrapped.corpus, ExpandedCorpus) and wrapped.corpus.limit == 3
    assert not isinstance(plain.corpus, ExpandedCorpus)


class TestThePolicy:
    """The default policy asks for confounds, gets synonyms, searches for all of them, then processes them."""

    def run_default(self, tmp_path, monkeypatch, env, *extra):
        monkeypatch.setattr(cli, "build_backend", fake_backend(env))
        out = tmp_path / "r.html"
        argv = [
            "run",
            *COMMON,
            "--corpus-file",
            "x",
            "--iv",
            "IT",
            "--dv",
            "perf",
            "--json",
            "--out",
            str(out),
        ]
        assert cli.main([*argv, *extra]) == 0
        return AuditTrail.load(tmp_path / "r.audit.jsonl")

    def test_a_run_with_no_policy_chosen_is_deterministic_and_ends_by_itself(
        self, tmp_path, monkeypatch, env
    ):
        trail = self.run_default(tmp_path, monkeypatch, env)
        started = trail.of_kind("run_started")[0].data
        assert started["policy"] == "DeterministicPolicy" and started["max_steps"] == cli.RUN_STEPS
        assert trail.of_kind("run_finished")[0].data["reason"] == "stop_action"
        asked = [e.data["messages"][-1]["content"] for e in trail.of_kind("model_call")]
        assert any('{"confounds"' in prompt for prompt in asked) and any('{"synonyms"' in p for p in asked)

    def test_the_options_reach_the_policy_and_the_log(self, tmp_path, monkeypatch, env):
        trail = self.run_default(tmp_path, monkeypatch, env, "--confounds", "1", "--synonyms", "0")
        settings = trail.of_kind("run_started")[0].data["policy_settings"]
        assert settings["max_confounds"] == 1 and settings["synonyms_per_term"] == 0
        assert settings["queries_per_side"] == 3 and settings["attach_per_side"] == cli.SYNTHESIS_ATTACH
        queries = [e.data["query"] for e in trail.of_kind("retrieved")]
        assert queries and all("firm size" in q for q in queries)  # only the one confound it was allowed

    def test_the_default_run_reads_what_it_retrieved_by_meaning_and_no_synthesis_turns_that_off(
        self, tmp_path, monkeypatch, env
    ):
        asked = lambda trail: [e.data["messages"][-1]["content"] for e in trail.of_kind("model_call")]  # noqa: E731
        trail = self.run_default(tmp_path, monkeypatch, env, "--confounds", "1")
        assert any("JSON lines only" in prompt for prompt in asked(trail))
        trail = self.run_default(tmp_path, monkeypatch, env, "--confounds", "1", "--no-synthesis")
        assert not any("JSON lines only" in prompt for prompt in asked(trail))

    def test_the_second_phase_asks_for_measures_and_mechanisms_unless_switched_off(
        self, tmp_path, monkeypatch, env
    ):
        asked = lambda trail: " ".join(e.data["messages"][-1]["content"] for e in trail.of_kind("model_call"))  # noqa: E731
        on = asked(self.run_default(tmp_path, monkeypatch, env, "--confounds", "1"))
        assert "channels or mechanisms" in on and "measures, forms or outcomes" in on
        for flag in ("--no-mechanisms", "--no-synthesis"):
            off = asked(self.run_default(tmp_path, monkeypatch, env, "--confounds", "1", flag))
            assert "channels or mechanisms" not in off

    def test_without_the_synthesis_it_cites_fewer_passages_per_link(self, tmp_path, monkeypatch, env):
        trail = self.run_default(tmp_path, monkeypatch, env, "--confounds", "1", "--no-synthesis")
        assert trail.of_kind("run_started")[0].data["policy_settings"]["attach_per_side"] == 4

    def test_queries_sets_how_many_searches_each_side_gets(self, tmp_path, monkeypatch, env):
        trail = self.run_default(tmp_path, monkeypatch, env, "--confounds", "1", "--queries", "5")
        settings = trail.of_kind("run_started")[0].data["policy_settings"]
        assert settings["queries_per_side"] == 5

    def test_epsilon_greedy_is_still_there_with_its_own_step_budget(self, tmp_path, monkeypatch, env):
        trail = self.run_default(tmp_path, monkeypatch, env, "--policy", "epsilon-greedy", "--seed", "1")
        started = trail.of_kind("run_started")[0].data
        assert started["policy"] == "EpsilonGreedy" and started["max_steps"] == 40

    def test_synonyms_default_per_policy(self):
        assert cli.synonym_count(parse("run", *COMMON, "--iv", "a", "--dv", "b", "--out", "o")) == 3
        epsilon = parse("run", *COMMON, "--iv", "a", "--dv", "b", "--out", "o", "--policy", "epsilon-greedy")
        assert cli.synonym_count(epsilon) == 0
        assert cli.synonym_count(parse("serve", *COMMON, "--synonyms", "5")) == 5

    def test_the_deterministic_backend_does_not_ask_the_model_to_propose_every_move(self, tmp_path):
        args = parse("serve", *COMMON, "--corpus-file", str(write_corpus(tmp_path)), "--api-key", "k")
        assert cli.build_backend(args).proposer.llm is None
        eps = parse(
            "serve",
            *COMMON,
            "--corpus-file",
            str(write_corpus(tmp_path)),
            "--api-key",
            "k",
            "--policy",
            "epsilon-greedy",
        )
        assert cli.build_backend(eps).proposer.llm is not None

    def test_serve_hands_the_manager_the_policy_and_its_step_budget(self, env, monkeypatch):
        seen = []

        class Server:
            server_port = 1

            def serve_forever(self):
                raise KeyboardInterrupt

            def server_close(self):
                pass

        monkeypatch.setattr(cli, "build_backend", fake_backend(env))
        monkeypatch.setattr(cli, "make_server", lambda manager, *a: seen.append(manager) or Server())
        cli.main(["serve", *COMMON, "--corpus-file", "x"])
        cli.main(["serve", *COMMON, "--corpus-file", "x", "--policy", "epsilon-greedy"])
        deterministic, epsilon = seen
        assert deterministic.default_steps == cli.RUN_STEPS and deterministic.policy_factory is not None
        assert epsilon.default_steps == 40 and epsilon.policy_factory is None
        from confoundsearch import DeterministicPolicy

        assert isinstance(deterministic.policy_factory(None), DeterministicPolicy)
