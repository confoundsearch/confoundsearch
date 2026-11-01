"""The Claude Code skill must describe the program as it really is: its flags, its output, its messages."""

from __future__ import annotations

import json
import re
import shlex
from pathlib import Path

import pytest

from confoundsearch.interface import cli
from tests.test_cli import RUN_ARGV, fake_backend

ROOT = Path(__file__).parent.parent
PACKAGE = ROOT / "confoundsearch"
SKILL = PACKAGE / "interface" / "skill" / "SKILL.md"
TEXT = SKILL.read_text(encoding="utf-8")


def frontmatter():
    head, _, body = TEXT.partition("\n---\n")
    assert head.startswith("---\n"), "SKILL.md must open with a --- frontmatter block"
    fields = dict(line.split(": ", 1) for line in head[len("---\n") :].splitlines())
    return fields, body


class TestFormat:
    def test_the_name_matches_the_folder_the_installer_creates(self):
        name = frontmatter()[0]["name"]
        assert name == "confoundsearch" and re.fullmatch(r"[a-z0-9-]{1,64}", name)

    def test_the_description_says_what_it_does_and_when_to_use_it(self):
        description = frontmatter()[0]["description"]
        assert 50 < len(description) <= 1024 and "\n" not in description
        assert "Use when" in description and "confound" in description

    def test_the_body_covers_the_whole_job(self):
        body = frontmatter()[1]
        for heading in (
            "When to use it", "Before you run it", "Run it", "Reading the result", "Report it", "If it fails",
        ):  # fmt: skip
            assert f"## {heading}" in body
        assert len(TEXT.splitlines()) < 500

    def test_it_is_shipped_in_the_package(self):
        pyproject = (ROOT / "pyproject.toml").read_text()
        assert '"confoundsearch.interface.skill"' in pyproject and "SKILL.md" in pyproject
        assert (SKILL.parent / "__init__.py").exists() and cli.skill_text() == TEXT

    def test_it_tells_the_agent_not_to_handle_keys_or_spend_unasked(self):
        lowered = TEXT.lower()
        assert "never ask the user to paste a key" in lowered and "never echo" in lowered
        assert "do not run it on your own" in lowered and "do not retry in a loop" in lowered
        assert "cite only passages that appear in the output" in lowered


class TestMatchesTheProgram:
    def options(self):
        parser = cli.build_parser()
        found = set(parser._option_string_actions)
        for action in parser._subparsers._group_actions:
            for sub in action.choices.values():
                found |= set(sub._option_string_actions)
        return found

    def test_every_flag_it_mentions_exists(self):
        mentioned = set(re.findall(r"--[a-z][a-z0-9-]*", TEXT))
        assert mentioned and mentioned <= self.options(), mentioned - self.options()

    def test_the_documented_command_parses(self):
        block = re.search(r"```bash\n(confoundsearch run.*?)```", TEXT, re.S).group(1)
        tokens = shlex.split(block.replace("\\\n", " "))
        args = cli.build_parser().parse_args(tokens[1:])
        assert (
            args.json
            and args.max_steps is None  # the default plan ends by itself: no step budget to pick
            and args.policy == "deterministic"
            and args.iv == "IT investment"
            and args.dv == "firm performance"
        )
        assert args.model == "gpt-4.1" and str(args.out) == "confoundsearch-result.html"

    def test_the_audit_file_name_it_promises_is_what_the_program_writes(self, tmp_path, monkeypatch, env):
        monkeypatch.setattr(cli, "build_backend", fake_backend(env))
        cli.main([*RUN_ARGV, "--out", str(tmp_path / "confoundsearch-result.html")])
        assert "confoundsearch-result.audit.jsonl" in TEXT
        assert (tmp_path / "confoundsearch-result.audit.jsonl").exists()

    def test_every_environment_variable_it_names_is_read_by_the_program(self):
        source = "".join(p.read_text() for p in PACKAGE.rglob("*.py"))
        names = set(re.findall(r"`([A-Z][A-Z0-9]*_[A-Z0-9_]+)`", TEXT))
        assert {"OPENAI_API_KEY", "ES_URL", "ES_API_KEY", "S2_KEY", "OPENALEX_EMAIL"} <= names
        assert all(name in source for name in names), [n for n in names if n not in source]

    @pytest.mark.parametrize(
        "phrase,where",
        [
            ("needs ES_URL", "cli.py"),
            ("HTTP", "http.py"),
            ("could not reach", "http.py"),
            ("semanticscholar.org/product/api", "sources.py"),
            ("OPENALEX_EMAIL", "sources.py"),
        ],
    )
    def test_the_failure_table_quotes_real_messages(self, phrase, where):
        assert phrase in next(PACKAGE.rglob(where)).read_text()
        assert phrase.split()[0] in TEXT

    def fields(self, marker):
        paragraph = re.search(rf"{re.escape(marker)}(.*?)\n\n", TEXT, re.S).group(1)
        return paragraph

    def names(self, text):
        return set(re.findall(r"`([a-z_]+)`", text))

    def test_the_documented_fields_are_exactly_the_real_ones(self, tmp_path, monkeypatch, capsys, env):
        monkeypatch.setattr(cli, "build_backend", fake_backend(env))
        cli.main([*RUN_ARGV, "--out", str(tmp_path / "p.html"), "--json"])
        result = json.loads(capsys.readouterr().out)
        top = self.names(self.fields("Top-level fields:"))
        assert top == set(result)
        each = self.fields("Each item of `confounds` has:")
        confound_part, passage_part = each.split("Each passage in the evidence lists has")
        confound = self.names(confound_part) - {"confounds"}
        assert confound == set(result["confounds"][0])
        passages = result["confounds"][0]["evidence_to_iv"] + result["confounds"][0]["evidence_to_dv"]
        assert passages and self.names(passage_part) == set(passages[0])
        assert {"events", "intact", "complete", "head"} <= self.names(TEXT) and set(result["audit"]) == {
            "events", "intact", "complete", "head",
        }  # fmt: skip

    def test_stdout_really_is_just_the_json(self, tmp_path, monkeypatch, capsys, env):
        monkeypatch.setattr(cli, "build_backend", fake_backend(env))
        cli.main([*RUN_ARGV, "--out", str(tmp_path / "p.html"), "--json"])
        json.loads(capsys.readouterr().out)  # raises if anything else was printed
        assert "single JSON document and nothing else" in TEXT


class TestTheInterfaceSkill:
    """/confoundsearchui only opens a page: it must name real commands and must not spend money."""

    UI = PACKAGE / "interface" / "skill" / "UI_SKILL.md"
    TEXT = UI.read_text(encoding="utf-8")

    def test_the_name_matches_the_folder_the_installer_creates(self):
        head = self.TEXT.partition("\n---\n")[0][len("---\n") :]
        fields = dict(line.split(": ", 1) for line in head.splitlines())
        assert fields["name"] == "confoundsearchui" and fields["description"]
        assert (
            cli.SKILLS["confoundsearchui"] == "UI_SKILL.md"
            and cli.skill_text("confoundsearchui") == self.TEXT
        )

    def test_it_is_shipped_in_the_package(self):
        assert "UI_SKILL.md" in (ROOT / "pyproject.toml").read_text()

    def test_the_commands_it_names_exist_and_it_does_not_run_a_search(self):
        parser = cli.build_parser()
        parser.parse_args(["review", "--in", "FILE.json"])
        parser.parse_args(["audit", "FILE"])
        assert "confoundsearch review --in confoundsearch-result.html" in self.TEXT
        assert "Do not start a run yourself" in self.TEXT
