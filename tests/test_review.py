"""`confoundsearch review --in FILE` and `run --review`: look at a result in the interface, or as text."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from confoundsearch.interface import cli, review
from confoundsearch.interface.export import export_zip
from confoundsearch.interface.known import known_to_share_data
from tests.test_cli import RUN_ARGV, fake_backend

KNOWN = {
    "iv": "Structure",
    "dv": "Performance",
    "controls": [
        {
            "name": "Firm age",
            "citations_to_IV": [
                {
                    "title": "Decentralization",
                    "doi": "10.1/a",
                    "authors": "A. Author",
                    "venue": "A Journal",
                    "year": 2007,
                    "quotation": "Younger firms decentralize.",
                }
            ],
            "citations_to_DV": [{"title": "Age and Performance", "quotation": "x" * 400}],
        },
        {"name": "Size", "citations_to_IV": [], "citations_to_DV": []},
    ],
}


@pytest.fixture
def opened(monkeypatch):
    """Records what the command would have opened in a browser, without opening one."""
    pages = []
    monkeypatch.setattr(cli, "open_page", lambda page: pages.append(Path(page)) or True)
    return pages


class TestTheText:
    def test_each_confound_lists_its_papers_for_both_links(self):
        text = review.review_text(known_to_share_data(KNOWN))
        assert text.startswith("Candidate confounds for Structure → Performance\n")
        assert "1. Firm age  (1 paper for Structure, 1 paper for Performance)" in text
        assert "Firm age affects Structure:" in text and "Firm age affects Performance:" in text
        assert "- Decentralization (A. Author · A Journal · 2007)" in text
        assert "https://doi.org/10.1/a" in text and "“Younger firms decentralize.”" in text

    def test_a_confound_with_no_papers_says_so_and_a_long_quote_is_cut(self):
        text = review.review_text(known_to_share_data(KNOWN))
        assert "2. Size  (0 papers for Structure, 0 papers for Performance)" in text and "(no papers)" in text
        assert "x" * 300 + "…”" in text and "x" * 301 not in text

    def test_an_empty_result_says_none_were_found(self):
        data = {"iv": "X", "dv": "Y", "dag_confounders": [], "all_md_names": []}
        assert review.review_text(data) == "Candidate confounds for X → Y\n\nNo confounds found.\n"


def test_a_paper_with_no_title_link_or_quote_is_still_listed():
    assert review.entry_lines({}) == ["     - Unknown"]


class TestLoading:
    def test_a_known_confounds_file_is_converted(self, tmp_path):
        path = tmp_path / "known.json"
        path.write_text(json.dumps(KNOWN))
        assert review.load_share(path) == known_to_share_data(KNOWN)

    def test_a_zip_export_is_read(self, tmp_path):
        path = tmp_path / "run.zip"
        path.write_bytes(export_zip(known_to_share_data(KNOWN)))
        assert review.load_share(path)["iv"] == "Structure"

    def test_a_page_is_already_a_page(self, tmp_path):
        assert review.load_share(tmp_path / "result.html") is None


class TestOpening:
    def test_it_opens_the_full_path_as_a_file_address(self, tmp_path):
        seen = []
        page = tmp_path / "p.html"
        assert review.open_page(page, opener=lambda url: seen.append(url) or True) is True
        assert seen == [page.resolve().as_uri()]

    def test_with_no_browser_it_says_where_the_file_is(self, tmp_path, capsys):
        page = tmp_path / "p.html"
        assert review.open_page(page, opener=lambda url: False) is False
        assert f"open this file yourself: {page.resolve()}" in capsys.readouterr().err

    def test_the_message_can_be_redirected(self, tmp_path):
        said = []
        review.open_page(tmp_path / "p.html", opener=lambda url: False, err=said.append)
        assert len(said) == 1


class TestTheClosingMessage:
    def test_it_names_the_folder_the_page_and_the_audit_trail_and_how_to_review(self, tmp_path):
        text = review.where_it_went(tmp_path / "r.html", tmp_path / "r.audit.jsonl", True)
        assert f"Results are in {tmp_path.resolve()}" in text
        assert f"page   {(tmp_path / 'r.html').resolve()}" in text
        assert "every query and verdict, hash chain intact" in text
        assert f"To review: confoundsearch review --in {(tmp_path / 'r.html').resolve()}" in text

    def test_a_broken_chain_is_said_loudly_and_the_hint_can_be_left_out(self, tmp_path):
        text = review.where_it_went(tmp_path / "r.html", tmp_path / "a.jsonl", False, review_hint=False)
        assert "hash chain BROKEN" in text and "To review" not in text

    def test_without_an_audit_trail_only_the_page_is_listed(self, tmp_path):
        text = review.where_it_went(tmp_path / "r.html", None, None)
        assert "  audit  " not in text and "  page  " in text


class TestTheReviewCommand:
    def known(self, tmp_path):
        path = tmp_path / "known.json"
        path.write_text(json.dumps(KNOWN))
        return path

    def test_a_page_is_opened_as_it_is(self, tmp_path, opened):
        page = tmp_path / "result.html"
        page.write_text("<html></html>")
        assert cli.main(["review", "--in", str(page)]) == 0
        assert opened == [page]

    def test_known_confounds_get_a_page_next_to_the_file_and_it_is_opened(self, tmp_path, opened):
        assert cli.main(["review", "--in", str(self.known(tmp_path))]) == 0
        page = tmp_path / "known.html"
        assert "Candidate Confounds" in page.read_text(encoding="utf-8") and opened == [page]

    def test_text_prints_the_confounds_and_opens_nothing(self, tmp_path, opened, capsys):
        assert cli.main(["review", "--in", str(self.known(tmp_path)), "--format", "text"]) == 0
        assert "Firm age affects Structure:" in capsys.readouterr().out and opened == []

    def test_text_needs_the_data_not_the_page(self, tmp_path, opened, capsys):
        page = tmp_path / "result.html"
        page.write_text("<html></html>")
        assert cli.main(["review", "--in", str(page), "--format", "text"]) == 1
        assert "needs the .zip or .json" in capsys.readouterr().err and opened == []

    @pytest.mark.parametrize("content", ["not json", "{}"])
    def test_a_file_that_is_not_a_result_fails_with_a_message(self, tmp_path, opened, capsys, content):
        path = tmp_path / "bad.json"
        path.write_text(content)
        assert cli.main(["review", "--in", str(path)]) == 1
        assert f"error: {path}" in capsys.readouterr().err and opened == []

    def test_a_missing_file_fails_with_a_message(self, tmp_path, capsys):
        assert cli.main(["review", "--in", str(tmp_path / "nope.html"), "--format", "text"]) == 1
        assert "error:" in capsys.readouterr().err

    def test_the_input_is_named_with_in(self, capsys):
        with pytest.raises(SystemExit):
            cli.main(["review"])
        assert "--in" in capsys.readouterr().err


class TestRunReview:
    def run(self, tmp_path, monkeypatch, env, *extra):
        monkeypatch.setattr(cli, "build_backend", fake_backend(env))
        return cli.main([*RUN_ARGV, "--out", str(tmp_path / "r.html"), *extra])

    def test_without_the_flag_nothing_is_opened_but_the_message_says_how(
        self, tmp_path, monkeypatch, capsys, env, opened
    ):
        assert self.run(tmp_path, monkeypatch, env) == 0
        assert opened == [] and "To review: confoundsearch review --in" in capsys.readouterr().out

    def test_review_opens_the_page_it_just_wrote(self, tmp_path, monkeypatch, env, opened):
        assert self.run(tmp_path, monkeypatch, env, "--review") == 0
        assert opened == [tmp_path / "r.html"]

    def test_review_text_prints_the_confounds(self, tmp_path, monkeypatch, capsys, env, opened):
        assert self.run(tmp_path, monkeypatch, env, "--review", "text") == 0
        assert "Candidate confounds for" in capsys.readouterr().out and opened == []

    def test_review_text_cannot_share_stdout_with_json(self, tmp_path, monkeypatch, capsys, env, opened):
        assert self.run(tmp_path, monkeypatch, env, "--review", "text", "--json") == 1
        assert "--json" in capsys.readouterr().err and not (tmp_path / "r.html").exists()

    def test_the_browser_can_open_while_json_owns_stdout(self, tmp_path, monkeypatch, capsys, env, opened):
        assert self.run(tmp_path, monkeypatch, env, "--review", "--json") == 0
        captured = capsys.readouterr()
        json.loads(captured.out)  # stdout is the JSON and nothing else
        assert "Opening the interface" in captured.err
        assert opened == [tmp_path / "r.html"]

    def test_opening_the_interface_is_said_and_the_findings_come_first(
        self, tmp_path, monkeypatch, capsys, env, opened
    ):
        assert self.run(tmp_path, monkeypatch, env, "--review") == 0
        out = capsys.readouterr().out
        assert "Candidate confounds for" in out and out.index("Candidate confounds") < out.index(" steps, ")
        assert out.startswith("\n\nCandidate confounds for")  # set apart by two blank lines
        assert out.rstrip().endswith("Opening the interface in your browser\u2026")


def test_a_share_data_json_is_used_as_it_is(tmp_path):
    data = known_to_share_data(KNOWN)
    path = tmp_path / "share.json"
    path.write_text(json.dumps(data))
    assert review.load_share(path) == data


def test_run_makes_the_output_folder_it_is_pointed_at(tmp_path, monkeypatch, env):
    monkeypatch.setattr(cli, "build_backend", fake_backend(env))
    out = tmp_path / "results" / "nested" / "org.html"
    assert cli.main([*RUN_ARGV, "--out", str(out)]) == 0
    assert out.exists() and (out.parent / "org.audit.jsonl").exists()
    audit = tmp_path / "audits" / "a.jsonl"
    assert cli.main([*RUN_ARGV, "--out", str(out), "--audit", str(audit)]) == 0 and audit.exists()


class TestFindingsText:
    def graph(self):
        from confoundsearch import Graph, Passage, Vertex

        g = Graph("IT", "perf")
        for label in ("firm size", "firm age", "weather"):
            g = g.add_confound(Vertex(label, frozenset({label})))
        for label, target in (("firm size", "X"), ("firm size", "Y"), ("firm age", "X")):
            g = g.add_edge(label, target).attach(label, target, Passage(f"{label}{target}", "t", "T", "u"))
        return g

    def test_each_confound_with_evidence_is_listed_with_its_paper_counts(self):
        lines = review.findings_text(self.graph()).splitlines()
        assert lines[0] == "Candidate confounds for IT → perf:"
        assert lines[1].strip().startswith("firm age") and lines[1].endswith("(one link only)")  # by name
        assert "0 papers \u2192 perf" in lines[1]
        assert lines[2].strip() == "firm size  1 paper \u2192 IT \u00b7 1 paper \u2192 perf"
        assert len(lines) == 3  # "weather" had no evidence

    def test_no_evidence_says_so(self):
        from confoundsearch import Graph

        assert review.findings_text(Graph("IT", "perf")).startswith("No candidate confound had")
