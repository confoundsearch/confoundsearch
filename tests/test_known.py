"""A JSON blob of known confounds -> the interface's page. The same page a run exports."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from confoundsearch.interface import cli
from confoundsearch.interface.known import known_to_share_data, load_known, page_from_json

ROOT = Path(__file__).resolve().parent.parent
KNOWN = ROOT / "examples" / "known_confounds.json"
PAGE = ROOT / "assets" / "known_confounds.html"

TINY = {
    "iv": "Structure",
    "dv": "Performance",
    "controls": [
        {
            "name": "Firm <age>",
            "citations_to_IV": [
                {"title": 'A "quoted" title', "doi": "10.1/a", "quotation": "Young firms decentralize."}
            ],
            "citations_to_DV": [{"title": "No DOI paper", "quotation": "Older firms earn less."}],
        },
        {
            "name": "Size",
            "citations_to_DV": [{"title": "Only DV", "doi": "10.1/b", "quotation": "Big firms."}],
        },
    ],
}


def blob(page: str) -> dict:
    return json.loads(re.search(r"var EVIDENCE_BLOB = (\{.*?\});\n", page, re.S).group(1))


class TestTheShareData:
    def test_each_control_is_one_meta_dimension_with_its_papers_on_each_leg(self):
        data = known_to_share_data(TINY)
        assert (data["iv"], data["dv"]) == ("Structure", "Performance")
        first, second = data["dag_confounders"]
        assert first["name"] == "Firm age"  # angle brackets never reach the page
        [to_iv], [to_dv] = first["evidence_to_iv"], first["evidence_to_dv"]
        assert to_iv == {
            "title": "A ”quoted” title",
            "url": "https://doi.org/10.1/a",
            "quote": "Young firms decentralize.",
            "llm_label": "Yes",
        }
        assert to_dv["url"] == ""  # no DOI, no link
        assert second["evidence_to_iv"] == []  # a leg an expert gave nothing for is empty, not an error
        assert [m["name"] for m in data["dag_grouped"]["meta_dimensions"]] == ["Firm age", "Size"]

    def test_every_control_is_listed_even_if_it_is_not_supported_on_both_legs(self):
        assert known_to_share_data(TINY)["all_md_names"] == ["Firm age", "Size"]


class TestThePage:
    def test_a_known_confounds_blob_becomes_the_standalone_page(self):
        page = page_from_json(TINY)
        assert "<title>ConfoundSearch — Structure → Performance</title>" in page
        assert "Young firms decentralize." in page

    def test_a_share_data_blob_is_shown_as_it_is(self):
        data = known_to_share_data(TINY)
        assert page_from_json(data) == page_from_json(TINY)

    def test_anything_else_is_refused(self):
        with pytest.raises(ValueError, match="not a known-confounds file"):
            page_from_json({"hello": 1})

    def test_the_papers_sit_on_the_leg_they_support(self):
        known = load_known(KNOWN)
        assert (known["iv"], known["dv"]) == ("Organizational structure", "Firm performance")
        evidence = blob(page_from_json(known))
        to_iv = evidence["__leg__:Market Competition:to_iv"]
        to_dv = evidence["__leg__:Market Competition:to_dv"]
        assert "The Flattening Firm" in to_iv and "Competition and Corporate Performance" not in to_iv
        assert "Competition and Corporate Performance" in to_dv and "The Flattening Firm" not in to_dv
        assert "<b>2 papers</b> suggest that Market Competition affects Firm performance." in to_dv
        ceo_dv = evidence["__leg__:CEO Style:to_dv"]
        assert "<b>1 paper</b> suggests that CEO Style affects Firm performance." in ceo_dv

    def test_every_confound_has_evidence_for_its_card_and_both_arrows(self):
        known = load_known(KNOWN)
        evidence = blob(page_from_json(known))
        assert [c["name"] for c in known["controls"]] == ["IT intensity", "CEO Style", "Market Competition"]
        for control in known["controls"]:
            for key in ("__meta__", "__leg__"):
                assert any(k.startswith(f"{key}:{control['name']}") for k in evidence)

    def test_counts_say_paper_or_papers(self):
        html = page_from_json(load_known(KNOWN))
        assert "<small>1 paper</small>" in html and "<small>2 papers</small>" in html
        assert "1 papers" not in html

    def test_the_quoted_passage_is_the_main_text(self):
        leg = blob(page_from_json(TINY))["__leg__:Firm age:to_iv"]
        assert '<p class="quote">\u201cYoung firms decentralize.\u201d</p>' in leg
        css = page_from_json(TINY)
        assert (
            ".quote{margin:0 0 8px;font-size:15px;line-height:1.55;font-style:italic;color:var(--ink);}"
            in css
        )

    def test_the_authors_journal_and_year_show_when_the_file_gives_them(self):
        known = load_known(KNOWN)
        evidence = blob(page_from_json(known))
        leg = evidence["__leg__:Market Competition:to_iv"]
        assert '<span class="authors">Nicholas Bloom, Raffaella Sadun, John Van Reenen</span>' in leg
        assert (
            '<span class="venue">The Quarterly Journal of Economics</span>' in leg and ">2012</span>" in leg
        )

    def test_the_example_page_is_current(self):
        assert PAGE.read_text(encoding="utf-8") == page_from_json(load_known(KNOWN))


class TestTheCommand:
    def test_show_writes_the_page_next_to_the_file_by_default(self, tmp_path, capsys):
        source = tmp_path / "answers.json"
        source.write_text(json.dumps(TINY))
        assert cli.main(["show", str(source)]) == 0
        assert (tmp_path / "answers.html").read_text(encoding="utf-8") == page_from_json(TINY)
        assert f"wrote {tmp_path / 'answers.html'}" in capsys.readouterr().out

    def test_the_output_path_can_be_chosen(self, tmp_path):
        source = tmp_path / "a.json"
        source.write_text(json.dumps(TINY))
        assert cli.main(["show", str(source), "--out", str(tmp_path / "page.html")]) == 0
        assert (tmp_path / "page.html").exists()

    def test_a_file_that_is_not_known_confounds_fails_with_a_message(self, tmp_path, capsys):
        source = tmp_path / "a.json"
        source.write_text("{}")
        assert cli.main(["show", str(source)]) == 1
        assert "not a known-confounds file" in capsys.readouterr().err
        assert not (tmp_path / "a.html").exists()
