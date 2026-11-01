# ruff: noqa: E501
"""The standalone page renders every kind of state the package can export, in the interface's one style."""

from __future__ import annotations

import json
import re

import pytest

from confoundsearch import AddConfound, AddEdge, AttachPassage, Retrieve, Vertex
from confoundsearch.interface.export import to_share_data
from confoundsearch.interface.ui import page, standalone_html
from tests.conftest import DV, IV
from tests.test_export import FULL, play


def render(env, *actions) -> str:
    return standalone_html(to_share_data(play(env, *actions)))


def blob(html: str) -> dict:
    return json.loads(re.search(r"var EVIDENCE_BLOB = (\{.*?\});\n", html, re.S).group(1))


def share(*confounds, names=None):
    """A share_data dict; each confound is (name, [to_iv entries], [to_dv entries])."""
    entries = [{"name": n, "evidence_to_iv": iv, "evidence_to_dv": dv} for n, iv, dv in confounds]
    return {
        "iv": "X var",
        "dv": "Y var",
        "dag_confounders": entries,
        "all_md_names": names or [c[0] for c in confounds],
    }


def paper(title="A paper", **fields):
    return {
        "title": title,
        "url": "https://doi.org/10.1/x",
        "quote": "Quoted words.",
        "llm_label": "Yes",
        **fields,
    }


class TestARun:
    def test_empty_graph_renders(self, env):
        html = render(env)
        assert html.startswith("<!DOCTYPE html>") and "No confounders found" in html

    def test_page_names_the_relationship(self, env):
        html = render(env, *FULL)
        assert f"<title>ConfoundSearch — {IV} → {DV}</title>" in html
        assert f"<b>{IV}</b> and <b>{DV}</b>" in html

    def test_page_embeds_the_graph_and_evidence(self, env):
        html = render(env, *FULL)
        assert "firm size" in html and "dag-svg" in html and "EVIDENCE_BLOB" in html
        assert "larger firms invest more in IT" in html  # the quote behind the P(C, X) edge

    def test_page_is_self_contained_apart_from_fonts(self, env):
        sources = re.findall(r'(?:src|href)="(https?://[^"]+)"', render(env, *FULL))
        assert all(
            s.startswith(("https://fonts.googleapis.com", "https://doi.org")) or "p1" in s for s in sources
        )

    @pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6])
    def test_renders_any_number_of_confounds_and_draws_at_most_five(self, env, n):
        actions = []
        for i in range(n):
            label = f"c{i}"
            actions += [AddConfound(label, frozenset({f"s{i}"})), AddEdge(label, "X"), AddEdge(label, "Y")]
        actions += [Retrieve("larger"), Retrieve("profits")]
        state = play(env, *actions)
        for i in range(n):
            for target, pid in (("X", "p1"), ("Y", "p2")):
                state = env.step(state, AttachPassage(f"c{i}", target, pid))
        html = standalone_html(to_share_data(state))
        assert html.count('class="md-card"') == n
        assert html.count('class="dag-md-circle"') == min(n, 5)
        assert all(
            f"__meta__:c{i}" in blob(html) for i in range(n)
        )  # a card past the fifth still has evidence

    def test_unconnected_confounds_render_without_error(self, env):
        state = play(env, AddConfound("a", frozenset({"a"})), AddEdge("a", "X"))
        assert standalone_html(to_share_data(state)).startswith("<!DOCTYPE html>")

    def test_hostile_names_do_not_inject_markup(self, env):
        state = play(env)
        state = state.with_graph(
            state.graph.add_confound(Vertex("<script>alert(1)</script>", frozenset({"a"})))
        )
        assert "<script>alert(1)" not in standalone_html(to_share_data(state))


class TestTheLayout:
    def html(self):
        return standalone_html(
            share(
                ("IT intensity", [paper("P1")], [paper("P2"), paper("P3")]), ("CEO Style", [paper("P4")], [])
            )
        )

    def test_each_card_lists_its_two_links_with_their_paper_counts(self):
        html = self.html()
        assert '<div class="card-name">IT intensity</div>' in html
        assert 'data-md="IT intensity" data-leg="to_iv">\u2192 X var <small>1 paper</small></button>' in html
        assert 'data-md="IT intensity" data-leg="to_dv">\u2192 Y var <small>2 papers</small></button>' in html
        assert 'data-md="CEO Style" data-leg="to_dv">\u2192 Y var <small>0 papers</small></button>' in html
        assert "Candidate Confounds</div>" in html

    def test_each_confound_has_its_own_colour_on_its_card_node_and_arrows(self):
        html = self.html()
        first, second = page.COLORS[0], page.COLORS[1]
        assert first != second
        assert f'class="md-card" data-md="IT intensity" data-color="{first}"' in html
        assert 'class="dag-md-circle" cx="' in html and f'fill="{second}"' in html
        assert f'data-leg="to_iv" data-color="{first}"' in html

    def test_the_diagram_has_one_node_and_two_arrows_per_drawn_confound(self):
        html = self.html()
        assert html.count('class="dag-md-circle"') == 2 and html.count("<line ") == 5  # 4 edges and X to Y
        assert (
            'data-md="IT intensity" data-leg="to_iv"' in html and "dagLegClick('CEO Style','to_dv')" in html
        )

    def test_a_count_shows_only_on_a_chosen_arrow(self):
        html = self.html()
        assert 'class="dag-count small"' not in html  # no little numbers on the arrows
        assert html.count('class="dag-count pill"') == 4  # one per arrow, hidden until it is chosen
        assert ">2 papers</text>" in html and ">1 paper</text>" in html and ">0 papers</text>" in html
        css = page.asset("page.css")
        assert (
            ".dag-count.pill{display:none;}" in css
            and ".dag-svg.has-sel .dag-count.pill.sel{display:inline;}" in css
        )

    def test_an_arrow_with_no_papers_is_dashed(self):
        html = self.html()
        lines = [
            line for line in html.splitlines() if line.startswith("<line") and 'data-md="CEO Style"' in line
        ]
        assert [("stroke-dasharray" in line) for line in lines] == [
            False,
            True,
        ]  # to_iv has a paper, to_dv none

    def test_the_variables_say_what_they_are_and_the_black_arrow_what_it_is(self):
        html = self.html()
        assert ">cause (IV)</text>" in html and ">outcome (DV)</text>" in html
        assert ">the relationship studied</text>" in html
        assert html.count('class="dag-ring"') == 2 and html.count('class="dag-var"') == 2

    def test_a_single_confound_sits_at_the_top_of_the_arc(self):
        html = standalone_html(share(("Only", [paper()], [paper()])))
        assert 'cx="320.0" cy="130.0"' in html

    def test_three_confounds_fan_out_over_the_arc_like_the_reference_page(self):
        html = standalone_html(share(*[(n, [paper()], [paper()]) for n in "abc"]))
        assert (
            'cx="135.0" cy="208.0"' in html
            and 'cx="320.0" cy="130.0"' in html
            and 'cx="505.0" cy="208.0"' in html
        )

    def test_variable_names_are_split_over_two_lines_at_the_middle(self):
        assert page.two_lines("Organizational structure") == ["Organizational", "structure"]
        assert page.two_lines("Firm") == ["Firm"]
        assert len(page.two_lines("A very long variable name")) == 2

    def test_names_with_quotes_still_click(self):
        html = standalone_html(share(("Owner's style", [paper()], [paper()])))
        assert "dagMetaClick('Owner\\&#x27;s style')" in html
        assert "dagLegClick('Owner\\&#x27;s style','to_iv')" in html


class TestTheEvidence:
    def evidence(self, to_iv, to_dv):
        return blob(standalone_html(share(("Age", to_iv, to_dv))))

    def test_a_link_names_its_edge_and_counts_its_papers(self):
        evidence = self.evidence([paper("One")], [paper("A"), paper("B")])
        iv_leg, dv_leg = evidence["__leg__:Age:to_iv"], evidence["__leg__:Age:to_dv"]
        assert (
            '<span class="chip">Age</span><span class="edge-arrow">\u2192</span><span class="var-pill">X var</span>'
            in iv_leg
        )
        assert "<b>1 paper</b> suggests that Age affects X var." in iv_leg
        assert "<b>2 papers</b> suggest that Age affects Y var." in dv_leg

    def test_the_edge_header_the_pill_outline_and_the_side_bar_share_the_confounds_colour(self):
        color = page.COLORS[0]
        leg = self.evidence([paper()], [paper()])["__leg__:Age:to_iv"]
        assert (
            f'data-color="{color}" style="--c:{color}"' in leg
        )  # the CSS draws chip, arrow, pill outline and bar from this
        css = page.asset("page.css")
        assert ".var-pill{border:2px solid var(--c" in css and ".chip{background:var(--c" in css
        assert "border-left:6px solid" in css and "box.style.borderLeftColor" in page.asset("page.js")

    def test_a_leg_points_to_the_confounds_other_edge(self):
        iv_leg = self.evidence([paper()], [paper("A"), paper("B")])["__leg__:Age:to_iv"]
        assert (
            "Also: <b>Age \u2192 Y var</b> \u00b7 2 papers" in iv_leg
            and "dagLegClick('Age','to_dv')" in iv_leg
        )

    def test_a_confound_shows_both_of_its_edges_one_after_the_other(self):
        meta = self.evidence([paper("One")], [paper("A")])["__meta__:Age"]
        assert (
            meta.count('class="edge-section"') == 2
            and "Age affects X var" in meta
            and "Age affects Y var" in meta
        )
        assert meta.index('var-pill">X var') < meta.index('var-pill">Y var')

    def test_a_paper_says_what_it_supports_then_the_quote_then_title_and_byline(self):
        entry = paper(
            "Firm Age", authors="A. Author, B. Writer", venue="A Journal", year=2012, quote="It's a result."
        )
        html = self.evidence([entry], [])["__leg__:Age:to_iv"]
        order = [
            html.index("Supports: Age \u2192 X var"),
            html.index("\u201cIt&#x27;s a result.\u201d"),
            html.index(">Firm Age</a>"),
            html.index("A. Author, B. Writer \u00b7 A Journal \u00b7 2012"),
        ]
        assert order == sorted(order)
        assert '<a href="https://doi.org/10.1/x" target="_blank">' in html

    def test_the_byline_and_link_are_left_out_when_unknown(self):
        entry = {"title": "Bare", "quote": "Q.", "llm_label": "Yes"}
        html = self.evidence([entry], [])["__leg__:Age:to_iv"]
        assert "<b>Bare</b>" in html and 'class="byline"' not in html
        assert page.byline({"venue": "Only a venue"}) == "Only a venue"

    def test_a_missing_title_and_quote_are_handled(self):
        html = self.evidence([{"url": "", "llm_label": "Yes"}], [])["__leg__:Age:to_iv"]
        assert "<b>Unknown</b>" in html and 'class="quote"' not in html

    def test_a_very_long_quotation_is_cut_with_an_ellipsis(self):
        html = self.evidence([paper(quote="word " * 400)], [])["__leg__:Age:to_iv"]
        assert "\u2026\u201d" in html and len(html) < 3000

    def test_only_accepted_papers_are_shown(self):
        evidence = self.evidence([paper("Kept"), paper("Rejected", llm_label="No")], [])
        assert "Kept" in evidence["__leg__:Age:to_iv"] and "Rejected" not in evidence["__leg__:Age:to_iv"]

    def test_a_link_with_no_papers_says_so(self):
        assert "No papers found that suggest Age affects X var." in self.evidence([], [])["__leg__:Age:to_iv"]

    def test_papers_are_separated_by_a_rule(self):
        html = self.evidence([paper("One"), paper("Two")], [])["__leg__:Age:to_iv"]
        assert html.count("border-top:1px solid #e6e9ee") == 1  # between the two papers
