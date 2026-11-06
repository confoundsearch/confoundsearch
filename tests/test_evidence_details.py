"""Who wrote each paper and how often it is cited, ranking by that, and marking the best sentence."""

from __future__ import annotations

import json
import sqlite3

import pytest

from confoundsearch import Graph, LLMHighlights, Passage, Vertex
from confoundsearch.data.cache import PaperDatabase
from confoundsearch.data.corpus import InMemoryCorpus as Corpus
from confoundsearch.data.multi import MultiCorpus
from confoundsearch.data.sources import OpenAlexCorpus, SemanticScholarCorpus, authors_text, venue_text
from confoundsearch.interface.export import choose_highlights, most_cited_first, to_share_data
from confoundsearch.interface.ui.page import byline_html, quotation
from confoundsearch.logic.state import State
from tests.test_sources import OA_BODY, S2_BODY, api  # noqa: F401  (the fixture)


def cited(pid, n, title=None, **more):
    return Passage(
        pid, f"Text of {pid}.", title or f"Paper {pid}", "https://doi.org/10.1/" + pid, cited_by=n, **more
    )


class TestWhatASourceKnows:
    def test_authors_come_as_the_first_three_then_et_al(self):
        assert authors_text(["A  One", "B Two", "C Three"]) == "A One, B Two, C Three"
        assert authors_text(["A", "B", "C", "D", " ", None]) == "A, B, C et al."
        assert authors_text([]) == ""

    @pytest.mark.parametrize(
        "paper, expected",
        [
            ({"venue": "Direct"}, "Direct"),
            ({"venue": "", "journal": {"name": "A Journal"}}, "A Journal"),
            ({"journal": None, "publicationVenue": {"name": "A Conference"}}, "A Conference"),
            ({}, ""),
        ],
    )
    def test_the_venue_is_the_first_name_the_source_gives(self, paper, expected):
        assert venue_text(paper) == expected

    def test_semantic_scholar_gives_authors_venue_year_and_citations(self, api):  # noqa: F811
        body = {
            "data": [
                {
                    "paperId": "a",
                    "corpusId": 1,
                    "title": "T",
                    "abstract": "Abstract.",
                    "url": "u",
                    "year": 1996,
                    "citationCount": 2668,
                    "authors": [{"name": "S. Nickell"}, None, {"name": None}],
                    "journal": {"name": "Journal of Political Economy"},
                }
            ]
        }
        server = api(body)
        (found,) = SemanticScholarCorpus(base_url=server.url).search("q", 1)
        assert (found.authors, found.venue, found.year, found.cited_by) == (
            "S. Nickell",
            "Journal of Political Economy",
            1996,
            2668,
        )
        assert {"authors", "journal", "publicationVenue"} <= set(
            server.requests[0]["query"]["fields"][0].split(",")
        )

    def test_openalex_gives_authors_venue_year_and_citations(self, api):  # noqa: F811
        work = dict(OA_BODY["results"][0])
        work.update(
            publication_year=2010,
            cited_by_count=321,
            authorships=[{"author": {"display_name": "Maria Guadalupe"}}, {"author": None}],
            primary_location={"source": {"display_name": "AEJ: Applied"}},
        )
        server = api({"results": [work]})
        (found,) = OpenAlexCorpus(base_url=server.url).search("q", 1)
        assert (found.authors, found.venue, found.year, found.cited_by) == (
            "Maria Guadalupe",
            "AEJ: Applied",
            2010,
            321,
        )
        select = server.requests[0]["query"]["select"][0]
        assert {"cited_by_count", "authorships", "primary_location"} <= set(select.split(","))

    def test_a_source_that_says_nothing_leaves_the_fields_empty(self, api):  # noqa: F811
        (found, _) = OpenAlexCorpus(base_url=api(OA_BODY).url).search("q", 5)
        assert (found.authors, found.venue, found.year, found.cited_by) == ("", "", None, None)

    def test_a_jsonl_file_may_carry_them(self, tmp_path):
        path = tmp_path / "c.jsonl"
        path.write_text(
            json.dumps({"id": 1, "text": "t", "authors": "A", "venue": "V", "year": 1999, "cited_by": 5})
            + "\n"
            + json.dumps({"id": 2, "text": "u"})
            + "\n"
        )
        first, second = Corpus.from_jsonl(path).passages
        assert (first.authors, first.venue, first.year, first.cited_by) == ("A", "V", 1999, 5)
        assert (second.authors, second.venue, second.year, second.cited_by) == ("", "", None, None)


class TestSavedPapers:
    def test_they_come_back_with_their_details(self, tmp_path):
        database = PaperDatabase(tmp_path / "d.sqlite3")
        database.save("s", "q", 3, [Passage("1", "text", "Title", "u", "A. B", "A Journal", 1996, 12)])
        (back,) = database.load("s", "q", 3)
        assert (back.authors, back.venue, back.year, back.cited_by) == ("A. B", "A Journal", 1996, 12)

    def test_a_later_copy_fills_what_the_first_lacked_and_updates_the_count(self, tmp_path):
        database = PaperDatabase(tmp_path / "d.sqlite3")
        database.save("s", "q", 3, [Passage("1", "text", "Title", "u")])
        database.save("s", "r", 3, [Passage("1", "text", "Title", "u", "A. B", "A Journal", 1996, 12)])
        database.save(
            "s", "t", 3, [Passage("1", "text", "Title", "u", "", "", None, 40)]
        )  # keeps what it had
        (back,) = database.load("s", "t", 3)
        assert (back.authors, back.venue, back.year, back.cited_by) == ("A. B", "A Journal", 1996, 40)

    def test_a_data_directory_from_before_gets_the_new_columns(self, tmp_path):
        path = tmp_path / "old.sqlite3"
        db = sqlite3.connect(path)
        db.executescript(
            "CREATE TABLE papers (key TEXT PRIMARY KEY, id TEXT NOT NULL, title TEXT NOT NULL, "
            "url TEXT NOT NULL, text TEXT NOT NULL, source TEXT NOT NULL);"
            "INSERT INTO papers VALUES ('k', '1', 'Old', 'u', 'text', 's');"
            "CREATE TABLE searches (source TEXT NOT NULL, query TEXT NOT NULL, k INTEGER NOT NULL, "
            "papers TEXT NOT NULL, saved TEXT NOT NULL, PRIMARY KEY (source, query, k));"
            "INSERT INTO searches VALUES ('s', 'q', 3, '[\"k\"]', 'now');"
        )
        db.commit()
        db.close()
        (old,) = PaperDatabase(path).load("s", "q", 3)
        assert (old.title, old.authors, old.venue, old.year, old.cited_by) == ("Old", "", "", None, None)
        PaperDatabase(path).save("s", "r", 3, [Passage("2", "new", "New", "u", "A", "V", 2000, 3)])
        assert PaperDatabase(path).load("s", "r", 3)[0].cited_by == 3

    def test_merging_sources_keeps_the_details_of_the_fuller_copy_or_the_other(self):
        class One:
            def __init__(self, found):
                self.found = found

            def search(self, query, k):
                return self.found

        full = Passage("a", "a longer abstract text", "Same Title", "u", "A. B", "A Journal", 1996, 12)
        bare = Passage("b", "short", "Same Title", "u2")
        longer_but_bare = Passage("c", "an even longer abstract text than any other", "Same Title", "u3")
        both = MultiCorpus({"x": One([full]), "y": One([bare])}).search("q", 5)
        assert [(p.authors, p.cited_by) for p in both] == [("A. B", 12)]
        later = MultiCorpus({"x": One([full]), "y": One([longer_but_bare])}).search("q", 5)
        assert (later[0].text, later[0].authors, later[0].year, later[0].cited_by) == (
            longer_but_bare.text,
            "A. B",
            1996,
            12,
        )


def state_with(*cited_papers):
    graph = Graph("IT", "perf").add_confound(Vertex("size", frozenset({"size"})))
    state = State(graph)
    for target in ("X", "Y"):
        state = state.with_graph(state.graph.add_edge("size", target))
    for passage in cited_papers:
        state = state.with_graph(state.graph.attach("size", "X", passage))
    return state


class TestTheResult:
    def test_papers_come_most_cited_first_and_those_without_a_count_last(self):
        state = state_with(
            cited("a", 5, "Alpha"), cited("b", 900, "Beta"), cited("c", None, "Gamma"), cited("d", 5, "Delta")
        )
        titles = [e["title"] for e in to_share_data(state)["dag_confounders"][0]["evidence_to_iv"]]
        assert titles == ["Beta", "Alpha", "Delta", "Gamma"]  # ties by title; unknown last
        assert most_cited_first(cited("a", 5)) < most_cited_first(cited("a", 4))

    def test_each_entry_carries_what_is_known_about_the_paper(self):
        passage = Passage("a", "Text.", "T", "https://doi.org/10.1/a", "A. B", "A Journal", 1996, 12)
        (entry,) = to_share_data(state_with(passage))["dag_confounders"][0]["evidence_to_iv"]
        assert (entry["authors"], entry["venue"], entry["year"], entry["cited_by"]) == (
            "A. B",
            "A Journal",
            1996,
            12,
        )
        bare = to_share_data(state_with(Passage("b", "Text.", "B")))["dag_confounders"][0]["evidence_to_iv"][
            0
        ]
        assert not {"authors", "venue", "year", "cited_by", "highlight"} & set(bare)

    def test_a_highlight_goes_with_its_link_and_paper(self):
        state = state_with(cited("a", 1), cited("b", 2))
        data = to_share_data(state, {("size", "X", "a"): "Text of a."})
        by_title = {e["title"]: e for e in data["dag_confounders"][0]["evidence_to_iv"]}
        assert by_title["Paper a"]["highlight"] == "Text of a." and "highlight" not in by_title["Paper b"]

    def test_the_best_sentence_is_asked_for_once_for_each_link_with_papers(self):
        state = state_with(cited("a", 1), cited("b", 9))
        asked = []

        def highlight(confound, target, passages):
            asked.append((confound, target, [p.id for p in passages]))
            return {"a": "Text of a."}

        assert choose_highlights(state, highlight) == {("size", "X", "a"): "Text of a."}
        assert asked == [("size", "IT", ["b", "a"])]  # most cited first; a link with no papers is not asked


class FakeClient:
    def __init__(self, *replies):
        self.replies, self.seen = list(replies), []

    def chat(self, messages):
        self.seen.append(messages[0]["content"])
        return self.replies.pop(0)


class TestPickingTheBestSentence:
    PASSAGES = (
        Passage("a", "Intro words. Competition raises growth, they find.  More words.", "A"),
        Passage("b", "Nothing relevant at all.", "B"),
    )

    def pick(self, reply):
        client = FakeClient(reply)
        return LLMHighlights(client)("competition", "performance", self.PASSAGES), client

    def test_the_sentence_comes_back_as_the_paper_wrote_it(self):
        reply = '{"id": "a", "sentence": "“Competition  raises growth, they find.…"}'
        found, client = self.pick(reply)
        assert found == {"a": "Competition raises growth, they find."}
        assert '"competition" affects "performance"' in client.seen[0] and "id: b" in client.seen[0]

    def test_it_ignores_what_it_cannot_check(self):
        reply = "\n".join(
            [
                "```json",
                "not json",
                '{"id": "zzz", "sentence": "Competition raises growth, they find."}',  # not one it was given
                '{"id": "a", "sentence": "A sentence the paper never wrote."}',
                '{"id": "a"}',
                '{"id": "a", "sentence": "   "}',
                '{"id": "a", "sentence": "COMPETITION RAISES GROWTH, they find."}',
                '{"id": "a", "sentence": "Intro words."}',  # only the first usable one counts
                "```",
            ]
        )
        found, _ = self.pick(reply)
        assert found == {"a": "Competition raises growth, they find."}

    def test_no_papers_asks_nothing(self):
        client = FakeClient()
        assert LLMHighlights(client)("c", "t", []) == {} and client.seen == []


class TestTheDisplay:
    LONG = "Start words. " * 90 + "The key result is that competition raises growth. " + "End words. " * 90

    def test_the_marked_sentence_is_shown_in_italics_inside_the_quotation(self):
        out = quotation("Before. Key sentence. After.", "Key sentence.")
        assert out == '“Before. <mark class="hl"><b>Key sentence.</b></mark> After.”'

    def test_a_marked_sentence_beyond_the_cut_is_kept_in_view(self):
        sentence = "The key result is that competition raises growth."
        out = quotation(self.LONG, sentence)
        assert f'<mark class="hl"><b>{sentence}</b></mark>' in out
        assert out.startswith("“…") and out.endswith("…”") and len(out) < 1000

    def test_an_early_sentence_in_a_long_text_just_cuts_the_end(self):
        out = quotation("Key sentence. " + "word " * 400, "Key sentence.")
        assert out.startswith("“<mark") and out.endswith("…”")

    def test_a_highlight_that_is_not_in_the_text_changes_nothing(self):
        assert (
            quotation("Plain <text>.", "something else")
            == quotation("Plain <text>.")
            == "“Plain &lt;text&gt;.”"
        )

    def test_without_a_space_to_snap_to_it_still_starts_before_the_sentence(self):
        text = "x" * 1500 + "KEY." + "y" * 50  # one very long word
        out = quotation(text, "KEY.")
        assert '<mark class="hl"><b>KEY.</b></mark>' in out

    def test_a_long_quotation_is_cut_at_a_whole_word(self):
        text = "alphabet gamma " * 100  # 1500 characters, with long words to be cut in half
        words = quotation(text)[1:-2].split()  # inside the quotation marks, without the ellipsis
        assert set(words) <= {"alphabet", "gamma"}
        marked = quotation(text + "KEY SENTENCE. " + text, "KEY SENTENCE.")
        assert '<mark class="hl"><b>KEY SENTENCE.</b></mark>' in marked
        assert set(marked.replace("\u2026", " ").replace("\u201c", " ").split()) - {"alphabet", "gamma"} <= {
            "<mark",
            'class="hl"><b>KEY',
            "SENTENCE.</b></mark>",
            "\u201d",
        } | {"SENTENCE.</b></mark>\u201d"}

    def test_the_byline_shows_authors_venue_year_and_citations(self):
        entry = {
            "authors": "S. Nickell",
            "venue": "Journal of Political Economy",
            "year": 1996,
            "cited_by": 2668,
        }
        out = byline_html(entry)
        assert out.count('<span class="sep">') == 3 and "cited 2,668\u00d7" in out
        assert '<span class="venue">Journal of Political Economy</span>' in out
        assert byline_html({}) == "" and byline_html({"cited_by": 0}) == ""
        assert byline_html({"authors": "<b>X</b>"}) == '<span class="authors">&lt;b&gt;X&lt;/b&gt;</span>'


class TestADemoFileFromBefore:
    """A file made before authors, venue, year and citations were saved has papers, and no such columns."""

    def make(self, tmp_path, with_example_set):
        path = tmp_path / "old-demo.sqlite3"
        db = sqlite3.connect(path)
        db.executescript(
            "CREATE TABLE papers (key TEXT PRIMARY KEY, id TEXT, title TEXT, url TEXT, text TEXT, "
            "source TEXT);"
            "INSERT INTO papers VALUES ('a', '1', 'Firm size', 'u', 'Big firms win', 's');"
            "INSERT INTO papers VALUES ('b', '2', 'Weather', 'u', 'Rain and tourism', 's');"
        )
        if with_example_set:
            db.executescript(
                "CREATE TABLE example_papers (key TEXT PRIMARY KEY); INSERT INTO example_papers VALUES ('a');"
            )
        db.commit()
        db.close()
        return path

    @pytest.mark.parametrize("with_example_set, expected", [(True, ["1"]), (False, ["1", "2"])])
    def test_its_papers_load_without_the_details(self, tmp_path, with_example_set, expected):
        from confoundsearch.data.demo import DemoCorpus

        corpus = DemoCorpus(self.make(tmp_path, with_example_set))
        found = corpus.search("firm size rain tourism", 5)
        assert sorted(p.id for p in found) == expected
        assert all((p.authors, p.venue, p.year, p.cited_by) == ("", "", None, None) for p in found)

    def test_a_file_that_is_not_a_papers_database_is_an_error(self, tmp_path):
        from confoundsearch.data.demo import DemoCorpus
        from confoundsearch.errors import CorpusError

        path = tmp_path / "other.sqlite3"
        db = sqlite3.connect(path)
        db.execute("CREATE TABLE other (x)")
        db.commit()
        db.close()
        with pytest.raises(CorpusError, match="could not be read"):
            DemoCorpus(path).search("q", 1)
