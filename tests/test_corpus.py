from __future__ import annotations

from confoundsearch import InMemoryCorpus, Passage
from tests.conftest import P1, P2, P3, P4


def test_ranks_by_word_overlap_then_id(corpus):
    hits = corpus.search("larger firms invest", 10)
    assert hits[0] == P1 and set(hits) == {P1, P2}


def test_ties_break_by_id():
    c = InMemoryCorpus([Passage("b", "alpha"), Passage("a", "alpha")])
    assert [p.id for p in c.search("alpha", 5)] == ["a", "b"]


def test_respects_k(corpus):
    assert len(corpus.search("firms", 1)) == 1


def test_non_positive_k_returns_nothing(corpus):
    assert corpus.search("firms", 0) == [] and corpus.search("firms", -3) == []


def test_no_match_returns_nothing(corpus):
    assert corpus.search("zebra", 5) == []


def test_matching_is_case_and_space_insensitive(corpus):
    assert corpus.search("  HEAD   COUNT ", 5) == [P3]


def test_irrelevant_passage_never_returned_for_other_queries(corpus):
    assert P4 not in corpus.search("firms performance profits", 5)


def test_from_jsonl(tmp_path):
    path = tmp_path / "c.jsonl"
    path.write_text(
        '{"id": 1, "text": "alpha beta", "title": "T", "url": "https://x"}\n\n{"id": "b", "text": "beta"}\n'
    )
    corpus = InMemoryCorpus.from_jsonl(path)
    assert corpus.passages == (Passage("1", "alpha beta", "T", "https://x"), Passage("b", "beta"))
    assert [p.id for p in corpus.search("beta", 5)] == ["1", "b"]
