import pytest

from confoundsearch import ExpandedCorpus, LLMRewordings, Passage
from confoundsearch.errors import LLMError
from tests.test_llm import FakeClient


def passage(pid):
    return Passage(pid, f"text {pid}", f"title {pid}", "")


class Lists:
    """A corpus that answers each query with a fixed list of ids, and remembers what was asked."""

    def __init__(self, answers):
        self.answers, self.asked = answers, []

    def search(self, query, k):
        self.asked.append((query, k))
        return [passage(pid) for pid in self.answers.get(query, [])][:k]


class TestExpandedCorpus:
    def test_a_paper_several_phrasings_return_comes_first(self):
        corpus = Lists({"a": ["x", "y"], "b": ["z", "y"]})
        found = ExpandedCorpus(corpus, lambda q: ["b"]).search("a", 3)
        assert [p.id for p in found] == ["y", "x", "z"]  # y is second twice; x and z are first once each
        assert corpus.asked == [("a", 3), ("b", 3)]

    def test_ties_break_by_id_and_the_list_is_cut_at_k(self):
        corpus = Lists({"a": ["m", "n"], "b": ["n", "m"]})
        assert [p.id for p in ExpandedCorpus(corpus, lambda q: ["b"]).search("a", 1)] == ["m"]

    def test_rewordings_are_cleaned_and_limited(self):
        corpus = Lists({})
        expanded = ExpandedCorpus(corpus, lambda q: ["A", " ", "b ", "B", "c", "d"], limit=2)
        assert expanded.queries("a") == ["a", "b", "c"]  # no blank, no repeat in another case, two at most

    def test_a_failing_model_leaves_the_plain_search(self):
        def broken(query):
            raise LLMError("down")

        corpus = Lists({"a": ["x"]})
        assert [p.id for p in ExpandedCorpus(corpus, broken).search("a", 5)] == ["x"]

    @pytest.mark.parametrize("query,k", [("", 5), ("  ", 5), ("a", 0)])
    def test_nothing_to_search_asks_nothing(self, query, k):
        corpus = Lists({"a": ["x"]})
        assert ExpandedCorpus(corpus, lambda q: ["b"]).search(query, k) == [] and corpus.asked == []


class TestLLMRewordings:
    def test_asks_for_queries_and_returns_the_strings(self):
        client = FakeClient('```json\n{"queries": ["corporate hierarchies", 3, "decentralization"]}\n```')
        assert LLMRewordings(client, 2)("organizational structure") == [
            "corporate hierarchies",
            "decentralization",
        ]
        assert (
            "organizational structure" in client.seen[0][0]["content"]
            and "2 other" in client.seen[0][0]["content"]
        )

    @pytest.mark.parametrize("reply", ["not json", "[]", '{"queries": "oops"}', '{"other": 1}', "null"])
    def test_an_unusable_reply_gives_no_rewordings(self, reply):
        assert LLMRewordings(FakeClient(reply))("q") == []


class TestLLMConfoundsAndSynonyms:
    def test_the_model_is_asked_for_confounds_of_this_relationship(self):
        from confoundsearch import LLMConfounds

        client = FakeClient('{"confounds": ["firm size", 7, "firm age"]}')
        assert LLMConfounds(client, 5)("IT investment", "firm performance") == ["firm size", "firm age"]
        prompt = client.seen[0][0]["content"]
        assert '"IT investment"' in prompt and '"firm performance"' in prompt and "List the 5 most" in prompt

    def test_the_model_is_asked_for_synonyms_of_one_term(self):
        from confoundsearch import LLMSynonyms

        client = FakeClient('```json\n{"synonyms": ["head count", "workforce size"]}\n```')
        assert LLMSynonyms(client, 2)("firm size") == ["head count", "workforce size"]
        assert (
            'for "firm size"' in client.seen[0][0]["content"]
            and "List 2 synonyms" in client.seen[0][0]["content"]
        )

    @pytest.mark.parametrize("reply", ["not json", "[]", '{"confounds": "oops"}', '{"other": 1}', "null"])
    def test_an_unusable_reply_gives_nothing_to_look_for_and_no_synonyms(self, reply):
        from confoundsearch import LLMConfounds, LLMSynonyms

        assert LLMConfounds(FakeClient(reply))("x", "y") == []
        assert LLMSynonyms(FakeClient(reply))("x") == []
