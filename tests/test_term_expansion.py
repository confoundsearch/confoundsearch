"""Expanding a construct into a long list of query terms: the model's variants and the vector neighbours."""

import json
from dataclasses import dataclass, field
from typing import ClassVar

import pytest

from confoundsearch import (
    Expansion,
    KeywordExpander,
    LLMError,
    LLMTermVariants,
    RemoteError,
    Term,
    expand_terms,
)
from confoundsearch.errors import VectorStoreError
from confoundsearch.model.expansion import ANGLES


class Chat:
    """A model that answers each question with the next reply, and remembers the questions."""

    def __init__(self, *replies):
        self.replies, self.asked = list(replies), []

    def chat(self, messages):
        self.asked.append(messages[0]["content"])
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def terms(*names):
    return json.dumps({"terms": list(names)})


class TestLLMTermVariants:
    def test_each_round_asks_from_a_different_angle(self):
        chat = Chat(*[terms(f"t{i}") for i in range(5)])
        LLMTermVariants(chat)("firm size")
        assert len(chat.asked) == 5
        for question, (_, what) in zip(chat.asked, ANGLES):
            assert '"firm size"' in question and what in question
        assert len(set(chat.asked)) == 5

    def test_more_rounds_than_angles_start_over(self):
        chat = Chat(*[terms("x")] * 7)
        LLMTermVariants(chat, rounds=7)("firm size")
        assert chat.asked[5] == chat.asked[0] and chat.asked[6] == chat.asked[1]

    def test_the_setting_is_part_of_the_question_when_given(self):
        chat = Chat(terms("x"))
        LLMTermVariants(chat, rounds=1, context="It is studied as a cause of organizational structure.")(
            "firm size"
        )
        assert "studied as a cause of organizational structure" in chat.asked[0]

    def test_the_answers_are_pooled_cleaned_and_deduplicated(self):
        chat = Chat(
            terms("Company Size", "company size", "number of employees", "  spaced   out  "),
            terms("FIRM SIZE", "x" * 81, "", "headcount"),
            "not json at all",
        )
        got = LLMTermVariants(chat, rounds=3)("firm size")
        assert got == [
            "Company Size",
            "number of employees",
            "spaced out",
            "headcount",
        ]  # not the construct, no repeats

    def test_a_failed_round_is_skipped_and_only_all_failing_is_an_error(self):
        chat = Chat(LLMError("busy"), terms("headcount"), LLMError("busy again"))
        assert LLMTermVariants(chat, rounds=3)("firm size") == ["headcount"]
        with pytest.raises(LLMError, match="down"):
            LLMTermVariants(Chat(LLMError("down")), rounds=1)("firm size")

    def test_no_rounds_asks_nothing(self):
        chat = Chat()
        assert LLMTermVariants(chat, rounds=0)("firm size") == [] and chat.asked == []

    def test_unusable_answers_give_an_empty_list_not_an_error(self):
        assert LLMTermVariants(Chat("[]", '{"terms": "x"}'), rounds=2)("firm size") == []


@dataclass
class Hit:
    id: str
    score: float
    metadata: dict = field(default_factory=dict)


def hit(name, score, works=100):
    return Hit(name.replace(" ", "-"), score, {"name": name, "works": works})


class Index:
    """A vector index that answers each term with a fixed list, and remembers what it was asked."""

    def __init__(self, answers):
        self.answers, self.asked = answers, []

    def __call__(self, term, k):
        self.asked.append((term, k))
        answer = self.answers.get(term)
        if isinstance(answer, Exception):
            raise answer
        return answer if answer is not None else []


class TestExpandTerms:
    def test_the_construct_comes_first_then_the_models_terms_then_the_neighbours(self):
        index = Index(
            {
                "firm size": [hit("large firms", 0.8), hit("enterprise scale", 0.7)],
                "headcount": [hit("staff numbers", 0.6)],
            }
        )
        result = expand_terms("  firm   size ", lambda c: ["headcount", "Firm Size"], index)
        assert isinstance(result, Expansion) and result.construct == "firm size"
        assert [(t.text, t.kind) for t in result.terms] == [
            ("firm size", "construct"),
            ("headcount", "synonym"),  # 'Firm Size' is the construct again: dropped
            ("large firms", "neighbor"),  # first for 'firm size', better scored than ...
            ("staff numbers", "neighbor"),  # ... first for 'headcount': an equal rank breaks the tie by score
            ("enterprise scale", "neighbor"),  # second for 'firm size'
        ]
        assert result.counts() == {"construct": 1, "synonym": 1, "neighbor": 3}

    def test_every_term_is_asked_for_its_neighbours_with_the_same_k(self):
        index = Index({})
        expand_terms("a", lambda c: ["b", "c"], index, per_term=37)
        assert sorted(index.asked) == [("a", 37), ("b", 37), ("c", 37)]

    def test_a_keyword_many_terms_bring_up_ranks_above_one_only_one_does(self):
        index = Index(
            {
                "a": [hit("only a", 0.99), hit("shared", 0.5)],
                "b": [hit("shared", 0.6), hit("only b", 0.58)],
                "c": [hit("shared", 0.4)],
            }
        )
        names = [t.text for t in expand_terms("a", lambda c: ["b", "c"], index).terms if t.kind == "neighbor"]
        assert names[0] == "shared" and names[1:] == ["only a", "only b"]

    def test_a_neighbour_keeps_its_best_score_its_papers_and_who_found_it(self):
        index = Index({"a": [hit("shared", 0.5, works=70)], "b": [hit("shared", 0.9, works=70)]})
        (term,) = [t for t in expand_terms("a", lambda c: ["b"], index).terms if t.kind == "neighbor"]
        assert term == Term("shared", "neighbor", 0.9, 70, ("a", "b"))

    def test_a_neighbour_that_is_already_a_term_is_not_listed_twice(self):
        index = Index({"a": [hit("A", 1.0), hit("b", 0.9), hit("new", 0.8)]})
        assert [t.text for t in expand_terms("a", lambda c: ["b"], index).terms] == ["a", "b", "new"]

    def test_the_list_is_cut_at_the_limit_but_always_keeps_the_construct(self):
        index = Index({"a": [hit(f"n{i}", 1 - i / 100) for i in range(20)]})
        assert len(expand_terms("a", lambda c: [], index, limit=7).terms) == 7
        assert [t.text for t in expand_terms("a", lambda c: [], index, limit=0).terms] == ["a"]

    def test_a_term_that_fails_is_recorded_and_the_others_still_count(self):
        index = Index(
            {
                "a": [hit("fine", 0.9)],
                "b": VectorStoreError("service down"),
                "c": LookupError("not a keyword"),
            }
        )
        result = expand_terms("a", lambda c: ["b", "c"], index)
        assert [t.text for t in result.terms if t.kind == "neighbor"] == ["fine"]
        assert result.skipped == {"b": "service down", "c": "not a keyword"}

    def test_no_model_still_gives_the_vector_neighbours(self):
        def broken(construct):
            raise LLMError("no key")

        result = expand_terms("a", broken, Index({"a": [hit("fine", 0.9)]}))
        assert [t.text for t in result.terms] == ["a", "fine"] and "no key" in result.skipped["synonyms"]

    def test_odd_metadata_is_tolerated(self):
        index = Index({"a": [Hit("no-name", 0.9, {}), Hit("x", 0.8, {"name": "papers?", "works": "many"})]})
        got = {t.text: t.works for t in expand_terms("a", lambda c: [], index).terms if t.kind == "neighbor"}
        assert got == {"no-name": None, "papers?": None}  # the id stands in for a missing name

    def test_nothing_to_expand_is_an_error(self):
        with pytest.raises(ValueError, match="no construct"):
            expand_terms("   ", lambda c: [], Index({}))

    def test_the_order_does_not_depend_on_which_thread_answers_first(self):
        index = Index({"a": [hit("x", 0.5)], "b": [hit("y", 0.5)], "c": [hit("z", 0.5)]})
        first = [t.text for t in expand_terms("a", lambda c: ["b", "c"], index, workers=1).terms]
        for _ in range(5):
            assert [t.text for t in expand_terms("a", lambda c: ["b", "c"], index, workers=8).terms] == first

    def test_remote_errors_from_the_models_step_are_remote_errors(self):
        assert issubclass(LLMError, RemoteError)  # which is what the graceful path catches

    def test_a_keyword_listed_twice_by_one_term_counts_for_each_listing_but_names_the_term_once(self):
        index = Index({"a": [hit("dup", 0.9), hit("Dup", 0.8)]})
        (term,) = [t for t in expand_terms("a", lambda c: [], index).terms if t.kind == "neighbor"]
        assert term.found_via == ("a",) and term.score == 0.9


class TestKeywordExpander:
    def test_it_is_made_once_and_used_for_any_number_of_constructs(self):
        from confoundsearch import KeywordExpander

        index = Index({"a": [hit("near a", 0.9)], "b": [hit("near b", 0.8)]})
        expander = KeywordExpander(lambda c: [], index, per_term=7, limit=50, workers=2)
        assert (expander.per_term, expander.limit, expander.workers) == (7, 50, 2)
        assert expander.expand("a").texts() == ["a", "near a"]
        assert expander("b").texts() == ["b", "near b"]  # calling it is expanding
        assert index.asked == [("a", 7), ("b", 7)]

    def test_the_settings_belong_to_the_object_so_each_expander_can_differ(self):
        from confoundsearch import KeywordExpander

        index = Index({"a": [hit(f"n{i}", 1 - i / 100) for i in range(10)]})
        small = KeywordExpander(lambda c: [], index, limit=3)
        large = KeywordExpander(lambda c: [], index, limit=50)
        assert len(small("a").terms) == 3 and len(large("a").terms) == 11

    def test_the_function_is_the_same_thing_in_one_call(self):
        from confoundsearch import KeywordExpander

        index = Index({"a": [hit("x", 0.5)]})
        assert expand_terms("a", lambda c: ["b"], index) == KeywordExpander(lambda c: ["b"], index).expand(
            "a"
        )

    def test_texts_lists_just_the_terms_in_order(self):
        index = Index({"a": [hit("x", 0.5)]})
        assert expand_terms("a", lambda c: ["b"], index).texts() == ["a", "b", "x"]


# -- a model that answers by what it is asked ---------------------------------------------------------------


class Model:
    """Answers the angle and field questions with terms named for the question, and the synonym questions
    with one synonym per listed term. Remembers every question."""

    def __init__(self, fail=()):
        self.asked, self.fail = [], list(fail)

    def chat(self, messages):
        question = messages[0]["content"]
        self.asked.append(question)
        for marker in self.fail:
            if marker in question:
                raise LLMError(f"cannot answer: {marker}")
        if '"synonyms": {' in question:
            listed = [line[2:] for line in question.splitlines() if line.startswith("- ")]
            return json.dumps({"synonyms": {term: [f"{term} alias"] for term in listed}})
        if '"hypernyms": {' in question:
            listed = [line[2:] for line in question.splitlines() if line.startswith("- ")]
            return json.dumps({"hypernyms": {term: [f"{term} category"] for term in listed}})
        if "channels or mechanisms" in question:
            return terms("delegation channel", "shared proxy")
        if "observable attributes" in question:
            return terms("tenure", "shared proxy")
        if "stand-in" in question:  # a field question
            field = question.split("papers in ")[1].split(" use for it")[0]
            return terms(f"{field} proxy", "shared proxy")
        return terms("angle term", "shared proxy")


class TestWideNet:
    def test_field_variants_ask_each_field_with_the_free_cash_flow_example(self):
        from confoundsearch.model.expansion import EXAMPLE, FIELDS, LLMFieldVariants

        model = Model()
        got = LLMFieldVariants(model, fields=("finance", "sociology"), per_field=5, context="Setting here.")(
            "firm size"
        )
        assert got == ["finance proxy", "shared proxy", "sociology proxy"]  # in field order, no repeats
        assert len(model.asked) == 2 and all(EXAMPLE in q and "free cash flow" in q for q in model.asked)
        assert all('"firm size"' in q and "up to 5" in q and "Setting here." in q for q in model.asked)
        assert (
            "papers in finance use for it" in model.asked[0]
            and "papers in sociology use for it" in model.asked[1]
        )
        assert len(FIELDS) == 9 and "finance" in FIELDS and "sociology" in FIELDS

    def test_a_field_that_fails_is_skipped_and_only_all_failing_is_an_error(self):
        from confoundsearch.model.expansion import LLMFieldVariants

        got = LLMFieldVariants(Model(fail=["papers in finance"]), fields=("finance", "sociology"))(
            "firm size"
        )
        assert got == ["sociology proxy", "shared proxy"]
        with pytest.raises(LLMError, match="cannot answer"):
            LLMFieldVariants(Model(fail=["stand-in"]), fields=("finance", "sociology"))("firm size")

    def test_the_construct_itself_is_not_a_variant(self):
        from confoundsearch.model.expansion import LLMFieldVariants

        class Echo:
            def chat(self, messages):
                return terms("Firm Size", "headcount")

        assert LLMFieldVariants(Echo(), fields=("finance",))("firm size") == ["headcount"]

    def test_synonyms_of_variants_are_asked_in_batches_and_pooled(self):
        from confoundsearch.model.expansion import LLMTermSynonyms

        model = Model()
        listed = [f"v{i}" for i in range(25)]
        got = LLMTermSynonyms(model, per_term=3, batch=10)(listed)
        assert len(model.asked) == 3 and "up to 3" in model.asked[0]  # 10 + 10 + 5
        assert got == [f"v{i} alias" for i in range(25)]  # and none of the listed terms themselves

    def test_a_failed_batch_is_skipped_and_all_failing_is_an_error(self):
        from confoundsearch.model.expansion import LLMTermSynonyms

        assert LLMTermSynonyms(Model(fail=["- v0"]), batch=1)(["v0", "v1"]) == ["v1 alias"]
        with pytest.raises(LLMError):
            LLMTermSynonyms(Model(fail=["synonyms"]), batch=1)(["v0", "v1"])

    @pytest.mark.parametrize(
        "reply", ["not json", "[]", '{"synonyms": []}', '{"synonyms": {"a": "x"}}', "{}"]
    )
    def test_unusable_synonym_replies_give_nothing(self, reply):
        from confoundsearch.model.expansion import LLMTermSynonyms

        class Fixed:
            def chat(self, messages):
                return reply

        assert LLMTermSynonyms(Fixed())(["a"]) == []

    def test_nothing_to_ask_about_asks_nothing(self):
        from confoundsearch.model.expansion import LLMTermSynonyms

        model = Model()
        assert LLMTermSynonyms(model)([]) == [] and model.asked == []

    def test_the_net_runs_the_three_stages_and_labels_where_each_term_came_from(self):
        from confoundsearch.model.expansion import LLMWideNet

        net = LLMWideNet(Model(), rounds=1, fields=("finance",), per_field=2, per_variant=1)
        assert net("firm size") == [
            ("angle term", "synonym"),
            ("shared proxy", "synonym"),
            ("finance proxy", "variant"),  # 'shared proxy' is already taken by the angles
            ("firm size category", "hypernym"),
            ("finance proxy category", "hypernym"),
            ("finance proxy alias", "variant synonym"),
            ("delegation channel", "mechanism"),
            ("tenure", "attribute"),
        ]
        assert net.problems == []

    def test_a_stage_that_fails_is_noted_and_the_others_still_count(self):
        from confoundsearch.model.expansion import LLMWideNet

        net = LLMWideNet(Model(fail=["stand-in"]), rounds=1, fields=("finance",))
        assert [kind for _, kind in net("firm size")] == [
            "synonym",
            "synonym",
            "hypernym",
            "mechanism",
            "attribute",
        ]
        assert len(net.problems) == 1 and net.problems[0].startswith("field variants: cannot answer")

    def test_all_stages_failing_is_an_error_that_names_them(self):
        from confoundsearch.model.expansion import LLMWideNet

        class Down:
            def chat(self, messages):
                raise LLMError("no key")

        net = LLMWideNet(Down(), rounds=1, fields=("finance",))
        with pytest.raises(LLMError, match="angles: no key; field variants: no key"):
            net("firm size")

    def test_it_asks_for_the_channels_the_idea_works_through_and_marks_them(self):
        from confoundsearch.model.expansion import LLMWideNet

        pairs = LLMWideNet(Model(), rounds=1, fields=(), per_hypernym=0)("firm size")
        assert ("delegation channel", "mechanism") in pairs

    def test_it_asks_for_the_characteristics_papers_measure_and_marks_them(self):
        from confoundsearch.model.expansion import LLMWideNet

        pairs = LLMWideNet(Model(), rounds=1, fields=(), per_hypernym=0, per_mechanism=0)("firm size")
        assert ("tenure", "attribute") in pairs
        none = LLMWideNet(Model(), rounds=1, fields=(), per_hypernym=0, per_mechanism=0, per_attribute=0)
        assert ("tenure", "attribute") not in none("firm size")

    def test_a_failing_mechanisms_stage_is_noted_and_skipped(self):
        from confoundsearch.model.expansion import LLMWideNet

        net = LLMWideNet(
            Model(fail=["channels or mechanisms"]), rounds=1, fields=(), per_hypernym=0, per_attribute=0
        )
        assert [k for _, k in net("firm size")] == ["synonym", "synonym"]
        assert net.problems and net.problems[0].startswith("mechanisms:")

    def test_switching_the_field_steps_off(self):
        from confoundsearch.model.expansion import LLMWideNet

        model = Model()
        kinds = [
            k
            for _, k in LLMWideNet(model, rounds=1, fields=(), per_mechanism=0, per_attribute=0)("firm size")
        ]
        assert kinds == ["synonym", "synonym", "hypernym"] and len(model.asked) == 2
        model = Model()
        kinds = [
            k
            for _, k in LLMWideNet(
                model, rounds=1, fields=(), per_hypernym=0, per_mechanism=0, per_attribute=0
            )("firm size")
        ]
        assert kinds == ["synonym", "synonym"] and len(model.asked) == 1
        model = Model()
        kinds = [
            k
            for _, k in LLMWideNet(
                model, rounds=1, fields=("finance",), per_variant=0, per_mechanism=0, per_attribute=0
            )("firm size")
        ]
        assert kinds == ["synonym", "synonym", "variant", "hypernym", "hypernym"]
        assert len(model.asked) == 3  # no synonym questions
        model = Model()
        LLMWideNet(model, rounds=1, fields=("finance",), per_field=0, per_mechanism=0, per_attribute=0)(
            "firm size"
        )
        assert len(model.asked) == 2  # the angles and the construct's hypernyms


class TestHypernyms:
    def test_asks_for_broader_terms_in_batches(self):
        from confoundsearch.model.expansion import LLMHypernyms

        model = Model()
        assert LLMHypernyms(model, per_term=2, batch=1)(["a", "b"]) == ["a category", "b category"]
        assert len(model.asked) == 2 and "broader terms" in model.asked[0] and "up to 2" in model.asked[0]


class TestNeighborSynonyms:
    def run(self, hop, **kw):
        index = Index({"a": [hit("n1", 0.9, works=5), hit("n2", 0.8, works=5)]})
        return KeywordExpander(lambda c: [], index, neighbor_synonyms=hop, **kw).expand("a")

    def test_synonyms_of_the_best_neighbours_join_the_list_once(self):
        asked = []

        def hop(names):
            asked.append(names)
            return ["n1 alias", "N1 Alias", "a", "n2", "  "]

        result = self.run(hop, neighbor_top=2)
        assert asked == [["n1", "n2"]]
        assert result.texts() == ["a", "n1", "n2", "n1 alias"]  # the repeats, echoes and blanks are dropped
        assert result.terms[-1].kind == "neighbor synonym" and result.counts()["neighbor synonym"] == 1

    def test_only_the_top_neighbours_are_asked_about(self):
        asked = []
        self.run(lambda names: asked.append(names) or [], neighbor_top=1)
        assert asked == [["n1"]]

    def test_a_failing_model_is_noted_and_the_neighbours_stay(self):
        def hop(names):
            raise LLMError("no key")

        result = self.run(hop)
        assert result.texts() == ["a", "n1", "n2"] and "no key" in result.skipped["neighbor synonyms"]

    def test_off_when_not_given_or_zero_or_nothing_was_found(self):
        called = []
        assert self.run(None).texts() == ["a", "n1", "n2"]
        self.run(lambda names: called.append(1) or [], neighbor_top=0)
        KeywordExpander(lambda c: [], Index({}), neighbor_synonyms=lambda n: called.append(1) or []).expand(
            "a"
        )
        assert called == []


class TestTaggedSeeds:
    def test_pairs_keep_their_kind_and_strings_are_synonyms(self):
        index = Index({})
        result = expand_terms(
            "a", lambda c: ["plain", ("proxy", "variant"), ("proxy alias", "variant synonym")], index
        )
        assert [(t.text, t.kind) for t in result.terms] == [
            ("a", "construct"),
            ("plain", "synonym"),
            ("proxy", "variant"),
            ("proxy alias", "variant synonym"),
        ]
        assert result.counts() == {"construct": 1, "synonym": 1, "variant": 1, "variant synonym": 1}

    def test_no_more_terms_are_fetched_for_than_max_seeds_in_the_order_given(self):
        index = Index({})
        result = expand_terms("a", lambda c: [f"t{i}" for i in range(50)], index, max_seeds=4)
        assert [t.text for t in result.terms] == ["a", "t0", "t1", "t2"]
        assert sorted(term for term, _ in index.asked) == ["a", "t0", "t1", "t2"]
        assert [t.text for t in expand_terms("a", lambda c: ["b"], index, max_seeds=0).terms] == [
            "a"
        ]  # the construct stays

    def test_a_repeated_seed_is_taken_once_with_its_first_kind(self):
        result = expand_terms("a", lambda c: [("x", "variant"), ("X", "synonym")], Index({}))
        assert [(t.text, t.kind) for t in result.terms] == [("a", "construct"), ("x", "variant")]


# -- the command ---------------------------------------------------------------------------------------------


class FakeService:
    """Stands in for the keyword service: neighbours by term, a manifest, and a record of what was asked."""

    made: ClassVar[list] = []

    def __init__(self, address, embedder=None):
        self.address, self.embedder, self.max_k, self.asked = address, embedder, 100, []
        FakeService.made.append(self)

    def learn_limits(self):
        self.max_k = 40
        return self.max_k

    def query(self, term, k, min_works=0):
        self.asked.append((term, k, min_works))
        if term == "angle term":
            raise VectorStoreError("no keyword named 'angle term'")
        return [hit(f"near {term}", 0.9, works=500), hit("shared idea", 0.8, works=900)]


class TestExpandCommand:
    @pytest.fixture(autouse=True)
    def setup(self, monkeypatch):
        from confoundsearch.interface import cli

        FakeService.made = []
        self.model = Model()
        monkeypatch.setattr(cli, "KeywordService", FakeService)
        monkeypatch.setattr(cli, "build_client", lambda args: self.model)
        monkeypatch.setattr(cli, "service_url", lambda setting=None: setting or "https://svc.example.test")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        self.cli = cli

    def run(self, *argv):
        return self.cli.main(["expand", "firm size", "--rounds", "1", "--fields", "finance", *argv])

    def test_it_prints_one_term_per_line_and_a_summary_on_stderr(self, capsys):
        assert self.run() == 0
        out, err = capsys.readouterr()
        lines = out.splitlines()
        assert lines[0].split()[:2] == ["construct", "firm"]
        assert any(line.startswith("variant ") and "finance proxy" in line for line in lines)
        assert any(line.startswith("variant synonym") and "finance proxy alias" in line for line in lines)
        assert any(
            line.startswith("neighbor") and "near firm size" in line and "0.900" in line for line in lines
        )
        assert "1 construct, 2 synonym, 1 variant, 2 hypernym, 1 variant synonym" in err
        assert "skipped 'angle term': no keyword named" in err and "step(s) skipped" in err

    def test_per_term_is_held_to_what_the_service_allows_and_the_options_are_passed_on(self):
        self.run("--per-term", "1000", "--min-works", "250")
        service = FakeService.made[0]
        assert service.address == "https://svc.example.test"
        assert {k for _, k, _ in service.asked} == {40} and {w for _, _, w in service.asked} == {250}

    def test_the_field_options_reach_the_model(self):
        self.run("--per-field", "3", "--per-variant", "2", "--context", "Setting.")
        field_questions = [q for q in self.model.asked if "stand-in" in q]
        assert (
            len(field_questions) == 1 and "up to 3" in field_questions[0] and "Setting." in field_questions[0]
        )
        assert any('"synonyms": {' in q and "up to 2" in q for q in self.model.asked)

    def test_hypernym_and_neighbour_options_reach_the_model(self):
        self.run("--per-hypernym", "5", "--neighbor-top", "1", "--per-neighbor", "3")
        assert any('"hypernyms": {' in q and "up to 5" in q for q in self.model.asked)
        hop = [q for q in self.model.asked if '"synonyms": {' in q and "up to 3" in q]
        assert len(hop) == 1 and sum(line.startswith("- ") for line in hop[0].splitlines()) == 1

    def test_fields_none_skips_the_field_steps_and_max_seeds_caps_the_fetching(self):
        self.cli.main(["expand", "firm size", "--rounds", "1", "--fields", "none", "--max-seeds", "2"])
        assert not any("stand-in" in q for q in self.model.asked)
        assert len(FakeService.made[0].asked) == 2  # the construct and one more

    def test_json_and_out_hold_the_same_thing(self, capsys, tmp_path):
        out = tmp_path / "terms.json"
        assert self.run("--json", "--limit", "12", "--out", str(out)) == 0
        printed = json.loads(capsys.readouterr().out)
        saved = json.loads(out.read_text())
        assert printed == saved and printed["construct"] == "firm size" and len(printed["terms"]) == 12
        assert printed["counts"]["construct"] == 1 and "angle term" in printed["skipped"]
        kinds = {t["kind"] for t in printed["terms"]}
        assert {"construct", "synonym", "variant", "variant synonym"} <= kinds

    def test_a_phrase_that_is_not_a_keyword_can_be_embedded_with_your_own_key(self):
        self.run()
        assert FakeService.made[0].embedder.api_key == "sk-test"

    def test_a_problem_in_a_model_step_is_reported(self, capsys):
        self.model.fail = ["stand-in"]
        assert self.run() == 0
        assert "the model step had a problem: field variants" in capsys.readouterr().err

    def test_it_needs_a_model_and_a_service(self, monkeypatch, capsys):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        assert self.run() == 1 and "give --model" in capsys.readouterr().err
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setattr(self.cli, "service_url", lambda setting=None: None)
        assert self.run() == 1 and "has no address" in capsys.readouterr().err


class TestBuildExpander:
    class Service:
        def __init__(self):
            self.max_k, self.learned, self.asked = 100, 0, []

        def learn_limits(self):
            self.learned += 1
            self.max_k = 40
            return self.max_k

        def query(self, term, k, min_works=0):
            self.asked.append((term, k, min_works))
            return [hit(f"near {term}", 0.9, works=500)]

    def test_it_asks_the_service_its_limits_and_never_asks_for_more(self):
        from confoundsearch import KeywordExpander, build_expander

        service = self.Service()
        expander = build_expander(
            Model(),
            service,
            rounds=1,
            min_works=250,
            per_term=1000,
            limit=9,
            fields=(),
            per_variant=0,
            per_hypernym=0,
            per_neighbor=0,
        )
        assert isinstance(expander, KeywordExpander) and service.learned == 1
        assert (expander.per_term, expander.limit) == (40, 9)  # the service's limit, not 1000
        result = expander.expand("firm size")
        assert result.texts()[:3] == ["firm size", "angle term", "shared proxy"]
        assert all(k == 40 and works == 250 for _, k, works in service.asked)

    def test_the_wide_net_is_the_default_and_its_options_are_passed_on(self):
        from confoundsearch import build_expander

        model, service = Model(), self.Service()
        expander = build_expander(
            model, service, rounds=1, fields=("finance", "sociology"), per_field=3, per_variant=2, max_seeds=5
        )
        result = expander.expand("firm size")
        assert expander.max_seeds == 5 and len(service.asked) == 5  # capped, though more were named
        assert {"construct", "synonym", "variant"} <= set(result.counts()) and "stand-in" in " ".join(
            model.asked
        )

    def test_the_model_questions_use_the_rounds_and_context_given(self):
        from confoundsearch import build_expander

        model = Model()
        build_expander(
            model,
            self.Service(),
            rounds=2,
            per_round=3,
            context="A setting.",
            fields=(),
            per_hypernym=0,
            per_mechanism=0,
            per_attribute=0,
            per_neighbor=0,
        ).expand("firm size")
        assert len(model.asked) == 2 and all("A setting." in q and "up to 3" in q for q in model.asked)


class TestMeasuresAndMechanisms:
    def test_measures_are_asked_for_by_name_and_cleaned(self):
        from confoundsearch.model.expansion import LLMMeasures

        model = FakeReply(
            '{"terms": ["return on equity", "Return on Equity", "firm size", "labor productivity"]}'
        )
        assert LLMMeasures(model, 3)("firm size") == ["return on equity", "labor productivity"]
        assert '"firm size"' in model.asked[0] and "List 3" in model.asked[0]

    def test_mechanisms_are_asked_about_the_confound_on_a_variable_or_alone(self):
        from confoundsearch.model.expansion import LLMMechanisms

        model = FakeReply('{"terms": ["delegation of authority"]}')
        assert LLMMechanisms(model, 2)("CEO style", "organizational structure") == ["delegation of authority"]
        assert 'could "CEO style" affect "organizational structure"?' in model.asked[0]
        LLMMechanisms(model, 2)("CEO style")
        assert 'could "CEO style" work?' in model.asked[1]


class TestAttributes:
    def test_attributes_are_asked_for_by_name_and_cleaned(self):
        from confoundsearch.model.expansion import LLMAttributes

        model = FakeReply('{"terms": ["CEO tenure", "ceo tenure", "CEO style", "functional background"]}')
        assert LLMAttributes(model, 3)("CEO style") == ["CEO tenure", "functional background"]
        assert 'of "CEO style"' in model.asked[0] and "List 3" in model.asked[0]


class FakeReply:
    def __init__(self, reply):
        self.reply, self.asked = reply, []

    def chat(self, messages):
        self.asked.append(messages[0]["content"])
        return self.reply
