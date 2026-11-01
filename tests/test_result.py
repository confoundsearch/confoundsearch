from __future__ import annotations

import json

from confoundsearch import (
    AddConfound,
    AddEdge,
    AttachPassage,
    AuditedClient,
    Environment,
    InMemoryCorpus,
    Passage,
    Retrieve,
    Stop,
)
from confoundsearch.interface.result import QUOTE_CHARS, run_result
from confoundsearch.logic.episode import run_episode
from tests.conftest import DV, IV, P1, P2, StubJudge
from tests.test_episode import SCRIPT, Scripted


def play(env, script=SCRIPT):
    agent = Scripted(*script)
    return run_episode(env, agent, agent, IV, DV)


def test_the_result_names_the_question_the_ending_and_the_score(env):
    result = run_result(play(env))
    assert result["question"] == {"independent_variable": IV, "dependent_variable": DV}
    assert result["ended"] == "stop_action" and result["steps"] == len(SCRIPT) and result["score"] > 1.0
    assert set(result) == {
        "question",
        "ended",
        "steps",
        "score",
        "confounds",
        "model_calls",
        "models",
        "files",
        "audit",
    }


def test_it_is_plain_json(env):
    assert json.loads(json.dumps(run_result(play(env))))["steps"] == len(SCRIPT)


def test_each_supported_confound_lists_its_evidence_with_provenance(env):
    (confound,) = run_result(play(env))["confounds"]
    assert confound["label"] == "firm size" and confound["phrasings"] == ["firm size"]
    assert confound["linked_to_both_variables"] and confound["supported_on_both_links"]
    assert confound["rejected_passages"] == 0
    (to_iv,), (to_dv,) = confound["evidence_to_iv"], confound["evidence_to_dv"]
    assert to_iv == {
        "id": "p1",
        "title": "",
        "url": "",
        "quote": P1.text,
        "found_by_query": "larger firms",
        "judged_at_step": 5,
    }
    assert to_dv["id"] == "p2" and to_dv["quote"] == P2.text and to_dv["judged_at_step"] == 6


def test_a_confound_linked_but_unsupported_is_reported_as_such(corpus):
    env = Environment(corpus, StubJudge({"p1"}))  # the judge refuses p2
    (confound,) = run_result(play(env))["confounds"]
    assert confound["linked_to_both_variables"] and not confound["supported_on_both_links"]
    assert len(confound["evidence_to_iv"]) == 1 and confound["evidence_to_dv"] == []
    assert confound["rejected_passages"] == 1


def test_a_confound_with_no_evidence_at_all_still_appears(env):
    script = [AddConfound("a", frozenset({"a"})), AddEdge("a", "X"), AddEdge("a", "Y"), Stop()]
    (confound,) = run_result(play(env, script))["confounds"]
    assert confound["linked_to_both_variables"] and not confound["supported_on_both_links"]
    assert (
        confound["evidence_to_iv"] == confound["evidence_to_dv"] == [] and confound["rejected_passages"] == 0
    )


def test_supported_confounds_come_first_then_the_better_evidenced(env):
    def confound(label, strings, edges, attach):
        return [AddConfound(label, frozenset(strings)), *[AddEdge(label, t) for t in edges], *attach]

    script = [
        *confound("weak", {"weak"}, ["X", "Y"], []),
        *confound("zeta", {"zeta"}, ["X", "Y"], []),
        *confound("firm size", {"firm size"}, ["X", "Y"], []),
        Retrieve("larger firms"),
        AttachPassage("firm size", "X", "p1"),
        AttachPassage("firm size", "Y", "p2"),
        Stop(),
    ]
    labels = [c["label"] for c in run_result(play(env, script))["confounds"]]
    assert labels == ["firm size", "weak", "zeta"]  # supported first, then alphabetical among the rest


def test_long_passages_are_quoted_in_part_and_short_ones_whole():
    long_text = "word " * 400
    corpus = InMemoryCorpus([Passage("L", long_text), Passage("S", "short word")])
    env = Environment(corpus, StubJudge({"L", "S"}))
    script = [
        AddConfound("c", frozenset({"c"})), AddEdge("c", "X"), Retrieve("word"),
        AttachPassage("c", "X", "L"), AttachPassage("c", "X", "S"), Stop(),
    ]  # fmt: skip
    quotes = {p["id"]: p["quote"] for p in run_result(play(env, script))["confounds"][0]["evidence_to_iv"]}
    assert quotes["S"] == "short word"
    assert quotes["L"].endswith("...") and len(quotes["L"]) <= QUOTE_CHARS + 3


def test_files_are_reported_when_given(env, tmp_path):
    result = run_result(play(env), tmp_path / "page.html", tmp_path / "log.jsonl")
    assert result["files"] == {"page": str(tmp_path / "page.html"), "audit": str(tmp_path / "log.jsonl")}
    assert run_result(play(env))["files"] == {"page": None, "audit": None}


def test_the_audit_status_comes_with_the_hash_to_keep(env):
    trajectory = play(env)
    audit = run_result(trajectory)["audit"]
    assert audit == {
        "events": len(trajectory.audit.events),
        "intact": True,
        "complete": True,
        "head": trajectory.audit.head,
    }


def test_model_calls_and_names_are_counted_from_the_log(env):
    class Model:
        model = "test-model"

        def chat(self, messages):
            return "ok"

    client = AuditedClient(Model())

    class Asks(Scripted):
        def propose(self, state):
            client.chat([{"role": "user", "content": "?"}])
            return super().propose(state)

    agent = Asks(*SCRIPT)
    result = run_result(run_episode(env, agent, agent, IV, DV))
    assert result["model_calls"] == len(SCRIPT) and result["models"] == ["test-model"]


def test_a_run_that_did_nothing_has_no_confounds(env):
    class Nothing:
        def propose(self, state):
            return []

    result = run_result(run_episode(env, Nothing(), Nothing(), IV, DV))
    assert result["confounds"] == [] and result["ended"] == "no_actions" and result["steps"] == 0
