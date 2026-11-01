from __future__ import annotations

import json

import pytest

from confoundsearch import AddConfound, AddEdge, AttachPassage, Environment, Passage, Retrieve, State, Vertex
from confoundsearch.interface.export import SCHEMA, export_zip, plain, save_export, to_share_data
from tests.conftest import DV, IV, P1, P2, P3, StubJudge


def play(env: Environment, *actions) -> State:
    state = env.initial_state(IV, DV)
    for action in actions:
        state = env.step(state, action)
    return state


FULL = (
    AddConfound("firm size", frozenset({"firm size", "head count"})),
    AddEdge("firm size", "X"),
    AddEdge("firm size", "Y"),
    Retrieve("larger firms"),
    AttachPassage("firm size", "X", "p1"),
    AttachPassage("firm size", "Y", "p2"),
)


def test_empty_state(env):
    data = to_share_data(env.initial_state(IV, DV))
    assert data["schema"] == SCHEMA and data["mode"] == "explore"
    assert (data["iv"], data["dv"]) == (IV, DV)
    assert data["dag_confounders"] == [] and data["all_md_names"] == [] and data["phi"] == 0.0
    assert data["dag_grouped"] == {"meta_dimensions": []} and data["done"] is False


def test_vertex_becomes_a_meta_dimension_with_one_confounder(env):
    data = to_share_data(play(env, *FULL))
    (conf,) = data["dag_confounders"]
    (md,) = data["dag_grouped"]["meta_dimensions"]
    assert conf["name"] == md["name"] == "firm size"
    assert md["groups"] == [{"name": "Evidence", "description": "", "confounders": ["firm size"]}]
    assert md["description"] == "Cluster of strings: firm size, head count."


def test_evidence_is_split_by_leg_and_labelled_yes(env):
    (conf,) = to_share_data(play(env, *FULL))["dag_confounders"]
    assert [e["llm_label"] for e in conf["evidence_to_iv"]] == ["Yes"]
    assert conf["evidence_to_iv"][0]["quote"] == P1.text
    assert [e["quote"] for e in conf["evidence_to_dv"]] == [P2.text]


def test_only_confounds_with_support_on_both_legs_are_listed_in_all_md_names(env):
    half = play(env, *FULL[:5])  # evidence on X only
    assert to_share_data(half)["all_md_names"] == []
    assert to_share_data(play(env, *FULL))["all_md_names"] == ["firm size"]


def test_rejected_passages_are_exported_as_no():
    from confoundsearch import InMemoryCorpus

    env = Environment(InMemoryCorpus([P1, P2, P3]), StubJudge(set()))
    state = play(env, *FULL)
    (conf,) = to_share_data(state)["dag_confounders"]
    assert [e["llm_label"] for e in conf["evidence_to_iv"]] == ["No"]
    assert to_share_data(state)["all_md_names"] == []


def test_title_defaults_to_the_start_of_the_text_and_urls_must_be_http(env):
    state = play(env, AddConfound("a", frozenset({"a"})), AddEdge("a", "X"))
    passage = Passage("z", "x" * 200, title="", url="javascript:alert(1)")
    state = state.with_observed(frozenset({passage}))
    graph = state.graph.attach("a", "X", passage)
    entry = to_share_data(state.with_graph(graph))["dag_confounders"][0]["evidence_to_iv"][0]
    assert entry["title"] == "x" * 80 and entry["url"] == ""
    ok = Passage("y", "t", title="T", url="https://example.org/p")
    state = state.with_graph(graph.add_edge("a", "Y").attach("a", "Y", ok))
    assert to_share_data(state)["dag_confounders"][0]["evidence_to_dv"][0]["url"] == "https://example.org/p"


def test_names_reaching_the_page_cannot_carry_markup(env):
    state = State.initial("<b>x</b>", 'y"onload="z')
    state = state.with_graph(
        state.graph.add_confound(Vertex("<img src=x onerror=1>", frozenset({"<s>a</s>"})))
    )
    data = to_share_data(state)
    blob = json.dumps(data)
    assert "<" not in blob.replace("\\u003c", "") and 'onload="' not in blob


@pytest.mark.parametrize(
    "raw,clean",
    [("a<b>c", "abc"), ('say "hi"', "say \u201dhi\u201d"), ("it's", "it\u2019s"), ("plain", "plain")],
)
def test_plain(raw, clean):
    assert plain(raw) == clean


def test_done_and_phi_are_reported(env):
    state = play(env, *FULL).stopped()
    data = to_share_data(state)
    assert data["done"] is True and data["phi"] > 1.0
    json.dumps(data)  # serializable


class TestSaveExport:
    def test_json(self, env, tmp_path):
        state = play(env, *FULL)
        target = save_export(state, tmp_path / "run.json")
        assert target == tmp_path / "run.json" and json.loads(
            target.read_text(encoding="utf-8")
        ) == to_share_data(state)

    def test_zip_holds_share_data_json_like_the_gradio_exports(self, env, tmp_path):
        import zipfile

        state = play(env, *FULL)
        target = save_export(state, str(tmp_path / "run.zip"))
        with zipfile.ZipFile(target) as archive:
            assert archive.namelist() == ["share_data.json"]
            assert json.loads(archive.read("share_data.json")) == to_share_data(state)

    def test_export_zip_is_just_the_bytes(self):
        import io
        import zipfile

        with zipfile.ZipFile(io.BytesIO(export_zip({"a": "é"}))) as archive:
            assert json.loads(archive.read("share_data.json")) == {"a": "é"}

    @pytest.mark.parametrize("name,shown", [("run.html", ".html"), ("run", "a file with no extension")])
    def test_other_formats_are_refused_with_a_clear_message(self, env, tmp_path, name, shown):
        with pytest.raises(ValueError, match=shown):
            save_export(play(env), tmp_path / name)
        assert not (tmp_path / name).exists()


class TestExportWithAudit:
    def test_the_zip_can_carry_the_audit_trail(self, env, tmp_path):
        import zipfile

        from confoundsearch import AuditLog

        log = AuditLog()
        log.record("note", n=1)
        target = save_export(play(env), tmp_path / "run.zip", audit=log)
        with zipfile.ZipFile(target) as archive:
            assert sorted(archive.namelist()) == ["audit.jsonl", "share_data.json"]
            assert archive.read("audit.jsonl").decode() == log.to_jsonl()

    def test_without_an_audit_the_zip_is_unchanged(self):
        import io
        import zipfile

        with zipfile.ZipFile(io.BytesIO(export_zip({"a": 1}))) as archive:
            assert archive.namelist() == ["share_data.json"]

    def test_a_json_export_refuses_an_audit_trail(self, env, tmp_path):
        from confoundsearch import AuditLog

        with pytest.raises(ValueError, match=r"audit trail goes in a \.zip"):
            save_export(play(env), tmp_path / "run.json", audit=AuditLog())
        assert not (tmp_path / "run.json").exists()
